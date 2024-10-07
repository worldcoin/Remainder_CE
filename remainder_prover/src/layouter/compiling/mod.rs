//! This is a module for constructing an [InstantiatedCircuit] given a
//! [LayouterCircuit], which is a closure provided by the circuit builder that
//! takes in one parameter, the context, along with all of the "circuit
//! builders" necessary in order to represent the data-dependency relationships
//! within the circuit. Additionally, the circuit builder in this closure also
//! provides the necessary input data to populate the circuits with.
//!
//! The important distinction in this compilation process is that first, using
//! the data-dependency relationships in the circuit, a [GKRCircuitDescription]
//! is generated, which represents the data-dependencies in the circuit along
//! with the shape of the circuit itself (the number of variables in the
//! different MLEs that make up a layer).
//!
//! Then, using this circuit description along with the data inputs, the
//! `instantiate()` function will create an [InstantiatedCircuit], where now
//! the circuit description is "filled in" with its associated data.

#[cfg(test)]
mod tests;

use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use std::collections::HashMap;
use std::marker::PhantomData;
use tracing::{instrument, span, Level};

use crate::claims::wlx_eval::WLXAggregator;
use crate::claims::ClaimAggregator;
use crate::input_layer::enum_input_layer::InputLayerEnum;
use crate::input_layer::{InputLayer, InputLayerDescription};
use crate::layer::layer_enum::LayerEnum;
use crate::layer::Layer;
use crate::mle::evals::MultilinearExtension;
use crate::mle::mle_enum::MleEnum;
use crate::mle::Mle;
use crate::prover::{
    generate_circuit_description, GKRCircuitDescription, GKRError, InstantiatedCircuit,
};
use ark_std::{end_timer, start_timer};
use log::info;
use remainder_shared_types::transcript::{ProverTranscript, Transcript, TranscriptWriter};
use remainder_shared_types::Field;

use super::nodes::circuit_inputs::InputLayerNodeData;
use super::nodes::NodeId;
use super::{
    component::Component,
    nodes::{node_enum::NodeEnum, Context},
};

/// The struct used in order to aid with proving, which contains the function
/// used in order to generate the circuit description itself, called the
/// `witness_builder`, which can then be used to generate an
/// [InstantiatedCircuit].
pub struct LayouterCircuit<
    F: Field,
    C: Component<NodeEnum<F>>,
    Fn: FnMut(&Context) -> (C, Vec<InputLayerNodeData<F>>),
> {
    witness_builder: Fn,
    _marker: PhantomData<F>,
}

impl<
        F: Field,
        C: Component<NodeEnum<F>>,
        Fn: FnMut(&Context) -> (C, Vec<InputLayerNodeData<F>>),
    > LayouterCircuit<F, C, Fn>
{
    /// Constructs a `LayouterCircuit` by taking in a closure whose parameter is
    /// a [Context], which determines the ID of each individual part of the
    /// circuit whose data output will be computed for future layers. The
    /// function itself returns a [Component], which is a set of circuit "nodes"
    /// which will be used to build the circuit itself. The closure also returns
    /// a [Vec<InputLayerNodeData>], which is a vector of all of the data that will
    /// be fed into the input layers of the circuit.
    pub fn new(witness_builder: Fn) -> Self {
        Self {
            witness_builder,
            _marker: PhantomData,
        }
    }
}

impl<
        F: Field,
        C: Component<NodeEnum<F>>,
        Fn: FnMut(&Context) -> (C, Vec<InputLayerNodeData<F>>),
    > LayouterCircuit<F, C, Fn>
{
    fn synthesize_and_commit(
        &mut self,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> (InstantiatedCircuit<F>, GKRCircuitDescription<F>) {
        let ctx = Context::new();
        let (component, input_layer_data) = (self.witness_builder)(&ctx);

        // Convert the input layer data into a map that maps the input shred ID
        // i.e. adapt witness builder output to the instantate() function.
        // This can be removed once witness builders are removed.
        let mut shred_id_to_data = HashMap::<NodeId, MultilinearExtension<F>>::new();
        input_layer_data.into_iter().for_each(|input_layer_data| {
            input_layer_data
                .data
                .into_iter()
                .for_each(|input_shred_data| {
                    shred_id_to_data.insert(
                        input_shred_data.corresponding_input_shred_id,
                        input_shred_data.data,
                    );
                });
        });

        let (circuit_description, input_builder, _) =
            generate_circuit_description(component.yield_nodes()).unwrap();

        let inputs = input_builder(shred_id_to_data).unwrap();

        // FIXME(Ben)
        // // Add the inputs to transcript.
        // // In the future flow, the inputs will be added to the transcript in the calling context.
        // circuit_description
        //     .input_layers
        //     .iter()
        //     .for_each(|input_layer| {
        //         let mle = inputs.get(&input_layer.layer_id).unwrap();
        //         transcript_writer.append_elements("Input values", mle.get_evals_vector());
        //     });

        let mut challenge_sampler =
            |size| transcript_writer.get_challenges("Verifier challenges", size);
        let instantiated_circuit = circuit_description.instantiate(&inputs, &mut challenge_sampler);

        (instantiated_circuit, circuit_description)
    }

    /// From the witness builder, generate a circuit description and then an
    /// instantiated circuit. Then, write the "proof" of this circuit to the
    /// `transcript_writer`. Return the circuit description for verifying.
    #[instrument(skip_all, err)]
    pub fn prove(
        &mut self,
        mut transcript_writer: TranscriptWriter<F, PoseidonSponge<F>>,
    ) -> Result<(Transcript<F>, GKRCircuitDescription<F>), GKRError> {
        let synthesize_commit_timer = start_timer!(|| "Circuit synthesize and commit");
        info!("Synethesizing circuit...");

        let (
            InstantiatedCircuit {
                input_layers,
                mut output_layers,
                layers,
                fiat_shamir_challenges: _fiat_shamir_challenges,
                layer_map,
            },
            circuit_description,
        ) = self.synthesize_and_commit(&mut transcript_writer);

        info!("Circuit synthesized and witness generated.");
        end_timer!(synthesize_commit_timer);

        // Claim aggregator to keep track of GKR-style claims across all layers.
        let mut aggregator = WLXAggregator::<F, LayerEnum<F>, InputLayerEnum<F>>::new();

        // --------- STAGE 1: Output Claim Generation ---------
        let claims_timer = start_timer!(|| "Output claims generation");
        let output_claims_span = span!(Level::DEBUG, "output_claims_span").entered();

        // Go through circuit output layers and grab claims on each.
        for output in output_layers.iter_mut() {
            let layer_id = output.layer_id();
            info!("Output Layer: {:?}", layer_id);

            // FIXME(Ben) we shouldn't need to append output values to transcript if they are always zero
            transcript_writer.append_elements("Output values", output.get_mle().bookkeeping_table());

            let challenges = transcript_writer.get_challenges("output layer binding", output.num_free_vars());
            output
                .fix_layer(&challenges)
                .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

            // Add the claim to either the set of current claims we're proving
            // or the global set of claims we need to eventually prove.
            aggregator
                .extract_claims(output)
                .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;
        }

        end_timer!(claims_timer);
        output_claims_span.exit();

        // --------- STAGE 2: Prove Intermediate Layers ---------
        let intermediate_layers_timer = start_timer!(|| "ALL intermediate layers proof generation");
        let all_layers_sumcheck_proving_span =
            span!(Level::DEBUG, "all_layers_sumcheck_proving_span").entered();

        // Collects all the prover messages for sumchecking over each layer, as
        // well as all the prover messages for claim aggregation at the
        // beginning of proving each layer.
        for mut layer in layers.layers.into_iter().rev() {
            let layer_id = layer.layer_id();

            let layer_timer = start_timer!(|| format!("Generating proof for layer {:?}", layer_id));
            let layer_id_trace_repr = format!("{:?}", layer_id);
            let layer_sumcheck_proving_span = span!(
                Level::DEBUG,
                "layer_sumcheck_proving_span",
                layer_id = layer_id_trace_repr
            )
            .entered();
            info!("Proving Intermediate Layer: {:?}", layer_id);

            info!("Starting claim aggregation...");
            let claim_aggr_timer =
                start_timer!(|| format!("Claim aggregation for layer {:?}", layer_id));

            let output_mles_from_layer = layer_map.get(&layer_id).unwrap();
            let layer_claim = aggregator.prover_aggregate_claims(
                &layer,
                output_mles_from_layer,
                &mut transcript_writer,
            )?;

            end_timer!(claim_aggr_timer);

            info!("Prove sumcheck message");
            let sumcheck_msg_timer = start_timer!(|| format!(
                "Compute sumcheck message for layer {:?}",
                layer.layer_id()
            ));

            // Compute all sumcheck messages across this particular layer.
            layer
                .prove_rounds(layer_claim, &mut transcript_writer)
                .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

            end_timer!(sumcheck_msg_timer);

            aggregator
                .extract_claims(&layer)
                .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

            end_timer!(layer_timer);
            layer_sumcheck_proving_span.exit();
        }

        end_timer!(intermediate_layers_timer);
        all_layers_sumcheck_proving_span.exit();

        // FIXME(Ben) move to calling context
        // // --------- STAGE 3: Prove Input Layers ---------
        // let input_layers_timer = start_timer!(|| "INPUT layers proof generation");
        // let input_layer_proving_span = span!(Level::DEBUG, "input_layer_proving_span").entered();

        // for input_layer in input_layers {
        //     let layer_id = input_layer.layer_id();

        //     info!("New Input Layer: {:?}", layer_id);
        //     let layer_timer =
        //         start_timer!(|| format!("proof generation for INPUT layer {:?}", layer_id));

        //     let claim_aggr_timer = start_timer!(|| format!(
        //         "claim aggregation for INPUT layer {:?}",
        //         input_layer.layer_id()
        //     ));

        //     let output_mles_from_layer = layer_map.get(&layer_id).unwrap();
        //     let layer_claim = aggregator.prover_aggregate_claims_input(
        //         &input_layer,
        //         output_mles_from_layer,
        //         &mut transcript_writer,
        //     )?;

        //     end_timer!(claim_aggr_timer);

        //     let eval_proof_timer =
        //         start_timer!(|| format!("evaluation proof for INPUT layer {:?}", layer_id));

        //     input_layer
        //         .open(&mut transcript_writer, layer_claim)
        //         .map_err(GKRError::InputLayerError)?;

        //     end_timer!(eval_proof_timer);

        //     end_timer!(layer_timer);
        // }

        // // TODO(Makis): What do we do with the input commitments? Put them into
        // // transcript?

        // end_timer!(input_layers_timer);
        // input_layer_proving_span.exit();

        // --------- STAGE 4: Verifier Challenges --------- There is nothing to
        // be done here, since the claims on verifier challenges are checked
        // directly by the verifier, without aggregation.

        Ok((transcript_writer.get_transcript(), circuit_description))
    }
}
