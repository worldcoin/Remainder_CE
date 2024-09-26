//! A module for defining how certain nodes can be Compiled into a GKR Witness

#[cfg(test)]
mod tests;

use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use tracing::{instrument, span, Level};

use crate::claims::wlx_eval::WLXAggregator;
use crate::claims::ClaimAggregator;
use crate::expression::circuit_expr::{filter_bookkeeping_table, CircuitMle};
use crate::input_layer::enum_input_layer::InputLayerEnum;
use crate::input_layer::fiat_shamir_challenge::FiatShamirChallenge;
use crate::input_layer::{CircuitInputLayer, InputLayer};
use crate::layer::layer_enum::LayerEnum;
use crate::layer::LayerId;
use crate::layer::{CircuitLayer, Layer};
use crate::layouter::layouting::CircuitMap;
use crate::layouter::nodes::circuit_inputs::compile_inputs::combine_input_mles;
use crate::mle::evals::MultilinearExtension;
use crate::output_layer::mle_output_layer::MleOutputLayer;
use crate::output_layer::OutputLayer;
use crate::prover::layers::Layers;
use crate::prover::{
    generate_circuit_description, GKRCircuitDescription, GKRError, InstantiatedCircuit,
};
use ark_std::{end_timer, start_timer};
use itertools::Itertools;
use log::info;
use remainder_shared_types::transcript::{ProverTranscript, Transcript, TranscriptWriter};
use remainder_shared_types::Field;

use super::layouting::{CircuitLocation, InputNodeMap};
use super::nodes::circuit_inputs::InputLayerData;
use super::nodes::NodeId;
use super::{
    component::Component,
    nodes::{node_enum::NodeEnum, Context},
};

/// A basic circuit that uses the Layouter to construct the witness
pub struct LayouterCircuit<
    F: Field,
    C: Component<NodeEnum<F>>,
    Fn: FnMut(&Context) -> (C, Vec<InputLayerData<F>>),
> {
    witness_builder: Fn,
    _marker: PhantomData<F>,
}

impl<F: Field, C: Component<NodeEnum<F>>, Fn: FnMut(&Context) -> (C, Vec<InputLayerData<F>>)>
    LayouterCircuit<F, C, Fn>
{
    /// Constructs a `LayouterCircuit` by taking in a closure that computes a Component
    /// that contains all the nodes that will be layedout and compiled into the witness
    pub fn new(witness_builder: Fn) -> Self {
        Self {
            witness_builder,
            _marker: PhantomData,
        }
    }
}

impl<F: Field, C: Component<NodeEnum<F>>, Fn: FnMut(&Context) -> (C, Vec<InputLayerData<F>>)>
    LayouterCircuit<F, C, Fn>
{
    fn populate_circuit(
        &mut self,
        gkr_circuit_description: &GKRCircuitDescription<F>,
        input_layer_to_node_map: InputNodeMap,
        data_input_layers: Vec<InputLayerData<F>>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> InstantiatedCircuit<F> {
        let GKRCircuitDescription {
            input_layers: input_layer_descriptions,
            fiat_shamir_challenges: fiat_shamir_challenge_descriptions,
            intermediate_layers: intermediate_layer_descriptions,
            output_layers: output_layer_descriptions,
        } = gkr_circuit_description;

        // Forward pass through input layer data to map input layer ID to the data that the circuit builder provides.
        let mut input_id_data_map = HashMap::<NodeId, &InputLayerData<F>>::new();
        data_input_layers.iter().for_each(|input_layer_data| {
            input_id_data_map.insert(
                input_layer_data.corresponding_input_node_id,
                input_layer_data,
            );
        });

        // Forward pass to get the map of circuit MLEs whose data is expected to be "compiled"
        // for future layers.
        let mut mle_claim_map = HashMap::<LayerId, HashSet<&CircuitMle<F>>>::new();
        intermediate_layer_descriptions
            .iter()
            .for_each(|intermediate_layer| {
                let layer_source_circuit_mles = intermediate_layer.get_circuit_mles();
                layer_source_circuit_mles
                    .into_iter()
                    .for_each(|circuit_mle| {
                        let layer_id = circuit_mle.layer_id();
                        if let Entry::Vacant(e) = mle_claim_map.entry(layer_id) {
                            e.insert(HashSet::from([circuit_mle]));
                        } else {
                            mle_claim_map
                                .get_mut(&layer_id)
                                .unwrap()
                                .insert(circuit_mle);
                        }
                    })
            });

        output_layer_descriptions.iter().for_each(|output_layer| {
            let layer_source_mle = &output_layer.mle;
            let layer_id = layer_source_mle.layer_id();
            if let Entry::Vacant(e) = mle_claim_map.entry(layer_id) {
                e.insert(HashSet::from([&output_layer.mle]));
            } else {
                mle_claim_map
                    .get_mut(&layer_id)
                    .unwrap()
                    .insert(&output_layer.mle);
            }
        });

        let mut circuit_map = CircuitMap::new();

        // input layers
        // go through input data, map it to the inputlayernode it corresponds to
        // for each input layer node, take the input data it corresponds to and combine it to form one big bookkeeping table,
        // we convert the circuit input layer into a prover input layer using this big bookkeeping table
        // we add the data in the input data corresopnding with the circuit location for each input data struct into the circuit map
        let mut prover_input_layers: Vec<InputLayerEnum<F>> = Vec::new();
        let mut fiat_shamir_challenges = Vec::new();
        input_layer_descriptions
            .iter()
            .for_each(|input_layer_description| {
                let input_layer_id = input_layer_description.layer_id();
                let input_node_id = input_layer_to_node_map.get_node_id(&input_layer_id).unwrap();
                let corresponding_input_data = *(input_id_data_map.get(input_node_id).unwrap());
                let input_mles = corresponding_input_data
                    .data
                    .iter()
                    .map(|input_shred_data| &input_shred_data.data);

                let combined_mle = combine_input_mles(&input_mles.collect_vec());
                let mle_outputs_necessary = mle_claim_map.get(&input_layer_id).unwrap();
                mle_outputs_necessary.iter().for_each(|mle_output| {
                    let prefix_bits = mle_output.prefix_bits();
                    let output = filter_bookkeeping_table(&combined_mle, &prefix_bits);
                    circuit_map
                        .add_node(CircuitLocation::new(input_layer_id, prefix_bits), output);
                });

                let mut prover_input_layer = input_layer_description
                    .convert_into_prover_input_layer(
                        combined_mle,
                        &corresponding_input_data.precommit,
                    );
                let commitment = prover_input_layer.commit().unwrap();
                InputLayerEnum::append_commitment_to_transcript(&commitment, transcript_writer);
                prover_input_layers.push(prover_input_layer);
            });

        fiat_shamir_challenge_descriptions
            .iter()
            .for_each(|fiat_shamir_challenge_description| {
                let fiat_shamir_challenge_mle = MultilinearExtension::new(transcript_writer.get_challenges(
                    "Verifier challenges",
                    1 << fiat_shamir_challenge_description.num_bits,
                ));
                circuit_map.add_node(
                    CircuitLocation::new(fiat_shamir_challenge_description.layer_id(), vec![]),
                    fiat_shamir_challenge_mle.clone(),
                );
                fiat_shamir_challenges.push(FiatShamirChallenge::new(
                    fiat_shamir_challenge_mle,
                    fiat_shamir_challenge_description.layer_id(),
                ));
            });

        // forward pass of the layers
        // convert the circuit layer into a prover layer using circuit map -> populate a GKRCircuit as you do this
        // prover layer ( mle_claim_map ) -> populates circuit map
        intermediate_layer_descriptions
            .iter()
            .for_each(|intermediate_layer_description| {
                let mle_outputs_necessary = mle_claim_map
                    .get(&intermediate_layer_description.layer_id())
                    .unwrap();
                intermediate_layer_description.compute_data_outputs(mle_outputs_necessary, &mut circuit_map);
            });

        let mut prover_intermediate_layers: Vec<LayerEnum<F>> =
            Vec::with_capacity(intermediate_layer_descriptions.len());
        intermediate_layer_descriptions
            .iter()
            .for_each(|intermediate_layer_description| {
                let prover_intermediate_layer =
                    intermediate_layer_description.convert_into_prover_layer(&circuit_map);
                prover_intermediate_layers.push(prover_intermediate_layer)
            });

        let mut prover_output_layers: Vec<MleOutputLayer<F>> = Vec::new();
        output_layer_descriptions
            .iter()
            .for_each(|output_layer_description| {
                let prover_output_layer =
                    output_layer_description.into_prover_output_layer(&circuit_map);
                prover_output_layers.push(prover_output_layer)
            });
        InstantiatedCircuit {
            input_layers: prover_input_layers,
            fiat_shamir_challenges,
            layers: Layers::new_with_layers(prover_intermediate_layers),
            output_layers: prover_output_layers,
        }
    }

    fn synthesize_and_commit(
        &mut self,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> (InstantiatedCircuit<F>, GKRCircuitDescription<F>) {
        let ctx = Context::new();
        let (component, input_layer_data) = (self.witness_builder)(&ctx);
        // TODO(vishady): ADD CIRCUIT DESCRIPTION TO TRANSCRIPT (maybe not here...)
        let (circuit_description, input_node_map) =
            generate_circuit_description(component, ctx).unwrap();

        let instantiated_circuit = self.populate_circuit(
            &circuit_description,
            input_node_map,
            input_layer_data,
            transcript_writer,
        );

        (instantiated_circuit, circuit_description)
    }

    /// The backwards pass, creating the GKRProof.
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

            output.append_mle_to_transcript(&mut transcript_writer);

            output
                .fix_layer(&mut transcript_writer)
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

            let layer_claim = aggregator.prover_aggregate_claims(&layer, &mut transcript_writer)?;

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

        // --------- STAGE 3: Prove Input Layers ---------
        let input_layers_timer = start_timer!(|| "INPUT layers proof generation");
        let input_layer_proving_span = span!(Level::DEBUG, "input_layer_proving_span").entered();

        for input_layer in input_layers {
            let layer_id = input_layer.layer_id();

            info!("New Input Layer: {:?}", layer_id);
            let layer_timer =
                start_timer!(|| format!("proof generation for INPUT layer {:?}", layer_id));

            let claim_aggr_timer = start_timer!(|| format!(
                "claim aggregation for INPUT layer {:?}",
                input_layer.layer_id()
            ));

            let layer_claim =
                aggregator.prover_aggregate_claims_input(&input_layer, &mut transcript_writer)?;

            end_timer!(claim_aggr_timer);

            let eval_proof_timer =
                start_timer!(|| format!("evaluation proof for INPUT layer {:?}", layer_id));

            input_layer
                .open(&mut transcript_writer, layer_claim)
                .map_err(GKRError::InputLayerError)?;

            end_timer!(eval_proof_timer);

            end_timer!(layer_timer);
        }

        // TODO(Makis): What do we do with the input commitments?
        // Put them into transcript?

        end_timer!(input_layers_timer);
        input_layer_proving_span.exit();

        // --------- STAGE 4: Verifier Challenges ---------
        // There is nothing to be done here, since the claims on verifier challenges are checked
        // directly by the verifier, without aggregation.

        Ok((transcript_writer.get_transcript(), circuit_description))
    }
}
