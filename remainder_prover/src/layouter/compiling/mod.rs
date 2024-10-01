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
//! `populate_circuit` function will create an [InstantiatedCircuit], where now
//! the circuit description is "filled in" with its associated data.

#[cfg(test)]
mod tests;

use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use tracing::{instrument, span, Level};

use crate::claims::claim_aggregation::prover_aggregate_claims;
use crate::claims::ClaimTracker;
use crate::expression::circuit_expr::{filter_bookkeeping_table, MleDescription};
use crate::input_layer::enum_input_layer::InputLayerEnum;
use crate::input_layer::fiat_shamir_challenge::FiatShamirChallenge;
use crate::input_layer::{InputLayer, InputLayerDescription};
use crate::layer::layer_enum::LayerEnum;
use crate::layer::LayerId;
use crate::layer::{Layer, LayerDescription};
use crate::layouter::layouting::CircuitMap;
use crate::layouter::nodes::circuit_inputs::compile_inputs::combine_input_mles;
use crate::mle::evals::MultilinearExtension;
use crate::output_layer::mle_output_layer::MleOutputLayer;
use crate::output_layer::OutputLayer;
use crate::prover::helpers::get_circuit_description_hash_as_field_elems;
use crate::prover::layers::Layers;
use crate::prover::{
    generate_circuit_description, GKRCircuitDescription, GKRError, InstantiatedCircuit,
};
use ark_std::{end_timer, start_timer};
use itertools::Itertools;
use log::info;
use remainder_shared_types::transcript::{ProverTranscript, Transcript, TranscriptWriter};
use remainder_shared_types::Field;

use super::layouting::{CircuitLocation, InputNodeMap, LayerMap};
use super::nodes::circuit_inputs::InputLayerData;
use super::nodes::NodeId;
use super::{
    component::Component,
    nodes::{node_enum::NodeEnum, Context},
};

/// Defines the type of hash used when adding a circuit description's hash
/// into the transcript.
pub enum CircuitHashType {
    /// This uses Rust's [DefaultHasher] implementation and uses the
    /// #[derive(Hash)] implementation. The hash function implemented
    /// underneath is not cryptographically secure, and thus this option
    /// is generally not recommended.
    DefaultRustHash,
    /// This converts the circuit description into a JSON string via
    /// [serde::Serialize] and hashes the bytes using [Sha3_256].
    Sha3_256,
    /// This converts the circuit description into a JSON string via
    /// [serde::Serialize] and hashes the bytes in chunks of 16, converting
    /// them first to field elements.
    Poseidon,
}

/// The struct used in order to aid with proving, which contains the function
/// used in order to generate the circuit description itself, called the
/// `witness_builder`, which can then be used to generate an
/// [InstantiatedCircuit].
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
    /// Constructs a `LayouterCircuit` by taking in a closure whose parameter is
    /// a [Context], which determines the ID of each individual part of the
    /// circuit whose data output will be computed for future layers. The
    /// function itself returns a [Component], which is a set of circuit "nodes"
    /// which will be used to build the circuit itself. The closure also returns
    /// a [Vec<InputLayerData>], which is a vector of all of the data that will
    /// be fed into the input layers of the circuit.
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
    /// Returns an [InstantiatedCircuit] by populating the
    /// [GKRCircuitDescription] with data.
    ///
    /// # Arguments:
    /// * gkr_circuit_description: type [GKRCircuitDescription], which is the
    ///     circuit description of the circuit we wish to populate
    /// * input_layer_to_node_map: type [InputNodeMap], which the corresponding
    ///     [super::nodes::circuit_inputs::InputLayerNode]'s [NodeId] to a
    ///     [LayerId], in order to associate the [InputLayerData] to the correct
    ///     [InputLayer].
    /// * data_input_layers: type [Vec<InputLayerData<F>>], which contains all
    ///     of the data needed in order to populate the circuit description.
    /// * transcript_writer: type implements [ProverTranscript<F>], which is
    ///     primarily used for [FiatShamirChallenge] in order to grab the
    ///     challenges from the transcript in order to generate the concretized
    ///     [FiatShamirChallenge] for the circuit.
    ///
    /// # Requires:
    /// The order of the data in the `data_input_layers` vector must match the
    /// order that the input shreds are returned in the [Component] returned
    /// when `self.witness_builder` is called given a context.
    fn populate_circuit(
        &mut self,
        gkr_circuit_description: &GKRCircuitDescription<F>,
        input_layer_to_node_map: InputNodeMap,
        data_input_layers: Vec<InputLayerData<F>>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> (InstantiatedCircuit<F>, LayerMap<F>) {
        let GKRCircuitDescription {
            input_layers: input_layer_descriptions,
            fiat_shamir_challenges: fiat_shamir_challenge_descriptions,
            intermediate_layers: intermediate_layer_descriptions,
            output_layers: output_layer_descriptions,
        } = gkr_circuit_description;

        // Create a map that maps the input layer's node ID to the input layer
        // data that corresponds input layer node by doing a forward pass of
        // `data_input_layers`.
        let mut input_id_data_map = HashMap::<NodeId, &InputLayerData<F>>::new();
        data_input_layers.iter().for_each(|input_layer_data| {
            input_id_data_map.insert(
                input_layer_data.corresponding_input_node_id,
                input_layer_data,
            );
        });

        // Create a map that maps layer ID to a set of MLE descriptions that are
        // expected to be compiled from its output. For example, if we have a
        // layer whose first "half" (when MSB is 0) is used in a future layer,
        // and its second half is also used in a future layer, we would expect
        // both of these to be represented as MLE descriptions in the HashSet
        // associated with this layer with the appropriate prefix bits.
        let mut mle_claim_map = HashMap::<LayerId, HashSet<&MleDescription<F>>>::new();
        // Do a forward pass through all of the intermediate layer descriptions
        // and look into the "future" to see which parts of each layer are
        // required for future layers.
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

        // Do a forward pass through all of the intermediate layer descriptions
        // and look into the "future" to see which parts of each layer are
        // required for output layers.
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

        // Step 1: populate the circuit map with all of the data necessary in
        // order to instantiate the circuit.
        let mut circuit_map = CircuitMap::new();
        let mut prover_input_layers: Vec<InputLayerEnum<F>> = Vec::new();
        let mut fiat_shamir_challenges = Vec::new();
        // Step 1a: populate the circuit map by compiling the necessary data
        // outputs for each of the input layers, while writing the commitments
        // to them into the transcript.
        input_layer_descriptions
            .iter()
            .for_each(|input_layer_description| {
                let input_layer_id = input_layer_description.layer_id();
                let input_node_id = input_layer_to_node_map.get_node_id(input_layer_id).unwrap();
                let corresponding_input_data = *(input_id_data_map.get(input_node_id).unwrap());
                let input_mles = corresponding_input_data
                    .data
                    .iter()
                    .map(|input_shred_data| &input_shred_data.data);
                // Combine all of the input data corresponding to this layer in
                // order to create the layerwise bookkeeping table for the input
                // layer.
                let combined_mle = combine_input_mles(&input_mles.collect_vec());
                let mle_outputs_necessary = mle_claim_map.get(&input_layer_id).unwrap();
                // Compute all data outputs necessary for future layers for each
                // input layer.
                mle_outputs_necessary.iter().for_each(|mle_output| {
                    let prefix_bits = mle_output.prefix_bits();
                    let output = filter_bookkeeping_table(&combined_mle, &prefix_bits);
                    circuit_map.add_node(CircuitLocation::new(input_layer_id, prefix_bits), output);
                });
                // Compute the concretized input layer since we have the
                // layerwise bookkeeping table.
                let mut prover_input_layer = input_layer_description
                    .convert_into_prover_input_layer(
                        combined_mle,
                        &corresponding_input_data.precommit,
                    );
                // Compute the commitment to the input layer combined MLE and
                // add it to transcript.
                let commitment = prover_input_layer.commit().unwrap();
                InputLayerEnum::append_commitment_to_transcript(&commitment, transcript_writer);
                prover_input_layers.push(prover_input_layer);
            });
        // Step 1b: for each of the fiat shamir challenges, use the transcript
        // in order to get the challenges and fill the layer.
        fiat_shamir_challenge_descriptions
            .iter()
            .for_each(|fiat_shamir_challenge_description| {
                let fiat_shamir_challenge_mle =
                    MultilinearExtension::new(transcript_writer.get_challenges(
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

        // Step 1c: Compute the data outputs, using the map from Layer ID to
        // which Circuit MLEs are necessary to compile for this layer, for each
        // of the intermediate layers.
        intermediate_layer_descriptions
            .iter()
            .for_each(|intermediate_layer_description| {
                let mle_outputs_necessary = mle_claim_map
                    .get(&intermediate_layer_description.layer_id())
                    .unwrap();
                intermediate_layer_description
                    .compute_data_outputs(mle_outputs_necessary, &mut circuit_map);
            });

        // Step 2: Using the fully populated circuit map, convert each of the
        // layer descriptions into concretized layers. Step 2a: Concretize the
        // intermediate layer descriptions.
        let mut prover_intermediate_layers: Vec<LayerEnum<F>> =
            Vec::with_capacity(intermediate_layer_descriptions.len());
        intermediate_layer_descriptions
            .iter()
            .for_each(|intermediate_layer_description| {
                let prover_intermediate_layer =
                    intermediate_layer_description.convert_into_prover_layer(&circuit_map);
                prover_intermediate_layers.push(prover_intermediate_layer)
            });

        // Step 2b: Concretize the output layer descriptions.
        let mut prover_output_layers: Vec<MleOutputLayer<F>> = Vec::new();
        output_layer_descriptions
            .iter()
            .for_each(|output_layer_description| {
                let prover_output_layer =
                    output_layer_description.into_prover_output_layer(&circuit_map);
                prover_output_layers.push(prover_output_layer)
            });
        let instantiated_circuit = InstantiatedCircuit {
            input_layers: prover_input_layers,
            fiat_shamir_challenges,
            layers: Layers::new_with_layers(prover_intermediate_layers),
            output_layers: prover_output_layers,
        };

        let layer_map = circuit_map.convert_to_layer_map();

        (instantiated_circuit, layer_map)
    }

    /// This function simply calls the [LayouterCircuit]'s `witness_builder`
    /// function and generates the corresponding circuit description from it.
    ///
    /// TODO(ryancao): This function is hacky and should be re-done correctly
    /// AFTER we fix the `self.witness_builder` function!! However, this is
    /// the most direct way to just grab the circuit description for
    /// now.
    pub fn get_circuit_description(&mut self) -> GKRCircuitDescription<F> {
        let ctx = Context::new();
        let (component, _input_layer_data) = (self.witness_builder)(&ctx);
        let (circuit_description, _input_node_map) =
            generate_circuit_description(component, ctx).unwrap();
        circuit_description
    }

    fn synthesize_and_commit(
        &mut self,
        transcript_writer: &mut impl ProverTranscript<F>,
        circuit_description_hash_type: CircuitHashType,
    ) -> (
        InstantiatedCircuit<F>,
        LayerMap<F>,
        GKRCircuitDescription<F>,
    ) {
        let ctx = Context::new();
        let (component, input_layer_data) = (self.witness_builder)(&ctx);

        let (circuit_description, input_node_map) =
            generate_circuit_description(component, ctx).unwrap();

        // --- Generate circuit description hash and add to transcript ---
        let hash_value_as_field_elems = get_circuit_description_hash_as_field_elems(
            &circuit_description,
            circuit_description_hash_type,
        );
        transcript_writer.append_elements("Circuit description hash", &hash_value_as_field_elems);

        let (instantiated_circuit, layer_map) = self.populate_circuit(
            &circuit_description,
            input_node_map,
            input_layer_data,
            transcript_writer,
        );

        (instantiated_circuit, layer_map, circuit_description)
    }

    /// From the witness builder, generate a circuit description and then an
    /// instantiated circuit. Then, write the "proof" of this circuit to the
    /// `transcript_writer`. Return the circuit description for verifying.
    #[instrument(skip_all, err)]
    pub fn prove(
        &mut self,
        mut transcript_writer: TranscriptWriter<F, PoseidonSponge<F>>,
        circuit_description_hash_type: CircuitHashType,
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
            layer_map,
            circuit_description,
        ) = self.synthesize_and_commit(&mut transcript_writer, circuit_description_hash_type);

        info!("Circuit synthesized and witness generated.");
        end_timer!(synthesize_commit_timer);

        // Maps a `LayerId` to a collection of claims made on that layer.
        let mut claim_tracker = ClaimTracker::new();

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

            for claim in output
                .get_claims()
                .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?
            {
                claim_tracker.insert(claim.get_to_layer_id(), claim);
            }
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
            let layer_claims = claim_tracker.get(layer_id).unwrap();
            let layer_claim = prover_aggregate_claims(
                layer_claims,
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
                .prove(layer_claim, &mut transcript_writer)
                .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

            end_timer!(sumcheck_msg_timer);

            for claim in layer
                .get_claims()
                .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?
            {
                claim_tracker.insert(claim.get_to_layer_id(), claim);
            }

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

            let output_mles_from_layer = layer_map.get(&layer_id).unwrap();
            let layer_claims = claim_tracker.get(layer_id).unwrap();
            let layer_claim = prover_aggregate_claims(
                layer_claims,
                &input_layer,
                output_mles_from_layer,
                &mut transcript_writer,
            )?;

            end_timer!(claim_aggr_timer);

            let eval_proof_timer =
                start_timer!(|| format!("evaluation proof for INPUT layer {:?}", layer_id));

            input_layer
                .open(&mut transcript_writer, layer_claim)
                .map_err(GKRError::InputLayerError)?;

            end_timer!(eval_proof_timer);

            end_timer!(layer_timer);
        }

        // TODO(Makis): What do we do with the input commitments? Put them into
        // transcript?

        end_timer!(input_layers_timer);
        input_layer_proving_span.exit();

        // --------- STAGE 4: Verifier Challenges --------- There is nothing to
        // be done here, since the claims on verifier challenges are checked
        // directly by the verifier, without aggregation.

        Ok((transcript_writer.get_transcript(), circuit_description))
    }
}
