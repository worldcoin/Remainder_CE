//!Modules that orchestrates creating a GKR Proof

/// Includes boilerplate for creating a GKR circuit, i.e. creating a transcript, proving, verifying, etc.
pub mod helpers;

/// Includes various traits that define interfaces of a GKR Prover
pub mod proof_system;

/// Struct for representing a list of layers
pub mod layers;

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use self::layers::Layers;
use crate::claims::wlx_eval::WLXAggregator;
use crate::claims::Claim;
use crate::expression::circuit_expr::filter_bookkeeping_table;
use crate::input_layer::enum_input_layer::{
    InputLayerDescriptionEnum, InputLayerEnum, InputLayerEnumVerifierCommitment,
};
use crate::input_layer::fiat_shamir_challenge::{
    FiatShamirChallenge, FiatShamirChallengeDescription,
};
use crate::input_layer::{InputLayer, InputLayerDescription, InputLayerDescriptionTrait, InputLayerTrait};
use crate::layer::layer_enum::{LayerDescriptionEnum, VerifierLayerEnum};
use crate::layer::{Layer, LayerDescription};
use crate::layouter::layouting::{
    layout, CircuitDescriptionMap, CircuitLocation, CircuitMap, InputNodeMap,
};
use crate::layouter::nodes::node_enum::NodeEnum;
use crate::layouter::nodes::{CircuitNode, NodeId};
use crate::mle::dense::DenseMle;
use crate::mle::evals::MultilinearExtension;
use crate::mle::mle_description::MleDescription;
use crate::output_layer::mle_output_layer::{MleOutputLayer, MleOutputLayerDescription};
use crate::output_layer::{OutputLayerTrait, OutputLayerDescription};
use crate::utils::mle::build_composite_mle;
use crate::{
    claims::ClaimAggregator,
    input_layer::InputLayerError,
    layer::{layer_enum::LayerEnum, LayerError, LayerId},
};
use ark_std::{end_timer, start_timer};
use itertools::Itertools;
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use remainder_shared_types::transcript::VerifierTranscript;
use remainder_shared_types::transcript::{
    ProverTranscript, TranscriptReaderError, TranscriptWriter,
};
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info};
use tracing::{instrument, span, Level};

#[derive(Error, Debug, Clone)]
/// Errors relating to the proving of a GKR circuit
pub enum GKRError {
    #[error("No claims were found for layer {0:?}")]
    /// No claims were found for layer
    NoClaimsForLayer(LayerId),
    #[error("Transcript during verifier's interaction with the transcript.")]
    /// Errors when reading from the transcript
    TranscriptError(TranscriptReaderError),
    #[error("Error when proving layer {0:?}: {1}")]
    /// Error when proving layer
    ErrorWhenProvingLayer(LayerId, LayerError),
    #[error("Error when proving input layer {0:?}: {1}")]
    /// Error when proving input layer
    ErrorWhenProvingInputLayer(LayerId, InputLayerError),
    #[error("Error when verifying layer {0:?}: {1}")]
    /// Error when verifying layer
    ErrorWhenVerifyingLayer(LayerId, LayerError),

    /// Error when verifying input layer
    #[error("Error when verifying input layer {0:?}: {1}")]
    ErrorWhenVerifyingInputLayer(LayerId, InputLayerError),

    #[error("Error when verifying output layer")]
    /// Error when verifying output layer
    ErrorWhenVerifyingOutputLayer,
    /// InputShred length mismatch
    #[error("InputShred with NodeId {0} should have {1} variables, but has {2}")]
    InputShredLengthMismatch(NodeId, usize, usize),
    /// Error for input layer commitment
    #[error("Error when commiting to InputLayer {0}")]
    InputLayerError(InputLayerError),
    #[error("Error when verifying circuit hash.")]
    /// Error when verifying circuit hash
    ErrorWhenVerifyingCircuitHash(TranscriptReaderError),

    /// Error generating the Verifier Key.
    #[error("Error generating the Verifier Key")]
    ErrorGeneratingVerifierKey,
}

/// A proof of the sumcheck protocol; Outer vec is rounds, inner vec is evaluations
/// this inner vec is none if there is no sumcheck proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckProof<F>(pub Vec<Vec<F>>);

impl<F: Field> From<Vec<Vec<F>>> for SumcheckProof<F> {
    fn from(value: Vec<Vec<F>>) -> Self {
        Self(value)
    }
}

/// The witness of a GKR circuit, used to actually prove the circuit
#[derive(Debug)]
pub struct InstantiatedCircuit<F: Field> {
    /// The intermediate layers of the circuit
    pub layers: Layers<F, LayerEnum<F>>,
    /// The output layers of the circuit
    pub output_layers: Vec<MleOutputLayer<F>>,
    /// The input layers of the circuit
    pub input_layers: Vec<InputLayer<F>>,
    /// The verifier challenges
    pub fiat_shamir_challenges: Vec<FiatShamirChallenge<F>>,
    /// FIXME(vishady) what actually is this :)
    pub layer_map: HashMap<LayerId, Vec<DenseMle<F>>>,
}

/// Assumes that the inputs have already been added to the transcript (if necessary).
/// Returns the vector of claims on the input layers.
pub fn prove_circuit<F: Field>(
    circuit_description: GKRCircuitDescription<F>,
    inputs: HashMap<LayerId, MultilinearExtension<F>>,
    mut transcript_writer: TranscriptWriter<F, PoseidonSponge<F>>,
) -> Result<Vec<Claim<F>>, GKRError> {
    // Note: no need to return the Transcript, since it is already in the TranscriptWriter!
    // Note(Ben): this can't be an instance method, because it consumes the intermediate layers!
    // Note(Ben): this is a GKR specific method.  So it makes sense for IT to define the challenge sampler, so that the circuit can be instantiated (rather than leaving this complexity to the calling context).

    let mut challenge_sampler =
        |size| transcript_writer.get_challenges("Verifier challenges", size);
    let instantiated_circuit = circuit_description.instantiate(&inputs, &mut challenge_sampler);

    let InstantiatedCircuit {
        input_layers,
        mut output_layers,
        layers,
        fiat_shamir_challenges: _fiat_shamir_challenges,
        layer_map,
    } = instantiated_circuit;

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
        let sumcheck_msg_timer =
            start_timer!(|| format!("Compute sumcheck message for layer {:?}", layer.layer_id()));

        // Compute all sumcheck messages across this particular layer.
        layer
            .prove_rounds(layer_claim, &mut transcript_writer)
            .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

        end_timer!(sumcheck_msg_timer);

        aggregator
            .extract_claims(&layer)
            .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

        end_timer!(layer_timer);
    }

    end_timer!(intermediate_layers_timer);
    all_layers_sumcheck_proving_span.exit();

    // FIXME(Ben) add back claim agg for input layers
    // (Wait until Makis has finished his YieldClaims refactor).
    let input_layer_claims = input_layers.iter().filter_map(|input_layer| aggregator.get_claims(input_layer.layer_id)).flatten().map(|claim_mle| claim_mle.get_claim().clone()).collect_vec();

    Ok(input_layer_claims)
}

/// Controls claim aggregation behavior.
pub const ENABLE_OPTIMIZATION: bool = true;

/// The Verifier Key associated with a GKR proof of a [ProofSystem].
/// It consists of consice GKR Circuit description to be use by the Verifier.
#[derive(Debug)]
pub struct GKRCircuitDescription<F: Field> {
    /// The circuit descriptions of the input layers.
    pub input_layers: Vec<InputLayerDescription>,
    /// The circuit descriptions of the verifier challengs
    pub fiat_shamir_challenges: Vec<FiatShamirChallengeDescription<F>>,
    /// The circuit descriptions of the intermediate layers.
    pub intermediate_layers: Vec<LayerDescriptionEnum<F>>,
    /// The circuit desriptions of the output layers.
    pub output_layers: Vec<MleOutputLayerDescription<F>>,
}

impl<F: Field> GKRCircuitDescription<F> {
    /// Label the MLE indices contained within a circuit description, starting
    /// each layer with the start_index.
    pub fn index_mle_indices(&mut self, start_index: usize) {
        let GKRCircuitDescription {
            input_layers: _,
            fiat_shamir_challenges: _,
            intermediate_layers,
            output_layers,
        } = self;
        intermediate_layers
            .iter_mut()
            .for_each(|intermediate_layer| {
                intermediate_layer.index_mle_indices(start_index);
            });
        output_layers.iter_mut().for_each(|output_layer| {
            output_layer.index_mle_indices(start_index);
        })
    }

    /// Returns an [InstantiatedCircuit] by populating the [GKRCircuitDescription] with data.
    /// Assumes that the input data has already been added to the transcript.
    ///
    /// # Arguments:
    /// * `input_data`: a [HashMap] mapping layer ids to the MLEs.
    /// * `challenge_sampler`: a closure that takes a string and a usize and returns that many field
    ///    elements; should be a wrapper of an instance method of the appropriate transcript.
    pub fn instantiate(
        &self,
        input_data: &HashMap<LayerId, MultilinearExtension<F>>,
        challenge_sampler: &mut impl FnMut(usize) -> Vec<F>,
    ) -> InstantiatedCircuit<F> {
        let GKRCircuitDescription {
            input_layers: input_layer_descriptions,
            fiat_shamir_challenges: fiat_shamir_challenge_descriptions,
            intermediate_layers: intermediate_layer_descriptions,
            output_layers: output_layer_descriptions,
        } = self;

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
        let mut prover_input_layers: Vec<InputLayer<F>> = Vec::new();
        let mut fiat_shamir_challenges = Vec::new();
        // Step 1a: populate the circuit map by compiling the necessary data
        // outputs for each of the input layers, while writing the commitments
        // to them into the transcript.
        input_layer_descriptions
            .iter()
            .for_each(|input_layer_description| {
                let input_layer_id = input_layer_description.layer_id;
                let combined_mle = input_data.get(&input_layer_id).unwrap();
                let mle_outputs_necessary = mle_claim_map.get(&input_layer_id).unwrap();
                // Compute all data outputs necessary for future layers for each
                // input layer.
                mle_outputs_necessary.iter().for_each(|mle_output| {
                    let prefix_bits = mle_output.prefix_bits();
                    let output = filter_bookkeeping_table(combined_mle, &prefix_bits);
                    circuit_map.add_node(CircuitLocation::new(input_layer_id, prefix_bits), output);
                });
                let prover_input_layer = InputLayer {
                    mle: combined_mle.clone(),
                    layer_id: input_layer_id,
                };
                prover_input_layers.push(prover_input_layer);
            });
        // Step 1b: for each of the fiat shamir challenges, use the transcript
        // in order to get the challenges and fill the layer.
        fiat_shamir_challenge_descriptions
            .iter()
            .for_each(|fiat_shamir_challenge_description| {
                let fiat_shamir_challenge_mle = MultilinearExtension::new(challenge_sampler(
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

        InstantiatedCircuit {
            input_layers: prover_input_layers,
            fiat_shamir_challenges,
            layers: Layers::new_with_layers(prover_intermediate_layers),
            output_layers: prover_output_layers,
            layer_map: circuit_map.convert_to_layer_map(),
        }
    }

    /// Verifies a GKR proof produced by the `prove` method.
    /// Assumes that all inputs and input commitments have already been added to transcript.
    /// # Arguments
    /// * `transcript_reader`: servers as the proof.
    #[instrument(skip_all, err)]
    fn verify(
        &mut self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<(), GKRError> {
        // TODO(Makis): Add circuit hash to Transcript.
        /*
        if let Some(circuit_hash) = maybe_circuit_hash {
            let transcript_circuit_hash = transcript_reader
                .consume_element("Circuit Hash")
                .map_err(GKRError::ErrorWhenVerifyingCircuitHash)?;
            assert_eq!(transcript_circuit_hash, circuit_hash);
        }
        */
        self.index_mle_indices(0);

        // Get the verifier challenges from the transcript.
        let fiat_shamir_challenges: Vec<FiatShamirChallenge<F>> = self
            .fiat_shamir_challenges
            .iter()
            .map(|fs_desc| {
                let values = transcript_reader
                    .get_challenges("Verifier challenges", 1 << fs_desc.num_bits)
                    .unwrap();
                fs_desc.instantiate(values)
            })
            .collect();

        // Claim aggregator to keep track of GKR-style claims across all layers.
        let mut aggregator = WLXAggregator::<F, LayerEnum<F>, InputLayerEnum<F>>::new();

        // --------- STAGE 1: Output Claim Generation ---------
        let claims_timer = start_timer!(|| "Output claims generation");
        let verifier_output_claims_span =
            span!(Level::DEBUG, "verifier_output_claims_span").entered();

        for circuit_output_layer in self.output_layers.iter() {
            let layer_id = circuit_output_layer.layer_id();
            info!("Verifying Output Layer: {:?}", layer_id);

            let verifier_output_layer = circuit_output_layer
                .retrieve_mle_from_transcript_and_fix_layer(transcript_reader)
                .map_err(|_| GKRError::ErrorWhenVerifyingOutputLayer)?;

            aggregator
                .extract_claims(&verifier_output_layer)
                .map_err(|_| GKRError::ErrorWhenVerifyingOutputLayer)?;
        }

        // dbg!(&aggregator);

        end_timer!(claims_timer);
        verifier_output_claims_span.exit();

        // --------- STAGE 2: Verify Intermediate Layers ---------
        let intermediate_layers_timer =
            start_timer!(|| "ALL intermediate layers proof verification");

        for layer in self.intermediate_layers.iter().rev() {
            let layer_id = layer.layer_id();

            info!("Intermediate Layer: {:?}", layer_id);
            let layer_timer =
                start_timer!(|| format!("Proof verification for layer {:?}", layer_id));

            let claim_aggr_timer =
                start_timer!(|| format!("Verify aggregated claim for layer {:?}", layer_id));

            let prev_claim = aggregator.verifier_aggregate_claims(layer_id, transcript_reader)?;
            debug!("Aggregated claim: {:#?}", prev_claim);

            end_timer!(claim_aggr_timer);

            info!("Verifier: about to verify layer");
            let sumcheck_msg_timer =
                start_timer!(|| format!("Verify sumcheck message for layer {:?}", layer_id));

            // Performs the actual sumcheck verification step.
            let verifier_layer: VerifierLayerEnum<F> = layer
                .verify_rounds(prev_claim, transcript_reader)
                .map_err(|err| {
                    GKRError::ErrorWhenVerifyingLayer(layer_id, LayerError::VerificationError(err))
                })?;

            end_timer!(sumcheck_msg_timer);

            aggregator
                .extract_claims(&verifier_layer)
                .map_err(|_| GKRError::ErrorWhenVerifyingOutputLayer)?;

            end_timer!(layer_timer);
        }

        end_timer!(intermediate_layers_timer);

        // FIXME(Ben) move to calling context
        // --------- STAGE 3: Verify Input Layers ---------
        // for (input_layer, commitment) in
        //     self.input_layers.iter().zip(input_layer_commitments.iter())
        // {
        //     let input_layer_claim =
        //         aggregator.verifier_aggregate_claims(input_layer_id, transcript_reader)?;

        //     input_layer
        //         .verify(commitment, input_layer_claim, transcript_reader)
        //         .map_err(GKRError::InputLayerError)?;
        // }

        // --------- STAGE 4: Verify claims on the verifier challenges ---------
        let fiat_shamir_challenges_timer = start_timer!(|| "Verifier challenges proof generation");
        for fiat_shamir_challenge in fiat_shamir_challenges {
            if let Some(claims) = aggregator.get_claims(fiat_shamir_challenge.layer_id()) {
                claims.iter().for_each(|claim_mle| {
                    fiat_shamir_challenge.verify(claim_mle.get_claim()).unwrap();
                });
            } else {
                return Err(GKRError::NoClaimsForLayer(fiat_shamir_challenge.layer_id()));
            }
        }
        end_timer!(fiat_shamir_challenges_timer);

        Ok(())
    }
}

/// Generate the circuit description given a set of nodes.
/// Returns a [GKRCircuitDescription] and a [InputNodeMap] that maps the ids of [InputLayerNode] to
/// their corresponding input layer id (this is a 1:1 correspondence).
pub fn generate_circuit_description<F: Field>(
    nodes: Vec<NodeEnum<F>>,
) -> Result<
    (
        GKRCircuitDescription<F>,
        InputNodeMap,
        impl Fn(
            HashMap<NodeId, MultilinearExtension<F>>,
        ) -> Result<HashMap<LayerId, MultilinearExtension<F>>, GKRError>,
    ),
    GKRError,
> {
    // FIXME(Ben) This doesn't seem well factored.  Pass in the return values of layout() as arguments to this function?  Inline layout here?
    let (
        input_layer_nodes,
        fiat_shamir_challenge_nodes,
        intermediate_nodes,
        lookup_nodes,
        output_nodes,
    ) = layout(nodes).unwrap();

    // Define counters for the layer ids
    let mut input_layer_id = LayerId::Input(0);
    let mut intermediate_layer_id = LayerId::Layer(0);
    let mut fiat_shamir_challenge_layer_id = LayerId::FiatShamirChallengeLayer(0);

    let mut intermediate_layers = Vec::<LayerDescriptionEnum<F>>::new();
    let mut output_layers = Vec::<MleOutputLayerDescription<F>>::new();
    let mut circuit_description_map = CircuitDescriptionMap::new();
    //FIXME(Ben) not needed anymore, we hope
    let mut input_layer_node_to_layer_map = InputNodeMap::new();

    let mut input_layer_id_to_input_shred_ids = HashMap::new();
    let mut input_layer_id_to_input_node_ids = HashMap::new();
    let input_layers = input_layer_nodes
        .iter()
        .map(|input_layer_node| {
            let input_layer_description = input_layer_node
                .generate_input_layer_description::<F>(
                    &mut input_layer_id,
                    &mut circuit_description_map,
                )
                .unwrap();
            input_layer_node_to_layer_map
                .add_node_layer_id(input_layer_description.layer_id, input_layer_node.id());
            input_layer_id_to_input_node_ids
                .insert(input_layer_description.layer_id(), input_layer_node.id());
            input_layer_id_to_input_shred_ids.insert(
                input_layer_description.layer_id,
                input_layer_node.subnodes().unwrap()
            );
            input_layer_description
        })
        .collect_vec();

    let fiat_shamir_challenges = fiat_shamir_challenge_nodes
        .iter()
        .map(|fiat_shamir_challenge_node| {
            fiat_shamir_challenge_node.generate_circuit_description::<F>(
                &mut fiat_shamir_challenge_layer_id,
                &mut circuit_description_map,
            )
        })
        .collect_vec();

    for node in &intermediate_nodes {
        let node_compiled_intermediate_layers = node
            .generate_circuit_description(&mut intermediate_layer_id, &mut circuit_description_map)
            .unwrap();
        intermediate_layers.extend(node_compiled_intermediate_layers);
    }

    // Get the contributions of each LookupTable to the circuit description.
    (intermediate_layers, output_layers) = lookup_nodes.iter().fold(
        (intermediate_layers, output_layers),
        |(mut lookup_intermediate_acc, mut lookup_output_acc), lookup_node| {
            let (intermediate_layers, output_layer) = lookup_node
                .generate_lookup_circuit_description(
                    &mut intermediate_layer_id,
                    &mut circuit_description_map,
                )
                .unwrap();
            lookup_intermediate_acc.extend(intermediate_layers);
            lookup_output_acc.push(output_layer);
            (lookup_intermediate_acc, lookup_output_acc)
        },
    );

    output_layers = output_nodes
        .iter()
        .fold(output_layers, |mut output_layer_acc, output_node| {
            output_layer_acc
                .extend(output_node.generate_circuit_description(&mut circuit_description_map));
            output_layer_acc
        });

    let circuit_description = GKRCircuitDescription {
        input_layers,
        fiat_shamir_challenges,
        intermediate_layers,
        output_layers,
    };

    // TODO(Ben) add the option to pass in input _layer_ node data as well
    let input_builder = move |input_node_data: HashMap<NodeId, MultilinearExtension<F>>| {
        let mut input_layer_data = HashMap::new();
        for (input_layer_id, input_shred_ids) in input_layer_id_to_input_shred_ids.iter() {
            let mut shred_mles_and_prefix_bits = vec![];
            for input_shred_id in input_shred_ids {
                let mle = input_node_data.get(input_shred_id).unwrap();
                let (circuit_location, num_vars) =
                    circuit_description_map.0.get(input_shred_id).unwrap();
                if *num_vars != mle.num_vars() {
                    return Err(GKRError::InputShredLengthMismatch(
                        *input_shred_id,
                        *num_vars,
                        mle.num_vars(),
                    ));
                }
                shred_mles_and_prefix_bits.push((mle, &circuit_location.prefix_bits))
            }
            let combined_mle = build_composite_mle(&shred_mles_and_prefix_bits);
            input_layer_data.insert(*input_layer_id, combined_mle);
        }
        Ok(input_layer_data)
    };

    Ok((
        circuit_description,
        input_layer_node_to_layer_map,
        input_builder,
    ))
}
