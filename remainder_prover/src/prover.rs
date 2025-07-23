//! Modules that orchestrate creating a GKR Proof
#![allow(clippy::type_complexity)]

/// Includes boilerplate for creating a GKR circuit, i.e. creating a transcript, proving, verifying, etc.
pub mod helpers;

/// Includes various traits that define interfaces of a GKR Prover
pub mod proof_system;

/// Struct for representing a list of layers
pub mod layers;

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use self::layers::Layers;
use crate::claims::claim_aggregation::{prover_aggregate_claims, verifier_aggregate_claims};
use crate::claims::{Claim, ClaimTracker};
use crate::expression::circuit_expr::filter_bookkeeping_table;
use crate::input_layer::fiat_shamir_challenge::{
    FiatShamirChallenge, FiatShamirChallengeDescription,
};
use crate::input_layer::ligero_input_layer::{LigeroCommitment, LigeroInputLayerDescription};
use crate::input_layer::{InputLayer, InputLayerDescription};
use crate::layer::layer_enum::{LayerDescriptionEnum, VerifierLayerEnum};
use crate::layer::{layer_enum::LayerEnum, LayerId};
use crate::layer::{Layer, LayerDescription, VerifierLayer};
use crate::layouter::layouting::{layout, CircuitDescriptionMap, CircuitLocation, CircuitMap};
use crate::layouter::nodes::node_enum::NodeEnum;
use crate::layouter::nodes::{CircuitNode, NodeId};
use crate::mle::dense::DenseMle;
use crate::mle::evals::MultilinearExtension;
use crate::mle::mle_description::MleDescription;
use crate::mle::mle_enum::MleEnum;
use crate::output_layer::{OutputLayer, OutputLayerDescription};
use crate::utils::mle::verify_claim;
use ark_std::{end_timer, start_timer};
use helpers::get_circuit_description_hash_as_field_elems;
use itertools::Itertools;
use remainder_ligero::ligero_commit::{
    remainder_ligero_commit, remainder_ligero_eval_prove, remainder_ligero_verify,
};
use remainder_shared_types::circuit_hash::CircuitHashType;
use remainder_shared_types::config::global_config::{
    get_current_global_prover_config, get_current_global_verifier_config, global_claim_agg_strategy,
};
use remainder_shared_types::config::{ClaimAggregationStrategy, ProofConfig};
use remainder_shared_types::field::Halo2FFTFriendlyField;
use remainder_shared_types::transcript::poseidon_sponge::PoseidonSponge;
use remainder_shared_types::transcript::VerifierTranscript;
use remainder_shared_types::transcript::{ProverTranscript, TranscriptWriter};
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info};
use tracing::{instrument, span, Level};

use anyhow::{anyhow, Result};

/// Errors that can be generated during GKR proving.
#[derive(Error, Debug, Clone)]
pub enum GKRError {
    #[error("No claims were found for layer {0:?}")]
    /// No claims were found for layer
    NoClaimsForLayer(LayerId),
    #[error("Error when proving layer {0:?}")]
    /// Error when proving layer
    ErrorWhenProvingLayer(LayerId),
    #[error("Error when verifying layer {0:?}")]
    /// Error when verifying layer
    ErrorWhenVerifyingLayer(LayerId),
    /// The evaluation of the input layer doesn't match the value of the claim from the layer
    #[error("Evaluation of input layer {0:?} doesn't match value of a claim originating from layer {0:?}.")]
    EvaluationMismatch(LayerId, LayerId),
    /// The public input layer values were not as expected by the verifier
    #[error("Values for public input layer {0:?} were not as expected by the verifier.")]
    PublicInputLayerValuesMismatch(LayerId),
    /// The verifier's claim tracker was not empty at the end of the verification process
    #[error("Verifier's claim tracker was not empty at the end of the verification process.")]
    ClaimTrackerNotEmpty,

    #[error("Error when verifying output layer")]
    /// Error when verifying output layer
    ErrorWhenVerifyingOutputLayer,
    /// InputShred length mismatch
    #[error("InputShred with NodeId {0} should have {1} variables, but has {2}")]
    InputShredLengthMismatch(NodeId, usize, usize),
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
    pub output_layers: Vec<OutputLayer<F>>,
    /// The input layers of the circuit
    pub input_layers: Vec<InputLayer<F>>,
    /// The verifier challenges
    pub fiat_shamir_challenges: Vec<FiatShamirChallenge<F>>,
    /// Maps LayerId to the MLE of its values
    pub layer_map: HashMap<LayerId, Vec<DenseMle<F>>>,
}

/// Write the GKR proof to transcript.
/// Appends inputs and Ligero input layer commitments to the transcript: public inputs first, then Ligero input layers, ordering by layer id in both cases.
/// Arguments:
/// * `inputs` - a map from input layer ID to the MLE of its values (in the clear) for _all_ input layers.
/// * `ligero_input_layers` - a vector of [LigeroInputLayerDescription]s, optionally paired with pre-computed commitments to their values (if provided, this are not checked, but simply used as is).
/// * `circuit_description` - the [GKRCircuitDescription] of the circuit to be proven.
pub fn prove<F: Halo2FFTFriendlyField>(
    inputs: &HashMap<LayerId, MultilinearExtension<F>>,
    ligero_input_layers: &HashMap<
        LayerId,
        (LigeroInputLayerDescription<F>, Option<LigeroCommitment<F>>),
    >,
    circuit_description: &GKRCircuitDescription<F>,
    circuit_description_hash_type: CircuitHashType,
    transcript_writer: &mut TranscriptWriter<F, PoseidonSponge<F>>,
) -> Result<ProofConfig> {
    // Grab proof config from global config
    let proof_config = ProofConfig::new_from_prover_config(&get_current_global_prover_config());

    // Generate circuit description hash and append to transcript
    let hash_value_as_field_elems = get_circuit_description_hash_as_field_elems(
        circuit_description,
        circuit_description_hash_type,
    );
    transcript_writer.append_elements("Circuit description hash", &hash_value_as_field_elems);

    // Add the input values of any public (i.e. non-ligero) input layers to transcript.
    // Select the public input layers from the input layers, and sort them by layer id, and append
    // their input values to the transcript.
    inputs
        .keys()
        .filter(|layer_id| !ligero_input_layers.contains_key(layer_id))
        .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
        .for_each(|layer_id| {
            let mle = inputs.get(layer_id).unwrap();
            transcript_writer
                .append_input_elements("Public input layer", &mle.iter().collect_vec());
        });

    // For each Ligero input layer, calculate commitments if not already provided, and then add each
    // commitment to the transcript.
    let mut ligero_input_commitments = HashMap::<LayerId, LigeroCommitment<F>>::new();
    ligero_input_layers
        .keys()
        .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
        .for_each(|layer_id| {
            // Commit to the Ligero input layer, if it is not already committed to.
            let (desc, maybe_precommitment) = ligero_input_layers.get(layer_id).unwrap();
            let commitment = if let Some(commitment) = maybe_precommitment {
                commitment.clone()
            } else {
                let input_mle = inputs.get(layer_id).unwrap();
                let (commitment, _) =
                    remainder_ligero_commit(&input_mle.iter().collect_vec(), &desc.aux);
                commitment
            };
            // Add the root of the commitment to the transcript.
            let root = commitment.get_root();
            transcript_writer.append_input_elements("Ligero commit", &[root.into_raw()]);
            // Store the commitment for later use.
            ligero_input_commitments.insert(*layer_id, commitment);
        });

    // Mutate the transcript to contain the proof of the intermediate layers of the circuit,
    // and return the claims on the input layer.
    let input_layer_claims = prove_circuit(circuit_description, inputs, transcript_writer).unwrap();

    // If in debug mode, then check the claims on all input layers.
    if cfg!(debug_assertions) {
        for claim in input_layer_claims.iter() {
            let input_mle = inputs.get(&claim.get_to_layer_id()).unwrap();
            let evaluation = input_mle.evaluate_at_point(claim.get_point());
            if evaluation != claim.get_eval() {
                return Err(anyhow!(GKRError::EvaluationMismatch(
                    claim.get_to_layer_id(),
                    claim.get_to_layer_id(),
                )));
            }
        }
    }

    // Create a Ligero evaluation proof for each claim on a Ligero input layer, writing it to transcript.
    for claim in input_layer_claims.iter() {
        let layer_id = claim.get_to_layer_id();
        if let Some((desc, _)) = ligero_input_layers.get(&layer_id) {
            let mle = inputs.get(&layer_id).unwrap();
            let commitment = ligero_input_commitments.get(&layer_id).unwrap();
            remainder_ligero_eval_prove(
                &mle.f.iter().collect_vec(),
                claim.get_point(),
                transcript_writer,
                &desc.aux,
                commitment,
            )
            .unwrap();
        }
    }

    Ok(proof_config)
}

/// Verify a GKR proof from a transcript.
pub fn verify<F: Halo2FFTFriendlyField>(
    public_inputs: &HashMap<LayerId, MultilinearExtension<F>>,
    ligero_inputs: &[LigeroInputLayerDescription<F>],
    circuit_description: &GKRCircuitDescription<F>,
    circuit_description_hash_type: CircuitHashType,
    transcript: &mut impl VerifierTranscript<F>,
    proof_config: &ProofConfig,
) -> Result<()> {
    // Check whether proof config matches current global verifier config
    if !get_current_global_verifier_config().matches_proof_config(proof_config) {
        panic!("Error: Attempted to verify a GKR proof whose config doesn't match that of the verifier.");
    }

    // Check the shape of the circuit description against the proof -- only needed to be done
    // for the input layers, because for intermediate and output layers, the proof is in the
    // transcript, which the verifier checks shape against the circuit description already.
    assert_eq!(
        public_inputs.len() + ligero_inputs.len(),
        circuit_description.input_layers.len()
    );

    // Generate circuit description hash and check against prover-provided circuit description hash
    let hash_value_as_field_elems = get_circuit_description_hash_as_field_elems(
        circuit_description,
        circuit_description_hash_type,
    );
    let prover_supplied_circuit_description_hash = transcript
        .consume_elements("Circuit description hash", hash_value_as_field_elems.len())
        .unwrap();
    assert_eq!(
        prover_supplied_circuit_description_hash,
        hash_value_as_field_elems
    );

    // Read and check public input values to transcript in order of layer id.
    public_inputs
        .keys()
        .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
        .map(|layer_id| {
            let layer_desc = circuit_description
                .input_layers
                .iter()
                .find(|desc| desc.layer_id == *layer_id)
                .unwrap();
            let (transcript_mle, _expected_input_hash_chain_digest) = transcript
                .consume_input_elements("Public input layer", 1 << layer_desc.num_vars)
                .unwrap();
            let expected_mle = public_inputs.get(layer_id).unwrap();
            if expected_mle.f.iter().collect_vec() != transcript_mle {
                Err(anyhow!(GKRError::PublicInputLayerValuesMismatch(*layer_id)))
            } else {
                Ok(())
            }
        })
        .collect::<Result<Vec<_>>>()?;

    // Read the Ligero input layer commitments from transcript in order of layer id.
    let mut ligero_commitments = HashMap::<LayerId, F>::new();
    ligero_inputs
        .iter()
        .sorted_by_key(|desc| desc.layer_id.get_raw_input_layer_id())
        .for_each(|desc| {
            let (commitment_as_vec, _expected_input_hash_chain_digest) = transcript
                .consume_input_elements("Ligero commit", 1)
                .unwrap();
            assert_eq!(commitment_as_vec.len(), 1);
            ligero_commitments.insert(desc.layer_id, commitment_as_vec[0]);
        });

    let input_layer_claims = circuit_description.verify(transcript).unwrap();

    // Every input layer claim is either for a public- or Ligero- input layer.
    let mut public_input_layer_claims = vec![];
    let mut ligero_input_layer_claims = vec![];
    input_layer_claims.into_iter().for_each(|claim| {
        let layer_id = claim.get_to_layer_id();
        if public_inputs.contains_key(&layer_id) {
            public_input_layer_claims.push(claim);
        } else if ligero_commitments.contains_key(&layer_id) {
            ligero_input_layer_claims.push(claim);
        } else {
            // This can only be a programming error on our part (since there was sufficient input
            // data to verify the proof of the circuit).
            panic!("Input layer {layer_id:?} has a claim but is not a public input layer nor a Ligero input layer.");
        }
    });

    // Check the claims on public input layers via explicit evaluation.
    for claim in public_input_layer_claims.iter() {
        let input_mle = public_inputs.get(&claim.get_to_layer_id()).unwrap();
        let evaluation = input_mle.evaluate_at_point(claim.get_point());
        if evaluation != claim.get_eval() {
            return Err(anyhow!(GKRError::EvaluationMismatch(
                claim.get_to_layer_id(),
                claim.get_from_layer_id(),
            )));
        }
    }

    // Check the claims on Ligero input layers via their evaluation proofs.
    for claim in ligero_input_layer_claims.iter() {
        let layer_id = claim.get_to_layer_id();
        let commitment = ligero_commitments.get(&layer_id).unwrap();
        let desc = ligero_inputs
            .iter()
            .find(|desc| desc.layer_id == layer_id)
            .unwrap();
        remainder_ligero_verify::<F>(
            commitment,
            &desc.aux,
            transcript,
            claim.get_point(),
            claim.get_eval(),
        );
    }

    Ok(())
}

/// Assumes that the inputs have already been added to the transcript (if necessary).
/// Returns the vector of claims on the input layers.
pub fn prove_circuit<F: Field>(
    circuit_description: &GKRCircuitDescription<F>,
    inputs: &HashMap<LayerId, MultilinearExtension<F>>,
    transcript_writer: &mut TranscriptWriter<F, PoseidonSponge<F>>,
) -> Result<Vec<Claim<F>>> {
    // Note: no need to return the Transcript, since it is already in the TranscriptWriter!
    // Note(Ben): this can't be an instance method, because it consumes the intermediate layers!
    // Note(Ben): this is a GKR specific method.  So it makes sense for IT to define the challenge sampler, so that the circuit can be instantiated (rather than leaving this complexity to the calling context).

    let mut challenge_sampler =
        |size| transcript_writer.get_challenges("Verifier challenges", size);
    let instantiated_circuit = circuit_description.instantiate(inputs, &mut challenge_sampler);

    let InstantiatedCircuit {
        input_layers,
        mut output_layers,
        layers,
        fiat_shamir_challenges: _fiat_shamir_challenges,
        mut layer_map,
    } = instantiated_circuit;

    // Maps a `LayerId` to a collection of claims made on that layer.
    let mut claim_tracker = ClaimTracker::new();

    // --------- STAGE 1: Output Claim Generation ---------
    let claims_timer = start_timer!(|| "Output claims generation");
    let output_claims_span = span!(Level::DEBUG, "output_claims_span").entered();

    // Go through circuit output layers and grab claims on each.
    for output in output_layers.iter_mut() {
        let layer_id = output.layer_id();
        info!("Output Layer: {:?}", layer_id);

        match output.get_mle() {
            MleEnum::Dense(_) => {
                panic!("We don't support DenseMLE as output layers for now")
            }
            // Just write a single zero into the transcript since the counts (layer size) are already included in the circuit description
            MleEnum::Zero(_) => {
                transcript_writer.append_elements("Output layer MLE evals", &[F::ZERO])
            }
        };

        let challenges = transcript_writer
            .get_challenges("Challenge on the output layer", output.num_free_vars());
        output.fix_layer(&challenges)?;

        let claim = output.get_claim()?;
        claim_tracker.insert(claim.get_to_layer_id(), claim);
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

        let output_mles_from_layer = layer_map.remove(&layer_id).unwrap();
        let layer_claims = claim_tracker.get(layer_id).unwrap();

        // We always want to perform interpolative claim aggregation on MatMult layers.
        if let LayerEnum::MatMult(_) = layer {
            let claim_aggr_timer =
                start_timer!(|| format!("Claim aggregation for layer {:?}", layer_id));
            let layer_claim =
                prover_aggregate_claims(layer_claims, output_mles_from_layer, transcript_writer)?;
            end_timer!(claim_aggr_timer);

            info!("Prove sumcheck message");
            let sumcheck_msg_timer = start_timer!(|| format!(
                "Compute sumcheck message for layer {:?}",
                layer.layer_id()
            ));

            // Compute all sumcheck messages across this particular layer.
            layer.prove(&[&layer_claim], transcript_writer)?;

            end_timer!(sumcheck_msg_timer);
        }
        // Otherwise, we perform claim aggregation specified by the claim agg strategy.
        else {
            match global_claim_agg_strategy() {
                ClaimAggregationStrategy::Interpolative => {
                    let claim_aggr_timer =
                        start_timer!(|| format!("Claim aggregation for layer {:?}", layer_id));
                    let layer_claim = prover_aggregate_claims(
                        layer_claims,
                        output_mles_from_layer,
                        transcript_writer,
                    )?;
                    end_timer!(claim_aggr_timer);

                    info!("Prove sumcheck message");
                    let sumcheck_msg_timer = start_timer!(|| format!(
                        "Compute sumcheck message for layer {:?}",
                        layer.layer_id()
                    ));

                    // Compute all sumcheck messages across this particular layer.
                    layer.prove(&[&layer_claim], transcript_writer)?;

                    end_timer!(sumcheck_msg_timer);
                }
                ClaimAggregationStrategy::RLC => {
                    let sumcheck_msg_timer = start_timer!(|| format!(
                        "Compute sumcheck message for layer {:?}",
                        layer.layer_id()
                    ));

                    layer.prove(
                        &layer_claims
                            .iter()
                            .map(|claim| claim.get_raw_claim())
                            .collect_vec(),
                        transcript_writer,
                    )?;
                    end_timer!(sumcheck_msg_timer);
                }
            }
        }

        for claim in layer.get_claims()? {
            claim_tracker.insert(claim.get_to_layer_id(), claim);
        }

        end_timer!(layer_timer);
    }

    end_timer!(intermediate_layers_timer);
    all_layers_sumcheck_proving_span.exit();

    let input_layer_claims = input_layers
        .iter()
        .filter_map(|input_layer| claim_tracker.get(input_layer.layer_id))
        .flatten()
        .cloned()
        .collect_vec();

    Ok(input_layer_claims)
}

/// The complete description of a layered circuit whose output validity can be
/// proven against a set of committed inputs.
#[derive(Debug, Serialize, Deserialize, Hash, Clone)]
#[serde(bound = "F: Field")]
pub struct GKRCircuitDescription<F: Field> {
    /// The circuit descriptions of the input layers.
    pub input_layers: Vec<InputLayerDescription>,
    /// The circuit descriptions of the verifier challengs
    pub fiat_shamir_challenges: Vec<FiatShamirChallengeDescription<F>>,
    /// The circuit descriptions of the intermediate layers.
    pub intermediate_layers: Vec<LayerDescriptionEnum<F>>,
    /// The circuit desriptions of the output layers.
    pub output_layers: Vec<OutputLayerDescription<F>>,
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
    ///   elements; should be a wrapper of an instance method of the appropriate transcript.
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
                fiat_shamir_challenges.push(FiatShamirChallenge {
                    mle: fiat_shamir_challenge_mle,
                    layer_id: fiat_shamir_challenge_description.layer_id(),
                });
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
        let mut prover_output_layers: Vec<OutputLayer<F>> = Vec::new();
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

    /// Verifies a GKR circuit proof produced by the `prove` method.
    /// Assumes that the circuit description, all inputs and input commitments have already been added to transcript.
    /// # Arguments
    /// * `transcript_reader`: servers as the proof.
    /// Returns claims on the input layers.
    #[instrument(skip_all, err)]
    fn verify(&self, transcript_reader: &mut impl VerifierTranscript<F>) -> Result<Vec<Claim<F>>> {
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

        // Claim tracker to keep track of GKR-style claims across all layers.
        let mut claim_tracker = ClaimTracker::new();

        // --------- Output Claim Generation ---------
        let claims_timer = start_timer!(|| "Output claims generation");
        let verifier_output_claims_span =
            span!(Level::DEBUG, "verifier_output_claims_span").entered();

        for circuit_output_layer in self.output_layers.iter() {
            let layer_id = circuit_output_layer.layer_id();
            info!("Verifying Output Layer: {:?}", layer_id);

            let verifier_output_layer = circuit_output_layer
                .retrieve_mle_from_transcript_and_fix_layer(transcript_reader)?;

            let claim = verifier_output_layer.get_claim()?;
            claim_tracker.insert(claim.get_to_layer_id(), claim);
        }

        end_timer!(claims_timer);
        verifier_output_claims_span.exit();

        // --------- Verify Intermediate Layers ---------
        let intermediate_layers_timer =
            start_timer!(|| "ALL intermediate layers proof verification");

        for layer in self.intermediate_layers.iter().rev() {
            let layer_id = layer.layer_id();

            info!("Intermediate Layer: {:?}", layer_id);
            let layer_timer =
                start_timer!(|| format!("Proof verification for layer {:?}", layer_id));

            let layer_claims = claim_tracker.remove(layer_id).unwrap();

            let verifier_layer = match global_claim_agg_strategy() {
                ClaimAggregationStrategy::Interpolative => {
                    let claim_aggr_timer =
                        start_timer!(|| format!("Claim aggregation for layer {:?}", layer_id));
                    let prev_claim = verifier_aggregate_claims(&layer_claims, transcript_reader)?;
                    debug!("Aggregated claim: {:#?}", prev_claim);
                    end_timer!(claim_aggr_timer);

                    info!("Prove sumcheck message");
                    let sumcheck_msg_timer = start_timer!(|| format!(
                        "Compute sumcheck message for layer {:?}",
                        layer.layer_id()
                    ));

                    // Performs the actual sumcheck verification step.
                    let verifier_layer: VerifierLayerEnum<F> =
                        layer.verify_rounds(&[&prev_claim], transcript_reader)?;

                    end_timer!(sumcheck_msg_timer);

                    verifier_layer
                }
                ClaimAggregationStrategy::RLC => {
                    let sumcheck_msg_timer = start_timer!(|| format!(
                        "Compute sumcheck message for layer {:?}",
                        layer.layer_id()
                    ));

                    let verifier_layer = layer.verify_rounds(
                        &layer_claims
                            .iter()
                            .map(|claim| claim.get_raw_claim())
                            .collect_vec(),
                        transcript_reader,
                    )?;
                    end_timer!(sumcheck_msg_timer);

                    verifier_layer
                }
            };

            for claim in verifier_layer.get_claims()? {
                claim_tracker.insert(claim.get_to_layer_id(), claim);
            }

            end_timer!(layer_timer);
        }

        end_timer!(intermediate_layers_timer);

        // --------- Verify claims on the verifier challenges ---------
        let fiat_shamir_challenges_timer = start_timer!(|| "Verifier challenges proof generation");
        for fiat_shamir_challenge in fiat_shamir_challenges {
            if let Some(claims) = claim_tracker.remove(fiat_shamir_challenge.layer_id()) {
                claims.iter().for_each(|claim| {
                    verify_claim(&fiat_shamir_challenge.mle.to_vec(), claim.get_raw_claim());
                });
            } else {
                return Err(anyhow!(GKRError::NoClaimsForLayer(
                    fiat_shamir_challenge.layer_id()
                )));
            }
        }
        end_timer!(fiat_shamir_challenges_timer);

        let input_layer_claims = self
            .input_layers
            .iter()
            .flat_map(|input_layer| claim_tracker.remove(input_layer.layer_id).unwrap())
            .collect_vec();

        // Verify that there are no claims remaining in the claim tracker.
        if !claim_tracker.is_empty() {
            return Err(anyhow!(GKRError::ClaimTrackerNotEmpty));
        }

        Ok(input_layer_claims)
    }
}

/// Generate the circuit description given a set of [NodeEnum]s.
/// Returns a [GKRCircuitDescription], and a function that takes a map of input shred data and returns a
/// map of input layer data.
/// The returned circuit description already has indices assigned to the MLEs.
pub fn generate_circuit_description<F: Field>(
    nodes: Vec<NodeEnum<F>>,
) -> Result<(
    GKRCircuitDescription<F>,
    HashMap<LayerId, Vec<NodeId>>,
    CircuitDescriptionMap,
)> {
    // FIXME This doesn't seem well factored.  Pass in the return values of layout() as arguments to this function?  Inline layout here?
    let (
        input_layer_nodes,
        fiat_shamir_challenge_nodes,
        intermediate_nodes,
        lookup_nodes,
        output_nodes,
    ) = layout(nodes).unwrap();
    let mut intermediate_layers = Vec::<LayerDescriptionEnum<F>>::new();
    let mut output_layers = Vec::<OutputLayerDescription<F>>::new();
    let mut circuit_description_map = CircuitDescriptionMap::new();

    let mut input_layer_id_to_input_shred_ids = HashMap::new();
    let input_layers = input_layer_nodes
        .iter()
        .map(|input_layer_node| {
            let input_layer_description = input_layer_node
                .generate_input_layer_description::<F>(&mut circuit_description_map)
                .unwrap();
            input_layer_id_to_input_shred_ids.insert(
                input_layer_description.layer_id,
                input_layer_node.subnodes().unwrap(),
            );
            input_layer_description
        })
        .collect_vec();

    let fiat_shamir_challenges = fiat_shamir_challenge_nodes
        .iter()
        .map(|fiat_shamir_challenge_node| {
            fiat_shamir_challenge_node
                .generate_circuit_description::<F>(&mut circuit_description_map)
        })
        .collect_vec();

    for node in &intermediate_nodes {
        let node_compiled_intermediate_layers = node
            .generate_circuit_description(&mut circuit_description_map)
            .unwrap();
        intermediate_layers.extend(node_compiled_intermediate_layers);
    }

    // Get the contributions of each LookupTable to the circuit description.
    (intermediate_layers, output_layers) = lookup_nodes.iter().fold(
        (intermediate_layers, output_layers),
        |(mut lookup_intermediate_acc, mut lookup_output_acc), lookup_node| {
            let (intermediate_layers, output_layer) = lookup_node
                .generate_lookup_circuit_description(&mut circuit_description_map)
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

    let mut circuit_description = GKRCircuitDescription {
        input_layers,
        fiat_shamir_challenges,
        intermediate_layers,
        output_layers,
    };
    circuit_description.index_mle_indices(0);

    Ok((
        circuit_description,
        input_layer_id_to_input_shred_ids,
        circuit_description_map,
    ))
}
