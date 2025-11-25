//! Defines the utilities for taking a list of nodes and turning it into a
//! layedout circuit
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;

use itertools::Itertools;
use ligero::ligero_commit::{ligero_commit, ligero_eval_prove};
use shared_types::circuit_hash::CircuitHashType;
use shared_types::config::global_config::get_current_global_prover_config;
use shared_types::config::ProofConfig;
use shared_types::transcript::{ProverTranscript, TranscriptSponge, TranscriptWriter};
use shared_types::Halo2FFTFriendlyField;

use crate::input_layer::ligero_input_layer::{
    LigeroCommitment, LigeroInputLayerDescriptionWithOptionalProverPrecommit,
    LigeroInputLayerDescriptionWithOptionalVerifierPrecommit,
};
use crate::prover::helpers::get_circuit_description_hash_as_field_elems;
use crate::prover::{prove_circuit, GKRCircuitDescription, GKRError};
use crate::utils::debug::sanitycheck_input_layers_and_claims;
use crate::verifiable_circuit::VerifiableCircuit;
use crate::{layer::LayerId, mle::evals::MultilinearExtension};

use anyhow::{anyhow, Result};

/// A circuit, along with all of its input data, ready to be proven using the vanila GKR proving
/// system which uses Ligero as a PCS for committed input layers, and provides _no_ zero-knowledge
/// guarantees.
#[derive(Clone, Debug)]
pub struct ProvableCircuit<F: Halo2FFTFriendlyField> {
    circuit_description: GKRCircuitDescription<F>,
    inputs: HashMap<LayerId, MultilinearExtension<F>>,
    committed_inputs: HashMap<LayerId, LigeroInputLayerDescriptionWithOptionalProverPrecommit<F>>,
    layer_label_to_layer_id: HashMap<String, LayerId>,
}

impl<F: Halo2FFTFriendlyField> ProvableCircuit<F> {
    /// Constructor
    pub fn new(
        circuit_description: GKRCircuitDescription<F>,
        inputs: HashMap<LayerId, MultilinearExtension<F>>,
        committed_inputs: HashMap<
            LayerId,
            LigeroInputLayerDescriptionWithOptionalProverPrecommit<F>,
        >,
        layer_label_to_layer_id: HashMap<String, LayerId>,
    ) -> Self {
        Self {
            circuit_description,
            inputs,
            committed_inputs,
            layer_label_to_layer_id,
        }
    }

    /// # WARNING
    /// To be used only for testing and debugging.
    ///
    /// Constructs a form of this circuit that can be verified when a proof is provided.
    /// This is done by erasing all input data associated with committed input layers, along with
    /// any commitments (the latter can be found in the proof).
    pub fn _gen_verifiable_circuit(&self) -> VerifiableCircuit<F> {
        let public_ids: HashSet<LayerId> = self.get_public_input_layer_ids().into_iter().collect();
        let predetermined_public_inputs: HashMap<LayerId, MultilinearExtension<F>> = self
            .inputs
            .clone()
            .into_iter()
            .filter(|(layer_id, _)| public_ids.contains(layer_id))
            .collect();

        let committed_inputs: HashMap<
            LayerId,
            LigeroInputLayerDescriptionWithOptionalVerifierPrecommit<F>,
        > = self
            .committed_inputs
            .clone()
            .into_iter()
            .map(|(layer_id, (desc, opt_commit))| {
                (layer_id, (desc, opt_commit.map(|commit| commit.get_root())))
            })
            .collect();

        VerifiableCircuit::new(
            self.circuit_description.clone(),
            predetermined_public_inputs,
            committed_inputs,
            self.layer_label_to_layer_id.clone(),
        )
    }

    /// Returns a reference to the [GKRCircuitDescription] of this circuit.
    ///
    /// TODO: This is only used by the back end. Do _not_ expose it to the circuit developer.
    pub fn get_gkr_circuit_description_ref(&self) -> &GKRCircuitDescription<F> {
        &self.circuit_description
    }

    /// Returns a vector of the [LayerId]s of all input layers with visibility [LayerVisibility::Public].
    ///
    /// TODO: Consider returning an iterator instead.
    pub fn get_public_input_layer_ids(&self) -> Vec<LayerId> {
        self.inputs
            .keys()
            .filter(|layer_id| !self.committed_inputs.contains_key(layer_id))
            .cloned()
            .collect()
    }

    /// Returns a vector of the [LayerId]s of all input layers with visibility
    /// [LayerVisibility::Committed].
    ///
    /// TODO: Consider returning an iterator instead.
    pub fn get_committed_input_layer_ids(&self) -> Vec<LayerId> {
        self.committed_inputs.keys().cloned().collect()
    }

    /// Returns the data associated with the input layer with ID `layer_id`, or an error if there is
    /// no input layer with this ID.
    pub fn get_input_mle(&self, layer_id: LayerId) -> Result<MultilinearExtension<F>> {
        self.inputs
            .get(&layer_id)
            .ok_or(anyhow!("Unrecognized Layer ID '{layer_id}'"))
            .cloned()
    }

    /// Returns the description of the committed input layer with ID `layer_id`, or an error if no
    /// such committed input layer exists.
    pub fn get_committed_input_layer(
        &self,
        layer_id: LayerId,
    ) -> Result<LigeroInputLayerDescriptionWithOptionalProverPrecommit<F>> {
        self.committed_inputs
            .get(&layer_id)
            .ok_or(anyhow!("Unrecognized Layer ID '{layer_id}'"))
            .cloned()
    }

    /// Get a reference to the mapping that maps a [LayerId] of layer (either public or committed)
    /// to the data that are associated with it.
    ///
    /// TODO: This is too transparent. Replace this with methods that answer the queries of the
    /// prover directly, and do _not_ expose it to the circuit developer.
    pub fn get_inputs_ref(&self) -> &HashMap<LayerId, MultilinearExtension<F>> {
        &self.inputs
    }

    /// Write the GKR proof to transcript.
    /// Appends inputs and Ligero input layer commitments to the transcript: public inputs first, then Ligero input layers, ordering by layer id in both cases.
    /// Arguments:
    /// * `inputs` - a map from input layer ID to the MLE of its values (in the clear) for _all_ input layers.
    /// * `ligero_input_layers` - a vector of [LigeroInputLayerDescription]s, optionally paired with pre-computed commitments to their values (if provided, this are not checked, but simply used as is).
    /// * `circuit_description` - the [GKRCircuitDescription] of the circuit to be proven.
    pub fn prove<Tr: TranscriptSponge<F>>(
        &self,
        circuit_description_hash_type: CircuitHashType,
        transcript_writer: &mut TranscriptWriter<F, Tr>,
    ) -> Result<ProofConfig> {
        // Grab proof config from global config
        let proof_config = ProofConfig::new_from_prover_config(&get_current_global_prover_config());

        // Generate circuit description hash and append to transcript
        let hash_value_as_field_elems = get_circuit_description_hash_as_field_elems(
            self.get_gkr_circuit_description_ref(),
            circuit_description_hash_type,
        );
        transcript_writer.append_elements("Circuit description hash", &hash_value_as_field_elems);

        // Add the input values of any public (i.e. non-ligero) input layers to transcript.
        // Select the public input layers from the input layers, and sort them by layer id, and append
        // their input values to the transcript.
        self.get_public_input_layer_ids()
            .into_iter()
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .for_each(|layer_id| {
                let mle = self.get_input_mle(layer_id).unwrap();
                transcript_writer
                    .append_input_elements("Public input layer", &mle.iter().collect_vec());
            });

        // For each Ligero input layer, calculate commitments if not already provided, and then add each
        // commitment to the transcript.
        let mut ligero_input_commitments = HashMap::<LayerId, LigeroCommitment<F>>::new();

        self.get_committed_input_layer_ids()
            .into_iter()
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .for_each(|layer_id| {
                // Commit to the Ligero input layer, if it is not already committed to.
                let (desc, maybe_precommitment) = self.get_committed_input_layer(layer_id).unwrap();
                let commitment = if let Some(commitment) = maybe_precommitment {
                    commitment.clone()
                } else {
                    let input_mle = self.get_input_mle(layer_id).unwrap();
                    let (commitment, _) = ligero_commit(&input_mle.iter().collect_vec(), &desc.aux);
                    commitment
                };
                // Add the root of the commitment to the transcript.
                let root = commitment.get_root();
                transcript_writer.append_input_elements("Ligero commit", &[root.into_raw()]);
                // Store the commitment for later use.
                ligero_input_commitments.insert(layer_id, commitment);
            });

        // Mutate the transcript to contain the proof of the intermediate layers of the circuit,
        // and return the claims on the input layer.
        let input_layer_claims = prove_circuit(self, transcript_writer).unwrap();

        // If in performance debugging mode, print the number of claims on input
        // layers and input layer sizes.
        if cfg!(feature = "performance-debug") {
            sanitycheck_input_layers_and_claims(
                &input_layer_claims,
                self.get_gkr_circuit_description_ref(),
            );
        }

        // If in debug mode, then check the claims on all input layers.
        if cfg!(debug_assertions) {
            for claim in input_layer_claims.iter() {
                let input_mle = self.get_input_mle(claim.get_to_layer_id()).unwrap();
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
            if let Ok((desc, _)) = self.get_committed_input_layer(layer_id) {
                let mle = self.get_input_mle(layer_id).unwrap();
                let commitment = ligero_input_commitments.get(&layer_id).unwrap();
                ligero_eval_prove(
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
}
