//! Defines the utilities for taking a list of nodes and turning it into a
//! layedout circuit
use std::collections::HashMap;
use std::fmt::Debug;

use itertools::Itertools;
use ligero::ligero_commit::ligero_verify;
use shared_types::circuit_hash::CircuitHashType;
use shared_types::config::global_config::get_current_global_verifier_config;
use shared_types::config::ProofConfig;
use shared_types::transcript::VerifierTranscript;
use shared_types::Halo2FFTFriendlyField;

use crate::input_layer::ligero_input_layer::{
    LigeroInputLayerDescriptionWithOptionalVerifierPrecommit, LigeroRoot,
};
use crate::prover::helpers::get_circuit_description_hash_as_field_elems;
use crate::prover::{GKRCircuitDescription, GKRError};
use crate::{layer::LayerId, mle::evals::MultilinearExtension};

use anyhow::{anyhow, Result};

/// A circuit that contains a [GKRCircuitDescription] alongside a description of
/// the committed input layers.
#[derive(Clone, Debug)]
pub struct VerifiableCircuit<F: Halo2FFTFriendlyField> {
    circuit_description: GKRCircuitDescription<F>,
    /// A partial mapping of public input layers to MLEs.
    /// Some (or all) public input layer IDs may be missing.
    /// The input layers present in this mapping are public input data that the verifier placed in
    /// the circuit, and which will be checked for equality with the respective public inputs in the
    /// proof during verification.
    predetermined_public_inputs: HashMap<LayerId, MultilinearExtension<F>>,
    committed_inputs: HashMap<LayerId, LigeroInputLayerDescriptionWithOptionalVerifierPrecommit<F>>,
    layer_label_to_layer_id: HashMap<String, LayerId>,
}

impl<F: Halo2FFTFriendlyField> VerifiableCircuit<F> {
    /// Returns a [VerifiableCircuit] initialized with the given data.
    pub fn new(
        circuit_description: GKRCircuitDescription<F>,
        predetermined_public_inputs: HashMap<LayerId, MultilinearExtension<F>>,
        committed_inputs: HashMap<
            LayerId,
            LigeroInputLayerDescriptionWithOptionalVerifierPrecommit<F>,
        >,
        layer_label_to_layer_id: HashMap<String, LayerId>,
    ) -> Self {
        Self {
            circuit_description,
            predetermined_public_inputs,
            committed_inputs,
            layer_label_to_layer_id,
        }
    }

    /// Returns a reference to the mapping which maps a [LayerId] of a committed input layer to its
    /// description.
    ///
    /// TODO: This is too transparent. Replace this with methods that answer the queries of the
    /// prover directly, and do _not_ expose it to the circuit developer.
    pub fn get_committed_inputs_ref(
        &self,
    ) -> &HashMap<LayerId, LigeroInputLayerDescriptionWithOptionalVerifierPrecommit<F>> {
        &self.committed_inputs
    }

    /// Returns a reference to the circuit description.
    ///
    /// TODO: This is only used by the back end. Do _not_ expose it to the circuit developer.
    pub fn get_gkr_circuit_description_ref(&self) -> &GKRCircuitDescription<F> {
        &self.circuit_description
    }

    /// Returns a vector of the [LayerId]s of all input layers which are public.
    pub fn get_public_input_layer_ids(&self) -> Vec<LayerId> {
        self.circuit_description
            .input_layers
            .iter()
            .filter(|input_layer_description| {
                // All input layers which are not committed are public by default.
                !self
                    .committed_inputs
                    .contains_key(&input_layer_description.layer_id)
            })
            .map(|public_input_layer_description| public_input_layer_description.layer_id)
            .collect()
    }

    /// Returns a vector of the [LayerId]s of all input layers which are committed
    /// to using a PCS.
    pub fn get_committed_input_layer_ids(&self) -> Vec<LayerId> {
        self.committed_inputs.keys().cloned().collect_vec()
    }

    /// Returns the data associated with the public input layer with ID `layer_id`, or None if
    /// no such public input layer exists.
    pub fn get_predetermined_public_input_mle_ref(
        &self,
        layer_id: &LayerId,
    ) -> Option<&MultilinearExtension<F>> {
        self.predetermined_public_inputs.get(layer_id)
    }

    /// Sets the pre-commitment of the input layer with label `layer_label` to `commitment`.
    /// Pre-commitments allow the verifier to check that an input layer commitment in the
    /// proof is the same as the expected `commitment` provided here.
    ///
    /// # Panics
    /// If `layer_label` is not a valid committed input layer label.
    pub fn set_pre_commitment(
        &mut self,
        layer_label: &str,
        commitment: LigeroRoot<F>,
    ) -> Result<()> {
        let layer_id = self.layer_label_to_layer_id.get(layer_label).unwrap();

        let (_, optional_commitment) = self
            .committed_inputs
            .get_mut(layer_id)
            .expect("Layer {layer_id} either does not exist, or is not a committed input layer.");

        *optional_commitment = Some(commitment);

        Ok(())
    }

    /// Verify a GKR proof from a transcript.
    pub fn verify(
        &self,
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
            self.get_public_input_layer_ids().len() + self.get_committed_inputs_ref().len(),
            self.get_gkr_circuit_description_ref().input_layers.len()
        );

        // Generate circuit description hash and check against prover-provided circuit description hash
        let hash_value_as_field_elems = get_circuit_description_hash_as_field_elems(
            self.get_gkr_circuit_description_ref(),
            circuit_description_hash_type,
        );
        let prover_supplied_circuit_description_hash = transcript
            .consume_elements("Circuit description hash", hash_value_as_field_elems.len())
            .unwrap();
        assert_eq!(
            prover_supplied_circuit_description_hash,
            hash_value_as_field_elems
        );

        let mut public_inputs = self.predetermined_public_inputs.clone();

        // Read and check public input values to transcript in order of layer id.
        self.get_public_input_layer_ids()
            .into_iter()
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .map(|layer_id| {
                let layer_desc = self
                    .get_gkr_circuit_description_ref()
                    .input_layers
                    .iter()
                    .find(|desc| desc.layer_id == layer_id)
                    .unwrap();
                let (transcript_evaluations, _expected_input_hash_chain_digest) = transcript
                    .consume_input_elements("Public input layer", 1 << layer_desc.num_vars)
                    .unwrap();
                match self.get_predetermined_public_input_mle_ref(&layer_id) {
                    Some(predetermined_public_input) => {
                        // If the verifier already knows what the input should be
                        // ahead of time, check against the transcript evaluations
                        // sent by the prover.
                        if predetermined_public_input.f.iter().collect_vec()
                            != transcript_evaluations
                        {
                            Err(anyhow!(GKRError::PublicInputLayerValuesMismatch(layer_id)))
                        } else {
                            Ok(())
                        }
                    }
                    None => {
                        // Otherwise, we append the proof values read from
                        // transcript to the public inputs.
                        public_inputs
                            .insert(layer_id, MultilinearExtension::new(transcript_evaluations));
                        Ok(())
                    }
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Sanitycheck: ensure that all of the public inputs are populated
        assert_eq!(public_inputs.len(), self.get_public_input_layer_ids().len());

        // Read the Ligero input layer commitments from transcript in order of layer id.
        let mut ligero_commitments = HashMap::<LayerId, F>::new();
        self.get_committed_input_layer_ids()
            .into_iter()
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .for_each(|layer_id| {
                let (commitment_as_vec, _expected_input_hash_chain_digest) = transcript
                    .consume_input_elements("Ligero commit", 1)
                    .unwrap();
                assert_eq!(commitment_as_vec.len(), 1);
                ligero_commitments.insert(layer_id, commitment_as_vec[0]);
            });

        let input_layer_claims = self
            .get_gkr_circuit_description_ref()
            .verify(transcript)
            .unwrap();

        // Every input layer claim is either for a public- or Ligero- input layer.
        let mut public_input_layer_claims = vec![];
        let mut ligero_input_layer_claims = vec![];
        input_layer_claims.into_iter().for_each(|claim| {
            let layer_id = claim.get_to_layer_id();
            if self.get_public_input_layer_ids().contains(&layer_id) {
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
            let claim_layer_id = claim.get_to_layer_id();
            let commitment = ligero_commitments.get(&claim_layer_id).unwrap();

            let (_, (desc, opt_pre_commitment)) = self
                .get_committed_inputs_ref()
                .iter()
                .find(|(layer_id, _)| **layer_id == claim_layer_id)
                .unwrap();

            if let Some(pre_commitment) = opt_pre_commitment {
                assert_eq!(pre_commitment.root, *commitment);
            }

            ligero_verify::<F>(
                commitment,
                &desc.aux,
                transcript,
                claim.get_point(),
                claim.get_eval(),
            );
        }

        Ok(())
    }
}
