//! Defines the utilities for taking a list of nodes and turning it into a
//! layedout circuit
use std::collections::HashSet;
use std::collections::{hash_map::Entry, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

use itertools::Itertools;
use remainder_ligero::ligero_commit::{
    remainder_ligero_commit, remainder_ligero_eval_prove, remainder_ligero_verify,
};
use remainder_shared_types::circuit_hash::CircuitHashType;
use remainder_shared_types::config::global_config::{
    get_current_global_prover_config, get_current_global_verifier_config,
};
use remainder_shared_types::config::ProofConfig;
use remainder_shared_types::transcript::{
    ProverTranscript, TranscriptSponge, TranscriptWriter, VerifierTranscript,
};
use remainder_shared_types::{Field, Halo2FFTFriendlyField};
use serde::{Deserialize, Serialize};

use crate::input_layer::ligero_input_layer::{
    LigeroCommitment, LigeroInputLayerDescriptionWithOptionalProverPrecommit,
    LigeroInputLayerDescriptionWithOptionalVerifierPrecommit, LigeroRoot,
};
use crate::prover::helpers::get_circuit_description_hash_as_field_elems;
use crate::prover::{prove_circuit, GKRCircuitDescription, GKRError};
use crate::utils::debug::sanitycheck_input_layers_and_claims;
use crate::{
    layer::LayerId,
    mle::{dense::DenseMle, evals::MultilinearExtension, mle_description::MleDescription},
};

use anyhow::{anyhow, Result};

/// A HashMap that records during circuit compilation where nodes live in the
/// circuit and what data they yield.
#[derive(Debug)]
pub struct CircuitEvalMap<F: Field>(pub(crate) HashMap<CircuitLocation, MultilinearExtension<F>>);
/// A map that maps layer ID to all the MLEs that are output from that layer.
/// Together these MLEs are combined along with the information from their
/// prefix bits to form the layerwise bookkeeping table.
pub type LayerMap<F> = HashMap<LayerId, Vec<DenseMle<F>>>;

impl<F: Field> CircuitEvalMap<F> {
    /// Create a new circuit map, which maps circuit location to the data stored
    /// at that location.removing
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Using the circuit location, which is a layer_id and prefix_bits tuple,
    /// get the data that exists here.
    pub fn get_data_from_circuit_mle(
        &self,
        circuit_mle: &MleDescription<F>,
    ) -> Result<&MultilinearExtension<F>> {
        let circuit_location =
            CircuitLocation::new(circuit_mle.layer_id(), circuit_mle.prefix_bits());
        let result = self
            .0
            .get(&circuit_location)
            .ok_or(anyhow!("Circuit location not found!"));
        if let Ok(actual_result) = result {
            assert_eq!(actual_result.num_vars(), circuit_mle.num_free_vars());
        }
        result
    }

    /// Adds a new node to the CircuitMap
    pub fn add_node(&mut self, circuit_location: CircuitLocation, value: MultilinearExtension<F>) {
        self.0.insert(circuit_location, value);
    }

    /// Destructively convert this into a map that maps LayerId to the
    /// [MultilinearExtension]s that generate claims on this area. This is to
    /// aid in claim aggregation, so we know the parts of the layerwise
    /// bookkeeping table in order to aggregate claims on this layer.
    pub fn convert_to_layer_map(mut self) -> LayerMap<F> {
        let mut layer_map = HashMap::<LayerId, Vec<DenseMle<F>>>::new();
        self.0.drain().for_each(|(circuit_location, data)| {
            let corresponding_mle = DenseMle::new_with_prefix_bits(
                data,
                circuit_location.layer_id,
                circuit_location.prefix_bits,
            );
            if let Entry::Vacant(e) = layer_map.entry(circuit_location.layer_id) {
                e.insert(vec![corresponding_mle]);
            } else {
                layer_map
                    .get_mut(&circuit_location.layer_id)
                    .unwrap()
                    .push(corresponding_mle);
            }
        });
        layer_map
    }
}

impl<F: Field> Default for CircuitEvalMap<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// The location of a Node in the circuit
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CircuitLocation {
    /// The LayerId this node has been placed into
    pub layer_id: LayerId,
    /// Any prefix_bits neccessary to differenciate claims made onto this node
    /// from other nodes in the same layer
    pub prefix_bits: Vec<bool>,
}

impl CircuitLocation {
    /// Creates a new CircuitLocation
    pub fn new(layer_id: LayerId, prefix_bits: Vec<bool>) -> Self {
        Self {
            layer_id,
            prefix_bits,
        }
    }
}

/// A circuit, along with all of its input data, ready to be proven using the vanila GKR proving
/// system which uses Ligero as a PCS for private input layers, and provides no zero-knowledge
/// guarantees.
#[derive(Clone, Debug)]
pub struct ProvableCircuit<F: Halo2FFTFriendlyField> {
    circuit_description: GKRCircuitDescription<F>,
    inputs: HashMap<LayerId, MultilinearExtension<F>>,
    private_inputs: HashMap<LayerId, LigeroInputLayerDescriptionWithOptionalProverPrecommit<F>>,
    layer_label_to_layer_id: HashMap<String, LayerId>,
}

impl<F: Halo2FFTFriendlyField> ProvableCircuit<F> {
    /// Constructor
    pub fn new(
        circuit_description: GKRCircuitDescription<F>,
        inputs: HashMap<LayerId, MultilinearExtension<F>>,
        private_inputs: HashMap<LayerId, LigeroInputLayerDescriptionWithOptionalProverPrecommit<F>>,
        layer_label_to_layer_id: HashMap<String, LayerId>,
    ) -> Self {
        Self {
            circuit_description,
            inputs,
            private_inputs,
            layer_label_to_layer_id,
        }
    }

    /// # WARNING
    /// To be used only for testing and debugging.
    ///
    /// Constructs a form of this circuit that can be verified when a proof is provided.
    /// This is done by erasing all input data associated with private input layers, along with any
    /// commitments (the latter can be found in the proof).
    pub fn _gen_verifiable_circuit(&self) -> VerifiableCircuit<F> {
        let public_ids: HashSet<LayerId> = self.get_public_input_layer_ids().into_iter().collect();
        let predetermined_public_inputs: HashMap<LayerId, MultilinearExtension<F>> = self
            .inputs
            .clone()
            .into_iter()
            .filter(|(layer_id, _)| public_ids.contains(layer_id))
            .collect();

        let private_inputs: HashMap<
            LayerId,
            LigeroInputLayerDescriptionWithOptionalVerifierPrecommit<F>,
        > = self
            .private_inputs
            .clone()
            .into_iter()
            .map(|(layer_id, (desc, opt_commit))| {
                (layer_id, (desc, opt_commit.map(|commit| commit.get_root())))
            })
            .collect();

        VerifiableCircuit {
            circuit_description: self.circuit_description.clone(),
            predetermined_public_inputs,
            private_inputs,
            layer_label_to_layer_id: self.layer_label_to_layer_id.clone(),
        }
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
            .filter(|layer_id| !self.private_inputs.contains_key(layer_id))
            .cloned()
            .collect()
    }

    /// Returns a vector of the [LayerId]s of all input layers with visibility [LayerVisibility::Private].
    ///
    /// TODO: Consider returning an iterator instead.
    pub fn get_private_input_layer_ids(&self) -> Vec<LayerId> {
        self.private_inputs.keys().cloned().collect()
    }

    /// Returns the data associated with the input layer with ID `layer_id`, or an error if there is
    /// no input layer with this ID.
    pub fn get_input_mle(&self, layer_id: LayerId) -> Result<MultilinearExtension<F>> {
        self.inputs
            .get(&layer_id)
            .ok_or(anyhow!("Unrecognized Layer ID '{layer_id}'"))
            .cloned()
    }

    /// Returns the description of the private input layer with ID `layer_id`, or an error if no
    /// such private input layer exists.
    pub fn get_private_input_layer(
        &self,
        layer_id: LayerId,
    ) -> Result<LigeroInputLayerDescriptionWithOptionalProverPrecommit<F>> {
        self.private_inputs
            .get(&layer_id)
            .ok_or(anyhow!("Unrecognized Layer ID '{layer_id}'"))
            .cloned()
    }

    /// Get a reference to the mapping that maps a [LayerId] of layer (either public or private) to
    /// the data that are associated with it.
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

        self.get_private_input_layer_ids()
            .into_iter()
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .for_each(|layer_id| {
                // Commit to the Ligero input layer, if it is not already committed to.
                let (desc, maybe_precommitment) = self.get_private_input_layer(layer_id).unwrap();
                let commitment = if let Some(commitment) = maybe_precommitment {
                    commitment.clone()
                } else {
                    let input_mle = self.get_input_mle(layer_id).unwrap();
                    let (commitment, _) =
                        remainder_ligero_commit(&input_mle.iter().collect_vec(), &desc.aux);
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
            if let Ok((desc, _)) = self.get_private_input_layer(layer_id) {
                let mle = self.get_input_mle(layer_id).unwrap();
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
}

/// A circuit that contains a [GKRCircuitDescription] alongside a description of
/// the private input layers.
#[derive(Clone, Debug)]
pub struct VerifiableCircuit<F: Halo2FFTFriendlyField> {
    circuit_description: GKRCircuitDescription<F>,
    predetermined_public_inputs: HashMap<LayerId, MultilinearExtension<F>>,
    private_inputs: HashMap<LayerId, LigeroInputLayerDescriptionWithOptionalVerifierPrecommit<F>>,
    layer_label_to_layer_id: HashMap<String, LayerId>,
}

impl<F: Halo2FFTFriendlyField> VerifiableCircuit<F> {
    /// Returns a [VerifiableCircuit] initialized with the given data.
    pub fn new(
        circuit_description: GKRCircuitDescription<F>,
        predetermined_public_inputs: HashMap<LayerId, MultilinearExtension<F>>,
        private_inputs: HashMap<
            LayerId,
            LigeroInputLayerDescriptionWithOptionalVerifierPrecommit<F>,
        >,
        layer_label_to_layer_id: HashMap<String, LayerId>,
    ) -> Self {
        Self {
            circuit_description,
            predetermined_public_inputs,
            private_inputs,
            layer_label_to_layer_id,
        }
    }

    /// Returns a reference to the mapping which maps a [LayerId] of a private input layer to its
    /// description.
    ///
    /// TODO: This is too transparent. Replace this with methods that answer the queries of the
    /// prover directly, and do _not_ expose it to the circuit developer.
    pub fn get_private_inputs_ref(
        &self,
    ) -> &HashMap<LayerId, LigeroInputLayerDescriptionWithOptionalVerifierPrecommit<F>> {
        &self.private_inputs
    }

    /// Returns a reference to the circuit description.
    ///
    /// TODO: This is only used by the back end. Do _not_ expose it to the circuit developer.
    pub fn get_gkr_circuit_description_ref(&self) -> &GKRCircuitDescription<F> {
        &self.circuit_description
    }

    /// Returns a vector of the [LayerId]s of all input layers with visibility
    /// [LayerVisibility::Public].
    pub fn get_public_input_layer_ids(&self) -> Vec<LayerId> {
        self.circuit_description
            .input_layers
            .iter()
            .filter(|input_layer_description| {
                // All input layers which are not private are public by default.
                !self
                    .private_inputs
                    .contains_key(&input_layer_description.layer_id)
            })
            .map(|public_input_layer_description| public_input_layer_description.layer_id)
            .collect()
    }

    /// Returns a vector of the [LayerId]s of all input layers with visibility
    /// [LayerVisibility::Private].
    pub fn get_private_input_layer_ids(&self) -> Vec<LayerId> {
        self.private_inputs.keys().cloned().collect_vec()
    }

    /// Returns the data associated with the public input layer with ID `layer_id`, or None if
    /// no such public input layer exists.
    pub fn get_public_input_mle_ref(&self, layer_id: &LayerId) -> Option<&MultilinearExtension<F>> {
        self.predetermined_public_inputs.get(layer_id)
    }

    /// Sets the pre-commitment of the input layer with label `layer_label` to `commitment`.
    /// Pre-commitments allow the verifier to check that an input layer commitment in the
    /// proof is the same as the expected `commitment` provided here.
    ///
    /// # Panics
    /// If `layer_label` is not a valid private input layer label.
    pub fn set_pre_commitment(
        &mut self,
        layer_label: &str,
        commitment: LigeroRoot<F>,
    ) -> Result<()> {
        let layer_id = self.layer_label_to_layer_id.get(layer_label).unwrap();

        let (_, optional_commitment) = self
            .private_inputs
            .get_mut(layer_id)
            .expect("Layer {layer_id} either does not exist, or is not a private input layer.");

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
            self.get_public_input_layer_ids().len() + self.get_private_inputs_ref().len(),
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
                match self.get_public_input_mle_ref(&layer_id) {
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
        self.get_private_input_layer_ids()
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
                .get_private_inputs_ref()
                .iter()
                .find(|(layer_id, _)| **layer_id == claim_layer_id)
                .unwrap();

            if let Some(pre_commitment) = opt_pre_commitment {
                assert_eq!(pre_commitment.root, *commitment);
            }

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
}
