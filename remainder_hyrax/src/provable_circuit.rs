//! Defines the utilities for taking a list of nodes and turning it into a
//! layedout circuit
use std::collections::HashMap;
use std::fmt::Debug;

use ark_std::{end_timer, start_timer};
use itertools::Itertools;
use rand::{CryptoRng, RngCore};
use remainder::mle::evals::MultilinearExtension;
use remainder::prover::helpers::get_circuit_description_hash_as_field_elems;
use remainder::{layer::LayerId, prover::GKRCircuitDescription};
use remainder_shared_types::config::global_config::{
    get_current_global_prover_config, global_prover_circuit_description_hash_type,
};
use remainder_shared_types::config::ProofConfig;
use remainder_shared_types::curves::PrimeOrderCurve;

use anyhow::{anyhow, bail, Result};
use remainder_shared_types::pedersen::{CommittedScalar, PedersenCommitter};
use remainder_shared_types::transcript::ec_transcript::ECTranscriptTrait;

use crate::hyrax_gkr::hyrax_input_layer::{
    commit_to_input_values, HyraxInputLayerDescription, HyraxInputLayerProof,
    HyraxProverInputCommitment, HyraxVerifierInputCommitment,
};
use crate::hyrax_gkr::hyrax_layer::HyraxClaim;
use crate::hyrax_gkr::{HyraxCircuitProof, HyraxProof};
use crate::utils::vandermonde::VandermondeInverse;
use crate::verifiable_circuit::HyraxVerifiableCircuit;

pub type HyraxInputLayerDescriptionWithOptionalProverPrecommit<C> = (
    HyraxInputLayerDescription,
    Option<HyraxProverInputCommitment<C>>,
);

pub type HyraxInputLayerDescriptionWithOptionalVerifierPrecommit<C> = (
    HyraxInputLayerDescription,
    Option<HyraxVerifierInputCommitment<C>>,
);

/// A circuit, along with all of its input data, ready to be proven using the Hyrax-GKR proving
/// system which uses Hyrax as a PCS for private input layers, and provides zero-knowledge
/// guarantees.
#[derive(Clone, Debug)]
pub struct HyraxProvableCircuit<C: PrimeOrderCurve> {
    circuit_description: GKRCircuitDescription<C::Scalar>,
    inputs: HashMap<LayerId, MultilinearExtension<C::Scalar>>,
    private_inputs: HashMap<LayerId, HyraxInputLayerDescriptionWithOptionalProverPrecommit<C>>,
    pub layer_label_to_layer_id: HashMap<String, LayerId>,
}

impl<C: PrimeOrderCurve> HyraxProvableCircuit<C> {
    /// Constructor
    pub fn new(
        circuit_description: GKRCircuitDescription<C::Scalar>,
        inputs: HashMap<LayerId, MultilinearExtension<C::Scalar>>,
        private_inputs: HashMap<LayerId, HyraxInputLayerDescriptionWithOptionalProverPrecommit<C>>,
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
    pub fn _gen_hyrax_verifiable_circuit(&self) -> HyraxVerifiableCircuit<C> {
        let public_inputs: HashMap<LayerId, MultilinearExtension<C::Scalar>> = self
            .inputs
            .clone()
            .into_iter()
            .filter(|(layer_id, _)| !self.private_inputs.contains_key(layer_id))
            .collect();

        let private_inputs: HashMap<
            LayerId,
            HyraxInputLayerDescriptionWithOptionalVerifierPrecommit<_>,
        > = self
            .private_inputs
            .clone()
            .into_iter()
            .map(|(layer_id, (desc, _))| (layer_id, (desc, None)))
            .collect();

        // Ensure the union of public input Layer IDs with private input layer IDs equal
        // the set of all input layer IDs.
        debug_assert_eq!(
            public_inputs
                .keys()
                .chain(self.private_inputs.keys())
                .cloned()
                .sorted()
                .collect_vec(),
            self.inputs.keys().cloned().sorted().collect_vec()
        );

        HyraxVerifiableCircuit {
            circuit_description: self.circuit_description.clone(),
            predetermined_public_inputs: public_inputs,
            private_inputs,
            layer_label_to_layer_id: self.layer_label_to_layer_id.clone(),
        }
    }

    /// Returns a reference to the [GKRCircuitDescription] of this circuit.
    ///
    /// TODO: This is only used by the back end. Do _not_ expose it to the circuit developer.
    pub fn get_gkr_circuit_description_ref(&self) -> &GKRCircuitDescription<C::Scalar> {
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
    pub fn get_input_mle(&self, layer_id: LayerId) -> Result<MultilinearExtension<C::Scalar>> {
        self.inputs
            .get(&layer_id)
            .ok_or(anyhow!("Unrecognized Layer ID '{layer_id}'"))
            .cloned()
    }

    /// Returns a reference to the description of the private input layer with ID `layer_id`, or an
    /// error if no such private input layer exists.
    pub fn get_private_input_layer_ref(
        &self,
        layer_id: LayerId,
    ) -> Result<&HyraxInputLayerDescriptionWithOptionalProverPrecommit<C>> {
        self.private_inputs
            .get(&layer_id)
            .ok_or(anyhow!("Unrecognized Layer ID '{layer_id}'"))
    }

    /// Returns a mutable reference to the description of the private input layer with ID
    /// `layer_id`, or an error if no such private input layer exists.
    pub fn get_private_input_layer_mut_ref(
        &mut self,
        layer_id: LayerId,
    ) -> Result<&mut HyraxInputLayerDescriptionWithOptionalProverPrecommit<C>> {
        self.private_inputs
            .get_mut(&layer_id)
            .ok_or(anyhow!("Unrecognized Layer ID '{layer_id}'"))
    }

    /// Get a reference to the mapping that maps a [LayerId] of layer (either public or private) to
    /// the data that are associated with it.
    ///
    /// TODO: This is too transparent. Replace this with methods that answer the queries of the
    /// prover directly, and do _not_ expose it to the circuit developer.
    pub fn get_inputs_ref(&self) -> &HashMap<LayerId, MultilinearExtension<C::Scalar>> {
        &self.inputs
    }

    /// Returns true if `layer_id` corresponds to the ID of a private input layer in this circuit.
    pub fn is_private_input_layer(&self, layer_id: LayerId) -> bool {
        self.private_inputs.contains_key(&layer_id)
    }

    /// Sets the commitment of layer with ID `layer_id` to `commitment`, if no
    /// commitment has already been set for this layer. Returns error otherwise.
    ///
    /// # Panics
    /// If `layer_id` is not a valid private input layer ID.
    pub fn set_commitment(
        &mut self,
        layer_id: LayerId,
        commitment: HyraxProverInputCommitment<C>,
        log_num_cols: Option<usize>,
    ) -> Result<()> {
        let (layer_descr, comm_opt) = self
            .private_inputs
            .get_mut(&layer_id)
            .expect("Layer {layer_id} either does not exist, or is not a private input layer.");

        if comm_opt.is_some() {
            bail!("Commitment already exists for Layer with ID {layer_id}.");
        } else {
            if let Some(log_num_cols) = log_num_cols {
                layer_descr.log_num_cols = log_num_cols;
            }
            *comm_opt = Some(commitment);
        }

        Ok(())
    }

    /// Sets the pre-commitment of layer with label `layer_label` to `pre_commitment`, if no
    /// pre-commitment has already been set for this layer. Returns error otherwise.
    ///
    /// # Panics
    /// If `layer_label` is not a valid private input layer label.
    pub fn set_pre_commitment(
        &mut self,
        layer_label: &str,
        pre_commitment: HyraxProverInputCommitment<C>,
        log_num_cols: Option<usize>,
    ) -> Result<()> {
        let layer_id = self.layer_label_to_layer_id.get(layer_label).unwrap();

        self.set_commitment(*layer_id, pre_commitment, log_num_cols)
    }

    pub fn get_commitment_mut_ref(
        &mut self,
        layer_id: &LayerId,
    ) -> Result<&mut HyraxProverInputCommitment<C>> {
        let (_, comm_opt) = self.private_inputs.get_mut(layer_id).unwrap();

        comm_opt
            .as_mut()
            .ok_or(anyhow!("Layer does not contain a commitment"))
    }

    pub fn get_commitment_ref(&self, layer_id: &LayerId) -> Result<&HyraxProverInputCommitment<C>> {
        let (_, comm_opt) = self.private_inputs.get(layer_id).unwrap();

        comm_opt
            .as_ref()
            .ok_or(anyhow!("Layer does not contain a commitment"))
    }

    pub fn get_commitment_ref_by_label(
        &self,
        layer_label: &str,
    ) -> Result<&HyraxProverInputCommitment<C>> {
        let layer_id = self
            .layer_label_to_layer_id
            .get(layer_label)
            .unwrap_or_else(|| {
                panic!(
                "Layer with label '{layer_label}' does not exist, or is not a private input layer."
            )
            });

        self.get_commitment_ref(layer_id)
    }

    pub fn commit(
        &mut self,
        committer: &PedersenCommitter<C>,
        mut rng: &mut (impl CryptoRng + RngCore),
        transcript: &mut impl ECTranscriptTrait<C>,
    ) {
        // For each hyrax input layer, calculate commitments if not already
        // provided, and then append each commitment to the transcript.
        let mut hyrax_input_commitments = HashMap::<LayerId, HyraxProverInputCommitment<C>>::new();
        self.get_private_input_layer_ids()
            .into_iter()
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .for_each(|layer_id| {
                // Commit to the Hyrax input layer, if it is not already
                // committed to.
                let (desc, maybe_precommitment) =
                    self.get_private_input_layer_ref(layer_id).unwrap();
                let prover_commitment: HyraxProverInputCommitment<C> =
                    if let Some(commitment) = maybe_precommitment {
                        commitment.clone()
                    } else {
                        // Commit to the values of the input layer
                        let input_mle = self.get_input_mle(layer_id).unwrap();
                        let hyrax_il_commit_timer =
                            start_timer!(|| format!("commit to hyrax input layer {layer_id}"));
                        let commitment = commit_to_input_values(
                            desc.num_vars,
                            desc.log_num_cols,
                            &input_mle,
                            committer,
                            &mut rng,
                        );
                        end_timer!(hyrax_il_commit_timer);
                        self.set_commitment(layer_id, commitment.clone(), None)
                            .unwrap();
                        commitment
                    };

                // Add the verifier's view of the commitment to transcript
                transcript.append_input_ec_points(
                    "Hyrax input layer commitment",
                    prover_commitment.commitment.clone(),
                );

                // Store the prover's view for later use in the evaluation
                // proofs.
                hyrax_input_commitments.insert(layer_id, prover_commitment);
            });
    }

    /// Create a [HyraxProof]. Values of public input layers are appended to
    /// transcript in order of `LayerId` value, ascending. Then Hyrax
    /// commitments are appended to transcript in order of `LayerId` value,
    /// ascending; this is also the ordering of `HyraxProof.hyrax_input_proofs`.
    ///
    /// # Arguments:
    /// * `inputs` - The MLEs of _all_ inputs (including Hyrax inputs), along
    ///   with their layer ids.
    /// * `hyrax_input_layers` - The descriptions of the Hyrax input layers,
    ///   along with (optionally) precommits.
    /// * `circuit_description` - The description of the circuit to be proven.
    /// * `committer` - The Pedersen committer to be used for commitments.
    /// * `rng` - The random number generator to be used for randomness.
    /// * `converter` - The Vandermonde inverse converter to be used for the
    ///   proof.
    /// * `transcript` - The transcript to be used for Fiat-Shamir challenges.
    ///
    /// # Requires:
    ///   * `circuit_description.index_mle_indices(0)` has been called
    pub fn prove(
        &mut self,
        committer: &PedersenCommitter<C>,
        mut rng: &mut (impl CryptoRng + RngCore),
        converter: &mut VandermondeInverse<C::Scalar>,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) -> (HyraxProof<C>, ProofConfig) {
        // Get proof config from global config
        let proof_config = ProofConfig::new_from_prover_config(&get_current_global_prover_config());

        // Generate circuit description hash and append to transcript
        let hash_value_as_field_elems = get_circuit_description_hash_as_field_elems(
            self.get_gkr_circuit_description_ref(),
            global_prover_circuit_description_hash_type(),
        );
        transcript
            .append_scalar_field_elems("Circuit description hash", &hash_value_as_field_elems);

        // Add the input values of any public (i.e. non-hyrax) input layers to
        // transcript. Select the public input layers from the input layers, and
        // sort them by layer id, and append their input values to the
        // transcript.
        self.get_inputs_ref()
            .keys()
            .filter(|layer_id| !self.get_private_input_layer_ids().contains(layer_id))
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .for_each(|layer_id| {
                let mle = self.get_input_mle(*layer_id).unwrap();
                let public_il_to_transcript_timer =
                    start_timer!(|| format!("adding il elements to transcript for {layer_id}"));
                transcript.append_input_scalar_field_elems(
                    "Public input layer values",
                    &mle.f.iter().collect_vec(),
                );
                end_timer!(public_il_to_transcript_timer);
            });

        self.commit(committer, rng, transcript);

        // Get the verifier challenges from the transcript.
        let mut challenge_sampler =
            |size| transcript.get_scalar_field_challenges("Verifier challenges", size);
        let instantiation_timer = start_timer!(|| "instantiate circuit");

        // Instantiate the circuit description given the data from sampling
        // verifier challenges.
        let mut instantiated_circuit = self
            .get_gkr_circuit_description_ref()
            .clone()
            .instantiate(self.get_inputs_ref(), &mut challenge_sampler);
        end_timer!(instantiation_timer);

        // Generate the circuit proof, which is, starting from claims generated
        // on the output layers, is the proof of the intermediate layers,
        // resulting in claims on the input layers.
        //
        // NOTE: The `claims_on_input_layers` are in a deterministic order;
        // namely the claims are in reverse order of the layers making the
        // claim. The verifier has a claim tracker populated in the same order.
        // Additionally, claims generated are always made from "left to right"
        // when viewing a layer as an expression in terms of other layers.
        let layer_proving_timer = start_timer!(|| "proving intermediate layers");
        let (circuit_proof, claims_on_input_layers) = HyraxCircuitProof::prove(
            &mut instantiated_circuit,
            committer,
            &mut rng,
            converter,
            transcript,
        );
        end_timer!(layer_proving_timer);

        // Collect the values of the public inputs
        let public_inputs = self
            .get_inputs_ref()
            .keys()
            .filter(|layer_id| !self.is_private_input_layer(**layer_id))
            .map(|layer_id| {
                let mle = self.get_input_mle(*layer_id).unwrap();
                (*layer_id, Some(mle.clone()))
            })
            .collect_vec();

        // Separate the claims on input layers into claims on public input
        // layers vs claims on Hyrax input layers
        let mut claims_on_public_values = vec![];
        let mut claims_on_hyrax_input_layers =
            HashMap::<LayerId, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>>::new();
        claims_on_input_layers.iter().for_each(|claim| {
            if self.is_private_input_layer(claim.to_layer_id) {
                if let Some(curr_claims) = claims_on_hyrax_input_layers.get_mut(&claim.to_layer_id)
                {
                    curr_claims.push(claim.clone());
                } else {
                    claims_on_hyrax_input_layers.insert(claim.to_layer_id, vec![claim.clone()]);
                }
            } else {
                claims_on_public_values.push(claim.clone());
            }
        });

        // If in debug mode, then check the claims on all input layers.
        if cfg!(debug_assertions) {
            for claim in claims_on_input_layers.iter() {
                let input_mle = self.get_input_mle(claim.to_layer_id).unwrap();
                let public_il_verification_timer =
                    start_timer!(|| format!("public il eval for {0}", claim.to_layer_id));
                let evaluation = input_mle.evaluate_at_point(&claim.point);
                if evaluation != claim.evaluation.value {
                    panic!(
                        "Claim on input layer {} does not match evaluation",
                        claim.to_layer_id
                    );
                }
                end_timer!(public_il_verification_timer);
            }
        }

        // Prove the claims on the Hyrax input layers
        let hyrax_input_proofs = self
            .get_private_input_layer_ids()
            .iter()
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .map(|layer_id| {
                if let (desc, Some(commitment)) =
                    self.get_private_input_layer_mut_ref(*layer_id).unwrap()
                {
                    // let commitment = provable_circuit.get_commitment_mut_ref(layer_id).unwrap();
                    let committed_claims = claims_on_hyrax_input_layers.remove(layer_id).unwrap();

                    let hyrax_il_proving_timer = start_timer!(|| format!(
                        "HyraxInputLayer::prove for {0} with {1} claims",
                        layer_id,
                        committed_claims.len()
                    ));
                    let il_proof = HyraxInputLayerProof::prove(
                        desc,
                        commitment,
                        &committed_claims,
                        committer,
                        &mut rng,
                        transcript,
                    );
                    end_timer!(hyrax_il_proving_timer);
                    il_proof
                } else {
                    panic!("Input layer with ID {layer_id} missing committment!")
                }
            })
            .collect_vec();

        // Check that now Hyrax input layer claims remain
        assert!(claims_on_hyrax_input_layers.is_empty());

        (
            HyraxProof {
                public_inputs,
                circuit_proof,
                claims_on_public_values,
                hyrax_input_proofs,
            },
            proof_config,
        )
    }
}
