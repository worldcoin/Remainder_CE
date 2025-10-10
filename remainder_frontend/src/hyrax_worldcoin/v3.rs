#![allow(warnings)]
use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;

use crate::{
    layouter::builder::Circuit,
    worldcoin_mpc::circuits::MPC_IRISCODE_INPUT_LAYER,
    zk_iriscode_ss::{
        circuits::{
            V3_AUXILIARY_LAYER, V3_DIGITS_LAYER, V3_INPUT_IMAGE_LAYER, V3_RH_MATMULT_SHRED,
            V3_SIGN_BITS_LAYER, V3_TO_SUB_MATMULT_SHRED,
        },
        data::IriscodeCircuitAuxData,
        parameters::IRISCODE_COMMIT_LOG_NUM_COLS,
        v3::{circuit_description_and_inputs, load_worldcoin_data},
    },
};
use clap::error;
use rand::rngs::ThreadRng;
use rand::{CryptoRng, Rng, RngCore};
use remainder::layer::LayerId;
use remainder::mle::evals::MultilinearExtension;
use remainder::prover::GKRCircuitDescription;
use remainder::utils::mle::pad_with;
use remainder::{
    circuit_building_context::CircuitBuildingContext, circuit_layout::VerifiableCircuit,
};
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::halo2curves::{bn256::G1 as Bn256Point, group::Group};
use remainder_shared_types::pedersen::PedersenCommitter;
use remainder_shared_types::transcript::ec_transcript::{ECTranscript, ECTranscriptTrait};
use remainder_shared_types::transcript::poseidon_sponge::PoseidonSponge;
use remainder_shared_types::transcript::Transcript;
use remainder_shared_types::{
    config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig, ProofConfig},
    halo2curves::grumpkin::G1,
};
use remainder_shared_types::{
    perform_function_under_prover_config, perform_function_under_verifier_config, Fq, Fr,
};
use remainder_shared_types::{Field, Zeroizable};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use super::orb::SerializedImageCommitment;
use crate::hyrax_worldcoin::orb::{IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING};
use remainder_hyrax::utils::vandermonde::VandermondeInverse;
use remainder_hyrax::{
    circuit_layout::HyraxProvableCircuit,
    hyrax_gkr::{self, verify_hyrax_proof, HyraxProof},
};
use remainder_hyrax::{
    circuit_layout::HyraxVerifiableCircuit,
    hyrax_gkr::hyrax_input_layer::{
        commit_to_input_values, HyraxInputLayerDescription, HyraxProverInputCommitment,
    },
};
use sha256::digest as sha256_digest;
use thiserror::Error;

use crate::zk_iriscode_ss::circuits::iriscode_ss_attach_input_data;
use anyhow::{anyhow, Result};

type Scalar = Fr;
type Base = Fq;

#[derive(Debug, Error)]
pub enum IriscodeError {
    #[error("Non-zero padding bits in iris/mask code")]
    NonZeroPaddingBits,
    #[error("Non-binary iris/mask code")]
    NonBinaryIrisMaskCode,
    #[error("Incorrect kernel values or thresholds")]
    IncorrectKernelValuesOrThresholds,
    #[error("Image commitment does not match expected hash")]
    WrongHash,
}

/// Verify that the iris/mask code in the supplied proof is correct, and return the unpadded iris/mask code, along with the
/// commitment to the image.
/// Checks, in particular:
/// * That the [HyraxProof] verifies.
/// * That the correct kernel values and thresholds are being used in the supplied proof.
/// * That the MLE encoding the iris/mask code has only 0s in the padding region.
/// * That the unpadded iris/mask code consists only of 0s and 1s.
/// * That the image commitment hash matches the expected hash.
/// This is a helper function for verifying the v3 masked iriscode is correct.
pub(crate) fn verify_v3_iriscode_proof_and_hash(
    proof: &HyraxProof<Bn256Point>,
    verifiable_circuit: &HyraxVerifiableCircuit<Bn256Point>,
    /*
    ic_circuit_desc: &IriscodeCircuitDescription<Fr>,
    auxiliary_mle: &MultilinearExtension<Fr>,
    */
    expected_commitment_hash: &str,
    committer: &PedersenCommitter<Bn256Point>,
    proof_config: &ProofConfig,
) -> Result<Vec<Bn256Point>> {
    /*
    let mut hyrax_input_layers = HashMap::new();

    // The image, with the precommit (must use the same number of columns as were used at the time of committing!)
    let image_hyrax_input_layer_desc = HyraxInputLayerDescription {
        layer_id: ic_circuit_desc.image_input_layer.layer_id,
        num_vars: ic_circuit_desc.image_input_layer.num_vars,
        log_num_cols: IMAGE_COMMIT_LOG_NUM_COLS,
    };
    hyrax_input_layers.insert(
        ic_circuit_desc.image_input_layer.layer_id,
        image_hyrax_input_layer_desc,
    );

    // The iris code
    let code_hyrax_input_layer_desc = HyraxInputLayerDescription {
        layer_id: ic_circuit_desc.code_input_layer.layer_id,
        num_vars: ic_circuit_desc.code_input_layer.num_vars,
        log_num_cols: IRISCODE_COMMIT_LOG_NUM_COLS,
    };
    hyrax_input_layers.insert(
        ic_circuit_desc.code_input_layer.layer_id,
        code_hyrax_input_layer_desc,
    );

    // The digits layer
    hyrax_input_layers.insert(
        ic_circuit_desc.digits_input_layer.layer_id,
        ic_circuit_desc.digits_input_layer.clone().into(),
    );
    */

    // Create a fresh transcript.
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("V3 Iriscode Circuit Pipeline");

    // Verify the relationship between iris/mask code and image.
    verify_hyrax_proof(
        &proof,
        &verifiable_circuit,
        &committer,
        &mut transcript,
        proof_config,
    );

    let image_layer_id = verifiable_circuit
        .get_private_input_layer_id(V3_INPUT_IMAGE_LAYER)
        .unwrap();
    let code_layer_id = verifiable_circuit
        .get_private_input_layer_id(V3_SIGN_BITS_LAYER)
        .unwrap();

    let image_commitment = proof.get_commitment_ref(image_layer_id).cloned().unwrap();
    // dbg!(&image_commitment);
    let code_commitment = proof.get_commitment_ref(code_layer_id).cloned().unwrap();
    /*
    // Extract the iris/mask code commitment from the proof.
    let code_commitment = proof
        .hyrax_input_proofs
        .iter()
        .find(|proof| proof.layer_id == ic_circuit_desc.code_input_layer.layer_id)
        .unwrap()
        .input_commitment
        .clone();

    // Extract the commitment to the image from the proof.
    let image_commitment = proof
        .hyrax_input_proofs
        .iter()
        .find(|proof| proof.layer_id == ic_circuit_desc.image_input_layer.layer_id)
        .unwrap()
        .input_commitment
        .clone();
    let code_commitment = todo!();
    let image_commitment: Vec<Bn256Point> = todo!();
    */

    // Check that the image commitment matches the expected hash.
    let commitment_hash = sha256_digest(
        &image_commitment
            .iter()
            .flat_map(|p| p.to_bytes_compressed())
            .collect::<Vec<u8>>(),
    );

    if commitment_hash != *expected_commitment_hash {
        return Err(anyhow!(IriscodeError::WrongHash));
    }

    // Return the commitments to the code and the image.
    Ok(code_commitment)
}

/// Prove a single instance of the iriscode circuit using the Hyrax proof system and the provided image precommit.
/// This is a helper function for proving the v3 masked iriscode is correct.
/// # Arguments:
/// * `inputs`: _all_ inputs to the circuit, as MLEs.
/// * `image_precommit`: The precommitment to the image (derived from the image using `committer`).
/// ///
/// This function is assumed to be called *with the prover config set*!
pub fn prove_with_image_precommit(
    mut provable_circuit: HyraxProvableCircuit<Bn256Point>,
    /*
    ic_circuit_desc: &IriscodeCircuitDescription<Fr>,
    inputs: HashMap<LayerId, MultilinearExtension<Fr>>,
    */
    image_layer_label: &str,
    code_layer_label: &str,
    image_precommit: HyraxProverInputCommitment<Bn256Point>,
    committer: &PedersenCommitter<Bn256Point>,
    blinding_rng: &mut (impl CryptoRng + RngCore),
    converter: &mut VandermondeInverse<Scalar>,
) -> ((
    HyraxProof<Bn256Point>,
    ProofConfig,
    HyraxProverInputCommitment<Bn256Point>,
)) {
    // Set up Hyrax input layer specification.
    let mut hyrax_input_layers: HashMap<
        LayerId,
        (
            HyraxInputLayerDescription,
            Option<HyraxProverInputCommitment<Bn256Point>>,
        ),
    > = HashMap::new();

    provable_circuit
        .set_pre_commitment(
            image_layer_label,
            image_precommit,
            Some(IMAGE_COMMIT_LOG_NUM_COLS),
        )
        .unwrap();
    /*
    // The image, with the precommit (must use the same number of columns as were used at the time of committing!)
    let image_hyrax_input_layer_desc = HyraxInputLayerDescription {
        layer_id: ic_circuit_desc.image_input_layer.layer_id,
        num_vars: ic_circuit_desc.image_input_layer.num_vars,
        log_num_cols: IMAGE_COMMIT_LOG_NUM_COLS,
    };
    hyrax_input_layers.insert(
        ic_circuit_desc.image_input_layer.layer_id,
        (image_hyrax_input_layer_desc, Some(image_precommit)),
    );

    // The iris code, with a precommit (that we compute here)
    let code_hyrax_input_layer_desc = HyraxInputLayerDescription {
        layer_id: ic_circuit_desc.code_input_layer.layer_id,
        num_vars: ic_circuit_desc.code_input_layer.num_vars,
        log_num_cols: IRISCODE_COMMIT_LOG_NUM_COLS,
    };
    // Build the commitment to the iris/mask code.
    let code_mle = inputs
        .get(&ic_circuit_desc.code_input_layer.layer_id)
        .unwrap();
    let code_commit = commit_to_input_values(
        &code_hyrax_input_layer_desc,
        &code_mle,
        committer,
        blinding_rng,
    );
    hyrax_input_layers.insert(
        ic_circuit_desc.code_input_layer.layer_id,
        (code_hyrax_input_layer_desc, Some(code_commit.clone())),
    );

    // The digit multiplicities, without the precommit
    hyrax_input_layers.insert(
        ic_circuit_desc.digits_input_layer.layer_id,
        (ic_circuit_desc.digits_input_layer.clone().into(), None),
    );
    */

    // Create a fresh transcript.
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("V3 Iriscode Circuit Pipeline");

    // Prove the relationship between iris/mask code and image.
    let (proof, proof_config) = HyraxProof::prove(
        &mut provable_circuit,
        &committer,
        blinding_rng,
        converter,
        &mut transcript,
    );

    let code_layer_id = *provable_circuit
        .layer_label_to_layer_id
        .get(code_layer_label)
        .unwrap();

    let code_commit_in_proof = &proof
        .hyrax_input_proofs
        .iter()
        .find(|hyrax_input_proof| hyrax_input_proof.layer_id == code_layer_id)
        .unwrap()
        .input_commitment;
    let code_commit = provable_circuit
        .get_commitment_ref_by_label(code_layer_label)
        .unwrap()
        .clone();
    assert_eq!(*code_commit_in_proof, code_commit.commitment);

    /*
    // Zeroize each value in the HashMap
    for (_, mut mle) in inputs {
        mle.zeroize();
    }

    // Zeroize the image precommit.
    let (image_hyrax_input_layer_desc, image_precommit) = hyrax_input_layers
        .get_mut(&ic_circuit_desc.image_input_layer.layer_id)
        .unwrap();

    image_precommit.as_mut().unwrap().zeroize();
    */

    (proof, proof_config, code_commit)
}

#[derive(Error, Debug)]
pub enum V3ProofError {
    #[error("Found empty proof field")]
    EmptyProofField,

    #[error("Verification produced an iriscode error")]
    IriscodeError(#[from] IriscodeError),

    #[error("Verifier returned an iriscode of the wrong length")]
    IriscodeLengthMismatch,
}

/// A serializable struct containing the 4 Hyrax proofs needed for the
/// V3 iris/mask circuit.
/// To allow for incremental proving, this struct can be initialized empty
/// and then proofs can be added as the become available.
/// Serialization is possible at any stage, even if a proof is missing.
/// Once all proofs are generated, the struct can be finalized into
/// a `V3Proof` which can be then passed to the verifier.
#[derive(Serialize, Deserialize)]
pub struct V3Prover {
    #[serde(skip)]
    #[serde(default = "V3Prover::default_committer")]
    committer: PedersenCommitter<Bn256Point>,

    #[serde(skip)]
    #[serde(default = "VandermondeInverse::new")]
    converter: VandermondeInverse<Scalar>,

    /// The Iriscode computation circuit description, along with auxiliary inputs (parameters which
    /// are constant for both iris and mask computation), but no image/mask inputs.
    v3_circuit: V3CircuitAndAuxData<Fr>,

    prover_config: GKRCircuitProverConfig,

    left_iris_proof: Option<HyraxProof<Bn256Point>>,
    left_mask_proof: Option<HyraxProof<Bn256Point>>,
    right_iris_proof: Option<HyraxProof<Bn256Point>>,
    right_mask_proof: Option<HyraxProof<Bn256Point>>,
}

impl V3Prover {
    pub fn default_committer() -> PedersenCommitter<Bn256Point> {
        PedersenCommitter::new(1 << IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING, None)
    }

    /// Generate an empty v3 prover with the given configuration.
    pub fn new(prover_config: GKRCircuitProverConfig, circuit: V3CircuitAndAuxData<Fr>) -> Self {
        Self {
            committer: Self::default_committer(),
            converter: VandermondeInverse::new(),
            v3_circuit: circuit,
            prover_config,
            left_iris_proof: None,
            left_mask_proof: None,
            right_iris_proof: None,
            right_mask_proof: None,
        }
    }

    /*
    pub fn remove_auxiliary_input_layer(&mut self, is_mask: bool, is_left_eye: bool) {
        let auxiliary_input_layer_id = self.circuit.circuit.auxiliary_input_layer.layer_id;

        self.get_as_mut(is_mask, is_left_eye)
            .remove_public_input_layer_by_id(auxiliary_input_layer_id);
    }
    */

    /// Generate a v3 prover with the given configuration, initialized
    /// with optional proofs.
    pub fn new_from_proofs(
        prover_config: GKRCircuitProverConfig,
        circuit: V3CircuitAndAuxData<Fr>,
        left_image_proof: HyraxProof<Bn256Point>,
        left_mask_proof: HyraxProof<Bn256Point>,
        right_image_proof: HyraxProof<Bn256Point>,
        right_mask_proof: HyraxProof<Bn256Point>,
    ) -> Self {
        Self {
            committer: Self::default_committer(),
            converter: VandermondeInverse::new(),
            v3_circuit: circuit,
            prover_config,
            left_iris_proof: Some(left_image_proof),
            left_mask_proof: Some(left_mask_proof),
            right_iris_proof: Some(right_image_proof),
            right_mask_proof: Some(right_mask_proof),
        }
    }

    pub fn prove(
        &mut self,
        is_mask: bool,
        is_left_eye: bool,
        image_bytes: Vec<u8>,
        image_precommit: HyraxProverInputCommitment<Bn256Point>,
        rng: &mut (impl CryptoRng + RngCore),
    ) -> HyraxProverInputCommitment<Bn256Point> {
        // Load the inputs to the circuit (these are all MLEs, i.e. in the clear).
        let input_data = load_worldcoin_data::<Fr>(image_bytes, is_mask);

        // Clone the circuit to start attaching data.
        let mut circuit = self.v3_circuit.get_circuit().clone();

        let aux_data = if is_mask {
            self.v3_circuit.get_mask_aux_data_ref().clone()
        } else {
            self.v3_circuit.get_iris_aux_data_ref().clone()
        };

        let circuit_with_inputs = iriscode_ss_attach_input_data::<
            _,
            { crate::zk_iriscode_ss::parameters::BASE },
        >(circuit, input_data, aux_data)
        .unwrap();

        let provable_circuit = circuit_with_inputs.finalize_hyrax().unwrap();

        // Prove the iriscode circuit with the image precommit.
        let (proof, _, code_commit) = prove_with_image_precommit(
            provable_circuit,
            V3_INPUT_IMAGE_LAYER,
            V3_SIGN_BITS_LAYER,
            image_precommit,
            &mut self.committer,
            rng,
            &mut self.converter,
        );

        self.set(is_mask, is_left_eye, proof);

        code_commit
    }

    /// Set the field indicated by `is_mask` and `is_left_eye` to `proof`,
    /// overwritting any existing value.
    pub fn set(&mut self, is_mask: bool, is_left_eye: bool, proof: HyraxProof<Bn256Point>) {
        match (is_mask, is_left_eye) {
            (false, false) => self.set_right_iris_proof(proof),
            (false, true) => self.set_left_iris_proof(proof),
            (true, false) => self.set_right_mask_proof(proof),
            (true, true) => self.set_left_mask_proof(proof),
        }
    }

    /// Set the left image proof to `proof`, overwritting any existing value.
    pub fn set_left_iris_proof(&mut self, proof: HyraxProof<Bn256Point>) {
        self.left_iris_proof = Some(proof)
    }

    /// Set the left mask proof to `proof`, overwritting any existing value.
    pub fn set_left_mask_proof(&mut self, proof: HyraxProof<Bn256Point>) {
        self.left_mask_proof = Some(proof)
    }

    /// Set the right image proof to `proof`, overwritting any existing value.
    pub fn set_right_iris_proof(&mut self, proof: HyraxProof<Bn256Point>) {
        self.right_iris_proof = Some(proof)
    }

    /// Set the right mask proof to `proof`, overwritting any existing value.
    pub fn set_right_mask_proof(&mut self, proof: HyraxProof<Bn256Point>) {
        self.right_mask_proof = Some(proof)
    }

    /// Returns whether the proof corresponding to `is_mask` and `is_left_eye`
    /// is present. If `true`, then `self.get()` is guaranteed to return `Some`
    /// value.
    pub fn is_set(&self, is_mask: bool, is_left_eye: bool) -> bool {
        match (is_mask, is_left_eye) {
            (false, false) => self.right_iris_proof.is_some(),
            (false, true) => self.left_iris_proof.is_some(),
            (true, false) => self.right_mask_proof.is_some(),
            (true, true) => self.left_mask_proof.is_some(),
        }
    }

    /// Get the proof indicated by `is_mask` and `is_left_eye`, if any,
    /// otherwise return `None`.
    pub fn get(&self, is_mask: bool, is_left_eye: bool) -> Option<&HyraxProof<Bn256Point>> {
        match (is_mask, is_left_eye) {
            (false, false) => self.get_right_iris_proof(),
            (false, true) => self.get_left_iris_proof(),
            (true, false) => self.get_right_mask_proof(),
            (true, true) => self.get_left_mask_proof(),
        }
    }

    /// Return a reference to the left image proof, if any, otherwise return
    /// `None`.
    pub fn get_left_iris_proof(&self) -> Option<&HyraxProof<Bn256Point>> {
        self.left_iris_proof.as_ref()
    }

    /// Return a reference to the left mask proof, if any, otherwise return
    /// `None`.
    pub fn get_left_mask_proof(&self) -> Option<&HyraxProof<Bn256Point>> {
        self.left_mask_proof.as_ref()
    }

    /// Return a reference to the right image proof, if any, otherwise return
    /// `None`.
    pub fn get_right_iris_proof(&self) -> Option<&HyraxProof<Bn256Point>> {
        self.right_iris_proof.as_ref()
    }

    /// Return a reference to the right mask proof, if any, otherwise return
    /// `None`.
    pub fn get_right_mask_proof(&self) -> Option<&HyraxProof<Bn256Point>> {
        self.right_mask_proof.as_ref()
    }

    /// Serializes `self` into a binary representation.
    pub fn serialize(&self) -> Vec<u8> {
        let serialized_proof = bincode::serialize(self).unwrap();

        serialized_proof
    }

    /// Checks whether `self` is ready to be finalized, i.e. whether all 4
    /// proofs are present.`
    fn is_ready_to_finalize(&self) -> bool {
        self.is_set(false, false)
            && self.is_set(false, true)
            && self.is_set(true, false)
            && self.is_set(true, true)
    }
    /// If `self` is ready to be finalized, it generates a `V3Proof` containing
    /// all 4 proofs in `self` along with the `ProofConfig` used to generate
    /// them.
    /// Returns `None` if not all proofs are present.
    pub fn finalize(&self) -> Result<V3Proof, V3ProofError> {
        if self.is_ready_to_finalize() {
            let proof_config = ProofConfig::new_from_prover_config(&self.prover_config);

            Ok(V3Proof::new(
                proof_config,
                self.left_iris_proof.as_ref().unwrap().clone(),
                self.left_mask_proof.as_ref().unwrap().clone(),
                self.right_iris_proof.as_ref().unwrap().clone(),
                self.right_mask_proof.as_ref().unwrap().clone(),
            ))
        } else {
            Err(V3ProofError::EmptyProofField)
        }
    }

    /// Deserializes `serialized_proof` and returns it.
    pub fn deserialize(serialized_prover: &[u8]) -> Self {
        bincode::deserialize(serialized_prover).unwrap()
    }

    /// Get a mutable reference to the proof indicated by `is_mask` and `is_left_eye`.
    pub fn get_as_mut(&mut self, is_mask: bool, is_left_eye: bool) -> &mut HyraxProof<Bn256Point> {
        match (is_mask, is_left_eye) {
            (false, false) => self.right_iris_proof.as_mut().unwrap(),
            (false, true) => self.left_iris_proof.as_mut().unwrap(),
            (true, false) => self.right_mask_proof.as_mut().unwrap(),
            (true, true) => self.left_mask_proof.as_mut().unwrap(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct V3Proof {
    #[serde(skip)]
    #[serde(default = "V3Prover::default_committer")]
    committer: PedersenCommitter<Bn256Point>,

    proof_config: ProofConfig,

    left_iris_proof: HyraxProof<Bn256Point>,
    left_mask_proof: HyraxProof<Bn256Point>,
    right_iris_proof: HyraxProof<Bn256Point>,
    right_mask_proof: HyraxProof<Bn256Point>,
}

impl V3Proof {
    pub fn new(
        proof_config: ProofConfig,
        left_iris_proof: HyraxProof<Bn256Point>,
        left_mask_proof: HyraxProof<Bn256Point>,
        right_iris_proof: HyraxProof<Bn256Point>,
        right_mask_proof: HyraxProof<Bn256Point>,
    ) -> Self {
        Self {
            committer: V3Prover::default_committer(),
            proof_config,
            left_iris_proof,
            left_mask_proof,
            right_iris_proof,
            right_mask_proof,
        }
    }

    pub fn insert_aux_public_data(
        &mut self,
        aux_mle: &MultilinearExtension<Fr>,
        is_mask: bool,
        is_left_eye: bool,
        aux_layer_id: LayerId,
    ) {
        let proof = self.get_as_mut(is_mask, is_left_eye);
        proof.insert_aux_public_data_by_id(aux_mle, aux_layer_id);
    }

    /// Get the proof indicated by `is_mask` and `is_left_eye`.
    pub fn get_as_mut(&mut self, is_mask: bool, is_left_eye: bool) -> &mut HyraxProof<Bn256Point> {
        match (is_mask, is_left_eye) {
            (false, false) => &mut self.right_iris_proof,
            (false, true) => &mut self.left_iris_proof,
            (true, false) => &mut self.right_mask_proof,
            (true, true) => &mut self.left_mask_proof,
        }
    }

    /// Get the proof indicated by `is_mask` and `is_left_eye`.
    pub fn get(&self, is_mask: bool, is_left_eye: bool) -> &HyraxProof<Bn256Point> {
        match (is_mask, is_left_eye) {
            (false, false) => self.get_right_iris_proof(),
            (false, true) => self.get_left_iris_proof(),
            (true, false) => self.get_right_mask_proof(),
            (true, true) => self.get_left_mask_proof(),
        }
    }

    /// Return a reference to the left image proof.
    pub fn get_left_iris_proof(&self) -> &HyraxProof<Bn256Point> {
        &self.left_iris_proof
    }

    /// Return a reference to the left mask proof.
    pub fn get_left_mask_proof(&self) -> &HyraxProof<Bn256Point> {
        &self.left_mask_proof
    }

    /// Return a reference to the right image proof.
    pub fn get_right_iris_proof(&self) -> &HyraxProof<Bn256Point> {
        &self.right_iris_proof
    }

    /// Return a reference to the right mask proof.
    pub fn get_right_mask_proof(&self) -> &HyraxProof<Bn256Point> {
        &self.right_mask_proof
    }

    /// Serializes `self` into a binary representation.
    pub fn serialize(&self) -> Vec<u8> {
        let serialized_proof = bincode::serialize(self).unwrap();

        serialized_proof
    }

    /// Deserializes `serialized_proof` and returns it.
    pub fn deserialize(serialized_proof: &[u8]) -> Self {
        bincode::deserialize(serialized_proof).unwrap()
    }

    /// NOTE: This function calls other internal functions which will
    /// enforce the verifier config, but itself does not do so. It therefore
    /// should be wrapped in a `perform_function_under_verifier_config!` macro.
    pub fn verify(
        &self,
        is_mask: bool,
        is_left_eye: bool,
        verifiable_circuit: &HyraxVerifiableCircuit<Bn256Point>,
        commitment_hash: &str,
    ) -> Result<()> {
        let proof = self.get(is_mask, is_left_eye);

        verify_v3_iriscode_proof_and_hash(
            proof,
            verifiable_circuit,
            commitment_hash,
            &self.committer,
            &self.proof_config,
        )?;

        Ok(())
    }
}

/// A circuit computing the V3 iriscode from an iris image and a mask.
/// The computation when the input is an iris image is essentially the same as the when the input is
/// a mask, the only difference being the values of some parameters which are passed as inputs to
/// the circuit, called auxiliary MLEs.
/// This struct holds a [Circuit] which has _already_ been initialized with input parameters which
/// are common between the cases of an iris image and a mask, and separatelly contains entries for
/// the auxiliary MLEs encoding the parameters specific to the iris image computation and the mask
/// computation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct V3CircuitAndAuxData<F: Field> {
    circuit: Circuit<F>,
    iris_aux_data: IriscodeCircuitAuxData<F>,
    mask_aux_data: IriscodeCircuitAuxData<F>,
}

impl<F: Field> V3CircuitAndAuxData<F> {
    pub fn new(
        circuit: Circuit<F>,
        iris_aux_data: IriscodeCircuitAuxData<F>,
        mask_aux_data: IriscodeCircuitAuxData<F>,
    ) -> Self {
        // Verify `circuit` contains the necessary input layers, which do _not_ contain any data
        // yet.
        assert!(!circuit
            .input_layer_contains_data(V3_INPUT_IMAGE_LAYER)
            .unwrap());
        assert!(!circuit.input_layer_contains_data(V3_DIGITS_LAYER).unwrap());
        assert!(!circuit
            .input_layer_contains_data(V3_SIGN_BITS_LAYER)
            .unwrap());
        assert!(!circuit
            .input_layer_contains_data(V3_AUXILIARY_LAYER)
            .unwrap());

        Self {
            circuit,
            iris_aux_data,
            mask_aux_data,
        }
    }

    pub fn get_circuit(&self) -> &Circuit<F> {
        &self.circuit
    }

    pub fn get_iris_aux_data_ref(&self) -> &IriscodeCircuitAuxData<F> {
        &self.iris_aux_data
    }

    pub fn get_mask_aux_data_ref(&self) -> &IriscodeCircuitAuxData<F> {
        &self.mask_aux_data
    }

    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(&self).expect("Failed to serialize V3CircuitAndAuxData")
    }

    pub fn deserialize(bytes: &[u8]) -> Self {
        bincode::deserialize(bytes).expect("Failed to deserialize V3CircuitAndAuxData")
    }
}
