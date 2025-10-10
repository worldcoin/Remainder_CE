use crate::{
    hyrax_worldcoin::{
        orb::PUBLIC_STRING,
        v3::{V3CircuitAndAuxData, V3Proof, V3ProofError, V3Prover},
    },
    layouter::builder::{Circuit, LayerVisibility},
    worldcoin_mpc::{
        circuits::{build_circuit, mpc_attach_data, MPC_SLOPES_LAYER},
        data::{gen_mpc_encoding_matrix, gen_mpc_evaluation_points, gen_mpc_input_data},
        parameters::{GR4_MODULUS, MPC_NUM_IRIS_4_CHUNKS},
    },
    zk_iriscode_ss::parameters::{IRISCODE_LEN, SHAMIR_SECRET_SHARE_SLOPE_LOG_NUM_COLS},
};
use remainder_hyrax::{
    circuit_layout::{HyraxProvableCircuit, HyraxVerifiableCircuit},
    hyrax_gkr::{
        hyrax_input_layer::{commit_to_input_values, HyraxProverInputCommitment},
        verify_hyrax_proof, HyraxProof,
    },
    utils::vandermonde::VandermondeInverse,
};

use itertools::Itertools;
use rand::{CryptoRng, Rng, RngCore};
use remainder::{layer::LayerId, mle::evals::MultilinearExtension};
use remainder_shared_types::{
    config::{
        global_config::{
            global_prover_claim_agg_constant_column_optimization, global_prover_enable_bit_packing,
            global_prover_lazy_beta_evals,
        },
        GKRCircuitProverConfig, ProofConfig,
    },
    curves::PrimeOrderCurve,
    pedersen::PedersenCommitter,
    transcript::{ec_transcript::ECTranscript, poseidon_sponge::PoseidonSponge},
    Base, Bn256Point, Field, Fr, Scalar,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors when verifying the MPC Proof
#[derive(Debug, Error)]
pub enum MPCError {
    #[error("Incorrect MLEs that should be invariant between circuits")]
    IncorrectInvariantAuxMles,
    #[error("Couldn't find the shares input layer MLE")]
    NoSharesInputLayerMle,
}

// Errors when producing the MPC Proof
#[derive(Error, Debug)]
pub enum MPCProofError {
    #[error("Found empty proof field")]
    EmptyProofField,

    #[error("Verification produced an MPC error")]
    MPCError(#[from] MPCError),
}

// Errors when producing the V3-MPC Proof
#[derive(Error, Debug)]
pub enum V3MPCProofError {
    #[error("Found empty proof field")]
    EmptyProofField,

    #[error("Verification produced an V3 error")]
    V3Error(#[from] V3ProofError),

    #[error("Verification produced an MPC error")]
    MPCError(#[from] MPCProofError),
}

pub fn print_features_status() {
    const STATUS_STR: [&str; 2] = ["OFF", "ON"];

    println!("=== FEATURES ===");
    println!(
        "Parallel feature for remainder_prover: {}",
        STATUS_STR[remainder::utils::is_parallel_feature_on() as usize]
    );
    println!(
        "Parallel feature for remainder_hyrax: {}",
        STATUS_STR[remainder_hyrax::utils::is_parallel_feature_on() as usize]
    );
    println!(
        "Lazy beta evaluation: {}",
        STATUS_STR[global_prover_lazy_beta_evals() as usize]
    );
    println!(
        "BitPackedVector: {}",
        STATUS_STR[global_prover_enable_bit_packing() as usize]
    );
    println!(
        "Claim aggregation constant column optimization for remainder_prover: {}",
        STATUS_STR[global_prover_claim_agg_constant_column_optimization() as usize]
    );
    println!("================\n");
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct V3MPCCommitments<C: PrimeOrderCurve> {
    left_iris_code: Vec<C>,
    left_mask_code: Vec<C>,
    right_iris_code: Vec<C>,
    right_mask_code: Vec<C>,
    slope_left_iris_code: Vec<C>,
    slope_right_iris_code: Vec<C>,
}

impl<C: PrimeOrderCurve> V3MPCCommitments<C> {
    pub fn new(
        left_iris_code: Vec<C>,
        left_mask_code: Vec<C>,
        right_iris_code: Vec<C>,
        right_mask_code: Vec<C>,
        slope_left_iris_code: Vec<C>,
        slope_right_iris_code: Vec<C>,
    ) -> Self {
        Self {
            left_iris_code,
            left_mask_code,
            right_iris_code,
            right_mask_code,
            slope_left_iris_code,
            slope_right_iris_code,
        }
    }

    pub fn get_code_commit_ref(&self, is_mask: bool, is_left_eye: bool) -> &Vec<C> {
        if is_mask {
            if is_left_eye {
                &self.left_mask_code
            } else {
                &self.right_mask_code
            }
        } else if is_left_eye {
            &self.left_iris_code
        } else {
            &self.right_iris_code
        }
    }

    pub fn get_slope_commit_ref(&self, is_left_eye: bool) -> &Vec<C> {
        if is_left_eye {
            &self.slope_left_iris_code
        } else {
            &self.slope_right_iris_code
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap()
    }

    pub fn deserialize(bytes: &[u8]) -> Self {
        bincode::deserialize(bytes).unwrap()
    }
}

/*
#[allow(clippy::too_many_arguments)]
pub fn prove_mpc_with_precommits(
    circuit: &Circuit<Fr>,
    iris_precommit: &HyraxProverInputCommitment<Bn256Point>,
    mask_precommit: &HyraxProverInputCommitment<Bn256Point>,
    slope_precommit: &HyraxProverInputCommitment<Bn256Point>,
    committer: &PedersenCommitter<Bn256Point>,
    blinding_rng: &mut (impl CryptoRng + RngCore),
    converter: &mut VandermondeInverse<Scalar>,
) -> HyraxProof<Bn256Point> {
    // Set up Hyrax input layer specification.
    let mut hyrax_input_layers = HashMap::new();

    hyrax_input_layers.insert(
        mpc_circuit_desc.iris_code_input_layer.layer_id,
        (
            mpc_circuit_desc.iris_code_input_layer.clone().into(),
            Some(iris_precommit.clone()),
        ),
    );

    hyrax_input_layers.insert(
        mpc_circuit_desc.mask_code_input_layer.layer_id,
        (
            mpc_circuit_desc.mask_code_input_layer.clone().into(),
            Some(mask_precommit.clone()),
        ),
    );

    hyrax_input_layers.insert(
        mpc_circuit_desc.slope_input_layer.layer_id,
        (
            mpc_circuit_desc.slope_input_layer.clone().into(),
            Some(slope_precommit.clone()),
        ),
    );

    hyrax_input_layers.insert(
        mpc_circuit_desc.auxilary_input_layer.layer_id,
        (mpc_circuit_desc.auxilary_input_layer.clone().into(), None),
    );

    // Create a fresh transcript.
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("V3 Iriscode Circuit Pipeline");

    // Prove the relationship between iris/mask code and image.
    let (proof, _proof_config) = HyraxProof::prove(
        &inputs,
        &hyrax_input_layers,
        &mpc_circuit_desc.circuit_description,
        committer,
        blinding_rng,
        converter,
        &mut transcript,
    );

    proof
}
*/

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct MPCCircuitConstData<F: Field> {
    pub encoding_matrix: MultilinearExtension<F>,
    pub evaluation_points: MultilinearExtension<F>,
    pub lookup_table_values: MultilinearExtension<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct MPCCircuitsAndConstData<F: Field> {
    pub mpc_circuit: Circuit<F>,
    pub encoding_matrix: MultilinearExtension<F>,
    pub evaluation_points: [MultilinearExtension<F>; 3],
}

#[derive(Serialize, Deserialize)]
pub struct MPCProver {
    #[serde(skip)]
    #[serde(default = "V3Prover::default_committer")]
    committer: PedersenCommitter<Bn256Point>,

    #[serde(skip)]
    #[serde(default = "VandermondeInverse::new")]
    converter: VandermondeInverse<Scalar>,

    slope_commitments: [HyraxProverInputCommitment<Bn256Point>; 2],

    prover_config: GKRCircuitProverConfig,
    mpc_circuit_and_const_mles_all_3_parties: MPCCircuitsAndConstData<Fr>,

    left_eye_proofs_all_3_parties: Option<Vec<HyraxProof<Bn256Point>>>,
    right_eye_proofs_all_3_parties: Option<Vec<HyraxProof<Bn256Point>>>,
}

impl MPCProver {
    /// Computes a set of random Shamir secret share slopes and creates a Hyrax
    /// commitment for them.
    ///
    /// Note that `rng` should be a CSPRNG seeded with a strong source of
    /// device entropy! Recommended to call this with `OsRng` and `ChaCha20Rng`.
    fn generate_secure_shamir_secret_share_slopes_and_commitments<C: PrimeOrderCurve>(
        num_vars: usize,
        log_num_cols: usize,
        pedersen_committer: &PedersenCommitter<C>,
        rng: &mut impl Rng,
    ) -> HyraxProverInputCommitment<C> {
        // The slopes must be in the range [0, 2^{16} - 1]
        let slopes_mle = MultilinearExtension::new(
            (0..IRISCODE_LEN)
                .map(|_idx| C::Scalar::from(rng.gen_range(0..GR4_MODULUS)))
                .collect_vec(),
        );

        commit_to_input_values(num_vars, log_num_cols, &slopes_mle, pedersen_committer, rng)
    }

    pub fn default_committer() -> PedersenCommitter<Bn256Point> {
        PedersenCommitter::new(512, PUBLIC_STRING, None)
    }

    pub fn new(
        prover_config: GKRCircuitProverConfig,
        mpc_circuit_and_aux_mles_all_3_parties: MPCCircuitsAndConstData<Fr>,
        rng: &mut (impl CryptoRng + RngCore),
    ) -> Self {
        let committer = Self::default_committer();

        let slope_input_layer_description = &mpc_circuit_and_aux_mles_all_3_parties
            .mpc_circuit
            .get_input_layer_description_ref(MPC_SLOPES_LAYER)
            .clone();

        let slopes_precommitment_left_eye =
            Self::generate_secure_shamir_secret_share_slopes_and_commitments(
                slope_input_layer_description.num_vars,
                SHAMIR_SECRET_SHARE_SLOPE_LOG_NUM_COLS,
                &committer,
                rng,
            );
        let slopes_precommitment_right_eye =
            Self::generate_secure_shamir_secret_share_slopes_and_commitments(
                slope_input_layer_description.num_vars,
                SHAMIR_SECRET_SHARE_SLOPE_LOG_NUM_COLS,
                &committer,
                rng,
            );

        Self {
            committer,
            converter: VandermondeInverse::new(),
            slope_commitments: [
                slopes_precommitment_left_eye,
                slopes_precommitment_right_eye,
            ],
            prover_config,
            mpc_circuit_and_const_mles_all_3_parties: mpc_circuit_and_aux_mles_all_3_parties,
            left_eye_proofs_all_3_parties: None,
            right_eye_proofs_all_3_parties: None,
        }
    }

    pub fn get_committer_ref(&self) -> &PedersenCommitter<Bn256Point> {
        &self.committer
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prove_mpc_with_precommits(
        mut mpc_provable_circuit: HyraxProvableCircuit<Bn256Point>,
        _iris_precommit: &HyraxProverInputCommitment<Bn256Point>,
        _mask_precommit: &HyraxProverInputCommitment<Bn256Point>,
        _slope_precommit: &HyraxProverInputCommitment<Bn256Point>,
        committer: &PedersenCommitter<Bn256Point>,
        blinding_rng: &mut (impl CryptoRng + RngCore),
        converter: &mut VandermondeInverse<Scalar>,
    ) -> HyraxProof<Bn256Point> {
        // TODO: Restore the precommits and fix the resulting bug!!
        /*
        mpc_provable_circuit
            .set_pre_commitment(MPC_IRISCODE_INPUT_LAYER, iris_precommit.clone(), None)
            .unwrap();
        mpc_provable_circuit
            .set_pre_commitment(MPC_MASKCODE_INPUT_LAYER, mask_precommit.clone(), None)
            .unwrap();
        mpc_provable_circuit
            .set_pre_commitment(
                MPC_SLOPES_LAYER,
                slope_precommit.clone(),
                Some(SHAMIR_SECRET_SHARE_SLOPE_LOG_NUM_COLS),
            )
            .unwrap();
        */

        // Create a fresh transcript.
        let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
            ECTranscript::new("MPC Circuit Pipeline");

        // Prove the relationship between iris/mask code and image.
        let (proof, _proof_config) = HyraxProof::prove(
            &mut mpc_provable_circuit,
            committer,
            blinding_rng,
            converter,
            &mut transcript,
        );

        proof
    }

    /*
    pub fn remove_auxiliary_input_layer(&mut self, is_left_eye: bool) {
        let three_proofs = if is_left_eye {
            &mut self.left_eye_proofs_all_3_parties
        } else {
            &mut self.right_eye_proofs_all_3_parties
        };

        (0..NUM_PARTIES).for_each(|party_idx| {
            let auxiliary_invariant_public_input_layer_id = self
                .mpc_circuit_and_aux_mles_all_3_parties[party_idx]
                .circuit_description
                .auxiliary_invariant_public_input_layer
                .layer_id;

            three_proofs.as_mut().unwrap()[party_idx]
                .remove_public_input_layer_by_id(auxiliary_invariant_public_input_layer_id)
        });
    }
    */

    pub fn prove(
        &mut self,
        is_left_eye: bool,
        iris_code_precommit: HyraxProverInputCommitment<Bn256Point>,
        mask_code_precommit: HyraxProverInputCommitment<Bn256Point>,
        rng: &mut (impl CryptoRng + RngCore),
    ) {
        assert_eq!(
            iris_code_precommit.blinding_factors_matrix.len(),
            iris_code_precommit.commitment.len()
        );
        // Ryan suggests asserting:
        //       iris_code_precommit.blinding_factors.len() ==
        //       iris_code_precommit.commitments.len() ==
        //       1 << LOG_NUM_COLS

        let iris_code_mle = &iris_code_precommit.mle;
        let mask_code_mle = &mask_code_precommit.mle;
        let slope_commitment = &self.slope_commitments[!is_left_eye as usize];

        /*
        let common_mpc_data = gen_mpc_common_aux_data::<Fr, MPC_NUM_IRIS_4_CHUNKS, 0>(iris_codes, masks, slopes, encoding_matrix, evaluation_points)

        let mpc_data_all_3_parties = [
            gen_mpc_input_data::<Fr, MPC_NUM_IRIS_4_CHUNKS, 0>(
                iris_code_mle,
                mask_code_mle,
                slope_mle,
            ),
            create_ss_circuit_inputs::<Fr, MPC_NUM_IRIS_4_CHUNKS, 1>(
                iris_code_mle,
                mask_code_mle,
                slope_mle,
            ),
            create_ss_circuit_inputs::<Fr, MPC_NUM_IRIS_4_CHUNKS, 2>(
                iris_code_mle,
                mask_code_mle,
                slope_mle,
            ),
        ];
        */

        let encoding_matrix = &self
            .mpc_circuit_and_const_mles_all_3_parties
            .encoding_matrix;

        let lookup_table_values =
            MultilinearExtension::new((0..GR4_MODULUS).map(Fr::from).collect());

        let proofs_all_3_parties: Vec<_> = (0..3)
            .map(|party_idx| {
                let mut circuit = self
                    .mpc_circuit_and_const_mles_all_3_parties
                    .mpc_circuit
                    .clone();

                let evaluation_points = &self
                    .mpc_circuit_and_const_mles_all_3_parties
                    .evaluation_points[party_idx];

                let input_data = gen_mpc_input_data::<Fr, MPC_NUM_IRIS_4_CHUNKS>(
                    &iris_code_mle,
                    &mask_code_mle,
                    &slope_commitment.mle,
                    encoding_matrix,
                    evaluation_points,
                );
                let const_data = MPCCircuitConstData {
                    encoding_matrix: encoding_matrix.clone(),
                    evaluation_points: evaluation_points.clone(),
                    lookup_table_values: lookup_table_values.clone(),
                };

                mpc_attach_data(&mut circuit, const_data, input_data);

                let provable_circuit = circuit
                    .finalize_hyrax()
                    .expect("Failed to finalize circuit");

                Self::prove_mpc_with_precommits(
                    provable_circuit,
                    &iris_code_precommit,
                    &mask_code_precommit,
                    &slope_commitment,
                    &self.committer,
                    rng,
                    &mut self.converter,
                )
            })
            .collect();

        self.set(is_left_eye, proofs_all_3_parties);
    }

    /// Set the field indicated by `is_mask` and `is_left_eye` to `proof`,
    /// overwritting any existing value.
    pub fn set(&mut self, is_left_eye: bool, proofs_all_3_parties: Vec<HyraxProof<Bn256Point>>) {
        match is_left_eye {
            false => self.set_right_proof(proofs_all_3_parties),
            true => self.set_left_proof(proofs_all_3_parties),
        }
    }

    /// Set the left image proof to `proof`, overwritting any existing value.
    pub fn set_left_proof(&mut self, proofs_all_3_parties: Vec<HyraxProof<Bn256Point>>) {
        self.left_eye_proofs_all_3_parties = Some(proofs_all_3_parties)
    }

    /// Set the right image proof to `proof`, overwritting any existing value.
    pub fn set_right_proof(&mut self, proofs_all_3_parties: Vec<HyraxProof<Bn256Point>>) {
        self.right_eye_proofs_all_3_parties = Some(proofs_all_3_parties)
    }

    pub fn is_set(&self, is_left_eye: bool) -> bool {
        match is_left_eye {
            true => {
                self.left_eye_proofs_all_3_parties.is_some()
                    && self.left_eye_proofs_all_3_parties.as_ref().unwrap().len() == 3
            }
            false => {
                self.right_eye_proofs_all_3_parties.is_some()
                    && self.right_eye_proofs_all_3_parties.as_ref().unwrap().len() == 3
            }
        }
    }

    /// Checks whether `self` is ready to be finalized, i.e. whether all 4
    /// proofs are present.`
    fn is_ready_to_finalize(&self) -> bool {
        self.is_set(true) && self.is_set(false)
    }

    pub fn finalize(&self) -> Result<MPCProof, MPCProofError> {
        if self.is_ready_to_finalize() {
            let proof_config = ProofConfig::new_from_prover_config(&self.prover_config);

            Ok(MPCProof::new(
                proof_config,
                self.left_eye_proofs_all_3_parties.as_ref().unwrap().clone(),
                self.right_eye_proofs_all_3_parties
                    .as_ref()
                    .unwrap()
                    .clone(),
            ))
        } else {
            Err(MPCProofError::EmptyProofField)
        }
    }
}

/// A Wrapper around the `V3Prover` and the `MPCProver`.
/// Includes the commitments to the iris / mask codes, so that the mpc4
/// circuits can use the pre-commitments as inputs.
#[derive(Serialize, Deserialize)]
pub struct V3MPCProver {
    /// The prover for the iris circuit.
    v3_prover: V3Prover,

    /// The prover for the mpc circuit.
    mpc_prover: MPCProver,

    /// The commitments to the iris / mask codes.
    left_iris_commit: Option<HyraxProverInputCommitment<Bn256Point>>,
    left_mask_commit: Option<HyraxProverInputCommitment<Bn256Point>>,
    right_iris_commit: Option<HyraxProverInputCommitment<Bn256Point>>,
    right_mask_commit: Option<HyraxProverInputCommitment<Bn256Point>>,
}

impl V3MPCProver {
    /// Generate an empty v3-mpc prover with the given configuration.
    pub fn new(
        prover_config: GKRCircuitProverConfig,
        iris_circuit: V3CircuitAndAuxData<Fr>,
        mpc_circuits: MPCCircuitsAndConstData<Fr>,
        rng: &mut (impl CryptoRng + RngCore),
    ) -> Self {
        Self {
            v3_prover: V3Prover::new(prover_config.clone(), iris_circuit),
            mpc_prover: MPCProver::new(prover_config, mpc_circuits, rng),

            left_iris_commit: None,
            left_mask_commit: None,
            right_iris_commit: None,
            right_mask_commit: None,
        }
    }

    /*
    fn remove_mpc_auxiliary_input_layer(&mut self, is_left_eye: bool) {
        self.mpc_prover.remove_auxiliary_input_layer(is_left_eye);
    }

    fn remove_v3_auxiliary_input_layer(&mut self, is_mask: bool, is_left_eye: bool) {
        self.v3_prover
            .remove_auxiliary_input_layer(is_mask, is_left_eye);
    }
    */

    pub fn prove_v3(
        &mut self,
        is_mask: bool,
        is_left_eye: bool,
        image_bytes: Vec<u8>,
        image_precommit: HyraxProverInputCommitment<Bn256Point>,
        rng: &mut (impl CryptoRng + RngCore),
    ) {
        let code_commitment =
            self.v3_prover
                .prove(is_mask, is_left_eye, image_bytes, image_precommit, rng);

        self.set_commit(is_mask, is_left_eye, code_commitment);

        // Remove the public `auxiliary_input_layer` from the proofs, as these are already
        // incorporated in the `CircuitAndAuxMles` as `iris_aux_mle` and `mask_aux_mle`.
        // self.remove_v3_auxiliary_input_layer(is_mask, is_left_eye);
    }

    pub fn prove_mpc(&mut self, is_left_eye: bool, rng: &mut (impl CryptoRng + RngCore)) {
        let iris_code_commitment = self.get_commit(false, is_left_eye).clone();
        let mask_code_commitment = self.get_commit(true, is_left_eye).clone();

        self.mpc_prover
            .prove(is_left_eye, iris_code_commitment, mask_code_commitment, rng);

        // Remove the invariant public `auxiliary_input_layer` from the proofs, as these are already
        // incorporated in the `MPCCircuitAndAuxMles` as `aux_mle`.
        // self.remove_mpc_auxiliary_input_layer(is_left_eye);
    }

    pub fn finalize(&self) -> Result<V3MPCProof, V3MPCProofError> {
        let v3_proof = self.v3_prover.finalize()?;
        let mpc_proof = self.mpc_prover.finalize()?;
        let commitments = V3MPCCommitments::new(
            self.get_commit_left_iris().commitment.clone(),
            self.get_commit_left_mask().commitment.clone(),
            self.get_commit_right_iris().commitment.clone(),
            self.get_commit_right_mask().commitment.clone(),
            self.mpc_prover.slope_commitments[0].commitment.clone(),
            self.mpc_prover.slope_commitments[1].commitment.clone(),
        );

        Ok(V3MPCProof {
            commitments,
            v3_proof,
            mpc_proof,
        })
    }

    /// Set the field indicated by `is_mask` and `is_left_eye` to `proof`,
    /// overwritting any existing value.
    pub fn set_commit(
        &mut self,
        is_mask: bool,
        is_left_eye: bool,
        commitment: HyraxProverInputCommitment<Bn256Point>,
    ) {
        match (is_mask, is_left_eye) {
            (false, false) => self.set_commit_right_iris(commitment),
            (false, true) => self.set_commit_left_iris(commitment),
            (true, false) => self.set_commit_right_mask(commitment),
            (true, true) => self.set_commit_left_mask(commitment),
        }
    }

    /// Set the left image commitment to `commitment`, overwritting any existing value.
    pub fn set_commit_left_iris(&mut self, commitment: HyraxProverInputCommitment<Bn256Point>) {
        self.left_iris_commit = Some(commitment)
    }

    /// Set the left mask commitment to `commitment`, overwritting any existing value.
    pub fn set_commit_left_mask(&mut self, commitment: HyraxProverInputCommitment<Bn256Point>) {
        self.left_mask_commit = Some(commitment)
    }

    /// Set the right image commitment to `commitment`, overwritting any existing value.
    pub fn set_commit_right_iris(&mut self, commitment: HyraxProverInputCommitment<Bn256Point>) {
        self.right_iris_commit = Some(commitment)
    }

    /// Set the right mask commitment to `commitment`, overwritting any existing value.
    pub fn set_commit_right_mask(&mut self, commitment: HyraxProverInputCommitment<Bn256Point>) {
        self.right_mask_commit = Some(commitment)
    }

    /// Set the field indicated by `is_mask` and `is_left_eye` to `proof`,
    /// overwritting any existing value.
    pub fn get_commit(
        &mut self,
        is_mask: bool,
        is_left_eye: bool,
    ) -> &HyraxProverInputCommitment<Bn256Point> {
        match (is_mask, is_left_eye) {
            (false, false) => self.get_commit_right_iris(),
            (false, true) => self.get_commit_left_iris(),
            (true, false) => self.get_commit_right_mask(),
            (true, true) => self.get_commit_left_mask(),
        }
    }

    /// Set the left image commitment to `commitment`, overwritting any existing value.
    pub fn get_commit_left_iris(&self) -> &HyraxProverInputCommitment<Bn256Point> {
        self.left_iris_commit.as_ref().unwrap()
    }

    /// Set the left mask commitment to `commitment`, overwritting any existing value.
    pub fn get_commit_left_mask(&self) -> &HyraxProverInputCommitment<Bn256Point> {
        self.left_mask_commit.as_ref().unwrap()
    }

    /// Set the right image commitment to `commitment`, overwritting any existing value.
    pub fn get_commit_right_iris(&self) -> &HyraxProverInputCommitment<Bn256Point> {
        self.right_iris_commit.as_ref().unwrap()
    }

    /// Set the right mask commitment to `commitment`, overwritting any existing value.
    pub fn get_commit_right_mask(&self) -> &HyraxProverInputCommitment<Bn256Point> {
        self.right_mask_commit.as_ref().unwrap()
    }

    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Failed to serialize V3MPCProver")
    }

    pub fn deserialize(bytes: &[u8]) -> Self {
        bincode::deserialize(bytes).expect("Failed to deserialize V3MPCProver")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MPCPartyProof {
    #[serde(skip)]
    #[serde(default = "V3Prover::default_committer")]
    committer: PedersenCommitter<Bn256Point>,

    proof_config: ProofConfig,
    left_eye_proof: HyraxProof<Bn256Point>,
    right_eye_proof: HyraxProof<Bn256Point>,
}

impl MPCPartyProof {
    pub fn new(
        proof_config: ProofConfig,
        left_eye_proof: HyraxProof<Bn256Point>,
        right_eye_proof: HyraxProof<Bn256Point>,
    ) -> Self {
        Self {
            committer: V3Prover::default_committer(),
            proof_config,
            left_eye_proof,
            right_eye_proof,
        }
    }

    #[cfg(feature = "print-trace")]
    pub fn print_size(&self) {
        println!("Left eye proof stats:");
        self.left_eye_proof.print_size();

        println!("Right eye proof stats:");
        self.right_eye_proof.print_size();
    }

    /*
    pub fn decompose(self) -> (ProofConfig, HyraxProof<Bn256Point>, HyraxProof<Bn256Point>) {
        (self.proof_config, self.left_eye_proof, self.right_eye_proof)
    }
    */

    pub fn insert_aux_public_data_by_id(
        &mut self,
        is_left_eye: bool,
        aux_mle: &MultilinearExtension<Fr>,
        id_to_insert: LayerId,
    ) {
        if is_left_eye {
            self.left_eye_proof
                .insert_aux_public_data_by_id(aux_mle, id_to_insert);
        } else {
            self.right_eye_proof
                .insert_aux_public_data_by_id(aux_mle, id_to_insert);
        }
    }

    pub fn verify_mpc_proof(
        &self,
        is_left_eye: bool,
        // mpc_circuit_desc: &MPCCircuitDescription<Fr>,
        mpc_circuit: &HyraxVerifiableCircuit<Bn256Point>,
        secret_share_mle_layer_id: LayerId,
    ) -> Result<MultilinearExtension<Fr>, MPCError> {
        /*
        let mut verifier_hyrax_input_layers = HashMap::new();

        verifier_hyrax_input_layers.insert(
            mpc_circuit_desc.slope_input_layer.layer_id,
            mpc_circuit_desc.slope_input_layer.clone().into(),
        );
        verifier_hyrax_input_layers.insert(
            mpc_circuit_desc.iris_code_input_layer.layer_id,
            mpc_circuit_desc.iris_code_input_layer.clone().into(),
        );
        verifier_hyrax_input_layers.insert(
            mpc_circuit_desc.mask_code_input_layer.layer_id,
            mpc_circuit_desc.mask_code_input_layer.clone().into(),
        );
        verifier_hyrax_input_layers.insert(
            mpc_circuit_desc.auxilary_input_layer.layer_id,
            mpc_circuit_desc.auxilary_input_layer.clone().into(),
        );
        */

        // Create a fresh transcript.
        let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
            ECTranscript::new("V3 Iriscode Circuit Pipeline");

        let proof = if is_left_eye {
            &self.left_eye_proof
        } else {
            &self.right_eye_proof
        };

        // Verify the relationship between iris/mask code and image.
        verify_hyrax_proof(
            proof,
            mpc_circuit,
            &self.committer,
            &mut transcript,
            &self.proof_config,
        );

        // Here we need to grab the actual secret share MLE and return it.
        let secret_share_mle_should_be_singleton = proof
            .public_inputs
            .iter()
            .filter(|(input_layer_id, _)| *input_layer_id == secret_share_mle_layer_id)
            .collect_vec();
        if secret_share_mle_should_be_singleton.len() != 1 {
            Err(MPCError::NoSharesInputLayerMle)
        } else {
            Ok(secret_share_mle_should_be_singleton[0].clone().1.unwrap())
        }
    }

    pub fn get_proof_config_ref(&self) -> &ProofConfig {
        &self.proof_config
    }

    pub fn get_left_eye_proof_ref(&self) -> &HyraxProof<Bn256Point> {
        &self.left_eye_proof
    }

    pub fn get_right_eye_proof_ref(&self) -> &HyraxProof<Bn256Point> {
        &self.right_eye_proof
    }

    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap()
    }

    pub fn deserialize(serialized_proof: &[u8]) -> Self {
        bincode::deserialize(serialized_proof).unwrap()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MPCProof {
    party_proofs: [MPCPartyProof; 3],
}

impl MPCProof {
    pub fn new(
        proof_config: ProofConfig,
        mut left_eye_proofs_all_3_parties: Vec<HyraxProof<Bn256Point>>,
        mut right_eye_proofs_all_3_parties: Vec<HyraxProof<Bn256Point>>,
    ) -> Self {
        assert_eq!(left_eye_proofs_all_3_parties.len(), 3);
        assert_eq!(right_eye_proofs_all_3_parties.len(), 3);

        let party_2_proof = MPCPartyProof::new(
            proof_config,
            left_eye_proofs_all_3_parties.pop().unwrap(),
            right_eye_proofs_all_3_parties.pop().unwrap(),
        );
        let party_1_proof = MPCPartyProof::new(
            proof_config,
            left_eye_proofs_all_3_parties.pop().unwrap(),
            right_eye_proofs_all_3_parties.pop().unwrap(),
        );
        let party_0_proof = MPCPartyProof::new(
            proof_config,
            left_eye_proofs_all_3_parties.pop().unwrap(),
            right_eye_proofs_all_3_parties.pop().unwrap(),
        );

        Self {
            party_proofs: [party_0_proof, party_1_proof, party_2_proof],
        }
    }

    pub fn get_party_proof_ref(&self, party_idx: usize) -> &MPCPartyProof {
        assert!(party_idx < 3);
        &self.party_proofs[party_idx]
    }

    /// Get the mpc proofs for all 3 parties indicated by `is_left_eye`.
    pub fn get(&self, is_left_eye: bool) -> Vec<&HyraxProof<Bn256Point>> {
        match is_left_eye {
            true => self.get_left_proof_refs(),
            false => self.get_right_proof_refs(),
        }
    }

    /// Return a reference to the proofs to the party_idx.
    pub fn get_proof_refs_for_party(
        &self,
        party_idx: usize,
    ) -> (&HyraxProof<Bn256Point>, &HyraxProof<Bn256Point>) {
        assert!(party_idx < 3);
        (
            self.party_proofs[party_idx].get_left_eye_proof_ref(),
            self.party_proofs[party_idx].get_right_eye_proof_ref(),
        )
    }

    /// Return a reference to the proofs to the left eye.
    pub fn get_left_proof_refs(&self) -> Vec<&HyraxProof<Bn256Point>> {
        vec![
            self.party_proofs[0].get_left_eye_proof_ref(),
            self.party_proofs[1].get_left_eye_proof_ref(),
            self.party_proofs[2].get_left_eye_proof_ref(),
        ]
    }

    /// Return a reference to the proofs to the right eye.
    pub fn get_right_proof_refs(&self) -> Vec<&HyraxProof<Bn256Point>> {
        vec![
            self.party_proofs[0].get_right_eye_proof_ref(),
            self.party_proofs[1].get_right_eye_proof_ref(),
            self.party_proofs[2].get_right_eye_proof_ref(),
        ]
    }

    /// Serializes `self` into a binary representation.
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap()
    }

    /// Deserializes `serialized_proof` and returns it.
    pub fn deserialize(serialized_proof: &[u8]) -> Self {
        bincode::deserialize(serialized_proof).unwrap()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct V3MPCProof {
    commitments: V3MPCCommitments<Bn256Point>,
    v3_proof: V3Proof,
    mpc_proof: MPCProof,
}

impl V3MPCProof {
    pub fn get_commitments_ref(&self) -> &V3MPCCommitments<Bn256Point> {
        &self.commitments
    }

    pub fn get_v3_proof_ref(&self) -> &V3Proof {
        &self.v3_proof
    }

    pub fn get_party_proof_ref(&self, party_idx: usize) -> &MPCPartyProof {
        self.mpc_proof.get_party_proof_ref(party_idx)
    }

    /// Serializes `self` into a binary representation.
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap()
    }

    /// Deserializes `serialized_proof` and returns it.
    pub fn deserialize(serialized_proof: &[u8]) -> Self {
        bincode::deserialize(serialized_proof).unwrap()
    }
}

/*
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct MPCCircuitAndAuxMles<F: Field> {
    circuit_description: MPCCircuitDescription<F>,
    input_builder_metadata: MPCInputBuilderMetadata,
    aux_mle: MultilinearExtension<F>,
}

impl<F: Field> MPCCircuitAndAuxMles<F> {
    pub fn get_input_builder_metadata(&self) -> &MPCInputBuilderMetadata {
        &self.input_builder_metadata
    }

    pub fn get_circuit(&self) -> &MPCCircuitDescription<F> {
        &self.circuit_description
    }

    pub fn get_aux_mle(&self) -> &MultilinearExtension<F> {
        &self.aux_mle
    }

    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(&self).expect("Failed to serialize MPCCircuitAndAuxMles")
    }

    pub fn deserialize(bytes: &[u8]) -> Self {
        bincode::deserialize(bytes).expect("Failed to deserialize MPCCircuitAndAuxMles")
    }
}
*/

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct V3MPCCircuitAndAuxMles<F: Field> {
    pub v3_circuit_and_aux_data: V3CircuitAndAuxData<F>,
    pub mpc_circuit_and_aux_mles_all_3_parties: MPCCircuitsAndConstData<F>,
}

impl<F: Field> V3MPCCircuitAndAuxMles<F> {
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(&self).expect("Failed to serialize CircuitAndAuxMles")
    }

    pub fn deserialize(bytes: &[u8]) -> Self {
        bincode::deserialize(bytes).expect("Failed to deserialize CircuitAndAuxMles")
    }
}

// Generate the circuit description and input builder used to generate the auxiliary MLEs.
pub fn generate_mpc_circuit_and_aux_mles_all_3_parties<F: Field>() -> MPCCircuitsAndConstData<F> {
    let mpc_circuit = build_circuit::<F, MPC_NUM_IRIS_4_CHUNKS>(LayerVisibility::Private);

    MPCCircuitsAndConstData {
        mpc_circuit,
        encoding_matrix: gen_mpc_encoding_matrix::<F, MPC_NUM_IRIS_4_CHUNKS>(),
        evaluation_points: [
            gen_mpc_evaluation_points::<F, MPC_NUM_IRIS_4_CHUNKS, 0>(),
            gen_mpc_evaluation_points::<F, MPC_NUM_IRIS_4_CHUNKS, 1>(),
            gen_mpc_evaluation_points::<F, MPC_NUM_IRIS_4_CHUNKS, 2>(),
        ],
    }
}
