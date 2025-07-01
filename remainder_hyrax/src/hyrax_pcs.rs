use crate::{
    hyrax_gkr::hyrax_layer::HyraxClaim, hyrax_primitives::proof_of_dot_prod::ProofOfDotProduct,
};
use ark_std::cfg_iter;
use itertools::Itertools;
use ndarray::Array2;
use rand::{CryptoRng, RngCore};
use remainder::mle::evals::{bit_packed_vector::num_bits, MultilinearExtension};
use remainder_shared_types::{
    config::global_config::global_prover_hyrax_batch_opening,
    curves::PrimeOrderCurve,
    pedersen::{CommittedScalar, PedersenCommitter},
    utils::bookkeeping_table::{initialize_tensor, initialize_tensor_rlc},
    Zeroizable,
};
use remainder_shared_types::{ff_field, transcript::ec_transcript::ECTranscriptTrait};
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
#[cfg(test)]
pub mod tests;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
/// Struct that contains all the information needed in a Hyrax evaluation proof.
/// The verifier uses the information in this struct in order to verify that the
/// prover indeed knows the evaluation of an MLE at a random challenge point.
pub struct HyraxPCSEvaluationProof<C: PrimeOrderCurve> {
    /// This is the proof of dot product evaluation proof from the vectors L*T,
    /// and R where T is the matrix of MLE coefficients, and L and R are the
    /// corresponding equality vectors for the random challenge points.
    podp_evaluation_proof: ProofOfDotProduct<C>,
    /// This is the commitment to what the prover claims the evaluation of the
    /// MLE at the random challenge point is.
    pub commitment_to_evaluation: C,
}

impl<C: PrimeOrderCurve> HyraxPCSEvaluationProof<C> {
    /// This function computes the commitments to the rows of the matrix.
    /// essentially, this is the vector of commitments that the prover should be
    /// sending over to the verifier.
    ///
    /// An optimized version of the vector commitment is used if and only if
    /// every value in the `data` [MultilinearExtension] can be fit within 128
    /// bits. The maximum number of bits needed is used to decide which unsigned
    /// integer primitive to cast all the field elements to.
    pub fn compute_matrix_commitments(
        // The log-size of the matrix rows. Both the row size and the column
        // size need to be powers of 2 for the Hyrax commitment scheme.
        log_n_cols: usize,
        data: &MultilinearExtension<C::Scalar>,
        committer: &PedersenCommitter<C>,
        blinding_factors: &[C::Scalar],
    ) -> Vec<C> {
        // Checking that the matrix row size and the matrix column size are both
        // powers of two.
        assert!(data.len().is_power_of_two());
        // Check that the number of blinding factors is the same as the number
        // of rows in the matrix, as each row gets one blinding factor.
        assert_eq!(data.len(), (1 << log_n_cols) * blinding_factors.len());

        // We take the largest value in the MLE and compute the bits needed to
        // represent it.
        // The max number of bits needed to encode any element in this input.
        let data_vec = data.to_vec();
        let max_input_mle_value = cfg_iter!(data_vec).max().unwrap();
        let max_num_bits_needed = num_bits(*max_input_mle_value);

        if max_num_bits_needed > 128 {
            data.to_vec()
                .chunks(1 << log_n_cols)
                .zip(blinding_factors.iter())
                .map(|(chunk, blind)| committer.vector_commit(chunk, blind))
                .collect_vec()
        } else {
            // Determine whether the number of bits fits within a u8, u16, u32,
            // u64, or u128.
            if max_num_bits_needed <= 8 {
                data.convert_into_u8_vec()
                    .chunks(1 << log_n_cols)
                    .zip(blinding_factors.iter())
                    .map(|(chunk, blind)| committer.unsigned_integer_vector_commit(chunk, blind))
                    .collect_vec()
            } else if max_num_bits_needed <= 16 {
                data.convert_into_u16_vec()
                    .chunks(1 << log_n_cols)
                    .zip(blinding_factors.iter())
                    .map(|(chunk, blind)| committer.unsigned_integer_vector_commit(chunk, blind))
                    .collect_vec()
            } else if max_num_bits_needed <= 32 {
                data.convert_into_u32_vec()
                    .chunks(1 << log_n_cols)
                    .zip(blinding_factors.iter())
                    .map(|(chunk, blind)| committer.unsigned_integer_vector_commit(chunk, blind))
                    .collect_vec()
            } else if max_num_bits_needed <= 64 {
                data.convert_into_u64_vec()
                    .chunks(1 << log_n_cols)
                    .zip(blinding_factors.iter())
                    .map(|(chunk, blind)| committer.unsigned_integer_vector_commit(chunk, blind))
                    .collect_vec()
            } else {
                assert!(max_num_bits_needed <= 128);
                data.convert_into_u128_vec()
                    .chunks(1 << log_n_cols)
                    .zip(blinding_factors.iter())
                    .map(|(chunk, blind)| committer.unsigned_integer_vector_commit(chunk, blind))
                    .collect_vec()
            }
        }
    }

    #[allow(clippy::type_complexity)]
    /// This function computes the "L" and the "R" vectors where L \times T
    /// \times R is the evaluation of the MLE whose coefficients are represented
    /// in T at a random challenge point given in the function.
    ///
    /// The `challenge_coordinates_vec` are passed into this function
    /// such that the indices after `log_n_cols` of each vec are all the same
    /// challenge coordinates. Hence, we can just take the random
    /// linear combination of all the "L" vectors to take the RLC of the
    /// left side, and factor out the "R" vector since all the challenge
    /// coordinates are shared.
    pub fn compute_l_r_from_log_n_cols(
        // The log size of the rows.
        log_n_cols: usize,
        // The challenge coordinates we are evaluating the MLE at.
        challenge_coordinates_vec: &[&Vec<C::Scalar>],
        // The random coefficients for the "L" vector tensor.
        random_coefficients: &[C::Scalar],
    ) -> (Vec<C::Scalar>, Vec<C::Scalar>) {
        // We split the challenge at the log number of rows in the matrix.
        // This way, the product L * T * R = the matrix evaluation.
        let (l_challenge_coords, r_challenge_coords): (Vec<&[C::Scalar]>, Vec<&[C::Scalar]>) =
            challenge_coordinates_vec
                .iter()
                .map(|challenge_coordinate| {
                    challenge_coordinate.split_at(challenge_coordinate.len() - log_n_cols)
                })
                .unzip();
        let first_r_chal_coordinate = &r_challenge_coords[0];
        // Check that all the right-hand-side coordinates do indeed match so we can
        // factor out the R vector.
        assert!(r_challenge_coords
            .iter()
            .all(|chal_coor| chal_coor == first_r_chal_coordinate));
        // The L and R vectors are simply the \chi values for the challenge
        // points provided.
        (
            // We take the RLC of the left-hand-side vectors.
            initialize_tensor_rlc(&l_challenge_coords, random_coefficients),
            // We factor out the right-hand-side vector and just produce its tensor.
            initialize_tensor(first_r_chal_coordinate),
        )
    }

    /// This computes the product of a vector and a matrix, resulting in a
    /// single vector. Because the matrix s provided as a flattened vector, we have a
    /// log_n_cols which determines the log size of the rows of the matrix.
    fn vector_matrix_product(
        // The left vector in this matrix product.
        vector: &[C::Scalar],
        // The right matrix in this matrix product.
        flattened_matrix: &MultilinearExtension<C::Scalar>,
        // Log dimension of the row size.
        log_n_cols: usize,
    ) -> Vec<C::Scalar> {
        let matrix_as_array2 = Array2::from_shape_vec(
            (flattened_matrix.len() / (1 << log_n_cols), 1 << log_n_cols),
            flattened_matrix.to_vec(),
        )
        .unwrap();

        // We do a simple dot product by transposing the matrix and treating
        // each row as a vector and taking the dot product of that and the
        // vector provided in the function. This results in the matrix vector
        // product as a vector of scalar field elements.
        let vec_matrix_prod = if !vector.is_empty() {
            matrix_as_array2
                .reversed_axes()
                .outer_iter()
                .map(|row| {
                    row.iter()
                        .zip(vector.iter())
                        .fold(C::Scalar::ZERO, |acc, (row_elem, vec_elem)| {
                            acc + (*row_elem * vec_elem)
                        })
                })
                .collect_vec()
        } else {
            assert_eq!(flattened_matrix.len(), 1, "we can only have a valid matrix-vector product when the vector is len 0 if the matrix is len 1");
            vec![flattened_matrix.first()]
        };

        vec_matrix_prod
    }

    /// The function where we construct the hyrax evaluation proof that the
    /// verifier then uses to determine whether the prover truly knows the
    /// correct evaluation of an MLE at a certain random challenge.
    #[allow(clippy::too_many_arguments)]
    pub fn prove(
        // The log size of the rows of the matrix.
        log_n_cols: usize,
        // A vector representing the coefficients of the MLE.
        data: &MultilinearExtension<C::Scalar>,
        // The batch of challenge coordinates and their proported evaluations on the MLE.
        claims: &[HyraxClaim<C::Scalar, CommittedScalar<C>>],
        // The pedersen vector and scalar commitment generators needed in order
        // to commit to the rows of the matrix and for the proof of dot product.
        committer: &PedersenCommitter<C>,
        // The random generator that the prover needs for proof of dot product.
        mut rng: &mut (impl CryptoRng + RngCore),
        // Transcript that the prover needs for proof of dot product.
        transcript: &mut impl ECTranscriptTrait<C>,
        // The blinding factors to commit to each of the rows of the MLE
        // coefficient matrix.
        blinding_factors_matrix: &mut [C::Scalar],
    ) -> Self {
        let random_coefficients = if !global_prover_hyrax_batch_opening() {
            assert_eq!(claims.len(), 1);
            vec![C::Scalar::ONE]
        } else {
            transcript
                .get_scalar_field_challenges("random coefficients for batch opening", claims.len())
        };
        let (l_vector, r_vector) = HyraxPCSEvaluationProof::<C>::compute_l_r_from_log_n_cols(
            log_n_cols,
            &claims.iter().map(|claim| &claim.point).collect_vec(),
            &random_coefficients,
        );

        // Since the prover knows the T matrix (matrix of MLE coefficients), the
        // prover can simply do a vector matrix product to compute T' = L \times
        // T.
        let t_prime =
            HyraxPCSEvaluationProof::<C>::vector_matrix_product(&l_vector, data, log_n_cols);

        // FIXME we could make this nicer by using the CommittedVector for each
        // row
        // The blinding factor for the commitment to the T' vector is a
        // combination of the blinding factors of the commitments to the rows of
        // the T matrix. it is a dot product of the L vector and the blinding
        // factors of the matrix rows commits.
        let mut blinding_factor_t_prime = if !l_vector.is_empty() {
            blinding_factors_matrix
                .iter_mut()
                .zip(l_vector.iter())
                .fold(
                    C::Scalar::ZERO,
                    |acc: <C as PrimeOrderCurve>::Scalar, (row_blind, l_vec_exponent)| {
                        acc + (*row_blind * l_vec_exponent)
                    },
                )
        } else {
            assert_eq!(
                blinding_factors_matrix.len(),
                1,
                "can only have one blinding factor if the vector is empty"
            );
            blinding_factors_matrix[0]
        };

        // Commit to t_prime_vector.
        let t_prime_commit = committer.committed_vector(&t_prime, &blinding_factor_t_prime);
        blinding_factor_t_prime.zeroize();

        // Commit to the dot product, add this to the transcript.
        let mle_evaluation_committed_scalar = random_coefficients
            .iter()
            .zip(claims.iter())
            .fold(CommittedScalar::zero(), |acc, (random_coeff, claim)| {
                acc + (&claim.evaluation * *random_coeff)
            });
        transcript.append_ec_point(
            "commitment to y",
            mle_evaluation_committed_scalar.commitment,
        );

        // Now that we have a commitment to T', we can do a proof of dot product
        // which claims that the dot product of T' and R is indeed the MLE
        // evaluation point. This evaluation proof is what the hyrax verifier
        // uses to verify this statement.
        let podp_evaluation_proof = ProofOfDotProduct::<C>::prove(
            &t_prime_commit,
            &mle_evaluation_committed_scalar,
            &r_vector,
            committer,
            &mut rng,
            transcript,
        );

        // Return the hyrax evaluation proof which comprises of the PoDP
        // evaluation proof along with the commitments to the rows of the matrix
        // and the evaluation of the matrix at the random challenge.
        Self {
            podp_evaluation_proof,
            commitment_to_evaluation: mle_evaluation_committed_scalar.commitment,
        }
    }

    pub fn verify(
        &self,
        log_n_cols: usize,
        committer: &PedersenCommitter<C>,
        commitment_to_coeff_matrix: &[C],
        challenge_coordinates: &[&Vec<C::Scalar>],
        transcript: &mut impl ECTranscriptTrait<C>,
    ) {
        let random_coefficients = if !global_prover_hyrax_batch_opening() {
            assert_eq!(challenge_coordinates.len(), 1);
            vec![C::Scalar::ONE]
        } else {
            transcript.get_scalar_field_challenges(
                "random coefficients for batch opening",
                challenge_coordinates.len(),
            )
        };
        let Self {
            podp_evaluation_proof,
            commitment_to_evaluation,
        } = &self;
        let (l_vector, r_vector) = HyraxPCSEvaluationProof::<C>::compute_l_r_from_log_n_cols(
            log_n_cols,
            challenge_coordinates,
            &random_coefficients,
        );

        // The verifier uses the L vector and does a scalar multiplication to
        // each of the matrix row commits with the appropriate index of the L
        // vector. it then adds them all together to get the commitment to T' =
        // L \times T.
        assert_eq!(l_vector.len(), commitment_to_coeff_matrix.len());
        let t_prime_commit_from_t_commit = if !l_vector.is_empty() {
            commitment_to_coeff_matrix
                .iter()
                .zip(l_vector.iter())
                .fold(C::zero(), |acc, (row_commit, l_vec_elem)| {
                    acc + (*row_commit * *l_vec_elem)
                })
        } else {
            assert_eq!(commitment_to_coeff_matrix.len(), 1);
            commitment_to_coeff_matrix[0]
        };

        transcript.append_ec_point("commitment to y", *commitment_to_evaluation);

        // Using this commitment, the verifier can then do a proof of dot
        // product verification given the evaluation proof and the prover's
        // claimed commitment to the evaluation of the MLE.
        podp_evaluation_proof.verify(
            &t_prime_commit_from_t_commit,
            commitment_to_evaluation,
            &r_vector,
            committer,
            transcript,
        )
    }
}
