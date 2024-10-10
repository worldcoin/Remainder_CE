use crate::hyrax_primitives::proof_of_dot_prod::ProofOfDotProduct;
use ark_std::cfg_into_iter;
use itertools::Itertools;
use ndarray::Array2;
use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::iter::IntoParallelIterator;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use remainder::layouter::nodes::circuit_inputs::HyraxInputDType;
use remainder_shared_types::ff_field;
use remainder_shared_types::transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript};
use remainder_shared_types::{
    curves::PrimeOrderCurve,
    pedersen::{CommittedScalar, PedersenCommitter},
    HasByteRepresentation,
};
use serde::{Deserialize, Serialize};

#[cfg(test)]
pub mod tests;

#[derive(Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct HyraxAuxInfo<C: PrimeOrderCurve> {
    pub log_n_cols: usize,
    pub committer: PedersenCommitter<C>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
/// struct that contains all the information needed in a hyrax evaluation proof. the verifier
/// uses the information in this struct in order to verify that the prover indeed knows the
/// evaluation of an MLE at a random challenge point.
pub struct HyraxPCSProof<C: PrimeOrderCurve> {
    /// this is the proof of dot product evaluation proof from the vectors L*T, and R where T
    /// is the matrix of MLE coefficients, and L and R are the corresponding equality vectors
    /// for the random challenge points
    podp_evaluation_proof: ProofOfDotProduct<C>,
    /// this is the commitment to what the prover claims the evaluation of the MLE at the random
    /// challenge point is.
    pub commitment_to_evaluation: CommittedScalar<C>,
    /// this is the blinding factor that is used to compute `commitment_to_evaluation` above.
    /// TODO(vishady) NOTE: this needs to be removed once Hyrax IP is implemented as we shouldn't be revealing
    /// the actual evaluation at that point.
    pub blinding_factor_evaluation: C::Scalar,
    /// this is the auxiliary information needed by the input layer in order to verify the opening proof
    pub aux: HyraxAuxInfo<C>,
}

/// an enum representing how the user can specify their MLE coefficients. at least for our pedersen
/// commitments, the distinction matters between u8, i8, and scalar field elements because of the
/// precomputations.
#[derive(Clone, Debug)]
pub enum MleCoefficientsVector<C: PrimeOrderCurve> {
    U8Vector(Vec<u8>),
    I8Vector(Vec<i8>),
    ScalarFieldVector(Vec<C::Scalar>),
}

impl<C: PrimeOrderCurve> MleCoefficientsVector<C> {
    /// From scalar field elements to the desired data type
    pub fn convert_from_scalar_field(
        scalar_field_vec: &[C::Scalar],
        dtype: &HyraxInputDType,
    ) -> Self {
        match dtype {
            HyraxInputDType::U8 => {
                let coeffs_vec = scalar_field_vec
                    .iter()
                    .map(|elem| {
                        let elem_bytes = elem.to_bytes_le();
                        // Ryan's note: This doesn't work because even if we don't have that many nonzero bytes we will always return 32
                        // assert_eq!(elem_bytes.len(), 1);
                        elem_bytes[0]
                    })
                    .collect_vec();
                Self::U8Vector(coeffs_vec)
            }
            HyraxInputDType::I8 => {
                todo!()
            }
        }
    }
    /// converts the vector of coefficients into all scalar field elements. this is useful in the step of hyrax
    /// where the prover needs to do a vector matrix multiplicaation with a vector of scalar field elements.
    pub fn convert_to_scalar_field(&self) -> Vec<C::Scalar> {
        match &self {
            // for u8s and i8s, because the scalar field size is going to be much larger than 2^8 we can always
            // just directly cast these to scalar field elements.
            MleCoefficientsVector::U8Vector(vec) => vec
                .iter()
                .map(|elem| C::Scalar::from(*elem as u64))
                .collect_vec(),
            MleCoefficientsVector::I8Vector(vec) => vec
                .iter()
                .map(|elem| C::Scalar::from(*elem as u64))
                .collect_vec(),
            MleCoefficientsVector::ScalarFieldVector(vec) => vec.to_vec(),
        }
    }
    pub fn len(&self) -> usize {
        match &self {
            MleCoefficientsVector::U8Vector(vec) => vec.len(),
            MleCoefficientsVector::I8Vector(vec) => vec.len(),
            MleCoefficientsVector::ScalarFieldVector(vec) => vec.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self {
            MleCoefficientsVector::U8Vector(vec) => vec.is_empty(),
            MleCoefficientsVector::I8Vector(vec) => vec.is_empty(),
            MleCoefficientsVector::ScalarFieldVector(vec) => vec.is_empty(),
        }
    }
}

impl<C: PrimeOrderCurve> HyraxPCSProof<C> {
    /// this function computes the commitments to the rows of the matrix. essentially, this is the vector of
    /// commitments that the prover should be sending over to the verifier.
    pub fn compute_matrix_commitments(
        // the log-size of the matrix rows. both the row size and the column size need to be powers of 2
        // for hyrax to work!
        log_n_cols: usize,
        data: &MleCoefficientsVector<C>,
        committer: &PedersenCommitter<C>,
        blinding_factors: &[C::Scalar],
    ) -> Vec<C> {
        // checking that the matrix row size and the matrix column size are both powers of two! otherwise hyrax does not work
        assert!(data.len().is_power_of_two());

        // this appropriately computes the commitments to the coefficients matrix based on its internal type. if it is a u8
        // or an i8, we can use precomputed bit decompositions in order to speed up the pedersen commitments!!
        let commits: Vec<C> = match data {
            MleCoefficientsVector::U8Vector(coeff_vector_u8) => {
                let u8committer: PedersenCommitter<C> = PedersenCommitter::new_with_generators(
                    committer.generators.clone(),
                    committer.blinding_generator,
                    Some(8),
                );
                // we are using the u8_vector_commit to commit to each of the rows of the matrix, which are determined by
                // the log_n_cols!
                coeff_vector_u8
                    .chunks(1 << log_n_cols)
                    .zip(blinding_factors.iter())
                    .map(|(chunk, blind)| u8committer.u8_vector_commit(chunk, blind))
                    .collect_vec()
            }
            MleCoefficientsVector::I8Vector(coeff_vector_i8) => {
                let i8committer: PedersenCommitter<C> = PedersenCommitter::new_with_generators(
                    committer.generators.clone(),
                    committer.blinding_generator,
                    Some(8),
                );
                // we are using the i8_vector_commit to commit to each of the rows of the matrix
                coeff_vector_i8
                    .chunks(1 << log_n_cols)
                    .zip(blinding_factors.iter())
                    .map(|(chunk, blind)| i8committer.i8_vector_commit(chunk, blind))
                    .collect_vec()
            }
            MleCoefficientsVector::ScalarFieldVector(coeff_vector_scalar_field) => {
                // we are using the regular vector_commit to commit to the rows of the matrix
                coeff_vector_scalar_field
                    .chunks(1 << log_n_cols)
                    .zip(blinding_factors.iter())
                    .map(|(chunk, blind)| committer.vector_commit(chunk, blind))
                    .collect_vec()
            }
        };
        commits
    }

    /// this computes a vector over all b of \chi_b which is \prod_{i}{(1-r_i)(1-b_i) + r_ib_i} where r_i is the bit
    /// decomposition of the random challenge point and b_i is the bit decomposition of the current index. essentially
    /// this is an MLE whose coefficients represent whether the index is equal to the random challenge point.
    /// NOTE: this is computed in little endian in order to work with the rest of our MLEs.
    fn compute_equality_chi_values(challenge_coordinates: &[C::Scalar]) -> Vec<C::Scalar> {
        // accounting for the case where we in fact don't want a matrix but just a row vector or a column vector. then
        // our challenge coordinates might be empty!
        if !challenge_coordinates.is_empty() {
            // dynamic programming algorithm in Tha13 for computing these equality values and returning them as a vector
            let (one_minus_r, r) = (
                C::Scalar::ONE - challenge_coordinates[0],
                challenge_coordinates[0],
            );
            let mut cur_table = vec![one_minus_r, r];

            for challenge in challenge_coordinates.iter().skip(1) {
                let (one_minus_r, r) = (C::Scalar::ONE - challenge, challenge);
                let mut firsthalf: Vec<C::Scalar> = cfg_into_iter!(cur_table.clone())
                    .map(|eval| eval * one_minus_r)
                    .collect();
                let secondhalf: Vec<C::Scalar> =
                    cfg_into_iter!(cur_table).map(|eval| eval * r).collect();
                firsthalf.extend(secondhalf.iter());
                cur_table = firsthalf;
            }

            cur_table
        } else {
            vec![]
        }
    }

    /// this function computes the "L" and the "R" vectors where L \times T \times R is the evaluation of the MLE whose
    /// coefficients are represented in T at a random challenge point given in the function.
    pub fn compute_l_r_from_log_n_cols(
        // the log size of the rows
        log_n_cols: usize,
        // the challenge coordinates we are evaluating the MLE at!
        challenge_coordinates: &[C::Scalar],
    ) -> (Vec<C::Scalar>, Vec<C::Scalar>) {
        // because of endian-ness, we need to actually reverse the order of the two. the r_challenge_coords are the most significant bits
        // and the l_challenge_coords are the least significant bits.
        let (r_challenge_coords, l_challenge_coords) = challenge_coordinates.split_at(log_n_cols);

        // the L and R vectors are simply the \chi values for the challenge points provided.
        (
            HyraxPCSProof::<C>::compute_equality_chi_values(l_challenge_coords),
            HyraxPCSProof::<C>::compute_equality_chi_values(r_challenge_coords),
        )
    }

    /// this computes the product of a vector and a matrix, resulting in a single vector. because the matrix is provided
    /// as a vector, we have a log_n_cols which determines the log size of the rows of the matrix.
    fn vector_matrix_product(
        // the left vector in this matrix product
        vector: &[C::Scalar],
        // the right matrix in this matrix product
        matrix: &MleCoefficientsVector<C>,
        // log dimension of the row size
        log_n_cols: usize,
    ) -> Vec<C::Scalar> {
        // the vector is a vector of scalar field elements, so we need the matrix to also be of scalar field elements.
        // because it is either u8, i8, or already scalar field elements we can do this conversion straightforwardly. see
        // the function comments for more details.
        let matrix_as_field_elem = matrix.convert_to_scalar_field();
        let matrix_as_array2 = Array2::from_shape_vec(
            (
                matrix_as_field_elem.len() / (1 << log_n_cols),
                1 << log_n_cols,
            ),
            matrix_as_field_elem.clone(),
        )
        .unwrap();

        // we do a simple dot product by transposing the matrix and treating each row as a vector and taking the dot product
        // of that and the vector provided in the function. this results in the matrix vector product as a vector of
        // scalar field elements.
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
            assert_eq!(matrix.len(), 1, "we can only have a valid matrix-vector product when the vector is len 0 if the matrix is len 1");
            vec![matrix_as_field_elem[0]]
        };

        vec_matrix_prod
    }

    /// the function where we construct the hyrax evaluation proof that the verifier then uses to determine whether the
    /// prover truly knows the correct evaluation of an MLE at a certain random challenge.
    #[allow(clippy::too_many_arguments)]
    pub fn prove(
        // the log size of the rows of the matrix
        log_n_cols: usize,
        // a vector representing the coefficients of the MLE. can be u8, i8, or scalar field elements
        data: &MleCoefficientsVector<C>,
        // the challenge coordinates we are evaluating the MLE at
        challenge_coordinates: &[C::Scalar],
        // what the prover claims the MLE evaluates to at the challenge point
        mle_evaluation_at_challenge: &C::Scalar,
        // the pedersen vector and scalar commitment generators needed in order to commit to the rows of the matrix
        // and for the proof of dot product
        committer: &PedersenCommitter<C>,
        // the blinding factor to commit to the evaluation of the MLE at the random challenge
        blinding_factor_evaluation: C::Scalar,
        // the random generator that the prover needs for proof of dot product
        prover_random_generator: &mut impl Rng,
        // transcript that the prover needs for proof of dot product
        prover_transcript: &mut impl ECProverTranscript<C>,
        // the blinding factors to commit to each of the rows of the MLE coefficient matrix
        blinding_factors_matrix: &[C::Scalar],
    ) -> Self {
        let (l_vector, r_vector) =
            HyraxPCSProof::<C>::compute_l_r_from_log_n_cols(log_n_cols, challenge_coordinates);

        // since the prover knows the T matrix (matrix of MLE coefficients), the prover can simply do a vector matrix product
        // to compute T' = L \times T
        let t_prime = HyraxPCSProof::<C>::vector_matrix_product(&l_vector, data, log_n_cols);

        // FIXME we could make this nicer by using the CommittedVector for each row
        // the blinding factor for the commitment to the T' vector is a combination of the blinding factors of the commitments
        // to the rows of the T matrix. it is a dot product of the L vector and the blinding factors of the matrix rows commits.
        let blinding_factor_t_prime = if !l_vector.is_empty() {
            blinding_factors_matrix.iter().zip(l_vector.iter()).fold(
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

        // commit to t_prime_vector and add to the transcript
        let t_prime_commit = committer.committed_vector(&t_prime, &blinding_factor_t_prime);
        prover_transcript.append_ec_point("commitment to x", t_prime_commit.commitment);

        // commit to the dot product, add this to the transcript
        let mle_eval_commit =
            committer.committed_scalar(mle_evaluation_at_challenge, &blinding_factor_evaluation);
        prover_transcript.append_ec_point("commitment to y", mle_eval_commit.commitment);

        // now that we have a commitment to T', we can do a proof of dot product which claims that the dot product of
        // T' and R is indeed the MLE evaluation point. this evaluation proof is what the hyrax verifier uses to verify
        // this statement.
        let podp_evaluation_proof = ProofOfDotProduct::<C>::prove(
            &t_prime_commit,
            &mle_eval_commit,
            &r_vector,
            committer,
            prover_random_generator,
            prover_transcript,
        );

        let aux = HyraxAuxInfo {
            log_n_cols,
            committer: committer.clone(),
        };

        // return the hyrax evaluation proof which comprises of the PoDP evaluation proof along with the commitments
        // to the rows of the matrix and the evaluation of the matrix at the random challenge.
        Self {
            podp_evaluation_proof,
            commitment_to_evaluation: mle_eval_commit,
            aux,
            blinding_factor_evaluation,
        }
    }

    pub fn verify_hyrax_evaluation_proof(
        &self,
        log_n_cols: usize,
        committer: &PedersenCommitter<C>,
        commitment_to_coeff_matrix: &[C],
        challenge_coordinates: &[C::Scalar],
        verifier_transcript: &mut impl ECVerifierTranscript<C>,
    ) {
        let Self {
            podp_evaluation_proof,
            commitment_to_evaluation,
            aux: _,
            blinding_factor_evaluation: _,
        } = &self;
        let (l_vector, r_vector) =
            HyraxPCSProof::<C>::compute_l_r_from_log_n_cols(log_n_cols, challenge_coordinates);

        // the verifier uses the L vector and does a scalar multiplication to each of the matrix row commits with the
        // appropriate index of the L vector. it then adds them all together to get the commitment to T' = L \times T.
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

        // add PoDP commitments to the transcript
        let transcript_t_prime_commit = verifier_transcript
            .consume_ec_point("commitment to x")
            .unwrap();
        assert_eq!(t_prime_commit_from_t_commit, transcript_t_prime_commit);

        let transcript_commitment_to_evaluation = verifier_transcript
            .consume_ec_point("commitment to y")
            .unwrap();
        assert_eq!(
            commitment_to_evaluation.commitment,
            transcript_commitment_to_evaluation
        );

        // using this commitment, the verifier can then do a proof of dot product verification given the evaluation proof
        // and the prover's claimed commitment to the evaluation of the MLE.
        podp_evaluation_proof.verify(
            &t_prime_commit_from_t_commit,
            &commitment_to_evaluation.commitment,
            &r_vector,
            committer,
            verifier_transcript,
        )
    }
}
