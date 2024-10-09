use itertools::Itertools;
use rand::Rng;
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::ff_field;
use remainder_shared_types::pedersen::{CommittedScalar, CommittedVector, PedersenCommitter};
use remainder_shared_types::transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript};
use serde::{Deserialize, Serialize};

#[cfg(test)]
/// Tests for the hyrax primitives.
mod tests;

/// Struct that holds all the information necessary for the evaluation proof of a proof of dot product.
/// The verifier can then use the information in this struct in order to determine whether the prover
/// knows the dot product of two vectors, x and a.
#[derive(Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct ProofOfDotProduct<C: PrimeOrderCurve> {
    /// the commitment to the d vector that the prover randomly samples
    commit_d: C,
    /// commitment to the dot product of the d vector and the a vector, which is public.
    commit_d_dot_a: C,
    /// the z vector that the prover computes based on its knowledge of the hidden vector, x.
    z_vector: Vec<C::Scalar>,
    /// the blinding challenge that goes with the commitment of the z vector
    z_delta: C::Scalar,
    /// the blinding challenge that goes with the commitment of the dot product of the z vector and the a vector
    z_beta: C::Scalar,
}

impl<C: PrimeOrderCurve> ProofOfDotProduct<C> {
    /// Generate a proof that the dot product of `x_vector` and `a_vector` is `dot_prod_val_y`. The
    /// Calling context is responsible for adding commitments to x_vector and dot_prod_val_y to the
    /// transcript (if this is necessary).
    pub fn prove(
        // the hidden x vector that only the prover knows
        x: &CommittedVector<C>,
        // the dot product value of x (hidden) and a (public) that only the prover knows
        y: &CommittedScalar<C>,
        // public vector that is responsible for the dot product
        a: &[C::Scalar],
        // pedersen generators necessary for the prover to make commitments
        committer: &PedersenCommitter<C>,
        // random generator that the prover uses to sample the d vector and consisting of blinding factors
        mut rng: &mut impl Rng,
        // transcript for Fiat-Shamir in order to generate the challenge c that should be "sent" by the verifier
        // in the interactive version
        transcript: &mut impl ECProverTranscript<C>,
    ) -> Self {
        // the prover randomly samples the d vector
        let d_vec: Vec<C::Scalar> = (0..x.value.len())
            .map(|_idx| C::Scalar::random(&mut rng))
            .collect_vec();
        // blinding factor to commit to d
        let r_delta = C::Scalar::random(&mut rng);
        // blinding factor to commit to the dot product of d and a
        let r_beta = C::Scalar::random(&mut rng);

        let d_dot_a: C::Scalar = ProofOfDotProduct::<C>::compute_dot_product(&d_vec, a);
        // commitments to d, and <d, a> inner product. add these commitments to the transcript.
        let commit_d = committer.vector_commit(&d_vec, &r_delta);
        let commit_d_dot_a = committer.scalar_commit(&d_dot_a, &r_beta);

        transcript.append_ec_point("commitment to d", commit_d);
        transcript.append_ec_point("commitment to d dot a", commit_d_dot_a);

        let c: <C as PrimeOrderCurve>::Scalar =
            transcript.get_scalar_field_challenge("challenge c");

        // the z vector that is cx + d. only the prover can compute this because it knows x.
        let z_vector = x
            .value
            .iter()
            .zip(d_vec.iter())
            .map(|(x_elem, d_elem)| c * x_elem + d_elem)
            .collect_vec();
        transcript.append_scalar_points("PoDP z_vector", &z_vector);
        // the blinding factor to commit to z
        let z_delta = c * x.blinding + r_delta;
        transcript.append_scalar_point("PoDP z_delta", z_delta);
        // the blinding factor to commit to <z, a>
        let z_beta = c * y.blinding + r_beta;
        transcript.append_scalar_point("PoDP z_beta", z_beta);

        // we send over this information to the verifier in order to verify that the prover indeed knows the
        // dot product of x and a.
        Self {
            commit_d,
            commit_d_dot_a,
            z_vector,
            z_delta,
            z_beta,
        }
    }

    /// the function that uses the evaluation proof (that is stored in &self) and the inputs to the "circuit" which
    /// are com_x (commitment to x), com_y (commitment to <x, a>), and a_vector (which is the public vector a).
    /// Panics iff the proof is invalid.
    /// Calling context is responsible for adding commitments to x_vector and dot_prod_val_y to the
    /// transcript (if this is necessary).
    pub fn verify(
        // evaluation proof
        &self,
        // commitment to x (blinded)
        com_x: &C,
        // commitment to <x, a> (blinded)
        com_y: &C,
        // public vector
        a: &[C::Scalar],
        // generators needed in order to produce commitments
        committer: &PedersenCommitter<C>,
        // transcript in order to generate the challenge c that would normally be produced interactively.
        transcript: &mut impl ECVerifierTranscript<C>,
    ) {
        let Self {
            commit_d,
            commit_d_dot_a,
            z_vector,
            z_delta,
            z_beta,
        } = &self;

        // --- Read transcript-generated prover messages and compare against proof-supplied messages ---
        // Messages for d and <d, a>
        let transcript_commit_d = transcript.consume_ec_point("commitment to d").unwrap();
        let transcript_commit_d_dot_a = transcript
            .consume_ec_point("commitment to d dot a")
            .unwrap();
        assert_eq!(*commit_d, transcript_commit_d);
        assert_eq!(*commit_d_dot_a, transcript_commit_d_dot_a);

        // now the verifier can sample the same random challenge. once again, this is in the base field so we
        // truncate in order to convert it into the scalar field.
        let c = transcript
            .get_scalar_field_challenge("challenge c")
            .unwrap();

        assert_eq!(
            &transcript
                .consume_scalar_points("PoDP z_vector", z_vector.len())
                .unwrap(),
            z_vector
        );
        assert_eq!(
            &transcript.consume_scalar_point("PoDP z_delta").unwrap(),
            z_delta
        );
        assert_eq!(
            &transcript.consume_scalar_point("PoDP z_beta").unwrap(),
            z_beta
        );

        // we compute <z, a> and then commitments to z and <z, a> based off of the blinding factors and values in
        // the evaluation proof.
        let z_dot_a = ProofOfDotProduct::<C>::compute_dot_product(z_vector, a);
        let com_z = committer.vector_commit(z_vector, z_delta);
        let com_z_dot_a = committer.scalar_commit(&z_dot_a, z_beta);

        // these are the two checks necessary in order to verify that the prover indeed knows the dot product
        // of x and a
        // TODO(vishady): make this an actual check and throw an error if it fails
        assert_eq!(*com_x * c + *commit_d, com_z);
        assert_eq!(*com_y * c + *commit_d_dot_a, com_z_dot_a);
    }

    /// given two vectors of elements in the scalar field, compute their dot product and return a single scalar
    /// field element.
    fn compute_dot_product(first_vector: &[C::Scalar], second_vector: &[C::Scalar]) -> C::Scalar {
        if first_vector.len() == 1 && second_vector.is_empty() {
            first_vector[0]
        } else if second_vector.len() == 1 && first_vector.is_empty() {
            return second_vector[0];
        } else {
            return first_vector
                .iter()
                .zip(second_vector.iter())
                .fold(C::Scalar::ZERO, |acc, (first, second)| {
                    acc + (*first * second)
                });
        }
    }
}
