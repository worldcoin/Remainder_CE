use rand::Rng;
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::ff_field;
use remainder_shared_types::transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript};

use crate::pedersen::{CommittedScalar, PedersenCommitter};

#[cfg(test)]
/// Tests for proof of product.
mod tests;

/// "Proof of product" i.e. a proof that if X, Y and Z are commitments, committing to x, y and z,
/// then x*y = z.  See Appendix A of the Hyrax paper.
pub struct ProofOfProduct<C: PrimeOrderCurve> {
    pub alpha: C,
    pub beta: C,
    pub delta: C,
    pub z1: C::Scalar,
    pub z2: C::Scalar,
    pub z3: C::Scalar,
    pub z4: C::Scalar,
    pub z5: C::Scalar,
}

impl<C: PrimeOrderCurve> ProofOfProduct<C> {
    /// Generate a proof that the x.value * y.value = z.value.
    /// The calling context is responsible for adding the commitments to the transcript (if
    /// required).
    pub fn prove(
        x: &CommittedScalar<C>,
        y: &CommittedScalar<C>,
        z: &CommittedScalar<C>,
        committer: &PedersenCommitter<C>,
        mut rng: &mut impl Rng,
        transcript: &mut impl ECProverTranscript<C>,
    ) -> Self {
        // Sample $b_1, .. b_5$ from the random tape.
        let b_1 = C::Scalar::random(&mut rng);
        let b_2 = C::Scalar::random(&mut rng);
        let b_3 = C::Scalar::random(&mut rng); // FIXME verifies if this is zero, otherwise not!
        let b_4 = C::Scalar::random(&mut rng);
        let b_5 = C::Scalar::random(&mut rng);

        // Compute $\alpha = g^{b_1}\cdot h^{b_2}, \beta = g^{b_3} \cdot h^{b_4}, \delta = X^{b_3} \cdot h^{b_5}$.
        let alpha = committer.scalar_commit(&b_1, &b_2);
        let beta = committer.scalar_commit(&b_3, &b_4);
        let delta = committer.scalar_commit(&(x.value * b_3), &(x.blinding * b_3 + b_5));

        // $\alpha, \beta, \delta$ are added to the transcript.
        transcript.append_ec_point("PoP alpha", alpha);
        transcript.append_ec_point("PoP beta", beta);
        transcript.append_ec_point("PoP delta", delta);

        // Scalar field element $c$ is sampled from the transcript.
        let c: <C as PrimeOrderCurve>::Scalar = transcript.get_scalar_field_challenge("PoP c");

        // Compute $z_1 = b_1 + c \cdot x,\quad z_2 = b_2 + c \cdot r_X, \quad z_3 = b_3 + c \cdot y,\quad z_4 = b_4 + c \cdot r_Y, \\ z_5 = b_5 + c \cdot (r_Z - r_Xy)
        let z1 = b_1 + c * x.value;
        let z2 = b_2 + c * x.blinding;
        let z3 = b_3 + c * y.value;
        let z4 = b_4 + c * y.blinding;
        let z5 = b_5 + c * (z.blinding - x.blinding * y.value);

        transcript.append_scalar_point("PoP z1", z1);
        transcript.append_scalar_point("PoP z2", z2);
        transcript.append_scalar_point("PoP z3", z3);
        transcript.append_scalar_point("PoP z4", z4);
        transcript.append_scalar_point("PoP z5", z5);

        Self {
            alpha,
            beta,
            delta,
            z1,
            z2,
            z3,
            z4,
            z5,
        }
    }

    /// The calling context is responsible for adding the commitments to the transcript (if
    /// required).
    pub fn verify(
        &self,
        commit_x: C,
        commit_y: C,
        commit_z: C,
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECVerifierTranscript<C>,
    ) {
        // $\alpha, \beta, \delta$ are added to transcript
        let transcript_alpha = transcript.consume_ec_point("PoP alpha").unwrap();
        let transcript_beta = transcript.consume_ec_point("PoP beta").unwrap();
        let transcript_delta = transcript.consume_ec_point("PoP delta").unwrap();

        assert_eq!(self.alpha, transcript_alpha);
        assert_eq!(self.beta, transcript_beta);
        assert_eq!(self.delta, transcript_delta);

        // A scalar field element $c$ is sampled from the transcript.
        let c = transcript.get_scalar_field_challenge("PoP c").unwrap();

        let z1 = transcript.consume_scalar_point("PoO z1").unwrap();
        assert_eq!(z1, self.z1);
        let z2 = transcript.consume_scalar_point("PoO z2").unwrap();
        assert_eq!(z2, self.z2);
        let z3 = transcript.consume_scalar_point("PoO z3").unwrap();
        assert_eq!(z3, self.z3);
        let z4 = transcript.consume_scalar_point("PoO z4").unwrap();
        assert_eq!(z4, self.z4);
        let z5 = transcript.consume_scalar_point("PoO z5").unwrap();
        assert_eq!(z5, self.z5);

        // Check the following:
        // \alpha \cdot X^c \overset{?}{=} g^{z_1}\cdot h^{z_2} \\
        // \beta \cdot Y^c \overset{?}{=} g^{z_3}\cdot h^{z_4} \\
        // \delta \cdot Z^c \overset{?}{=} X^{z_3}\cdot h^{z_5}
        let g = committer.scalar_commit_generator();
        let h = committer.blinding_generator;
        assert_eq!(self.alpha + commit_x * c, g * self.z1 + h * self.z2);
        assert_eq!(self.beta + commit_y * c, g * self.z3 + h * self.z4);
        assert_eq!(self.delta + commit_z * c, commit_x * self.z3 + h * self.z5);
    }
}
