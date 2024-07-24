use rand::Rng;
use remainder_shared_types::halo2curves::ff::Field;
use remainder_shared_types::{
    curves::PrimeOrderCurve,
    transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript},
};

use crate::pedersen::{CommittedScalar, PedersenCommitter};

pub mod tests;

/// TODO(vishady) riad audit comments: maybe instantiate proof structs with a transcript
/// maybe with a type system that is proof + transcript (and differentiate between weak and strong FS)
/// or at least add a comment that says what is expected to be in the transcript :D

/// Proof of equality shows that two messages are equal, via their commitments.
pub struct ProofOfEquality<C: PrimeOrderCurve> {
    pub alpha: C,
    pub z: C::Scalar,
}

impl<C: PrimeOrderCurve> ProofOfEquality<C> {
    /// The caller is responsible for adding commitments to the transcript, if that is required.
    pub fn prove(
        commit0: &CommittedScalar<C>,
        commit1: &CommittedScalar<C>,
        committer: &PedersenCommitter<C>,
        // TODO(vishady) riad audit comments: probably try to mark this as a "cryptographic rng" CryptoRng?,
        // try to allow different trait bound in testing
        mut rng: &mut impl Rng,
        transcript: &mut impl ECProverTranscript<C>,
    ) -> Self {
        // From its random tape, P samples $r$ from the scalar field.
        let r = C::Scalar::random(&mut rng);

        // Compute $\alpha = h^r$
        let alpha = committer.blinding_generator * r;

        // EC group element $\alpha$ is added to the transcript.
        transcript.append_ec_point("PoE alpha", alpha);

        // A scalar field element $c$ is sampled from the transcript.
        let c: <C as PrimeOrderCurve>::Scalar = transcript.get_scalar_field_challenge("PoE c");

        // Compute $z = c \cdot(s_1 - s_2) + r$.
        let z = c * (commit0.blinding - commit1.blinding) + r;

        Self { alpha, z }
    }

    /// The caller is responsible for adding commitments to the transcript, if that is required.
    pub fn verify(
        &self,
        commit0: C,
        commit1: C,
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECVerifierTranscript<C>,
    ) {
        // EC group element $\alpha$ is added to the transcript.
        let transcript_alpha = transcript.consume_ec_point("PoE alpha").unwrap();
        assert_eq!(self.alpha, transcript_alpha);

        // A scalar field element $c$ is sampled from the transcript.
        let c = transcript.get_scalar_field_challenge("PoE c").unwrap();

        // Check: $h^z \overset{?}{=} (c_0 \cdot (c_1)^{-1})^c\cdot \alpha$
        let h = committer.blinding_generator;
        let lhs = h * self.z;
        let rhs = (commit0 - commit1) * c + self.alpha;
        assert_eq!(lhs, rhs);
    }
}
