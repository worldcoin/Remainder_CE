use rand::{CryptoRng, RngCore};
use shared_types::curves::PrimeOrderCurve;
use shared_types::transcript::ec_transcript::ECTranscriptTrait;
use shared_types::{ff_field, Zeroizable};

use serde::{Deserialize, Serialize};
use shared_types::pedersen::{CommittedScalar, PedersenCommitter};

#[cfg(test)]
/// Tests for proof of equality.
mod tests;

/// Proof of equality shows that two messages are equal, via their commitments.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct ProofOfEquality<C: PrimeOrderCurve> {
    pub alpha: C,
    pub z: C::Scalar,
}

impl<C: PrimeOrderCurve> ProofOfEquality<C> {
    /// As a part of weak FS, the caller is responsible for adding commitments
    /// to the transcript, if that is required.
    pub fn prove(
        commit0: &CommittedScalar<C>,
        commit1: &CommittedScalar<C>,
        committer: &PedersenCommitter<C>,
        rng: &mut (impl CryptoRng + RngCore),
        transcript: &mut impl ECTranscriptTrait<C>,
    ) -> Self {
        // From its random tape, P samples $r$ from the scalar field.
        let mut r = C::Scalar::random(rng);

        // Compute $\alpha = h^r$
        let alpha = committer.blinding_generator * r;

        // EC group element $\alpha$ is added to the transcript.
        transcript.append_ec_point("Commitment to random value", alpha);

        // A scalar field element $c$ is sampled from the transcript.
        let c: <C as PrimeOrderCurve>::Scalar = transcript.get_scalar_field_challenge("PoE c");

        // Compute $z = c \cdot(s_1 - s_2) + r$.
        let z = c * (commit0.blinding - commit1.blinding) + r;
        transcript.append_scalar_field_elem("Blinded response", z);

        // Zeroize all randomly generated values.
        r.zeroize();

        Self { alpha, z }
    }

    /// As a part of weak FS, the caller is responsible for adding commitments
    /// to the transcript, if that is required.
    pub fn verify(
        &self,
        commit0: C,
        commit1: C,
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) {
        // EC group element $\alpha$ is added to the transcript.
        transcript.append_ec_point("Commitment to random value", self.alpha);

        // A scalar field element $c$ is sampled from the transcript.
        let c = transcript.get_scalar_field_challenge("PoE c");

        transcript.append_scalar_field_elem("Blinded response", self.z);

        // Check: $h^z \overset{?}{=} (c_0 \cdot (c_1)^{-1})^c\cdot \alpha$
        let h = committer.blinding_generator;
        let lhs = h * self.z;
        let rhs = (commit0 - commit1) * c + self.alpha;
        assert_eq!(lhs, rhs);
    }
}
