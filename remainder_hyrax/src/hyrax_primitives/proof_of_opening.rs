use rand::{CryptoRng, RngCore};
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::ff_field;
use remainder_shared_types::pedersen::{CommittedScalar, PedersenCommitter};
use remainder_shared_types::transcript::ec_transcript::ECTranscriptTrait;
use remainder_shared_types::Zeroizable;
use serde::{Deserialize, Serialize};

#[cfg(test)]
/// Tests for proof of opening.
mod tests;

/// "Proof of Opening", i.e. a proof that the prover knows an opening of a
/// scalar commitment. See Appendix A of the Hyrax paper.
///
/// As a part of weak FS, the calling context is required to add any
/// required elements into the transcript, if that is necessary.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct ProofOfOpening<C: PrimeOrderCurve> {
    pub z1: C::Scalar,
    pub z2: C::Scalar,
    pub alpha: C,
}

impl<C: PrimeOrderCurve> ProofOfOpening<C> {
    /// Generate a proof that the prover knows the opening of a scalar commitment. The calling
    /// context is responsible for adding the commitment to the transcript (if required).
    pub fn prove(
        x: &CommittedScalar<C>,
        committer: &PedersenCommitter<C>,
        mut rng: &mut (impl CryptoRng + RngCore),
        transcript: &mut impl ECTranscriptTrait<C>,
    ) -> Self {
        // Sample $t_1, t_2$ from the random tape.
        let mut t_1 = C::Scalar::random(&mut rng);
        let mut t_2 = C::Scalar::random(&mut rng);

        // Compute $\alpha = g^{t_1} \cdot h^{t_2}$.
        let alpha = committer.scalar_commit(&t_1, &t_2);

        // alpha is added to the transcript.
        transcript.append_ec_point("Commitment to random values", alpha);

        // A scalar field element $c$ is sampled from the transcript.
        let c = transcript.get_scalar_field_challenge("PoO c");

        // Compute $z_1 = x\cdot c + t_1$ and $z_2 = r \cdot c + t_2$.
        let z1 = x.value * c + t_1;
        let z2 = x.blinding * c + t_2;

        // Zeroize randomly generated values.
        t_1.zeroize();
        t_2.zeroize();

        transcript.append_scalar_field_elem("Blinded response 1", z1);
        transcript.append_scalar_field_elem("Blinded response 2", z2);

        Self { z1, z2, alpha }
    }

    /// Verify that the prover knows an opening of the commitment `x`.
    ///
    /// As a part of weak FS, the calling context is responsible for adding the
    /// commitment to the transcript (if required).
    pub fn verify(
        &self,
        x: C,
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) {
        // EC group element $\alpha$ is added to the transcript
        transcript.append_ec_point("Commitment to random values", self.alpha);

        // A scalar field element $c$ is sampled from the transcript.
        let c = transcript.get_scalar_field_challenge("PoO c");

        transcript.append_scalar_field_elem("Blinded response 1", self.z1);
        transcript.append_scalar_field_elem("Blinded response 2", self.z2);

        // Check: $g^{z_1} \cdot h^{z_2} \overset{?}{=} C_0^c \cdot \alpha$.
        assert_eq!(
            committer.scalar_commit(&self.z1, &self.z2),
            x * c + self.alpha
        );
    }
}
