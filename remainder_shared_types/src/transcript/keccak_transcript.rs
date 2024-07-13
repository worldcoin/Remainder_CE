use itertools::Itertools;
use sha3::{Digest, Keccak512};

use crate::FieldExt;

use super::TranscriptSponge;

#[derive(Clone, Debug, Default)]
pub struct KeccakTranscript {
    hasher: Keccak512,
}

impl<F> TranscriptSponge<F> for KeccakTranscript
where
    F: FieldExt<Repr = [u8; 32]>,
{
    fn absorb(&mut self, elem: F) {
        self.hasher.update(elem.to_repr())
    }

    fn absorb_elements(&mut self, elements: &[F]) {
        let bytes = elements.iter().flat_map(F::to_repr).collect_vec();
        self.hasher.update(&bytes)
    }

    fn squeeze(&mut self) -> F {
        let out = self.hasher.finalize_reset();
        self.hasher.update(out);

        F::from_uniform_bytes(out.as_slice().try_into().unwrap())
    }

    fn squeeze_elements(&mut self, num_elements: usize) -> Vec<F> {
        (0..num_elements)
            .map(|_| {
                let out = self.hasher.finalize_reset();
                self.hasher.update(out);

                F::from_uniform_bytes(out.as_slice().try_into().unwrap())
            })
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    use halo2curves::bn256::{Fq, Fr};

    use crate::transcript::{
        ec_transcript::{ECProverTranscript, ECTranscriptWriter},
        ProverTranscript,
    };

    use super::KeccakTranscript;

    #[test]
    fn test_keccak() {
        let mut transcript =
            ECTranscriptWriter::<halo2curves::bn256::G1Affine, KeccakTranscript>::new(
                "new transcript",
            );
        transcript.append("test", Fr::from(3));
        transcript.append("test2", Fq::one());
        let one = halo2curves::bn256::G1Affine::generator();
        transcript.append_ec_point("ec_test", one);
        let _: Fr = transcript.get_challenge("test_challenge");
        let _: Fq = transcript.get_challenge("test_challenge_2");
    }
}
