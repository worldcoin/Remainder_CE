//! An implementation of a `TranscriptSponge` that uses the Poseidon hash
//! function; Useful for recursive proving.

use super::TranscriptSponge;
use crate::Field;
use itertools::Itertools;
use poseidon::Poseidon;

/// A Poseidon implementation of a transcript sponge.
#[derive(Clone, Debug)]
pub struct PoseidonSponge<F: Field> {
    /// The specific poseidon sponge configuration.
    sponge: Poseidon<F, 3, 2>,
}

impl<F: Field> Default for PoseidonSponge<F> {
    fn default() -> Self {
        Self {
            sponge: Poseidon::new(8, 57),
        }
    }
}

impl<F: Field> TranscriptSponge<F> for PoseidonSponge<F> {
    fn absorb(&mut self, elem: F) {
        self.sponge.update(&[elem]);
    }

    fn absorb_elements(&mut self, elements: &[F]) {
        self.sponge.update(elements);
    }

    fn squeeze(&mut self) -> F {
        self.sponge.squeeze()
    }

    fn squeeze_elements(&mut self, num_elements: usize) -> Vec<F> {
        (0..num_elements)
            .map(|_| self.sponge.squeeze())
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {

    use halo2curves::bn256::Fq as Base;
    use halo2curves::bn256::G1 as Bn256Point;

    use crate::transcript::ec_transcript::ECTranscript;
    use crate::transcript::ec_transcript::ECTranscriptTrait;
    use crate::transcript::ProverTranscript;

    use super::PoseidonSponge;

    #[test]
    fn test_poseidon() {
        let mut transcript =
            ECTranscript::<Bn256Point, PoseidonSponge<Base>>::new("new transcript");
        transcript.append("test2", Base::one());
        let one = halo2curves::bn256::G1::generator();
        transcript.append_ec_point("ec_test", one);
        // let _: Fr = transcript.get_challenge("test_challenge");
        let _: Base = transcript.get_challenge("test_challenge_2");
    }
}
