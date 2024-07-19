//! An implementation of a `TranscriptSponge` that uses the Poseidon hash
//! function; Useful for recursive proving.

use super::TranscriptSponge;
use crate::FieldExt;
use itertools::Itertools;
use poseidon::Poseidon;

/// A Poseidon implementation of a transcript sponge.
#[derive(Clone)]
pub struct PoseidonSponge<F: FieldExt> {
    /// The specific poseidon sponge configuration.
    sponge: Poseidon<F, 3, 2>,
}

impl<F: FieldExt> Default for PoseidonSponge<F> {
    fn default() -> Self {
        Self {
            sponge: Poseidon::new(8, 57),
        }
    }
}

impl<F: FieldExt> TranscriptSponge<F> for PoseidonSponge<F> {
    fn absorb(&mut self, elem: F) {
        self.sponge.update(&[elem]);
    }

    fn absorb_elements(&mut self, elements: &[F]) {
        self.sponge.update(elements);
    }

    fn squeeze(&mut self) -> F {
        self.sponge.squeeze()
        // F::ONE + F::ONE
    }

    fn squeeze_elements(&mut self, num_elements: usize) -> Vec<F> {
        (0..num_elements)
            .map(|_| self.sponge.squeeze())
            .collect_vec()
        // (0..num_elements).map(|_| F::ONE).collect_vec()
    }
}

#[cfg(test)]
mod tests {

    use halo2curves::bn256::Fq as Base;
    use halo2curves::bn256::Fr as Scalar;
    use halo2curves::bn256::G1 as Bn256Point;

    use crate::{
        curves::PrimeOrderCurve,
        transcript::{
            ec_transcript::{ECProverTranscript, ECTranscriptWriter},
            ProverTranscript,
        },
    };

    use super::PoseidonSponge;

    #[test]
    fn test_poseidon() {
        let mut transcript =
            ECTranscriptWriter::<Bn256Point, PoseidonSponge<Base>>::new("new transcript");
        transcript.append("test2", Base::one());
        let one = halo2curves::bn256::G1::generator();
        transcript.append_ec_point("ec_test", one);
        // let _: Fr = transcript.get_challenge("test_challenge");
        let _: Base = transcript.get_challenge("test_challenge_2");
    }
}
