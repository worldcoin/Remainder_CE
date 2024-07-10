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
