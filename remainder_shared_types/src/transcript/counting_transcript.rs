//! An implementation of a `TranscriptSponge` that increments an internal counter for each squeeze.
//! For testing purposes only!

use std::marker::PhantomData;

use super::TranscriptSponge;
use crate::Field;
use itertools::Itertools;

/// An implementation of a transcript sponge that increments an internal counter for each squeeze.
#[derive(Clone, Default)]
pub struct CountingSponge<F: Field> {
    counter: usize,
    _marker: PhantomData<F>,
}

impl<F: Field> TranscriptSponge<F> for CountingSponge<F> {
    fn absorb(&mut self, _elem: F) {}

    fn absorb_elements(&mut self, _elements: &[F]) {}

    fn squeeze(&mut self) -> F {
        let res = self.counter;
        self.counter += 1;
        F::from(res as u64)
    }

    fn squeeze_elements(&mut self, num_elements: usize) -> Vec<F> {
        (0..num_elements).map(|_| self.squeeze()).collect_vec()
    }
}
