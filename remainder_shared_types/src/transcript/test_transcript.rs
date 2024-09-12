//! An implementation of a `TranscriptSponge` that always returns 1.
//! For testing purposes only!

use std::marker::PhantomData;

use super::TranscriptSponge;
use crate::Field;
use itertools::Itertools;

/// An implementation of a transcript sponge that always returns 1.
#[derive(Clone, Default)]
pub struct TestSponge<F: Field> {
    counter: usize,
    _marker: PhantomData<F>,
}

impl<F: Field> TranscriptSponge<F> for TestSponge<F> {
    fn absorb(&mut self, _elem: F) {}

    fn absorb_elements(&mut self, _elements: &[F]) {}

    fn squeeze(&mut self) -> F {
        let res = self.counter;
        self.counter += 1;
        F::from(res as u64)
    }

    fn squeeze_elements(&mut self, num_elements: usize) -> Vec<F> {
        (0..num_elements).map(|_| F::ONE + F::ONE).collect_vec()
    }
}
