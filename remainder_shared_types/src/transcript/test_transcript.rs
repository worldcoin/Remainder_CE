//! An implementation of a `TranscriptSponge` that always returns 1.
//! For testing purposes only!

use std::marker::PhantomData;

use super::TranscriptSponge;
use crate::FieldExt;
use itertools::Itertools;

/// An implementation of a transcript sponge that always returns 1.
#[derive(Clone)]
pub struct TestSponge<F: FieldExt> {
    _marker: PhantomData<F>,
}

impl<F: FieldExt> TranscriptSponge<F> for TestSponge<F> {
    fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    fn absorb(&mut self, _elem: F) {}

    fn absorb_elements(&mut self, _elements: &[F]) {}

    fn squeeze(&mut self) -> F {
        F::ONE + F::ONE
    }

    fn squeeze_elements(&mut self, num_elements: usize) -> Vec<F> {
        (0..num_elements).map(|_| F::ONE).collect_vec()
    }
}
