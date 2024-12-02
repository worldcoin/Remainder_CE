//! An implementation of a `TranscriptSponge` that always returns 1.
//! For testing purposes only!

use std::marker::PhantomData;

use super::TranscriptSponge;
use crate::Field;
use itertools::Itertools;
use std::fmt::Debug;

/// An implementation of a transcript sponge that always returns 1.
#[derive(Clone, Default, Debug)]
pub struct TestSponge<F: Field + Debug> {
    _marker: PhantomData<F>,
}

impl<F: Field + Debug> TranscriptSponge<F> for TestSponge<F> {
    fn absorb(&mut self, _elem: F) {}

    fn absorb_elements(&mut self, _elements: &[F]) {}

    fn squeeze(&mut self) -> F {
        F::ONE
    }

    fn squeeze_elements(&mut self, num_elements: usize) -> Vec<F> {
        (0..num_elements).map(|_| self.squeeze()).collect_vec()
    }

    fn absorb_initialization_label(&mut self, _label: &str) {}
}
