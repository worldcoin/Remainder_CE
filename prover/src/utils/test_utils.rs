//! Utilities that are only useful for tests

use itertools::Itertools;
use shared_types::{transcript::TranscriptSponge, Field};
use std::fmt::Debug;

/// A dummy transcript that can have arbitrary values fed into it.
///
/// Useful for writing tests where you want the transcript to return
/// small readable and consistent values.
///
/// Will return the values in VALUES in order upon each squeeze request.
/// Once SIZE is reached values will wrap around.
#[derive(Clone, Default, Debug)]
pub struct DummySponge<F: Field + Debug, const VALUE: i32> {
    /// The current position in the values list.
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field + Debug, const VALUE: i32> TranscriptSponge<F> for DummySponge<F, VALUE> {
    fn absorb(&mut self, _: F) {}

    fn absorb_elements(&mut self, _: &[F]) {}

    fn squeeze(&mut self) -> F {
        let is_negative = VALUE.is_negative();
        let abs = VALUE.abs();
        let value = F::from(abs as u64);

        if is_negative {
            value.neg()
        } else {
            value
        }
    }

    fn squeeze_elements(&mut self, num_elements: usize) -> Vec<F> {
        (0..num_elements).map(|_| self.squeeze()).collect_vec()
    }

    fn absorb_initialization_label(&mut self, _label: &str) {}
}
