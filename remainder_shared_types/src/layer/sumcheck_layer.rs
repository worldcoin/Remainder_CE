use crate::FieldExt;

use super::Layer;

/// A trait for defining an interface for Layers that implement the Sumcheck protocol
pub trait SumcheckLayer<F: FieldExt>: Layer<F> {
    /// Initialize the sumcheck round by setting the beta table, computing the number of rounds, etc.
    fn initialize_sumcheck(&mut self, claim_point: &[F]) -> Result<(), Self::Error>;

    /// Return the evaluations of the univariate polynomial generated during this round of sumcheck.
    /// Then, mutate the underlying bookkeeping tables to "bind" the given `challenge` to the bit
    /// labeled with that `round_index`.
    ///
    /// This must be called with a steadily incrementing round_index & with a securely generated challenge
    fn prove_sumcheck_round(
        &mut self,
        round_index: usize,
        challenge: F,
    ) -> Result<Vec<F>, Self::Error>;

    /// How many sumcheck rounds this layer will take to prove
    fn num_sumcheck_rounds(&self) -> usize;
}
