use crate::{transcript::ProverTranscript, FieldExt};

use super::Layer;

/// A trait for defining an interface for Layers that implement the Sumcheck protocol
pub trait SumcheckLayer<F: FieldExt>: Layer<F> {
    /// Initialize the sumcheck round by setting the beta table, computing the number of rounds, etc.
    fn initialize_sumcheck(
        &mut self,
        transcript: impl ProverTranscript<F>,
        claim: &[F],
    ) -> Result<(), Self::Error>;

    /// Prove a particular round in the sumcheck protocol
    ///
    /// This must be called with a steadily incrementing round_index & with a securely generated challenge
    fn prove_sumcheck_round(
        &mut self,
        transcript: impl ProverTranscript<F>,
        round_index: usize,
        challenge: F,
    ) -> Result<(), Self::Error>;

    /// Prove the final sumcheck round and do any cleanup required
    fn finish_sumcheck(&mut self, transcript: impl ProverTranscript<F>) -> Result<(), Self::Error>;

    /// How many sumcheck rounds this layer will take to prove
    fn num_vars(&self) -> usize;
}
