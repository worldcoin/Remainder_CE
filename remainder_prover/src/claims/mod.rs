//! Utilities involving the claims a layer makes.

/// The default claim aggregator that uses wlx evaluations.
pub mod wlx_eval;

use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter},
    FieldExt,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    expression::{generic_expr::Expression, verifier_expr::VerifierExpr},
    input_layer::InputLayer,
    layer::{combine_mle_refs::CombineMleRefError, Layer, LayerError, LayerId},
    prover::GKRError,
};

/// Errors to do with aggregating and collecting claims.
#[derive(Error, Debug, Clone)]
pub enum ClaimError {
    /// The Layer has not finished the sumcheck protocol.
    #[error("The Layer has not finished the sumcheck protocol")]
    SumCheckNotComplete,

    /// MLE indices must all be fixed.
    #[error("MLE indices must all be fixed")]
    ClaimMleIndexError,

    /// Layer ID not assigned.
    #[error("Layer ID not assigned")]
    LayerMleError,

    /// MLE within MleRef has multiple values within it.
    #[error("MLE within MleRef has multiple values within it")]
    MleRefMleError,

    /// Error aggregating claims.
    #[error("Error aggregating claims")]
    ClaimAggroError,

    /// Should be evaluating to a sum.
    #[error("Should be evaluating to a sum")]
    ExpressionEvalError,

    /// All claims in a group should agree on the number of variables.
    #[error("All claims in a group should agree on the number of variables")]
    NumVarsMismatch,

    /// All claims in a group should agree the destination layer field.
    #[error("All claims in a group should agree the destination layer field")]
    LayerIdMismatch,

    /// Error while combining mle refs in order to evaluate challenge point.
    #[error("Error while combining mle refs in order to evaluate challenge point")]
    MleRefCombineError(#[from] CombineMleRefError),
}

/// A claim contains a `point \in F^n` along with the `result \in F` that an
/// associated layer MLE is expected to evaluate to. In other words, if
/// `\tilde{V} : F^n -> F` is the MLE of a layer, then this claim asserts:
/// `\tilde{V}(point) == result`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(bound = "F: FieldExt")]
pub struct Claim<F: FieldExt> {
    /// The point in F^n where the layer MLE is to be evaluated on.
    point: Vec<F>,

    /// The expected result of evaluating this layer's MLE on `point`.
    result: F,
}

impl<F: FieldExt> Claim<F> {
    /// Constructs a new `Claim` from a given `point` and `result`.
    pub fn new(point: Vec<F>, result: F) -> Self {
        Self { point, result }
    }

    /// Returns the length of the `point` vector.
    pub fn get_num_vars(&self) -> usize {
        self.point.len()
    }

    /// Returns the point vector in F^n.
    pub fn get_point(&self) -> &Vec<F> {
        &self.point
    }

    /// Returns the expected result.
    pub fn get_result(&self) -> F {
        self.result
    }
}

/// A trait that defines a protocol for the tracking/aggregation of many claims.
/// TODO(Makis): We are currently using the same trait for both the prover and
/// the verifier.  I think it's more appropriate to separate it out into two
/// different traits.
pub trait ClaimAggregator<F: FieldExt> {
    /// The struct the claim aggregator takes in.
    /// Is typically composed of the `Claim` struct and additional information.
    type Claim: std::fmt::Debug + Clone + Serialize + for<'a> Deserialize<'a>;

    /// The `Layer` this claim aggregator takes claims from, and aggregates
    /// claims for.
    type Layer: Layer<F>;

    /// The `InputLayer` this claim aggregator aggregates claims for.
    type InputLayer: InputLayer<F>;

    /// Creates an empty [ClaimAggregator], ready to track claims.
    fn new() -> Self;

    /// Retrieves claims from `layer` to keep track and later aggregate.
    /// Passes the `transcript_writer` to the `layer` which in turn is
    /// responsible for adding any necessary information to the transcript as
    /// claims are generated.
    ///
    /// # WARNING
    /// The "prover" variants of these methods should never be used in
    /// conjunction with the "verifier" variants!
    /// TODO(Makis): Separate this out to a `ProverClaimAggregator` trait.
    fn prover_extract_claims(
        &mut self,
        layer: &impl ProverYieldClaim<F, Self::Claim>,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<(), LayerError>;

    /// Retrieves claims from `layer` (using the `transcript_reader` as needed)
    /// to track and later aggregate them.
    ///
    /// `expr` is the fully-bound expression corresponding to `layer`.  This is
    /// temporary. We should separate the concept of a `VerifierLayer` from that
    /// of a `CircuitLayer` to avoid the need to pass the expression here.
    ///
    /// # WARNING
    /// The "verifier" variants of these methods should never be used in
    /// conjunction with the "prover" variants!
    /// TODO(Makis): Separate this out to a `VerifierClaimAggregator` trait.
    fn verifier_extract_claims(
        &mut self,
        layer: &impl VerifierYieldClaim<F, Self::Claim>,
        expr: &Expression<F, VerifierExpr>,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), LayerError>;

    /// Returns the claims made to layer with ID `layer_id` (if any).
    /// The claims must have already been retrieved using
    /// `prover_retrieve_claims` or `verifier_retrieve_claims`.
    /// This method is safe to use by either the prover or the verifier.
    ///
    /// TODO(Makis): Do we need to expose this method to the public interface?
    /// It seems it is only used by the `aggregate_claim` variants internally.
    fn get_claims(&self, layer_id: LayerId) -> Option<&[Self::Claim]>;

    /// Aggregates all the claims made on `layer` and returns a single claim.
    ///
    /// Should be called only after all claims for `layer` have been retrieved
    /// using `prover_retrieve_claims`/`verifier_retrieve_claims`.
    ///
    /// Adds any communication to the F-S Transcript as needed.
    fn prover_aggregate_claims(
        &self,
        layer: &Self::Layer,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<Claim<F>, GKRError>;

    /// Similar to `prover_aggregate_claims` but for an input layer.
    ///
    /// Aggregates all the claims made on `input_layer` and returns a single
    /// claim.
    ///
    /// Should be called only after all claims for `input_layer` have been
    /// retrieved using `prover_retrieve_claims`/`verifier_retrieve_claims`.
    ///
    /// Adds any communication to the F-S Transcript as needed.
    fn prover_aggregate_claims_input(
        &self,
        input_layer: &Self::InputLayer,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<Claim<F>, GKRError>;

    /// The verifier's variant of `prover_aggregate_claims`.
    ///
    /// Aggregates all the claims made on the Layer with ID `layer_id` (could be
    /// either an intermediate or input layer) and returns a single claim.
    ///
    /// Should be called only after all claims for this layer have been
    /// retrieved using `verifier_retrieve_claims`.
    ///
    /// Uses the `transcript_reader` to retrieve any necessary information.
    ///
    /// Note: The reason that the verifier version only needs the ID of the
    /// layer and not the layer itself is that is retrieves the WLX evaluations
    /// from the transcript, instead of asking the layer struct to generate
    /// them.
    fn verifier_aggregate_claims(
        &self,
        layer_id: LayerId,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<Claim<F>, GKRError>;
}

/// A trait that allows a layer-like type to yield claims for other layers.
/// Typically, a [ClaimAggregator] uses this trait's method to retrieve
/// and keep track of claims.
pub trait ProverYieldClaim<F: FieldExt, Claim> {
    /// Generate and return the claims that this layer makes onto other layers.
    fn get_claims(&self) -> Result<Vec<Claim>, LayerError>;
}

/// A verifier variant of the [ProverYieldClaim] trait.
/// It allows a layer-like type to yield claims for other layers.
/// Typically, a [ClaimAggregator] uses this trait's method to retrieve
/// and keep track of claims.
pub trait VerifierYieldClaim<F: FieldExt, Claim> {
    /// Generates and returns the claims that this layer makes onto other
    /// layers.
    ///
    /// `expr` is the expression associated with the current layer with all bits
    /// bound. We're passing this instead of using relying entirely on `self`
    /// because, at this point, the [crate::layer::VerifierLayer]
    /// implementations maintain an `Expression<F, CircuitExpr>` without any
    /// bound bits or evaluations, which are needed to generate the claims.
    /// TODO(Makis): Introduce a separate `CircuitLayer` to avoid the need
    /// for passing the expression here.
    fn get_claims(&self, expr: &Expression<F, VerifierExpr>) -> Result<Vec<Claim>, LayerError>;
}
