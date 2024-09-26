//! Utilities involving the claims a layer makes.

/// The default claim aggregator that uses wlx evaluations.
pub mod wlx_eval;

use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};
use thiserror::Error;

use crate::{
    input_layer::enum_input_layer::InputLayerEnum,
    layer::{combine_mle_refs::CombineMleRefError, layer_enum::LayerEnum, LayerError},
    mle::dense::DenseMle,
    prover::GKRError,
};
use serde::{Deserialize, Serialize};

use crate::{
    input_layer::InputLayer,
    layer::{Layer, LayerId},
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

    /// Zero MLE refs cannot be used as intermediate values within a circuit!
    #[error("Zero MLE refs cannot be used as intermediate values within a circuit")]
    IntermediateZeroMLERefError,
}

/// A claim contains a `point \in F^n` along with the `result \in F` that an
/// associated layer MLE is expected to evaluate to. In other words, if
/// `\tilde{V} : F^n -> F` is the MLE of a layer, then this claim asserts:
/// `\tilde{V}(point) == result`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(bound = "F: Field")]
pub struct Claim<F: Field> {
    /// The point in F^n where the layer MLE is to be evaluated on.
    point: Vec<F>,

    /// The expected result of evaluating this layer's MLE on `point`.
    result: F,
}

impl<F: Field> Claim<F> {
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
pub trait ClaimAggregator<F: Field> {
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

    /// Retrieves claims from `layer` to keep track internally and later
    /// aggregate.
    fn extract_claims(&mut self, layer: &impl YieldClaim<Self::Claim>) -> Result<(), LayerError>;

    /// Returns the claims made to layer with ID `layer_id` (if any).
    /// The claims must have already been retrieved using
    /// `extract_claims`.
    ///
    /// TODO(Makis): Do we need to expose this method to the public interface?
    /// It seems it is only used by the `aggregate_claim` variants internally.
    fn get_claims(&self, layer_id: LayerId) -> Option<&[Self::Claim]>;

    /// Aggregates all the claims made on `layer` and returns a single claim.
    //
    /// Should be called only after all claims for `input_layer` have been
    /// retrieved using `extract_claims`.
    ///
    /// Adds any communication to the F-S Transcript as needed.
    fn prover_aggregate_claims(
        &self,
        layer: &LayerEnum<F>,
        output_mles_from_layer: &[DenseMle<F>],
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<Claim<F>, GKRError>;

    /// Similar to `prover_aggregate_claims` but for an input layer.
    ///
    /// Aggregates all the claims made on `input_layer` and returns a single
    /// claim.
    ///
    /// Should be called only after all claims for `input_layer` have been
    /// retrieved using `extract_claims`.
    ///
    /// Adds any communication to the F-S Transcript as needed.
    fn prover_aggregate_claims_input(
        &self,
        layer: &InputLayerEnum<F>,
        output_mles_from_layer: &[DenseMle<F>],
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<Claim<F>, GKRError>;

    /// The verifier's variant of `prover_aggregate_claims`.
    ///
    /// Aggregates all the claims made on the Layer with ID `layer_id` (could be
    /// either an intermediate or input layer) and returns a single claim.
    ///
    /// Should be called only after all claims for this layer have been
    /// retrieved using `extract_claims`.
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
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Claim<F>, GKRError>;
}

/// A trait that allows a layer-like type to yield claims for other layers.
/// Typically, a [ClaimAggregator] uses this trait's method to extract
/// and keep track of claims.
pub trait YieldClaim<Claim> {
    /// Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<Claim>, LayerError>;
}
