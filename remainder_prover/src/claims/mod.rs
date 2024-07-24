//!Utilities involving the claims a layer makes
///The claim aggregator that uses wlx evaluations
pub mod wlx_eval;

use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    FieldExt,
};
use thiserror::Error;

use crate::layer::combine_mle_refs::CombineMleRefError;
use serde::{Deserialize, Serialize};

use crate::{
    input_layer::InputLayer,
    layer::{Layer, LayerId},
};

#[derive(Error, Debug, Clone)]
///Errors to do with aggregating and collecting claims
pub enum ClaimError {
    #[error("The Layer has not finished the sumcheck protocol")]
    ///The Layer has not finished the sumcheck protocol
    SumCheckNotComplete,
    #[error("MLE indices must all be fixed")]
    ///MLE indices must all be fixed
    ClaimMleIndexError,
    #[error("Layer ID not assigned")]
    ///Layer ID not assigned
    LayerMleError,
    #[error("MLE within MleRef has multiple values within it")]
    ///MLE within MleRef has multiple values within it
    MleRefMleError,
    #[error("Error aggregating claims")]
    ///Error aggregating claims
    ClaimAggroError,
    #[error("Should be evaluating to a sum")]
    ///Should be evaluating to a sum
    ExpressionEvalError,
    #[error("All claims in a group should agree on the number of variables")]
    ///All claims in a group should agree on the number of variables
    NumVarsMismatch,
    #[error("All claims in a group should agree the destination layer field")]
    ///All claims in a group should agree the destination layer field
    LayerIdMismatch,
    #[error("Error while combining mle refs in order to evaluate challenge point")]
    ///Error while combining mle refs in order to evaluate challenge point
    MleRefCombineError(CombineMleRefError),

    /// Zero MLE refs cannot be used as intermediate values within a circuit!
    #[error("Zero MLE refs cannot be used as intermediate values within a circuit")]
    IntermediateZeroMLERefError,
}

/// A claim contains a `point` \in F^n along with the `result` \in F that an
/// associated layer MLE is expected to evaluate to. In other words, if `W : F^n
/// -> F` is the MLE, then the claim asserts: `W(point) == result`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct Claim<F: FieldExt> {
    /// The point in F^n where the layer MLE is to be evaluated on.
    point: Vec<F>,
    /// The expected result of evaluating this layer's MLE on `point`.
    result: F,
}

impl<F: FieldExt> Claim<F> {
    /// Constructs a new `Claim`
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

/// A wrapper struct that represents the `Claim` a `ClaimAggregator`
/// generates and the proof that this claim is correct
#[derive(Clone, Debug)]
pub struct ClaimAndProof<F: FieldExt, P> {
    /// The `Claim` that was proven
    pub claim: Claim<F>,
    /// The proof that the `Claim` is correct
    pub proof: P,
}

/// A trait that defines a protocol for the tracking/aggregation of many claims
pub trait ClaimAggregator<F: FieldExt> {
    ///The struct the claim aggregator takes in.
    ///
    /// Is typically composed of the `Claim` struct and additional information
    type Claim: std::fmt::Debug + Clone + Serialize + for<'a> Deserialize<'a>;

    ///The proof that a verifier can use to check that the aggregation was done correctly
    type AggregationProof: std::fmt::Debug + Clone + Serialize + for<'a> Deserialize<'a>;

    ///The Layer this claim aggregator takes claims from, and aggregates claims for
    type Layer: Layer<F>;

    ///The InputLayer this claim aggregator aggregates claims for
    type InputLayer: InputLayer<F>;

    type Error: std::error::Error;

    ///Creates an empty ClaimAggregator, ready to track claims
    fn new() -> Self;

    ///Takes in claims to track and later aggregate
    fn add_claims<L: YieldClaim<Self::Claim>>(&mut self, layer: &L) -> Result<(), L::Error>;

    ///Retrieves claims for aggregation
    fn get_claims(&self, layer_id: LayerId) -> Option<&[Self::Claim]>;

    ///Takes in some claims from the prover and aggregates them into one claim
    ///
    /// Extracts additional information from the `Layer` the claims are made on if neccessary.
    ///
    /// Adds any communication to the F-S Transcript
    fn prover_aggregate_claims(
        &self,
        layer: &Self::Layer,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<ClaimAndProof<F, Self::AggregationProof>, Self::Error>;

    ///Takes in some claims from the prover and aggregates them into one claim
    ///
    /// Extracts additional information from the `InputLayer` the claims are made on if neccessary.
    ///
    /// Adds any communication to the F-S Transcript
    fn prover_aggregate_claims_input(
        &self,
        layer: &Self::InputLayer,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<ClaimAndProof<F, Self::AggregationProof>, Self::Error>;

    ///Reads an AggregationProof from the Transcript and uses it to verify the aggregation of some claims.
    fn verifier_aggregate_claims(
        &self,
        layer_id: LayerId,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Claim<F>, Self::Error>;
}

///A trait that allows the type to yield some claims to be added
/// to the claim tracker
pub trait YieldClaim<Claim> {
    type Error: std::error::Error;
    /// Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<Claim>, Self::Error>;
}
