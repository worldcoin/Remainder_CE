//! A layer is a combination of multiple MLEs with an expression

pub mod combine_mle_refs;

pub mod gate;
pub mod layer_enum;
pub mod regular_layer;
// mod gkr_layer;

use std::fmt::Debug;

use derive_more::Display;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    claims::{Claim, ClaimError},
    expression::expr_errors::ExpressionError,
    sumcheck::InterpError,
};
use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptReaderError, TranscriptSponge, TranscriptWriter},
    FieldExt,
};

#[derive(Error, Debug, Clone)]
/// Errors to do with working with a Layer
pub enum LayerError {
    #[error("Layer isn't ready to prove")]
    /// Layer isn't ready to prove
    LayerNotReady,
    #[error("Error with underlying expression: {0}")]
    /// Error with underlying expression: {0}
    ExpressionError(#[from] ExpressionError),
    #[error("Error with aggregating curr layer")]
    /// Error with aggregating curr layer
    AggregationError,
    #[error("Error with getting Claim: {0}")]
    /// Error with getting Claim
    ClaimError(#[from] ClaimError),
    #[error("Error with verifying layer: {0}")]
    /// Error with verifying layer
    VerificationError(#[from] VerificationError),
    #[error("InterpError: {0}")]
    /// InterpError
    InterpError(#[from] InterpError),
    #[error("Transcript Error: {0}")]
    /// Transcript Error
    TranscriptError(#[from] TranscriptReaderError),
}

#[derive(Error, Debug, Clone)]
/// Errors to do with verifying a Layer
pub enum VerificationError {
    #[error("The sum of the first evaluations do not equal the claim")]
    /// The sum of the first evaluations do not equal the claim
    SumcheckStartFailed,
    #[error("The sum of the current rounds evaluations do not equal the previous round at a random point")]
    /// The sum of the current rounds evaluations do not equal the previous round at a random point
    SumcheckFailed,
    #[error("The final rounds evaluations at r do not equal the oracle query")]
    /// The final rounds evaluations at r do not equal the oracle query
    FinalSumcheckFailed,
    #[error("The Oracle query does not match the final claim")]
    /// The Oracle query does not match the final claim
    GKRClaimCheckFailed,
    #[error(
        "The Challenges generated during sumcheck don't match the claims in the given expression"
    )]
    ///The Challenges generated during sumcheck don't match the claims in the given expression
    ChallengeCheckFailed,
}

#[derive(Clone, Debug, Display, PartialEq, Eq, Hash, Serialize, Deserialize, Copy, PartialOrd)]
/// The location of a layer within the GKR circuit
pub enum LayerId {
    /// A random mle input layer
    ///
    /// TODO!(nick) Remove this once new batching code is implemented
    RandomInput(usize),
    /// An Mle located in the input layer
    Input(usize),
    /// A layer between the output layer and input layers
    Layer(usize),
    /// An MLE located in the output layer.
    Output(usize),
}

/// A layer is the smallest component of the GKR protocol.
///
/// Each `Layer` is a sub-protocol that takes in some `Claim` and creates a proof
/// that the `Claim` is correct
pub trait Layer<F: FieldExt> {
    /// The struct that contains the proof this `Layer` generates
    type Proof: Debug + Serialize + for<'a> Deserialize<'a>;

    /// Creates a proof for this Layer
    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<Self::Proof, LayerError>;

    /// Verifies the `Layer`'s proof
    fn verify_rounds(
        &mut self,
        claim: Claim<F>,
        proof: Self::Proof,
        transcript: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), LayerError>;

    /// Gets this `Layer`'s `LayerId`
    fn id(&self) -> &LayerId;
}
