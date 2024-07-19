//! A layer is a combination of multiple MLEs with an expression.

pub mod combine_mle_refs;

pub mod gate;
pub mod layer_enum;
pub mod regular_layer;

use std::fmt::Debug;

use derive_more::Display;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    claims::{wlx_eval::WLXAggregator, Claim, ClaimAggregator, ClaimError},
    expression::{
        expr_errors::ExpressionError, generic_expr::Expression, verifier_expr::VerifierExpr,
    },
    sumcheck::InterpError,
};
use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptReaderError, TranscriptSponge, TranscriptWriter},
    FieldExt,
};

/// Errors to do with working with a type implementing [Layer].
#[derive(Error, Debug, Clone)]
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

/// Errors to do with verifying a layer while working with a type implementing
/// [VerifierLayer].
#[derive(Error, Debug, Clone)]
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

    /// Error with underlying expression: {0}
    #[error("Error with underlying expression: {0}")]
    ExpressionError(#[from] ExpressionError),

    // Error while reading the transcript proof.
    #[error("Error while reading the transcript proof")]
    TranscriptError(#[from] TranscriptReaderError),

    /// Interpolation Error.
    #[error("Interpolation Error: {0}")]
    InterpError(#[from] InterpError),
}

/// The location of a layer within the GKR circuit.
#[derive(Clone, Debug, Display, PartialEq, Eq, Hash, Serialize, Deserialize, Copy, PartialOrd)]
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
    /// TEMP
    // type VerifierLayer: VerifierLayer<F> + Debug + Serialize + for<'a> Deserialize<'a>;

    /// The associated type used to store a description of this layer as part
    /// of a [GKRVerifierKey].
    type CircuitLayer: CircuitLayer<F> + Debug + Serialize + for<'a> Deserialize<'a>;

    /// Generates a description of this layer.
    fn into_circuit_layer(&self) -> Result<Self::CircuitLayer, LayerError>;

    /// Gets this layer's ID.
    fn layer_id(&self) -> LayerId;

    /// Tries to prove `claim` for this layer.
    ///
    /// In the process of proving, it mutates itself binding the variables
    /// of the expression that define the layer.
    ///
    /// If successful, the proof is implicitly included in the modified
    /// transcript.
    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<(), LayerError>;
}

/// A circuit-description counterpart of the GKR [Layer] trait.
pub trait CircuitLayer<F: FieldExt> {
    /// The associated type that the verifier uses to work with a layer of this
    /// kind.
    type VerifierLayer: VerifierLayer<F> + Debug + Serialize + for<'a> Deserialize<'a>;

    /// Returns this layer's ID.
    fn layer_id(&self) -> LayerId;

    /// Tries to verify `claim` for this layer and returns a [VerifierLayer]
    /// with a fully bound and evaluated expression.
    ///
    /// The proof is implicitly included in the `transcript`.
    fn verify_rounds(
        &self,
        claim: Claim<F>,
        transcript: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<Self::VerifierLayer, VerificationError>;
}

/// A verifier counterpart of a GKR [Layer] trait.
pub trait VerifierLayer<F: FieldExt> {
    /// Returns this layer's ID.
    fn layer_id(&self) -> LayerId;
}
