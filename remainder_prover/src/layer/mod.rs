//! A layer is a combination of multiple MLEs with an expression.

pub mod combine_mle_refs;

pub mod gate;
pub mod identity_gate;
pub mod layer_enum;
pub mod matmult;
pub mod product;
pub mod regular_layer;

use std::fmt::Debug;

use product::PostSumcheckLayer;
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
    transcript::{ProverTranscript, TranscriptReaderError, VerifierTranscript},
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
        transcript: &mut impl ProverTranscript<F>,
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
        transcript: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer, VerificationError>;

    /// The number of sumcheck rounds of this layer.
    fn num_sumcheck_rounds(&self) -> usize;

    /// Turns this Circuit Layer into a Verifier Layer
    fn into_verifier_layer(
        &self,
        sumcheck_bindings: &[F],
        claim_point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer, VerificationError>;
}

/// A verifier counterpart of a GKR [Layer] trait.
pub trait VerifierLayer<F: FieldExt> {
    /// Returns this layer's ID.
    fn layer_id(&self) -> LayerId;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Copy, PartialOrd)]
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

impl LayerId {
    /// Gets a new LayerId which represents a layerid of the same type but with an incremented id number
    pub fn next(&self) -> LayerId {
        match self {
            LayerId::RandomInput(id) => LayerId::RandomInput(id + 1),
            LayerId::Input(id) => LayerId::Input(id + 1),
            LayerId::Layer(id) => LayerId::Layer(id + 1),
            LayerId::Output(id) => LayerId::Output(id + 1),
        }
    }
}

/// A trait for defining an interface for Layers that implement the Sumcheck protocol
pub trait SumcheckLayer<F: FieldExt>: Layer<F> {
    /// Initialize the sumcheck round by setting the beta table, computing the number of rounds, etc.
    fn initialize_sumcheck(&mut self, claim_point: &[F]) -> Result<(), LayerError>;

    /// Return the evaluations of the univariate polynomial generated during this round of sumcheck.
    ///
    /// This must be called with a steadily incrementing round_index & with a securely generated challenge.
    fn compute_round_sumcheck_message(&self, round_index: usize) -> Result<Vec<F>, LayerError>;

    /// Mutate the underlying bookkeeping tables to "bind" the given `challenge` to the bit.
    /// labeled with that `round_index`.
    fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<(), LayerError>;

    /// How many sumcheck rounds this layer will take to prove.
    fn num_sumcheck_rounds(&self) -> usize;

    /// The maximum degree for any univariate in the sumcheck protocol.
    fn max_degree(&self) -> usize;
}

/// An interface for constructing a [PostSumcheckLayer] for a layer.
pub trait PostSumcheckEvaluation<F: FieldExt> {
    /// Get the [PostSumcheckLayer] for a certain layer, which is a struct which represents
    /// the fully bound layer.
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, F>;
}
