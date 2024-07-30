use crate::sumcheck::MleError;
use remainder_shared_types::transcript::TranscriptReaderError;
use thiserror::Error;

/// Error for handling the parsing and evaluation of expressions.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ExpressionError {
    /// Error for when an InvalidMleIndex is found while evaluating an expression
    /// TODO!(add some diagnoistics here)
    #[error("")]
    InvalidMleIndex,

    /// Error for when something unlikely goes wrong while evaluating an expression
    /// TODO!(split this up into many error variants)
    #[error("Something went wrong while evaluating: {0}")]
    EvaluationError(&'static str),

    /// Error that wraps an MleError.
    #[error("Something went wrong while evaluating the MLE: {0}")]
    MleError(#[from] MleError),

    /// Selector bit not bound before final evaluation gather.
    #[error("Selector bit not bound before final evaluation gather")]
    SelectorBitNotBoundError,

    /// MLE ref with more than one element in its bookkeeping table.
    #[error("MLE ref with more than one element in its bookkeeping table")]
    EvaluateNotFullyBoundError,

    /// MLE contains indices other than indexed.
    #[error("MLE contains indices other than indexed.")]
    EvaluateNotFullyIndexedError,

    /// The bound indices of this expression don't match the indices passed in.
    #[error("The bound indices of this expression don't match the indices passed in")]
    EvaluateBoundIndicesDontMatch,

    /// Transcript Reader Error.
    #[error("Transcript Reader Error")]
    TranscriptError(#[from] TranscriptReaderError),
}
