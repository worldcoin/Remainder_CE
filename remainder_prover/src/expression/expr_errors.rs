use thiserror::Error;
use crate::sumcheck::MleError;

#[derive(Error, Debug, Clone, PartialEq)]
///Error for handling the parsing and evaluation of expressions
pub enum ExpressionError {
    ///Error for when an InvalidMleIndex is found while evaluating an expression
    /// TODO!(add some diagnoistics here)
    #[error("")]
    InvalidMleIndex,
    ///Error for when Something unlikely goes wrong while evaluating an expression
    /// TODO!(split this up into many error variants)
    #[error("Something went wrong while evaluating: {0}")]
    EvaluationError(&'static str),
    ///Error that wraps an MleError
    #[error("Something went wrong while evaluating the MLE: {0}")]
    MleError(MleError),
    #[error("Selector bit not bound before final evaluation gather")]
    ///Selector bit not bound before final evaluation gather
    SelectorBitNotBoundError,
    #[error("MLE ref with more than one element in its bookkeeping table")]
    ///MLE ref with more than one element in its bookkeeping table
    EvaluateNotFullyBoundError,
    #[error("The bound indices of this expression don't match the indices passed in")]
    ///The bound indices of this expression don't match the indices passed in
    EvaluateBoundIndicesDontMatch,
}