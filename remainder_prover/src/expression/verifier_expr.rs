use std::{
    cmp::max,
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    mle::{beta::*, dense::DenseMleRef, MleIndex, MleRef},
    sumcheck::MleError,
};

use remainder_shared_types::FieldExt;

use super::{expr_errors::ExpressionError, generic_expr::{Expression, ExpressionType}};


/// Verifier Expression
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VerifierExpression;
impl<F: FieldExt> ExpressionType<F> for VerifierExpression {
    type Container = F;
}

// generic, verifier

/// Helper function for `evaluate_expr` to traverse the expression and simply
/// gather all of the evaluations, combining them as appropriate.
/// Strictly speaking this doesn't need to be `&mut` but we call `self.evaluate()`
/// within.
pub fn gather_combine_all_evals_verifier<F: FieldExt>(
    expr: &Expression<F, VerifierExpression>,
) -> Result<F, ExpressionError> {
    let constant = |c| Ok(c);
    let selector_column =
        |idx: &MleIndex<F>, lhs: Result<F, ExpressionError>, rhs: Result<F, ExpressionError>| {
            // --- Selector bit must be bound ---
            if let MleIndex::Bound(val, _) = idx {
                return Ok(*val * rhs? + (F::one() - val) * lhs?);
            }
            Err(ExpressionError::SelectorBitNotBoundError)
        };
    let mle_eval = for<'a> |mle_ref: &'a <VerifierExpression as ExpressionType<F>>::Container| -> Result<F, ExpressionError> {
        Ok(mle_ref.clone())
    };
    let negated = |a: Result<F, ExpressionError>| match a {
        Err(e) => Err(e),
        Ok(val) => Ok(val.neg()),
    };
    let sum = |lhs: Result<F, ExpressionError>, rhs: Result<F, ExpressionError>| Ok(lhs? + rhs?);
    let product = for<'a, 'b> |mle_refs: &'a [<VerifierExpression as ExpressionType<F>>::Container]| -> Result<F, ExpressionError> {
        mle_refs.iter().try_fold(F::one(), |acc, new_mle_ref| {
            Ok(acc * new_mle_ref.clone())
        })
    };
    let scaled = |a: Result<F, ExpressionError>, scalar: F| Ok(a? * scalar);
    expr.evaluate(
        &constant,
        &selector_column,
        &mle_eval,
        &negated,
        &sum,
        &product,
        &scaled,
    )
}