use std::fmt::Debug;
use serde::{Deserialize, Serialize};
use crate::mle::MleIndex;

use remainder_shared_types::FieldExt;

use super::{expr_errors::ExpressionError, generic_expr::{Expression, ExpressionType}};


/// Verifier Expression
/// the leaf nodes of the expression tree are F (because they are already fixed/evaluated)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VerifierExpression;
impl<F: FieldExt> ExpressionType<F> for VerifierExpression {
    type Container = F;
}

impl<F: FieldExt> Expression<F, VerifierExpression> {
    /// Evaluate the polynomial using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<T>(
        &self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&<VerifierExpression as ExpressionType<F>>::Container) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[<VerifierExpression as ExpressionType<F>>::Container]) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T {
        match self {
            Expression::Constant(scalar) => constant(*scalar),
            Expression::Selector(index, a, b) => selector_column(
                index,
                a.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                ),
                b.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                ),
            ),
            Expression::Mle(query) => mle_eval(query),
            Expression::Negated(a) => {
                let a = a.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                negated(a)
            }
            Expression::Sum(a, b) => {
                let a = a.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                let b = b.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                sum(a, b)
            }
            Expression::Product(queries) => product(queries),
            Expression::Scaled(a, f) => {
                let a = a.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                scaled(a, *f)
            }
        }
    }

    /// used by `evaluate_expr` to traverse the expression and simply
    /// gather all of the evaluations, combining them as appropriate.
    pub fn gather_combine_all_evals(
        &self,
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
        self.evaluate(
            &constant,
            &selector_column,
            &mle_eval,
            &negated,
            &sum,
            &product,
            &scaled,
        )
    }
}