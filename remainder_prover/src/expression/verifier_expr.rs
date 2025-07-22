//! The verifier's "view" of a "fully-bound" polynomial relationship between
//! an output layer and those of its inputs (see documentation within
//! [crate::expression] for more general information).
//!
//! Specifically, the [VerifierExpr] is exactly the struct a GKR verifier
//! uses to compute its "oracle query" at the end of the sumcheck protocol,
//! wherein
//! * The prover has sent over the last sumcheck message, i.e. the univariate
//!   polynomial f_{n - 1}(x) = \sum_{b_1, ..., b_{n - 1}} f(b_1, ..., b_{n - 1}, x)
//! * The verifier samples r_n uniformly from \mathbb{F} and wishes to check that
//!   Expr(r_1, ..., r_n) = f_{n - 1}(r_n).
//!   Specifically, the verifier wishes to "plug in" r_1, ..., r_n to Expr(x_1, ..., x_n)
//!   and additionally add in prover-claimed values for each of the MLEs at the
//!   leaves of Expr(x_1, ..., x_n) to check the above. The [VerifierExpr] allows
//!   the verifier to do exactly this, as the conversion from a
//!   [super::circuit_expr::ExprDescription] to a [VerifierExpr] involves exactly the
//!   process of "binding" the sumcheck challenges and "populating" each leaf
//!   MLE with the prover-claimed value for the evaluation of that MLE at the
//!   bound sumcheck challenge points.

use crate::mle::{verifier_mle::VerifierMle, MleIndex};

use remainder_shared_types::Field;

use super::{
    expr_errors::ExpressionError,
    generic_expr::Expression,
};

use anyhow::{anyhow, Ok, Result};

impl<F: Field> Expression<F, VerifierMle<F>> {
    /// Evaluate this fully bound expression.
    pub fn evaluate(&self) -> Result<F> {
        let constant = |c| Ok(c);
        let selector_column = |idx: &MleIndex<F>, lhs: Result<F>, rhs: Result<F>| -> Result<F> {
            // Selector bit must be bound
            if let MleIndex::Bound(val, _) = idx {
                return Ok(*val * rhs? + (F::ONE - val) * lhs?);
            }
            Err(anyhow!(ExpressionError::SelectorBitNotBoundError))
        };
        let mle_eval = |verifier_mle: &VerifierMle<F>| -> Result<F> { Ok(verifier_mle.value()) };
        let sum = |lhs: Result<F>, rhs: Result<F>| Ok(lhs? + rhs?);
        let product = |verifier_mles: &[VerifierMle<F>]| -> Result<F> {
            verifier_mles
                .iter()
                .try_fold(F::ONE, |acc, verifier_mle| Ok(acc * verifier_mle.value()))
        };
        let scaled = |val: Result<F>, scalar: F| Ok(val? * scalar);

        self.expression_node.reduce(
            &constant,
            &selector_column,
            &mle_eval,
            &sum,
            &product,
            &scaled,
        )
    }

    /// Traverses the expression tree to get the indices of all the nonlinear
    /// rounds. Returns a sorted vector of indices.
    pub fn get_all_nonlinear_rounds(&mut self) -> Vec<usize> {
        let (expression_node, mle_vec) = self.deconstruct_mut();
        let mut nonlinear_rounds: Vec<usize> =
            expression_node.get_all_nonlinear_rounds(mle_vec);
        nonlinear_rounds.sort();
        nonlinear_rounds
    }
}
