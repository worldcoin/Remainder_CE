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
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
};

use remainder_shared_types::Field;

use super::{
    expr_errors::ExpressionError,
    generic_expr::{Expression, ExpressionNode, ExpressionType},
};

use anyhow::{anyhow, Ok, Result};

/// Placeholder type for defining `Expression<F, VerifierExpr>`, the type used
/// for representing expressions for the Verifier.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VerifierExpr;

// The leaves of an expression of this type contain a [VerifierMle], an analogue
// of [crate::mle::dense::DenseMle], storing fully bound MLEs.
// TODO(Makis): Consider allowing for re-use of MLEs, like in a [ProverExpr]:
// ```ignore
//     type MLENodeRepr = usize,
//     type MleVec = Vec<VerifierMle<F>>,
// ```
impl<F: Field> ExpressionType<F> for VerifierExpr {
    type MLENodeRepr = VerifierMle<F>;
    type MleVec = ();
}

impl<F: Field> Expression<F, VerifierExpr> {
    /// Create a mle Expression that contains one MLE
    pub fn mle(mle: VerifierMle<F>) -> Self {
        let mle_node = ExpressionNode::Mle(mle);

        Expression::new(mle_node, ())
    }

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

impl<F: std::fmt::Debug + Field> std::fmt::Debug for Expression<F, VerifierExpr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Circuit Expression")
            .field("Expression_Node", &self.expression_node)
            .finish()
    }
}

impl<F: std::fmt::Debug + Field> std::fmt::Debug for ExpressionNode<F, VerifierExpr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpressionNode::Constant(scalar) => f.debug_tuple("Constant").field(scalar).finish(),
            ExpressionNode::Selector(index, a, b) => f
                .debug_tuple("Selector")
                .field(index)
                .field(a)
                .field(b)
                .finish(),
            // Skip enum variant and print query struct directly to maintain backwards compatibility.
            ExpressionNode::Mle(mle) => f.debug_struct("Circuit Mle").field("mle", mle).finish(),
            ExpressionNode::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            ExpressionNode::Product(a) => f.debug_tuple("Product").field(a).finish(),
            ExpressionNode::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}
