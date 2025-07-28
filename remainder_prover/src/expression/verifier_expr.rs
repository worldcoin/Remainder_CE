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
use std::{cmp::max, fmt::Debug};

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
        let product = |lhs: Result<F>, rhs: Result<F>| Ok(lhs? * rhs?);
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
        let mut nonlinear_rounds: Vec<usize> = expression_node.get_all_nonlinear_rounds(mle_vec);
        nonlinear_rounds.sort();
        nonlinear_rounds
    }
}

impl<F: Field> ExpressionNode<F, VerifierExpr> {
    /// Evaluate the polynomial using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce<T>(
        &self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&<VerifierExpr as ExpressionType<F>>::MLENodeRepr) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T {
        match self {
            ExpressionNode::Constant(scalar) => constant(*scalar),
            ExpressionNode::Selector(index, a, b) => {
                let lhs = a.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                let rhs = b.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                selector_column(index, lhs, rhs)
            }
            ExpressionNode::Mle(query) => mle_eval(query),
            ExpressionNode::Sum(a, b) => {
                let a = a.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                let b = b.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                sum(a, b)
            }
            ExpressionNode::Product(a, b) => {
                let a = a.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                let b = b.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                product(a, b)
            }
            ExpressionNode::Scaled(a, f) => {
                let a = a.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                scaled(a, *f)
            }
        }
    }

    /// traverse an expression tree in order to get all of the nonlinear rounds in an expression.
    pub fn get_all_nonlinear_rounds(
        &self,
        _mle_vec: &<VerifierExpr as ExpressionType<F>>::MleVec,
    ) -> Vec<usize> {
        let degree_per_index = self.get_rounds_helper(_mle_vec);
        (0..degree_per_index.len())
            .filter(|&i| degree_per_index[i] > 1)
            .collect()
    }

    // a recursive helper for get_all_rounds, get_all_nonlinear_rounds, and get_all_linear_rounds
    fn get_rounds_helper(
        &self,
        _mle_vec: &<VerifierExpr as ExpressionType<F>>::MleVec,
    ) -> Vec<usize> {
        // degree of each index
        let mut degree_per_index = Vec::new();
        // set the degree of the corresponding index to max(OLD_DEGREE, NEW_DEGREE)
        let max_degree = |degree_per_index: &mut Vec<usize>, index: usize, new_degree: usize| {
            if degree_per_index.len() <= index {
                degree_per_index.extend(vec![0; index + 1 - degree_per_index.len()]);
            }
            if degree_per_index[index] < new_degree {
                degree_per_index[index] = new_degree;
            }
        };
        // set the degree of the corresponding index to OLD_DEGREE + NEW_DEGREE
        let add_degree = |degree_per_index: &mut Vec<usize>, index: usize, new_degree: usize| {
            if degree_per_index.len() <= index {
                degree_per_index.extend(vec![0; index + 1 - degree_per_index.len()]);
            }
            degree_per_index[index] += new_degree;
        };

        match self {
            // in a product, we need the union of all the indices in each of the individual mle refs.
            ExpressionNode::Product(a, b) => {
                let a_degree_per_index = a.get_rounds_helper(_mle_vec);
                let b_degree_per_index = b.get_rounds_helper(_mle_vec);
                // nonlinear operator -- sum over the degree
                for i in 0..max(a_degree_per_index.len(), b_degree_per_index.len()) {
                    if let Some(a_degree) = a_degree_per_index.get(i) {
                        add_degree(&mut degree_per_index, i, *a_degree);
                    }
                    if let Some(b_degree) = b_degree_per_index.get(i) {
                        add_degree(&mut degree_per_index, i, *b_degree);
                    }
                }
            }
            // in an mle, we need all of the mle indices in the mle.
            ExpressionNode::Mle(mle) => {
                mle.var_indices().iter().for_each(|mle_index| {
                    if let MleIndex::Indexed(i) = mle_index {
                        max_degree(&mut degree_per_index, *i, 1);
                    }
                });
            }
            // in selector, take the max degree of each children, and add 1 degree to the selector itself
            ExpressionNode::Selector(sel_index, a, b) => {
                if let MleIndex::Indexed(i) = sel_index {
                    add_degree(&mut degree_per_index, *i, 1);
                };
                let a_degree_per_index = a.get_rounds_helper(_mle_vec);
                let b_degree_per_index = b.get_rounds_helper(_mle_vec);
                // linear operator -- take the max degree
                for i in 0..max(a_degree_per_index.len(), b_degree_per_index.len()) {
                    if let Some(a_degree) = a_degree_per_index.get(i) {
                        max_degree(&mut degree_per_index, i, *a_degree);
                    }
                    if let Some(b_degree) = b_degree_per_index.get(i) {
                        max_degree(&mut degree_per_index, i, *b_degree);
                    }
                }
            }
            // in sum, take the max degree of each children
            ExpressionNode::Sum(a, b) => {
                let a_degree_per_index = a.get_rounds_helper(_mle_vec);
                let b_degree_per_index = b.get_rounds_helper(_mle_vec);
                // linear operator -- take the max degree
                for i in 0..max(a_degree_per_index.len(), b_degree_per_index.len()) {
                    if let Some(a_degree) = a_degree_per_index.get(i) {
                        max_degree(&mut degree_per_index, i, *a_degree);
                    }
                    if let Some(b_degree) = b_degree_per_index.get(i) {
                        max_degree(&mut degree_per_index, i, *b_degree);
                    }
                }
            }
            // scaled and negated, does not affect degree
            ExpressionNode::Scaled(a, _) => {
                degree_per_index = a.get_rounds_helper(_mle_vec);
            }
            // for a constant there are no new indices.
            ExpressionNode::Constant(_) => {}
        }
        degree_per_index
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
            ExpressionNode::Product(a, b) => f.debug_tuple("Product").field(a).field(b).finish(),
            ExpressionNode::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}
