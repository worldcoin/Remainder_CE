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

use remainder_shared_types::{extension_field::ExtensionField, Field};

use super::{expr_errors::ExpressionError, generic_expr::Expression};

use anyhow::{anyhow, Ok, Result};

impl<F: Field, E> Expression<F, VerifierMle<E>> 
where
    E: ExtensionField<BaseField = F>,
{
    /// Evaluate this fully bound expression.
    pub fn evaluate(&self, bind_list: &Vec<Option<E>>,) -> Result<E> {
        let constant = |c| Ok(E::from(c));
        let selector_column = |idx: &MleIndex, lhs: Result<E>, rhs: Result<E>| -> Result<E> {
            // Selector bit must be bound
            if let MleIndex::Bound(idx) = idx {
                let val = bind_list[*idx].unwrap();
                return Ok(val * rhs? + (E::ONE - val) * lhs?);
            }
            Err(anyhow!(ExpressionError::SelectorBitNotBoundError))
        };
        let mle_eval = |verifier_mle: &VerifierMle<E>| -> Result<E> { Ok(verifier_mle.value()) };
        let sum = |lhs: Result<E>, rhs: Result<E>| Ok(lhs? + rhs?);
        let product = |verifier_mles: &[&VerifierMle<E>]| -> Result<E> {
            verifier_mles
                .iter()
                .try_fold(E::ONE, |acc, verifier_mle| Ok(acc * verifier_mle.value()))
        };
        let scaled = |val: Result<E>, scalar: E::BaseField| Ok(val? * scalar);

        self.expression_node.reduce(
            &self.mle_vec,
            &constant,
            &selector_column,
            &mle_eval,
            &sum,
            &product,
            &scaled,
        )
    }
}
