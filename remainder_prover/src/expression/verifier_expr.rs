use crate::{layer::LayerId, mle::MleIndex};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
};

use remainder_shared_types::FieldExt;

use super::{
    expr_errors::ExpressionError,
    generic_expr::{Expression, ExpressionNode, ExpressionType},
};

/// The verifier's representation of a [crate::mle::dense::DenseMle]
/// in the context of the verifier's circuit description.
/// A [VerifierMle] is used on the leaves of a [VerifierExpr].
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct VerifierMle<F: FieldExt> {
    /// Layer whose data this MLE is a subset of.
    layer_id: LayerId,

    /// A list of indices where the free variables have been assigned an index.
    var_indices: Vec<MleIndex<F>>,
}

/// Placeholder type for defining `Expression<F, VerifierExpr>`, the type used
/// for representing expressions on the verifier's circuit description.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VerifierExpr;

/// The leaves of an expression contain a [VerifierMle], the verifier's
/// analogue of a [crate::mle::dense::DenseMle], storing only metadata related
/// to the MLE, not any evaluations.
impl<F: FieldExt> ExpressionType<F> for VerifierExpr {
    type MLENodeRepr = VerifierMle<F>;
    type MleVec = ();
}

impl<F: FieldExt> Expression<F, VerifierExpr> {
    /*
    /// Evaluate the polynomial using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<T>(
        &self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&<VerifierExpr as ExpressionType<F>>::MLENodeRepr) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[<VerifierExpr as ExpressionType<F>>::MLENodeRepr]) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T {
        self.expression_node.evaluate(
            constant,
            selector_column,
            mle_eval,
            negated,
            sum,
            product,
            scaled,
        )
    }

    /// used by `evaluate_expr` to traverse the expression and simply
    /// gather all of the evaluations, combining them as appropriate.
    pub fn gather_combine_all_evals(&self) -> Result<F, ExpressionError> {
        let constant = |c| Ok(c);
        let selector_column = |idx: &MleIndex<F>,
                               lhs: Result<F, ExpressionError>,
                               rhs: Result<F, ExpressionError>| {
            // --- Selector bit must be bound ---
            if let MleIndex::Bound(val, _) = idx {
                return Ok(*val * rhs? + (F::ONE - val) * lhs?);
            }
            Err(ExpressionError::SelectorBitNotBoundError)
        };
        let mle_eval = |mle_ref: & <VerifierExpr as ExpressionType<F>>::MLENodeRepr| -> Result<F, ExpressionError> {
            Ok(*mle_ref)
        };
        let negated = |a: Result<F, ExpressionError>| match a {
            Err(e) => Err(e),
            Ok(val) => Ok(val.neg()),
        };
        let sum =
            |lhs: Result<F, ExpressionError>, rhs: Result<F, ExpressionError>| Ok(lhs? + rhs?);
        let product = |mle_refs: & [<VerifierExpr as ExpressionType<F>>::MLENodeRepr]| -> Result<F, ExpressionError> {
            mle_refs.iter().try_fold(F::ONE, |acc, new_mle_ref| {
                Ok(acc * *new_mle_ref)
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
    */

    /// Traverses the expression tree to get the indices of all the nonlinear
    /// rounds. Returns a sorted vector of indices.
    pub fn get_all_nonlinear_rounds(&mut self) -> Vec<usize> {
        let (expression_node, mle_vec) = self.deconstruct_mut();
        let mut nonlinear_rounds: Vec<usize> =
            expression_node.get_all_nonlinear_rounds(&mut vec![], mle_vec);
        nonlinear_rounds.sort();
        nonlinear_rounds
    }
}

impl<F: FieldExt> ExpressionNode<F, VerifierExpr> {
    /*
        /// Evaluate the polynomial using the provided closures to perform the
        /// operations.
        #[allow(clippy::too_many_arguments)]
        pub fn evaluate<T>(
            &self,
            constant: &impl Fn(F) -> T,
            selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
            mle_eval: &impl Fn(&<VerifierExpr as ExpressionType<F>>::MLENodeRepr) -> T,
            negated: &impl Fn(T) -> T,
            sum: &impl Fn(T, T) -> T,
            product: &impl Fn(&[<VerifierExpr as ExpressionType<F>>::MLENodeRepr]) -> T,
            scaled: &impl Fn(T, F) -> T,
        ) -> T {
            match self {
                ExpressionNode::Constant(scalar) => constant(*scalar),
                ExpressionNode::Selector(index, a, b) => selector_column(
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
                ExpressionNode::Mle(query) => mle_eval(query),
                ExpressionNode::Negated(a) => {
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
                ExpressionNode::Sum(a, b) => {
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
                ExpressionNode::Product(queries) => product(queries),
                ExpressionNode::Scaled(a, f) => {
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
    */

    /// Traverse an expression tree in order and returns a vector of indices of
    /// all the nonlinear rounds in an expression (in no particular order).
    pub fn get_all_nonlinear_rounds(
        &self,
        curr_nonlinear_indices: &mut Vec<usize>,
        mle_vec: &<VerifierExpr as ExpressionType<F>>::MleVec,
    ) -> Vec<usize> {
        let nonlinear_indices_in_node = {
            match self {
                // The only case where an index is nonlinear is if it is present in multiple mle
                // refs that are part of a product. We iterate through all the indices in the
                // product nodes to look for repeated indices within a single node.
                ExpressionNode::Product(verifier_mles) => {
                    let mut product_nonlinear_indices: HashSet<usize> = HashSet::new();
                    let mut product_indices_counts: HashMap<MleIndex<F>, usize> = HashMap::new();

                    verifier_mles.into_iter().for_each(|verifier_mle| {
                        verifier_mle.var_indices.iter().for_each(|mle_index| {
                            let curr_count = {
                                if product_indices_counts.contains_key(mle_index) {
                                    product_indices_counts.get(mle_index).unwrap()
                                } else {
                                    &0
                                }
                            };
                            product_indices_counts.insert(mle_index.clone(), curr_count + 1);
                        })
                    });

                    product_indices_counts
                        .into_iter()
                        .for_each(|(mle_index, count)| {
                            if count > 1 {
                                if let MleIndex::IndexedBit(i) = mle_index {
                                    product_nonlinear_indices.insert(i);
                                }
                            }
                        });

                    product_nonlinear_indices
                }
                // for the rest of the types of expressions, we simply traverse through the expression node to look
                // for more leaves which are specifically product nodes.
                ExpressionNode::Selector(_sel_index, a, b) => {
                    let mut sel_nonlinear_indices: HashSet<usize> = HashSet::new();
                    let a_indices = a.get_all_nonlinear_rounds(curr_nonlinear_indices, mle_vec);
                    let b_indices = b.get_all_nonlinear_rounds(curr_nonlinear_indices, mle_vec);
                    a_indices
                        .into_iter()
                        .zip(b_indices)
                        .for_each(|(a_mle_idx, b_mle_idx)| {
                            sel_nonlinear_indices.insert(a_mle_idx);
                            sel_nonlinear_indices.insert(b_mle_idx);
                        });
                    sel_nonlinear_indices
                }
                ExpressionNode::Sum(a, b) => {
                    let mut sum_nonlinear_indices: HashSet<usize> = HashSet::new();
                    let a_indices = a.get_all_nonlinear_rounds(curr_nonlinear_indices, mle_vec);
                    let b_indices = b.get_all_nonlinear_rounds(curr_nonlinear_indices, mle_vec);
                    a_indices
                        .into_iter()
                        .zip(b_indices)
                        .for_each(|(a_mle_idx, b_mle_idx)| {
                            sum_nonlinear_indices.insert(a_mle_idx);
                            sum_nonlinear_indices.insert(b_mle_idx);
                        });
                    sum_nonlinear_indices
                }
                ExpressionNode::Scaled(a, _) => a
                    .get_all_nonlinear_rounds(curr_nonlinear_indices, mle_vec)
                    .into_iter()
                    .collect(),
                ExpressionNode::Negated(a) => a
                    .get_all_nonlinear_rounds(curr_nonlinear_indices, mle_vec)
                    .into_iter()
                    .collect(),
                ExpressionNode::Constant(_) | ExpressionNode::Mle(_) => HashSet::new(),
            }
        };
        // we grab all of the indices and take the union of all of them to return all nonlinear rounds in an expression tree.
        nonlinear_indices_in_node.into_iter().for_each(|index| {
            if !curr_nonlinear_indices.contains(&index) {
                curr_nonlinear_indices.push(index);
            }
        });
        curr_nonlinear_indices.clone()
    }
}

impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for Expression<F, VerifierExpr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Expression")
            .field("Expression_Node", &self.expression_node)
            .finish()
    }
}

impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for ExpressionNode<F, VerifierExpr> {
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
            ExpressionNode::Mle(mle_ref) => {
                f.debug_struct("Mle").field("mle_ref", mle_ref).finish()
            }
            ExpressionNode::Negated(poly) => f.debug_tuple("Negated").field(poly).finish(),
            ExpressionNode::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            ExpressionNode::Product(a) => f.debug_tuple("Product").field(a).finish(),
            ExpressionNode::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}
