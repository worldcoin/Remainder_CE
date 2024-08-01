use crate::{
    layer::{
        product::{PostSumcheckLayer, Product},
        LayerId,
    },
    mle::MleIndex,
};
use ndarray::SliceInfoElem;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
};
use thiserror::Error;

use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptReaderError, TranscriptSponge},
    FieldExt,
};

use super::{
    expr_errors::ExpressionError,
    generic_expr::{Expression, ExpressionNode, ExpressionType},
};

/// A version of [crate::mle::dense::DenseMle] used by the Verifier.
/// A [VerifierMle] stores a fully bound MLE along with its evaluation.
/// It is used to represent the leaves of an `Expression<F, VerifierExpr>`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(bound = "F: FieldExt")]
pub struct VerifierMle<F: FieldExt> {
    /// Layer whose data this MLE is a subset of.
    layer_id: LayerId,

    /// A list of bound indices.
    var_indices: Vec<MleIndex<F>>,

    /// The evaluation of this MLE when variables are bound according
    /// `var_indices`.
    eval: F,
}

impl<F: FieldExt> VerifierMle<F> {
    pub fn new(layer_id: LayerId, var_indices: Vec<MleIndex<F>>, eval: F) -> Self {
        Self {
            layer_id,
            var_indices,
            eval,
        }
    }

    pub fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    pub fn num_vars(&self) -> usize {
        self.var_indices.len()
    }

    pub fn mle_indices(&self) -> &[MleIndex<F>] {
        &self.var_indices
    }

    pub fn value(&self) -> F {
        self.eval
    }

    /// Returns the evaluation challenges for a fully-bound MLE.
    ///
    /// Note that this function panics if a particular challenge is neither
    /// fixed nor bound!
    pub fn get_bound_point(&self) -> Vec<F> {
        self.mle_indices()
            .iter()
            .map(|index| match index {
                MleIndex::Bound(chal, _) => *chal,
                MleIndex::Fixed(chal) => F::from(*chal as u64),
                _ => panic!("MLE index not bound"),
            })
            .collect()
    }
}

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
impl<F: FieldExt> ExpressionType<F> for VerifierExpr {
    type MLENodeRepr = VerifierMle<F>;
    type MleVec = ();
}

impl<F: FieldExt> Expression<F, VerifierExpr> {
    /// Create a mle Expression that contains one MLE
    pub fn mle(mle: VerifierMle<F>) -> Self {
        let mle_node = ExpressionNode::Mle(mle);

        Expression::new(mle_node, ())
    }

    /// Evaluate this fully bound expression.
    pub fn evaluate(&self) -> Result<F, ExpressionError> {
        let constant = |c| Ok(c);
        let selector_column = |idx: &MleIndex<F>,
                               lhs: Result<F, ExpressionError>,
                               rhs: Result<F, ExpressionError>|
         -> Result<F, ExpressionError> {
            // --- Selector bit must be bound ---
            if let MleIndex::Bound(val, _) = idx {
                return Ok(*val * rhs? + (F::ONE - val) * lhs?);
            }
            dbg!("It was here three");
            Err(ExpressionError::SelectorBitNotBoundError)
        };
        let mle_eval =
            |verifier_mle: &VerifierMle<F>| -> Result<F, ExpressionError> { Ok(verifier_mle.eval) };
        let negated = |val: Result<F, ExpressionError>| Ok((val?).neg());
        let sum =
            |lhs: Result<F, ExpressionError>, rhs: Result<F, ExpressionError>| Ok(lhs? + rhs?);
        let product = |verifier_mles: &[VerifierMle<F>]| -> Result<F, ExpressionError> {
            verifier_mles
                .iter()
                .try_fold(F::ONE, |acc, verifier_mle| Ok(acc * verifier_mle.eval))
        };
        let scaled = |val: Result<F, ExpressionError>, scalar: F| Ok(val? * scalar);

        self.expression_node.reduce(
            &constant,
            &selector_column,
            &mle_eval,
            &negated,
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
            expression_node.get_all_nonlinear_rounds(&mut vec![], mle_vec);
        nonlinear_rounds.sort();
        nonlinear_rounds
    }

    /// Get the [PostSumcheckLayer] for this expression, which represents the fully bound values of the expression.
    pub fn get_post_sumcheck_layer(&self, multiplier: F) -> PostSumcheckLayer<F, F> {
        self.expression_node
            .get_post_sumcheck_layer(multiplier, &self.mle_vec)
    }
}

impl<F: FieldExt> ExpressionNode<F, VerifierExpr> {
    /// Evaluate the polynomial using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce<T>(
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
            ExpressionNode::Selector(index, a, b) => {
                let lhs = a.reduce(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                let rhs = b.reduce(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                selector_column(index, lhs, rhs)
            }
            ExpressionNode::Mle(query) => mle_eval(query),
            ExpressionNode::Negated(a) => {
                let a = a.reduce(
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
                let a = a.reduce(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                let b = b.reduce(
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
                let a = a.reduce(
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

    /// Recursively get the [PostSumcheckLayer] for an Expression node, which is the fully bound
    /// representation of an expression.
    pub fn get_post_sumcheck_layer(
        &self,
        multiplier: F,
        mle_vec: &<VerifierExpr as ExpressionType<F>>::MleVec,
    ) -> PostSumcheckLayer<F, F> {
        let mut products: Vec<Product<F, F>> = vec![];
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                let left_side_acc = multiplier * (F::ONE - mle_index.val().unwrap());
                let right_side_acc = multiplier * (mle_index.val().unwrap());
                products.extend(a.get_post_sumcheck_layer(left_side_acc, mle_vec).0);
                products.extend(b.get_post_sumcheck_layer(right_side_acc, mle_vec).0);
            }
            ExpressionNode::Sum(a, b) => {
                products.extend(a.get_post_sumcheck_layer(multiplier, mle_vec).0);
                products.extend(b.get_post_sumcheck_layer(multiplier, mle_vec).0);
            }
            ExpressionNode::Mle(mle) => {
                products.push(Product::<F, F>::new_from_verifier_mle(
                    &[mle.clone()],
                    multiplier,
                ));
            }
            ExpressionNode::Product(mles) => {
                let product = Product::<F, F>::new_from_verifier_mle(&mles, multiplier);
                products.push(product);
            }
            ExpressionNode::Scaled(a, scale_factor) => {
                let acc = multiplier * scale_factor;
                products.extend(a.get_post_sumcheck_layer(acc, mle_vec).0);
            }
            ExpressionNode::Negated(a) => {
                let acc = multiplier.neg();
                products.extend(a.get_post_sumcheck_layer(acc, mle_vec).0);
            }
            ExpressionNode::Constant(constant) => {
                products.push(Product::<F, F>::new(&[], *constant * multiplier));
            }
        }
        PostSumcheckLayer(products)
    }
}

impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for Expression<F, VerifierExpr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Circuit Expression")
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
                f.debug_struct("Circuit Mle").field("mle", mle_ref).finish()
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
