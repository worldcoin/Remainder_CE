use crate::{
    layer::{
        product::{PostSumcheckLayer, Product},
        LayerId,
    },
    layouter::nodes::CircuitNode,
    mle::{dense::DenseMle, Mle, MleIndex},
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    collections::{HashMap, HashSet},
    fmt::Debug,
};

use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptSponge, VerifierTranscript},
    FieldExt,
};

use super::{
    expr_errors::ExpressionError,
    generic_expr::{Expression, ExpressionNode, ExpressionType},
    verifier_expr::{VerifierExpr, VerifierMle},
};

/// A metadata-only version of [crate::mle::dense::DenseMle] used in the Circuit
/// Descrption.  A [CircuitMle] is stored in the leaves of an `Expression<F,
/// CircuitExpr>` tree.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(bound = "F: FieldExt")]
pub struct CircuitMle<F: FieldExt> {
    /// Layer whose data this MLE is a subset of.
    layer_id: LayerId,

    /// A list of indices where the free variables have been assigned an index.
    var_indices: Vec<MleIndex<F>>,
}

impl<F: FieldExt> CircuitNode for CircuitMle<F> {
    fn id(&self) -> crate::layouter::nodes::NodeId {
        todo!()
    }

    fn sources(&self) -> Vec<crate::layouter::nodes::NodeId> {
        todo!()
    }

    fn get_num_vars(&self) -> usize {
        todo!()
    }
}

impl<F: FieldExt> CircuitMle<F> {
    pub fn new(layer_id: LayerId, var_indices: &[MleIndex<F>]) -> Self {
        Self {
            layer_id,
            var_indices: var_indices.to_vec(),
        }
    }

    pub fn set_mle_indices(&mut self, new_mle_indices: Vec<MleIndex<F>>) {
        self.var_indices = new_mle_indices;
    }

    pub fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    pub fn mle_indices(&self) -> &[MleIndex<F>] {
        &self.var_indices
    }

    pub fn num_iterated_vars(&self) -> usize {
        self.var_indices.iter().fold(0, |acc, idx| {
            acc + match idx {
                MleIndex::IndexedBit(_) => 1,
                _ => 0,
            }
        })
    }

    // Bind the variable with index `var_index` to `value`.
    pub fn fix_variable(&mut self, var_index: usize, value: F) {
        for mle_index in self.var_indices.iter_mut() {
            if *mle_index == MleIndex::IndexedBit(var_index) {
                mle_index.bind_index(value);
            }
        }
    }

    /// Gets the values of the bound and fixed MLE indices of this MLE, panicking if the MLE is not
    /// fully bound.
    pub fn get_claim_point(&self, challenges: &[F]) -> Vec<F> {
        self.var_indices
            .iter()
            .map(|index| match index {
                MleIndex::Bound(chal, _idx) => *chal,
                MleIndex::Fixed(chal) => F::from(*chal as u64),
                MleIndex::IndexedBit(i) => challenges[*i],
                _ => panic!("DenseMleRefDesc contained iterated bit!"),
            })
            .collect()
    }

    pub fn into_verifier_mle(
        &self,
        point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierMle<F>, ExpressionError> {
        let verifier_indices = self
            .var_indices
            .iter()
            .map(|mle_index| match mle_index {
                MleIndex::IndexedBit(idx) => Ok(MleIndex::Bound(point[*idx], *idx)),
                MleIndex::Fixed(val) => Ok(MleIndex::Fixed(*val)),
                _ => Err(ExpressionError::SelectorBitNotBoundError),
            })
            .collect::<Result<Vec<MleIndex<F>>, ExpressionError>>()?;

        let eval = transcript_reader
            .consume_element("MLE evaluation")
            .map_err(|err| ExpressionError::TranscriptError(err))?;

        Ok(VerifierMle::new(self.layer_id, verifier_indices, eval))
    }

    /// Generate a [CircuitMle] from a [DenseMle].
    pub fn from_dense_mle(mle: &DenseMle<F>) -> Result<Self, ExpressionError> {
        let layer_id = mle.get_layer_id();
        let mle_indices = mle.mle_indices();

        let all_indices_indexed = mle_indices.iter().all(|mle_index| match mle_index {
            MleIndex::IndexedBit(_) => true,
            MleIndex::Fixed(_) => true,
            MleIndex::Bound(..) => true,
            _ => false,
        });

        if !all_indices_indexed {
            return Err(ExpressionError::EvaluateNotFullyIndexedError);
        }

        Ok(CircuitMle::new(layer_id, mle_indices))
    }
}

/// Placeholder type for defining `Expression<F, CircuitExpr>`, the type used
/// for representing expressions in the circuit description.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CircuitExpr;

// The leaves of an expression of this type contain a [CircuitMle], an analogue
// of [crate::mle::dense::DenseMle], storing only metadata related to the MLE,
// without any evaluations.
// TODO(Makis): Consider allowing for re-use of MLEs, like in a [ProverExpr]:
// ```ignore
//     type MLENodeRepr = usize,
//     type MleVec = Vec<CircuitMle<F>>,
// ```
impl<F: FieldExt> ExpressionType<F> for CircuitExpr {
    type MLENodeRepr = CircuitMle<F>;
    type MleVec = ();
}

impl<F: FieldExt> Expression<F, CircuitExpr> {
    /// Binds the variables of this expression to `point`, and retrieves the
    /// leaf MLE values from the `transcript_reader`.  Returns a `Expression<F,
    /// VerifierExpr>` version of `self`.
    pub fn bind(
        &self,
        point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Expression<F, VerifierExpr>, ExpressionError> {
        Ok(Expression::new(
            self.expression_node
                .into_verifier_node(point, transcript_reader)?,
            (),
        ))
    }

    /// Traverses the expression tree to get the indices of all the nonlinear
    /// rounds. Returns a sorted vector of indices.
    pub fn get_all_nonlinear_rounds(&self) -> Vec<usize> {
        self.expression_node
            .get_all_nonlinear_rounds(&mut vec![], &self.mle_vec)
            .into_iter()
            .sorted()
            .collect()
    }

    /// Get the [PostSumcheckLayer] for this expression, which represents the fully bound values of the expression.
    pub fn get_post_sumcheck_layer(
        &self,
        multiplier: F,
        challenges: &[F],
    ) -> PostSumcheckLayer<F, Option<F>> {
        self.expression_node
            .get_post_sumcheck_layer(multiplier, challenges, &self.mle_vec)
    }

    /// Get the maximum degree of any variable in this expression.
    pub fn get_max_degree(&self) -> usize {
        self.expression_node.get_max_degree(&self.mle_vec)
    }

    /// Returns the maximum degree of b_{curr_round} within an expression
    /// (and therefore the number of prover messages we need to send)
    pub fn get_round_degree(&self, curr_round: usize) -> usize {
        // --- By default, all rounds have degree at least 2 (beta table included) ---
        let mut round_degree = 1;

        let mut get_degree_closure = |expr: &ExpressionNode<F, CircuitExpr>,
                                      mle_vec: &<CircuitExpr as ExpressionType<F>>::MleVec|
         -> Result<(), ()> {
            let round_degree = &mut round_degree;

            // --- The only exception is within a product of MLEs ---
            if let ExpressionNode::Product(circuit_mles) = expr {
                let mut product_round_degree: usize = 0;
                for circuit_mle in circuit_mles {
                    let mle_indices = &circuit_mle.var_indices;
                    for mle_index in mle_indices {
                        if *mle_index == MleIndex::IndexedBit(curr_round) {
                            product_round_degree += 1;
                            break;
                        }
                    }
                }
                if *round_degree < product_round_degree {
                    *round_degree = product_round_degree;
                }
            }
            Ok(())
        };

        self.traverse(&mut get_degree_closure).unwrap();
        // add 1 cuz beta table but idk if we would ever use this without a beta table
        round_degree + 1
    }
}

impl<F: FieldExt> ExpressionNode<F, CircuitExpr> {
    pub fn into_verifier_node(
        &self,
        point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<ExpressionNode<F, VerifierExpr>, ExpressionError> {
        match self {
            ExpressionNode::Constant(scalar) => Ok(ExpressionNode::Constant(*scalar)),
            ExpressionNode::Selector(index, lhs, rhs) => match index {
                MleIndex::IndexedBit(idx) => Ok(ExpressionNode::Selector(
                    MleIndex::Bound(point[*idx], *idx),
                    Box::new(lhs.into_verifier_node(point, transcript_reader)?),
                    Box::new(rhs.into_verifier_node(point, transcript_reader)?),
                )),
                _ => Err(ExpressionError::SelectorBitNotBoundError),
            },
            ExpressionNode::Mle(circuit_mle) => Ok(ExpressionNode::Mle(
                circuit_mle.into_verifier_mle(point, transcript_reader)?,
            )),
            ExpressionNode::Negated(a) => Ok(ExpressionNode::Negated(Box::new(
                a.into_verifier_node(point, transcript_reader)?,
            ))),
            ExpressionNode::Sum(lhs, rhs) => Ok(ExpressionNode::Sum(
                Box::new(lhs.into_verifier_node(point, transcript_reader)?),
                Box::new(rhs.into_verifier_node(point, transcript_reader)?),
            )),
            ExpressionNode::Product(circuit_mles) => {
                let verifier_mles: Vec<VerifierMle<F>> = circuit_mles
                    .iter()
                    .map(|circuit_mle| Ok(circuit_mle.into_verifier_mle(point, transcript_reader)?))
                    .collect::<Result<Vec<VerifierMle<F>>, ExpressionError>>()?;

                Ok(ExpressionNode::Product(verifier_mles))
            }
            ExpressionNode::Scaled(circuit_mle, scalar) => Ok(ExpressionNode::Scaled(
                Box::new(circuit_mle.into_verifier_node(point, transcript_reader)?),
                *scalar,
            )),
        }
    }

    /// Evaluate the polynomial using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce<T>(
        &self,
        constant: &mut impl FnMut(F) -> T,
        selector_column: &mut impl FnMut(&MleIndex<F>, T, T) -> T,
        mle_eval: &mut impl FnMut(&<CircuitExpr as ExpressionType<F>>::MLENodeRepr) -> T,
        negated: &mut impl FnMut(T) -> T,
        sum: &mut impl FnMut(T, T) -> T,
        product: &mut impl FnMut(&[<CircuitExpr as ExpressionType<F>>::MLENodeRepr]) -> T,
        scaled: &mut impl FnMut(T, F) -> T,
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
        mle_vec: &<CircuitExpr as ExpressionType<F>>::MleVec,
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
                                } else if let MleIndex::Bound(_, i) = mle_index {
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
        challenges: &[F],
        mle_vec: &<VerifierExpr as ExpressionType<F>>::MleVec,
    ) -> PostSumcheckLayer<F, Option<F>> {
        let mut products: Vec<Product<F, Option<F>>> = vec![];
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                let idx_val = match mle_index {
                    MleIndex::IndexedBit(idx) => challenges[*idx],
                    MleIndex::Bound(chal, _idx) => *chal,
                    // TODO(vishady): actually we should just have an assertion that circuit description only
                    // contains indexed bits
                    _ => panic!("should not have any other index here"),
                };
                let left_side_acc = multiplier * (F::ONE - idx_val);
                let right_side_acc = multiplier * (idx_val);
                products.extend(
                    a.get_post_sumcheck_layer(left_side_acc, challenges, mle_vec)
                        .0,
                );
                products.extend(
                    b.get_post_sumcheck_layer(right_side_acc, challenges, mle_vec)
                        .0,
                );
            }
            ExpressionNode::Sum(a, b) => {
                products.extend(a.get_post_sumcheck_layer(multiplier, challenges, mle_vec).0);
                products.extend(b.get_post_sumcheck_layer(multiplier, challenges, mle_vec).0);
            }
            ExpressionNode::Mle(mle) => {
                products.push(Product::<F, Option<F>>::new(
                    &vec![mle.clone()],
                    multiplier,
                    challenges,
                ));
            }
            ExpressionNode::Product(mles) => {
                let product = Product::<F, Option<F>>::new(mles, multiplier, challenges);
                products.push(product);
            }
            ExpressionNode::Scaled(a, scale_factor) => {
                let acc = multiplier * scale_factor;
                products.extend(a.get_post_sumcheck_layer(acc, challenges, mle_vec).0);
            }
            ExpressionNode::Negated(a) => {
                let acc = multiplier.neg();
                products.extend(a.get_post_sumcheck_layer(acc, challenges, mle_vec).0);
            }
            ExpressionNode::Constant(constant) => {
                products.push(Product::<F, Option<F>>::new(
                    &[],
                    *constant * multiplier,
                    challenges,
                ));
            }
        }
        PostSumcheckLayer(products)
    }

    /// Get the maximum degree of an ExpressionNode, recursively.
    fn get_max_degree(&self, mle_vec: &<CircuitExpr as ExpressionType<F>>::MleVec) -> usize {
        match self {
            ExpressionNode::Selector(_, a, b) | ExpressionNode::Sum(a, b) => {
                let a_degree = a.get_max_degree(mle_vec);
                let b_degree = b.get_max_degree(mle_vec);
                max(a_degree, b_degree)
            }
            ExpressionNode::Mle(_) => {
                // 1 for the current MLE
                1
            }
            ExpressionNode::Product(mle_refs) => {
                // max degree is the number of MLEs in a product
                mle_refs.len()
            }
            ExpressionNode::Scaled(a, _) | ExpressionNode::Negated(a) => a.get_max_degree(mle_vec),
            ExpressionNode::Constant(_) => 1,
        }
    }
}

impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for Expression<F, CircuitExpr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Circuit Expression")
            .field("Expression_Node", &self.expression_node)
            .finish()
    }
}

impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for ExpressionNode<F, CircuitExpr> {
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
