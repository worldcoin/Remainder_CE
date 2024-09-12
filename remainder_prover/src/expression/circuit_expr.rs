use crate::{
    layer::{
        gate::BinaryOperation,
        product::{PostSumcheckLayer, Product},
        LayerId,
    },
    layouter::{
        layouting::{CircuitLocation, CircuitMap},
        nodes::CircuitNode,
    },
    mle::{dense::DenseMle, evals::MultilinearExtension, Mle, MleIndex},
};
use ark_std::log2;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    collections::{HashMap, HashSet},
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptSponge, VerifierTranscript},
    FieldExt,
};

use super::{
    expr_errors::ExpressionError,
    generic_expr::{Expression, ExpressionNode, ExpressionType},
    prover_expr::ProverExpr,
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

    pub fn index_mle_indices(&mut self, start_index: usize) {
        let mut index_counter = start_index;
        self.var_indices
            .iter_mut()
            .for_each(|mle_index| match mle_index {
                MleIndex::Iterated => {
                    let indexed_mle_index = MleIndex::IndexedBit(index_counter);
                    index_counter += 1;
                    *mle_index = indexed_mle_index;
                }
                MleIndex::Fixed(_bit) => {}
                _ => panic!("We should not have indexed or bound bits at this point!"),
            });
    }

    pub fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    pub fn mle_indices(&self) -> &[MleIndex<F>] {
        &self.var_indices
    }

    pub fn expression(self) -> Expression<F, CircuitExpr> {
        Expression::new(ExpressionNode::<F, CircuitExpr>::Mle(self), ())
    }

    pub fn num_iterated_vars(&self) -> usize {
        self.var_indices.iter().fold(0, |acc, idx| {
            acc + match idx {
                MleIndex::IndexedBit(_) => 1,
                _ => 0,
            }
        })
    }

    pub fn prefix_bits(&self) -> Vec<bool> {
        self.var_indices
            .iter()
            .filter_map(|idx| match idx {
                MleIndex::Fixed(bit) => Some(*bit),
                _ => None,
            })
            .collect()
    }

    pub fn into_dense_mle<'a>(&self, circuit_map: &CircuitMap<F>) -> DenseMle<F> {
        let data = circuit_map.get_data_from_circuit_mle(&self).unwrap();
        DenseMle::new_with_prefix_bits((*data).clone(), self.layer_id(), self.prefix_bits())
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

    /// Get the [CircuitMle]s for this expression, which are at the leaves of the expression.
    pub fn get_circuit_mles(&self) -> Vec<&CircuitMle<F>> {
        let circuit_mles = self.expression_node.get_circuit_mles();
        circuit_mles
    }

    pub fn index_mle_indices(&mut self, start_index: usize) {
        self.expression_node.index_mle_indices(start_index);
    }

    /// Get the [Expression<F, ProverExpr>] corresponding to this [Expression<F, CircuitExpr>] using the
    /// associated data in the [CircuitMap].
    pub fn into_prover_expression<'a>(
        &self,
        circuit_map: &CircuitMap<F>,
    ) -> Expression<F, ProverExpr> {
        let circuit_mles = self.get_circuit_mles();

        self.expression_node.into_prover_expression(&circuit_map)
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

    pub fn compute_bookkeeping_table(
        &self,
        circuit_map: &CircuitMap<F>,
    ) -> MultilinearExtension<F> {
        let output_data: MultilinearExtension<F> = match self {
            ExpressionNode::Mle(circuit_mle) => {
                let mle = circuit_map.get_data_from_circuit_mle(circuit_mle);
                mle.unwrap().clone()
            }
            ExpressionNode::Product(circuit_mles) => {
                let mle_bookkeeping_tables = circuit_mles
                    .iter()
                    .map(|circuit_mle| {
                        circuit_map
                            .get_data_from_circuit_mle(circuit_mle)
                            .unwrap()
                            .get_evals_vector()
                            .as_slice()
                    })
                    .collect_vec();
                evaluate_bookkeeping_tables_given_operation(
                    &mle_bookkeeping_tables,
                    BinaryOperation::Mul,
                )
            }
            ExpressionNode::Sum(a, b) => {
                let a_bookkeeping_table = a.compute_bookkeeping_table(circuit_map);
                let b_bookkeeping_table = b.compute_bookkeeping_table(circuit_map);
                evaluate_bookkeeping_tables_given_operation(
                    &[
                        &a_bookkeeping_table.get_evals_vector(),
                        &b_bookkeeping_table.get_evals_vector(),
                    ],
                    BinaryOperation::Add,
                )
            }
            ExpressionNode::Negated(a) => {
                let a_bookkeeping_table = a.compute_bookkeeping_table(circuit_map);
                MultilinearExtension::new(
                    a_bookkeeping_table
                        .get_evals_vector()
                        .iter()
                        .map(|elem| elem.neg())
                        .collect_vec(),
                )
            }
            ExpressionNode::Scaled(a, scale) => {
                let a_bookkeeping_table = a.compute_bookkeeping_table(circuit_map);
                MultilinearExtension::new(
                    a_bookkeeping_table
                        .get_evals_vector()
                        .iter()
                        .map(|elem| *elem * scale)
                        .collect_vec(),
                )
            }
            ExpressionNode::Selector(_mle_index, a, b) => {
                let a_bookkeeping_table = a.compute_bookkeeping_table(circuit_map);
                let b_bookkeeping_table = b.compute_bookkeeping_table(circuit_map);
                assert_eq!(
                    a_bookkeeping_table.num_vars(),
                    b_bookkeeping_table.num_vars()
                );
                MultilinearExtension::new(
                    a_bookkeeping_table
                        .get_evals_vector()
                        .iter()
                        .zip(b_bookkeeping_table.get_evals_vector())
                        .flat_map(|(a, b)| vec![*a, *b])
                        .collect_vec(),
                )
            }
            ExpressionNode::Constant(value) => MultilinearExtension::new(vec![*value]),
        };

        output_data
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

    /// Get all the [CircuitMle]s, recursively, for this expression by adding the MLEs in the leaves into the vector of CircuitMles.
    pub fn get_circuit_mles(&self) -> Vec<&CircuitMle<F>> {
        let mut circuit_mles: Vec<&CircuitMle<F>> = vec![];
        match self {
            ExpressionNode::Selector(_mle_index, a, b) => {
                circuit_mles.extend(a.get_circuit_mles());
                circuit_mles.extend(b.get_circuit_mles());
            }
            ExpressionNode::Sum(a, b) => {
                circuit_mles.extend(a.get_circuit_mles());
                circuit_mles.extend(b.get_circuit_mles());
            }
            ExpressionNode::Mle(mle) => {
                circuit_mles.push(mle);
            }
            ExpressionNode::Product(mles) => mles.iter().for_each(|mle| circuit_mles.push(mle)),
            ExpressionNode::Scaled(a, _scale_factor) => {
                circuit_mles.extend(a.get_circuit_mles());
            }
            ExpressionNode::Negated(a) => {
                circuit_mles.extend(a.get_circuit_mles());
            }
            ExpressionNode::Constant(_constant) => {}
        }
        circuit_mles
    }

    pub fn index_mle_indices(&mut self, start_index: usize) {
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                match mle_index {
                    MleIndex::Iterated => *mle_index = MleIndex::IndexedBit(start_index),
                    MleIndex::Fixed(_bit) => {}
                    _ => panic!("should not have indexed or bound bits at this point!"),
                };
                a.index_mle_indices(start_index + 1);
                b.index_mle_indices(start_index + 1);
            }
            ExpressionNode::Sum(a, b) => {
                a.index_mle_indices(start_index);
                b.index_mle_indices(start_index);
            }
            ExpressionNode::Mle(mle) => {
                mle.index_mle_indices(start_index);
            }
            ExpressionNode::Product(mles) => {
                mles.iter_mut()
                    .for_each(|mle| mle.index_mle_indices(start_index));
            }
            ExpressionNode::Scaled(a, scale_factor) => {
                a.index_mle_indices(start_index);
            }
            ExpressionNode::Negated(a) => {
                a.index_mle_indices(start_index);
            }
            ExpressionNode::Constant(_constant) => {}
        }
    }

    /// Get the [ExpressionNode<F, ProverExpr>] recursively, for this expression.
    pub fn into_prover_expression<'a>(
        &self,
        circuit_map: &CircuitMap<F>,
    ) -> Expression<F, ProverExpr> {
        match self {
            ExpressionNode::Selector(_mle_index, a, b) => b
                .into_prover_expression(&circuit_map)
                .concat_expr(a.into_prover_expression(&circuit_map)),
            ExpressionNode::Sum(a, b) => {
                a.into_prover_expression(&circuit_map) + b.into_prover_expression(&circuit_map)
            }
            ExpressionNode::Mle(mle) => {
                let prover_mle = mle.into_dense_mle(&circuit_map);
                prover_mle.expression()
            }
            ExpressionNode::Product(mles) => {
                let dense_mles = mles
                    .iter()
                    .map(|mle| mle.into_dense_mle(&circuit_map))
                    .collect_vec();
                Expression::<F, ProverExpr>::products(dense_mles)
            }
            ExpressionNode::Scaled(a, scale_factor) => {
                a.into_prover_expression(&circuit_map) * *scale_factor
            }
            ExpressionNode::Negated(a) => {
                Expression::<F, ProverExpr>::negated(a.into_prover_expression(&circuit_map))
            }
            ExpressionNode::Constant(constant) => Expression::<F, ProverExpr>::constant(*constant),
        }
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

    /// Returns the total number of variables (i.e. number of rounds of sumcheck) within
    /// the MLE representing the output "data" of this particular expression.
    ///
    /// Note that unlike within the `AbstractExpr` case, we don't need to return
    /// a `Result` since all MLEs within a `CircuitExpr` are instantiated with their
    /// appropriate number of variables.
    fn get_num_vars(&self) -> usize {
        match self {
            ExpressionNode::Constant(_) => 0,
            ExpressionNode::Selector(_, lhs, rhs) => {
                max(lhs.get_num_vars() + 1, rhs.get_num_vars() + 1)
            }
            ExpressionNode::Mle(circuit_mle_desc) => circuit_mle_desc.num_iterated_vars(),
            ExpressionNode::Negated(expr) => expr.get_num_vars(),
            ExpressionNode::Sum(lhs, rhs) => max(lhs.get_num_vars(), rhs.get_num_vars()),
            ExpressionNode::Product(nodes) => nodes.iter().fold(0, |cur_max, circuit_mle_desc| {
                max(cur_max, circuit_mle_desc.num_iterated_vars())
            }),
            ExpressionNode::Scaled(expr, _) => expr.get_num_vars(),
        }
    }
}

impl<F: FieldExt> Expression<F, CircuitExpr> {
    /// Computes the num_vars of this expression (how many rounds of sumcheck it would take to prove)
    /// TODO(ryancao): Aight figure out either where this is or write it
    // pub fn num_vars(&self, num_vars_map: &HashMap<NodeId, usize>) -> Result<usize, DAGError> {
    //     self.expression_node.get_num_vars(num_vars_map)
    // }

    /// Builds the ProverExpression using the AbstractExpression as a template.
    ///
    /// Gets the information the prover needs by consulting the CircuitMap to get
    /// the data and the prefix_bits
    ///
    /// TODO(ryancao): We'll do this during the data round!
    // pub fn build_prover_expr(
    //     self,
    //     circuit_map: &CircuitMap<'_, F>,
    // ) -> Result<Expression<F, ProverExpr>, DAGError> {
    //     // First we get all the mles that this expression will need to store
    //     let mut nodes = self.expression_node.get_node_ids(vec![]);
    //     nodes.sort();
    //     nodes.dedup();

    //     let mut node_map = HashMap::<NodeId, usize>::new();

    //     let mle_vec: Result<Vec<_>, _> = nodes
    //         .into_iter()
    //         .enumerate()
    //         .map(|(idx, node_id)| {
    //             let (location, data) = circuit_map.get_node(&node_id)?;

    //             let data = (*data).clone();

    //             let data = DenseMle::new_with_prefix_bits(
    //                 data,
    //                 location.layer_id,
    //                 location.prefix_bits.clone(),
    //             );

    //             node_map.insert(node_id, idx);
    //             Ok(data)
    //         })
    //         .collect();
    //     let mle_vec = mle_vec?;

    //     // Then we replace the NodeIds in the AbstractExpr w/ indices of our stored MLEs

    //     let expression_node = self.expression_node.build_prover_node(&node_map)?;

    //     Ok(Expression::<F, ProverExpr> {
    //         expression_node,
    //         mle_vec,
    //     })
    // }

    /// Returns the total number of variables (i.e. number of rounds of sumcheck)
    /// within the MLE representing the output "data" of this particular expression.
    ///
    /// Note that unlike within the AbstractExpr case, we don't need to return
    /// a `Result` since all MLEs within a `CircuitExpr` are instantiated with their appropriate number of variables.
    pub fn num_vars(&self) -> usize {
        self.expression_node.get_num_vars()
    }

    /// Creates an `Expression<F, CircuitExpr>` which describes the polynomial relationship
    ///
    /// `circuit_mle_descs[0](x_1, ..., x_{n_0}) * circuit_mle_descs[1](x_1, ..., x_{n_1}) * ...`
    pub fn products(circuit_mle_descs: Vec<CircuitMle<F>>) -> Self {
        let product_node = ExpressionNode::Product(circuit_mle_descs);

        Expression::new(product_node, ())
    }

    /// Creates an `Expression<F, CircuitExpr>` which describes the polynomial relationship
    /// TODO(ryancao): Change this so that `Self` is the `lhs`, rather than the other way around!
    ///
    /// `(1 - x_0) * lhs(x_1, ..., x_{n_lhs}) + b_0 * Self(x_1, ..., x_{n_rhs})`
    pub fn concat_expr(self, lhs: Expression<F, CircuitExpr>) -> Self {
        let (lhs_node, _) = lhs.deconstruct();
        let (rhs_node, _) = self.deconstruct();

        let concat_node =
            ExpressionNode::Selector(MleIndex::Iterated, Box::new(lhs_node), Box::new(rhs_node));

        Expression::new(concat_node, ())
    }

    /// Create a nested selector Expression that selects between 2^k Expressions
    /// by creating a binary tree of Selector Expressions.
    /// The order of the leaves is the order of the input expressions.
    /// (Note that this is very different from calling concat_expr consecutively.)
    /// See also [calculate_selector_values].
    pub fn selectors(expressions: Vec<Self>) -> Self {
        // Ensure length is a power of two
        assert!(expressions.len().is_power_of_two());
        let mut expressions = expressions;
        while expressions.len() > 1 {
            // Iterate over consecutive pairs of expressions, creating a new expression that selects between them
            expressions = expressions
                .into_iter()
                .tuples()
                .map(|(lhs, rhs)| {
                    let (lhs_node, _) = lhs.deconstruct();
                    let (rhs_node, _) = rhs.deconstruct();

                    let selector_node = ExpressionNode::Selector(
                        MleIndex::Iterated,
                        Box::new(lhs_node),
                        Box::new(rhs_node),
                    );

                    Expression::new(selector_node, ())
                })
                .collect();
        }
        expressions[0].clone()
    }

    /// Literally just `constant` as a term, but as an "`Expression`"
    pub fn constant(constant: F) -> Self {
        let mle_node = ExpressionNode::Constant(constant);

        Expression::new(mle_node, ())
    }

    /// Literally just `-expression`, as an "`Expression`"
    pub fn negated(expression: Self) -> Self {
        let (node, _) = expression.deconstruct();

        let mle_node = ExpressionNode::Negated(Box::new(node));

        Expression::new(mle_node, ())
    }

    /// Literally just `lhs` + `rhs`, as an "`Expression`"
    pub fn sum(lhs: Self, rhs: Self) -> Self {
        let (lhs_node, _) = lhs.deconstruct();
        let (rhs_node, _) = rhs.deconstruct();

        let sum_node = ExpressionNode::Sum(Box::new(lhs_node), Box::new(rhs_node));

        Expression::new(sum_node, ())
    }

    /// scales an Expression by a field element
    pub fn scaled(expression: Expression<F, CircuitExpr>, scale: F) -> Self {
        let (node, _) = expression.deconstruct();

        Expression::new(ExpressionNode::Scaled(Box::new(node), scale), ())
    }
}

fn evaluate_bookkeeping_tables_given_operation<F: FieldExt>(
    mle_bookkeeping_tables: &[&[F]],
    binary_operation: BinaryOperation,
) -> MultilinearExtension<F> {
    let max_num_vars = mle_bookkeeping_tables
        .iter()
        .map(|bookkeeping_table| log2(bookkeeping_table.len()))
        .max()
        .unwrap();

    let mut output_table = vec![F::ZERO; 1 << max_num_vars];
    (0..1 << (max_num_vars)).for_each(|index| {
        let evaluated_data_point = mle_bookkeeping_tables
            .iter()
            .map(|mle_bookkeeping_table| {
                let zero = F::ZERO;
                let index = if log2(mle_bookkeeping_table.len()) < max_num_vars {
                    let max = 1 << log2(mle_bookkeeping_table.len());
                    (index) % max
                } else {
                    index
                };
                let value = *mle_bookkeeping_table.get(index).unwrap_or(&zero);
                value
            })
            .reduce(|acc, value| binary_operation.perform_operation(acc, value))
            .unwrap();
        output_table[index] = evaluated_data_point;
    });
    MultilinearExtension::new(output_table)
}

impl<F: FieldExt> Neg for Expression<F, CircuitExpr> {
    type Output = Expression<F, CircuitExpr>;
    fn neg(self) -> Self::Output {
        Expression::<F, CircuitExpr>::negated(self)
    }
}

/// implement the Add, Sub, and Mul traits for the Expression
impl<F: FieldExt> Add for Expression<F, CircuitExpr> {
    type Output = Expression<F, CircuitExpr>;
    fn add(self, rhs: Expression<F, CircuitExpr>) -> Expression<F, CircuitExpr> {
        Expression::<F, CircuitExpr>::sum(self, rhs)
    }
}

impl<F: FieldExt> Sub for Expression<F, CircuitExpr> {
    type Output = Expression<F, CircuitExpr>;
    fn sub(self, rhs: Expression<F, CircuitExpr>) -> Expression<F, CircuitExpr> {
        self.add(rhs.neg())
    }
}

impl<F: FieldExt> Mul<F> for Expression<F, CircuitExpr> {
    type Output = Expression<F, CircuitExpr>;
    fn mul(self, rhs: F) -> Self::Output {
        Expression::<F, CircuitExpr>::scaled(self, rhs)
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
