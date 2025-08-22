//! The "pure" polynomial relationship description between the MLE representing
//! a single layer of a "structured" circuit and those representing data from
//! other layers. See documentation in [crate::expression] for more details.

use crate::{
    circuit_layout::CircuitEvalMap,
    expression::expr_errors::ExpressionError,
    layer::{
        gate::BinaryOperation,
        product::{PostSumcheckLayer, Product},
    },
    mle::{
        dense::DenseMle, evals::MultilinearExtension, mle_description::MleDescription,
        verifier_mle::VerifierMle, MleIndex,
    },
};
use ark_std::log2;
use itertools::Itertools;

use remainder_shared_types::{
    extension_field::ExtensionField, transcript::VerifierTranscript, Field,
};

use super::generic_expr::{Expression, ExpressionNode};

use anyhow::{anyhow, Result};

impl<F: Field> Expression<F, MleDescription> {
    /// Binds the variables of this expression to `point`, and retrieves the
    /// leaf MLE values from the `transcript_reader`.  
    /// Returns a [Expression<E, VerifierMle<E>>] version of [self].
    pub fn bind<E>(
        &self,
        point: &[E],
        transcript_reader: &mut impl VerifierTranscript<E::BaseField>,
    ) -> Result<(Expression<F, VerifierMle<E>>, Vec<Option<E>>)>
    where
        E: ExtensionField<BaseField = F>,
    {
        // Bind selector and MLE indices
        let (mut expression_node, mle_vec) = self.clone().deconstruct();
        expression_node.bind_selector(point);

        let verifier_mles = mle_vec
            .into_iter()
            .map(|m| m.into_verifier_mle(point, transcript_reader))
            .collect::<Result<Vec<_>, _>>()?;
        let bind_list = point.iter().map(|p| Some(*p)).collect();
        Ok((Expression::new(expression_node, verifier_mles), bind_list))
    }

    /// Get the [MleDescription]s for this expression.
    pub fn get_circuit_mles(&self) -> Vec<&MleDescription> {
        self.mle_vec.iter().collect()
    }

    /// Label the free variables in an expression.
    pub fn index_mle_vars(&mut self, start_index: usize) {
        self.expression_node
            .index_mle_vars(start_index, &mut self.mle_vec);
    }

    /// Get the [Expression<E, DenseMle<E>>] corresponding to this [Expression<E, MleDescription>] using the
    /// associated data in the [CircuitMap].
    pub fn into_prover_expression<E>(
        &self,
        circuit_map: &CircuitEvalMap<E>,
    ) -> Expression<F, DenseMle<E>>
    where
        E: ExtensionField<BaseField = F>,
    {
        let (expression_node, mle_vec) = self.clone().deconstruct();
        let prover_mles: Vec<_> = mle_vec
            .into_iter()
            .map(|m| m.into_dense_mle(circuit_map))
            .collect();
        Expression::new(expression_node, prover_mles)
    }

    /// Get the [PostSumcheckLayer] for this expression, which represents the fully bound values of the expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    pub fn get_post_sumcheck_layer<E>(
        &self,
        multiplier: E,
        challenges: &[E],
    ) -> PostSumcheckLayer<E, Option<E>>
    where
        E: ExtensionField<BaseField = F>,
    {
        self.expression_node
            .get_post_sumcheck_layer_verifier(multiplier, challenges, &self.mle_vec)
    }
}

impl<F: Field> ExpressionNode<F> {
    /// Convert all selector to `Bound`
    pub fn bind_selector<E>(&mut self, _point: &[E])
    where
        E: ExtensionField<BaseField = F>,
    {
        let mut bind_selector_closure = |expr: &mut ExpressionNode<F>| -> Result<()> {
            match expr {
                ExpressionNode::Selector(index, ..) => match index {
                    MleIndex::Indexed(idx) => {
                        *index = MleIndex::Bound(*idx);
                        Ok(())
                    }
                    _ => Err(anyhow!(ExpressionError::SelectorBitNotBoundError)),
                },
                _ => Ok(()),
            }
        };

        self.traverse_node_mut(&mut bind_selector_closure).unwrap();
    }

    /// Compute the expression-wise bookkeeping table (coefficients of the MLE representing the expression)
    /// for a given [ExprDescription]. This uses a [CircuitMap] in order to grab the correct data
    /// corresponding to the [MleDescription].
    pub fn compute_bookkeeping_table<E>(
        &self,
        circuit_map: &CircuitEvalMap<E>,
        mle_vec: &[MleDescription],
    ) -> Option<MultilinearExtension<E>>
    where
        E: ExtensionField<BaseField = F>,
    {
        let output_data: Option<MultilinearExtension<E>> = match self {
            ExpressionNode::Mle(mle_vec_index) => {
                let maybe_mle =
                    circuit_map.get_data_from_circuit_mle(mle_vec_index.get_mle(mle_vec));
                if let Ok(mle) = maybe_mle {
                    Some(mle.clone())
                } else {
                    return None;
                }
            }
            ExpressionNode::Product(mle_vec_indices) => {
                let mle_bookkeeping_tables = mle_vec_indices
                    .iter()
                    .map(|mle_vec_index| {
                        circuit_map
                            .get_data_from_circuit_mle(mle_vec_index.get_mle(mle_vec)) // Returns Result
                            .map(|data| data.to_vec()) // Map Ok value to slice
                    })
                    .collect::<Result<Vec<Vec<E>>>>() // Collect all into a Result
                    .ok()?;
                Some(evaluate_bookkeeping_tables_given_operation(
                    &mle_bookkeeping_tables,
                    BinaryOperation::Mul,
                ))
            }
            ExpressionNode::Sum(a, b) => {
                let a_bookkeeping_table = a.compute_bookkeeping_table(circuit_map, mle_vec)?;
                let b_bookkeeping_table = b.compute_bookkeeping_table(circuit_map, mle_vec)?;
                Some(evaluate_bookkeeping_tables_given_operation(
                    &[
                        (a_bookkeeping_table.to_vec()),
                        (b_bookkeeping_table.to_vec()),
                    ],
                    BinaryOperation::Add,
                ))
            }
            ExpressionNode::Scaled(a, scale) => {
                let a_bookkeeping_table = a.compute_bookkeeping_table(circuit_map, mle_vec)?;
                Some(MultilinearExtension::new(
                    a_bookkeeping_table
                        .iter()
                        .map(|elem| elem * *scale)
                        .collect_vec(),
                ))
            }
            ExpressionNode::Selector(_mle_index, a, b) => {
                let a_bookkeeping_table = a.compute_bookkeeping_table(circuit_map, mle_vec)?;
                let b_bookkeeping_table = b.compute_bookkeeping_table(circuit_map, mle_vec)?;
                assert_eq!(
                    a_bookkeeping_table.num_vars(),
                    b_bookkeeping_table.num_vars()
                );
                Some(MultilinearExtension::new(
                    a_bookkeeping_table
                        .iter()
                        .chain(b_bookkeeping_table.iter())
                        .collect_vec(),
                ))
            }
            ExpressionNode::Constant(value) => {
                Some(MultilinearExtension::new(vec![(*value).into()]))
            }
        };

        output_data
    }

    /// Label the MLE indices of an expression, starting from the `start_index`.
    pub fn index_mle_vars(&mut self, start_index: usize, mle_vec: &mut [MleDescription]) {
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                match mle_index {
                    MleIndex::Free => *mle_index = MleIndex::Indexed(start_index),
                    MleIndex::Fixed(_bit) => {}
                    _ => panic!("should not have indexed or bound bits at this point!"),
                };
                a.index_mle_vars(start_index + 1, mle_vec);
                b.index_mle_vars(start_index + 1, mle_vec);
            }
            ExpressionNode::Sum(a, b) => {
                a.index_mle_vars(start_index, mle_vec);
                b.index_mle_vars(start_index, mle_vec);
            }
            ExpressionNode::Mle(index_vec) => {
                index_vec
                    .get_mle_mut(mle_vec)
                    .index_mle_indices(start_index);
            }
            ExpressionNode::Product(index_vecs) => index_vecs
                .iter()
                .for_each(|i| i.get_mle_mut(mle_vec).index_mle_indices(start_index)),
            ExpressionNode::Scaled(a, _scale_factor) => {
                a.index_mle_vars(start_index, mle_vec);
            }
            ExpressionNode::Constant(_constant) => {}
        }
    }

    /// Recursively get the [PostSumcheckLayer] for an Expression node, which is the fully bound
    /// representation of an expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    pub fn get_post_sumcheck_layer_verifier<E>(
        &self,
        multiplier: E,
        challenges: &[E],
        mle_vec: &[MleDescription],
    ) -> PostSumcheckLayer<E, Option<E>>
    where
        E: ExtensionField<BaseField = F>,
    {
        let mut products: Vec<Product<E, Option<E>>> = vec![];
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                let idx_val = match mle_index {
                    MleIndex::Indexed(idx) => challenges[*idx],
                    // TODO(vishady): actually we should just have an assertion that circuit description only
                    // contains indexed bits
                    _ => panic!("should not have any other index here"),
                };
                let left_side_acc = multiplier * (E::ONE - idx_val);
                let right_side_acc = multiplier * (idx_val);
                products.extend(
                    a.get_post_sumcheck_layer_verifier(left_side_acc, challenges, mle_vec)
                        .0,
                );
                products.extend(
                    b.get_post_sumcheck_layer_verifier(right_side_acc, challenges, mle_vec)
                        .0,
                );
            }
            ExpressionNode::Sum(a, b) => {
                products.extend(
                    a.get_post_sumcheck_layer_verifier(multiplier, challenges, mle_vec)
                        .0,
                );
                products.extend(
                    b.get_post_sumcheck_layer_verifier(multiplier, challenges, mle_vec)
                        .0,
                );
            }
            ExpressionNode::Mle(mle) => {
                products.push(Product::<E, Option<E>>::new(
                    &[mle.get_mle(mle_vec).clone()],
                    &Vec::new(),
                    multiplier,
                    challenges,
                ));
            }
            ExpressionNode::Product(mles) => {
                let product = Product::<E, Option<E>>::new(
                    &mles
                        .iter()
                        .map(|m| m.get_mle(mle_vec).clone())
                        .collect::<Vec<_>>(),
                    &Vec::new(),
                    multiplier,
                    challenges,
                );
                products.push(product);
            }
            ExpressionNode::Scaled(a, scale_factor) => {
                let acc = multiplier * *scale_factor;
                products.extend(
                    a.get_post_sumcheck_layer_verifier(acc, challenges, mle_vec)
                        .0,
                );
            }
            ExpressionNode::Constant(constant) => {
                products.push(Product::<E, Option<E>>::new(
                    &[],
                    &Vec::new(),
                    multiplier * *constant,
                    challenges,
                ));
            }
        }
        PostSumcheckLayer(products)
    }
}

/// Given a bookkeeping table, use the according prefix bits in order
/// to filter it to the correct "view" that we want to see, assuming
/// that the prefix bits are the most significant bits, and that
/// the bookkeeping tables are stored in "big endian" format.
pub fn filter_bookkeeping_table<F: Field>(
    bookkeeping_table: &MultilinearExtension<F>,
    unfiltered_prefix_bits: &[bool],
) -> MultilinearExtension<F> {
    let current_table = bookkeeping_table.to_vec();
    let mut current_table_len = current_table.len();
    let filtered_table = unfiltered_prefix_bits
        .iter()
        .fold(current_table, |acc, bit| {
            let acc = if *bit {
                acc.into_iter().skip(current_table_len / 2).collect_vec()
            } else {
                acc.into_iter().take(current_table_len / 2).collect_vec()
            };
            current_table_len /= 2;
            acc
        });
    MultilinearExtension::new(filtered_table)
}

/// Evaluate the bookkeeping tables by applying the element-wise operation,
/// which can either be addition or multiplication.
pub(crate) fn evaluate_bookkeeping_tables_given_operation<F: Field>(
    mle_bookkeeping_tables: &[Vec<F>],
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
                    let multiple = (1 << max_num_vars) / max;
                    index / multiple
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
