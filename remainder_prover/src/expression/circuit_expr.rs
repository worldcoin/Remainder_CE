//! The "pure" polynomial relationship description between the MLE representing
//! a single layer of a "structured" circuit and those representing data from
//! other layers. See documentation in [crate::expression] for more details.

use crate::{
    layer::{
        gate::BinaryOperation,
        product::{PostSumcheckLayer, Product},
    },
    layouter::layouting::CircuitMap,
    mle::{
        dense::DenseMle, evals::MultilinearExtension, mle_description::MleDescription, verifier_mle::VerifierMle, AbstractMle, MleIndex
    },
};
use ark_std::log2;
use itertools::Itertools;
use std::{
    cmp::max,
    ops::Neg,
};

use remainder_shared_types::{transcript::VerifierTranscript, Field};

use super::generic_expr::{Expression, ExpressionNode};

use anyhow::Result;

impl<F: Field> Expression<F, MleDescription<F>> {
    /// Binds the variables of this expression to `point`, and retrieves the
    /// leaf MLE values from the `transcript_reader`.  Returns a `Expression<F,
    /// VerifierMle<F>>` version of `self`.
    pub fn bind(
        &self,
        point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Expression<F, VerifierMle<F>>> {
        let (expression_node, mle_vec) = self.deconstruct();
        let verifier_mles = mle_vec.into_iter().map(|m|
            m.into_verifier_node(point, transcript_reader)?
        ).collect();
        Ok(Expression::new(
            expression_node,
            verifier_mles,
        ))
    }

    /// Convenience function which creates a trivial [Expression<F, MleDescription<F>>]
    /// referring to a single MLE.
    pub fn from_mle_desc(mle_desc: MleDescription<F>) -> Self {
        Self {
            expression_node: ExpressionNode::<F>::Mle(mle_desc),
            mle_vec: (),
        }
    }

    /// Traverses the expression tree to get the indices of all the nonlinear
    /// rounds. Returns a sorted vector of indices.
    pub fn get_all_nonlinear_rounds(&self) -> Vec<usize> {
        self.expression_node
            .get_all_nonlinear_rounds(&self.mle_vec)
            .into_iter()
            .sorted()
            .collect()
    }

    /// Traverses the expression tree to return all indices within the
    /// expression. Can only be used after indexing the expression.
    pub fn get_all_rounds(&self) -> Vec<usize> {
        self.expression_node
            .get_all_rounds(&self.mle_vec)
            .into_iter()
            .sorted()
            .collect()
    }

    /// Get the [MleDescription]s for this expression.
    pub fn get_circuit_mles(&self) -> Vec<&MleDescription<F>> {
        self.mle_vec.iter().collect()
    }

    /// Label the free variables in an expression.
    pub fn index_mle_vars(&mut self, start_index: usize) {
        self.expression_node.index_mle_vars(start_index);
    }

    /// Get the [Expression<F, DenseMle<F>>] corresponding to this [Expression<F, MleDescription<F>>] using the
    /// associated data in the [CircuitMap].
    pub fn into_prover_expression(&self, circuit_map: &CircuitMap<F>) -> Expression<F, DenseMle<F>> {
        let (expression_node, mle_vec) = self.deconstruct();
        let prover_mles = mle_vec.into_iter().map(|m|
            m.into_dense_mle(circuit_map)?
        ).collect();
        Ok(Expression::new(
            expression_node,
            prover_mles,
        ))
    }

    /// Get the [PostSumcheckLayer] for this expression, which represents the fully bound values of the expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
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
        // By default, all rounds have degree at least 2 (beta table included)
        let mut round_degree = 1;

        let mut get_degree_closure = |expr: &ExpressionNode<F, MleDescription<F>>,
                                      _mle_vec: &[MleDescription<F>]|
         -> Result<()> {
            let round_degree = &mut round_degree;

            // The only exception is within a product of MLEs
            if let ExpressionNode::Product(circuit_mles) = expr {
                let mut product_round_degree: usize = 0;
                for circuit_mle in circuit_mles {
                    let mle_indices = circuit_mle.mle_indices();
                    for mle_index in mle_indices {
                        if *mle_index == MleIndex::Indexed(curr_round) {
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

impl<F: Field> ExpressionNode<F> {
    /// Compute the expression-wise bookkeeping table (coefficients of the MLE representing the expression)
    /// for a given [ExprDescription]. This uses a [CircuitMap] in order to grab the correct data
    /// corresponding to the [MleDescription].
    pub fn compute_bookkeeping_table(
        &self,
        circuit_map: &CircuitMap<F>,
    ) -> Option<MultilinearExtension<F>> {
        let output_data: Option<MultilinearExtension<F>> = match self {
            ExpressionNode::Mle(circuit_mle) => {
                let maybe_mle = circuit_map.get_data_from_circuit_mle(circuit_mle);
                if let Ok(mle) = maybe_mle {
                    Some(mle.clone())
                } else {
                    return None;
                }
            }
            ExpressionNode::Product(circuit_mles) => {
                let mle_bookkeeping_tables = circuit_mles
                    .iter()
                    .map(|circuit_mle| {
                        circuit_map
                            .get_data_from_circuit_mle(circuit_mle) // Returns Result
                            .map(|data| data.to_vec()) // Map Ok value to slice
                    })
                    .collect::<Result<Vec<Vec<F>>>>() // Collect all into a Result
                    .ok()?;
                Some(evaluate_bookkeeping_tables_given_operation(
                    &mle_bookkeeping_tables,
                    BinaryOperation::Mul,
                ))
            }
            ExpressionNode::Sum(a, b) => {
                let a_bookkeeping_table = a.compute_bookkeeping_table(circuit_map)?;
                let b_bookkeeping_table = b.compute_bookkeeping_table(circuit_map)?;
                Some(evaluate_bookkeeping_tables_given_operation(
                    &[
                        (a_bookkeeping_table.to_vec()),
                        (b_bookkeeping_table.to_vec()),
                    ],
                    BinaryOperation::Add,
                ))
            }
            ExpressionNode::Scaled(a, scale) => {
                let a_bookkeeping_table = a.compute_bookkeeping_table(circuit_map)?;
                Some(MultilinearExtension::new(
                    a_bookkeeping_table
                        .iter()
                        .map(|elem| elem * scale)
                        .collect_vec(),
                ))
            }
            ExpressionNode::Selector(_mle_index, a, b) => {
                let a_bookkeeping_table = a.compute_bookkeeping_table(circuit_map)?;
                let b_bookkeeping_table = b.compute_bookkeeping_table(circuit_map)?;
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
            ExpressionNode::Constant(value) => Some(MultilinearExtension::new(vec![*value])),
        };

        output_data
    }

    /// Label the MLE indices of an expression, starting from the `start_index`.
    pub fn index_mle_vars(&mut self, start_index: usize) {
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                match mle_index {
                    MleIndex::Free => *mle_index = MleIndex::Indexed(start_index),
                    MleIndex::Fixed(_bit) => {}
                    _ => panic!("should not have indexed or bound bits at this point!"),
                };
                a.index_mle_vars(start_index + 1);
                b.index_mle_vars(start_index + 1);
            }
            ExpressionNode::Sum(a, b) => {
                a.index_mle_vars(start_index);
                b.index_mle_vars(start_index);
            }
            ExpressionNode::Mle(mle) => {
                mle.index_mle_indices(start_index);
            }
            ExpressionNode::Product(mles) => {
                mles.iter_mut()
                    .for_each(|mle| mle.index_mle_indices(start_index));
            }
            ExpressionNode::Scaled(a, _scale_factor) => {
                a.index_mle_vars(start_index);
            }
            ExpressionNode::Constant(_constant) => {}
        }
    }

    /// Recursively get the [PostSumcheckLayer] for an Expression node, which is the fully bound
    /// representation of an expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    pub fn get_post_sumcheck_layer(
        &self,
        multiplier: F,
        challenges: &[F],
        _mle_vec: &[VerifierMle<F>],
    ) -> PostSumcheckLayer<F, Option<F>> {
        let mut products: Vec<Product<F, Option<F>>> = vec![];
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                let idx_val = match mle_index {
                    MleIndex::Indexed(idx) => challenges[*idx],
                    MleIndex::Bound(chal, _idx) => *chal,
                    // TODO(vishady): actually we should just have an assertion that circuit description only
                    // contains indexed bits
                    _ => panic!("should not have any other index here"),
                };
                let left_side_acc = multiplier * (F::ONE - idx_val);
                let right_side_acc = multiplier * (idx_val);
                products.extend(
                    a.get_post_sumcheck_layer(left_side_acc, challenges, _mle_vec)
                        .0,
                );
                products.extend(
                    b.get_post_sumcheck_layer(right_side_acc, challenges, _mle_vec)
                        .0,
                );
            }
            ExpressionNode::Sum(a, b) => {
                products.extend(
                    a.get_post_sumcheck_layer(multiplier, challenges, _mle_vec)
                        .0,
                );
                products.extend(
                    b.get_post_sumcheck_layer(multiplier, challenges, _mle_vec)
                        .0,
                );
            }
            ExpressionNode::Mle(mle) => {
                products.push(Product::<F, Option<F>>::new(
                    &[mle.clone()],
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
                products.extend(a.get_post_sumcheck_layer(acc, challenges, _mle_vec).0);
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

    /// Returns the total number of variables (i.e. number of rounds of sumcheck) within
    /// the MLE representing the output "data" of this particular expression.
    ///
    /// Note that unlike within the `AbstractExpr` case, we don't need to return
    /// a `Result` since all MLEs extentiating `AbstractMle` are instantiated with their
    /// appropriate number of variables.
    fn get_num_vars(&self) -> usize {
        match self {
            ExpressionNode::Constant(_) => 0,
            ExpressionNode::Selector(_, lhs, rhs) => {
                max(lhs.get_num_vars() + 1, rhs.get_num_vars() + 1)
            }
            ExpressionNode::Mle(circuit_mle_desc) => circuit_mle_desc.num_free_vars(),
            ExpressionNode::Sum(lhs, rhs) => max(lhs.get_num_vars(), rhs.get_num_vars()),
            ExpressionNode::Product(nodes) => nodes.iter().fold(0, |cur_max, circuit_mle_desc| {
                max(cur_max, circuit_mle_desc.num_free_vars())
            }),
            ExpressionNode::Scaled(expr, _) => expr.get_num_vars(),
        }
    }
}

impl<F: Field> Expression<F, MleDescription<F>> {
    /// Returns the total number of variables (i.e. number of rounds of sumcheck)
    /// within the MLE representing the output "data" of this particular expression.
    ///
    /// Note that unlike within the AbstractExpr case, we don't need to return
    /// a `Result` since all MLEs within a `ExprDescription` are instantiated with their appropriate number of variables.
    pub fn num_vars(&self) -> usize {
        self.expression_node.get_num_vars()
    }

    /// Creates an `Expression<F, ExprDescription>` which describes the polynomial relationship
    ///
    /// `circuit_mle_descs[0](x_1, ..., x_{n_0}) * circuit_mle_descs[1](x_1, ..., x_{n_1}) * ...`
    pub fn products(circuit_mle_descs: Vec<MleDescription<F>>) -> Self {
        let product_node = ExpressionNode::Product(circuit_mle_descs);

        Expression::new(product_node, ())
    }

    /// Creates an [Expression<F, ExprDescription>] which describes the polynomial relationship
    /// `(1 - x_0) * Self(x_1, ..., x_{n_lhs}) + b_0 * rhs(x_1, ..., x_{n_rhs})`
    ///
    /// NOTE that by default, performing a `select()` over an LHS and an RHS
    /// with different numbers of variables will create a selector tree such that
    /// the side with fewer variables always falls down the left-most side of
    /// that subtree.
    ///
    /// For example, if we are calling `select()` on two MLEs,
    /// V_i(x_0, ..., x_4) and V_i(x_0, ..., x_6)
    /// then the resulting expression will have a single top-level selector, and
    /// will forcibly move the first MLE (with two fewer variables) to the left-most
    /// subtree with 5 variables:
    /// (1 - x_0) * (1 - x_1) * (1 - x_2) * V_i(x_3, ..., x_7) +
    /// x_0 * V_i(x_1, ..., x_7)
    pub fn select(self, rhs: Expression<F, MleDescription<F>>) -> Self {
        let (lhs_node, _) = self.deconstruct();
        let (rhs_node, _) = rhs.deconstruct();

        // Compute the difference in number of free variables, to add the appropriate number of selectors
        let num_left_selectors = max(0, rhs_node.get_num_vars() - lhs_node.get_num_vars());
        let num_right_selectors = max(0, lhs_node.get_num_vars() - rhs_node.get_num_vars());

        let lhs_subtree = if num_left_selectors > 0 {
            // Always "go left" and "select" against a constant zero
            (0..num_left_selectors).fold(lhs_node, |cur_subtree, _| {
                ExpressionNode::Selector(
                    MleIndex::Free,
                    Box::new(cur_subtree),
                    Box::new(ExpressionNode::Constant(F::ZERO)),
                )
            })
        } else {
            lhs_node
        };

        let rhs_subtree = if num_right_selectors > 0 {
            // Always "go left" and "select" against a constant zero
            (0..num_right_selectors).fold(rhs_node, |cur_subtree, _| {
                ExpressionNode::Selector(
                    MleIndex::Free,
                    Box::new(cur_subtree),
                    Box::new(ExpressionNode::Constant(F::ZERO)),
                )
            })
        } else {
            rhs_node
        };

        // Sanitycheck
        debug_assert_eq!(lhs_subtree.get_num_vars(), rhs_subtree.get_num_vars());

        // Finally, a selector against the two (equal-num-vars) sides!
        let concat_node =
            ExpressionNode::Selector(MleIndex::Free, Box::new(lhs_subtree), Box::new(rhs_subtree));

        Expression::new(concat_node, ())
    }

    /// Create a nested selector Expression that selects between 2^k Expressions
    /// by creating a binary tree of Selector Expressions.
    /// The order of the leaves is the order of the input expressions.
    /// (Note that this is very different from calling `select()` consecutively.)
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
                        MleIndex::Free,
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

        let mle_node = ExpressionNode::Scaled(Box::new(node), F::from(1).neg());

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
    pub fn scaled(expression: Expression<F, MleDescription<F>>, scale: F) -> Self {
        let (node, _) = expression.deconstruct();

        Expression::new(ExpressionNode::Scaled(Box::new(node), scale), ())
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