//! The prover's view of an [Expression] -- see file-level documentation within
//! [crate::expression] for more details.
//!
//! Conceptually, [ProverExpr] contains the "circuit structure" between the
//! layer whose values are the output of the given expression and those whose
//! values are the inputs to the given expression, i.e. the polynomial
//! relationship between them, as well as (in an ownership sense) the actual
//! data, stored in [DenseMle]s.

use super::{
    circuit_expr::evaluate_bookkeeping_tables_given_operation,
    expr_errors::ExpressionError,
    generic_expr::{Expression, ExpressionNode},
};
use crate::{
    expression::generic_expr::MleVecIndex,
    layer::product::Product,
    mle::{betavalues::BetaValues, dense::DenseMle, AbstractMle, MleIndex},
    sumcheck::{
        apply_updated_beta_values_to_evals, beta_cascade, beta_cascade_no_independent_variable,
        SumcheckEvals,
    },
};
use crate::{
    layer::{gate::BinaryOperation, product::PostSumcheckLayer},
    mle::{verifier_mle::VerifierMle, Mle},
};
use itertools::{repeat_n, Itertools};
use remainder_shared_types::{extension_field::ExtensionField, Field};
use std::{cmp::max, collections::HashSet};

use anyhow::{anyhow, Ok, Result};
pub type ProverMle<F> = DenseMle<F>;

/// this is what the prover manipulates to prove the correctness of the computation.
/// Methods here include ones to fix bits, evaluate sumcheck messages, etc.
impl<F: Field, E> Expression<F, ProverMle<E>>
where
    E: ExtensionField<BaseField = F>,
{
    /// Create a product Expression that raises one MLE to a given power
    pub fn pow(pow: usize, mle: ProverMle<E>) -> Self {
        let mle_vec_indices = (0..pow).map(|_index| MleVecIndex::new(0)).collect_vec();
        let product_node = ExpressionNode::Product(mle_vec_indices);

        Expression::new(product_node, vec![mle])
    }

    /// Transforms the prover expression to a verifier expression.
    ///
    /// Should only be called when the entire expression is fully bound.
    ///
    /// Traverses the expression and changes the ProverMle to VerifierMle,
    /// by grabbing their bookkeeping table's 1st and only element.
    ///
    /// If the bookkeeping table has more than 1 element, it
    /// throws an ExpressionError::EvaluateNotFullyBoundError
    /// 
    /// Might be over-cautious here, but make sure the bind_list is fully assigned
    pub fn transform_to_verifier_expression(self, bind_list: &mut Vec<Option<E>>) -> Result<Expression<F, VerifierMle<E>>> {
        let (expression_node, mle_vec) = self.deconstruct();
        // Check that every MLE is fully bounded
        let verifier_mles = mle_vec
            .into_iter()
            .map(|m| {
                if !m.is_fully_bounded(bind_list) {
                    return Err(anyhow!(ExpressionError::EvaluateNotFullyBoundError));
                }
                Ok(VerifierMle::new(
                    m.layer_id(),
                    m.mle_indices().to_vec(),
                    m.value(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Expression::new(expression_node, verifier_mles))
    }

    /// fix the variable at a certain round index, always MSB index
    pub fn fix_variable(&mut self, round_index: usize, challenge: E, bind_list: &mut Vec<Option<E>>) {
        let (expression_node, mle_vec) = self.deconstruct_mut();

        expression_node.fix_variable_node(round_index, challenge, mle_vec, bind_list)
    }

    /// fix the variable at a certain round index, arbitrary index
    pub fn fix_variable_at_index(&mut self, round_index: usize, challenge: E, bind_list: &mut Vec<Option<E>>) {
        let (expression_node, mle_vec) = self.deconstruct_mut();

        expression_node.fix_variable_at_index_node(round_index, challenge, mle_vec, bind_list)
    }

    /// evaluates an expression on the given challenges points, by fixing the variables
    pub fn evaluate_expr(
        &mut self, 
        challenges: Vec<E>,
        bind_list: &mut Vec<Option<E>>,
    ) -> Result<E> {
        // It's as simple as fixing all variables
        challenges
            .iter()
            .enumerate()
            .for_each(|(round_idx, &challenge)| {
                self.fix_variable(round_idx, challenge, bind_list);
            });

        // ----- this is a check -----
        let mut observer_fn = |exp: &ExpressionNode<F>, mle_vec: &[ProverMle<E>]| -> Result<()> {
            match exp {
                ExpressionNode::Mle(mle_vec_idx) => {
                    let mle = mle_vec_idx.get_mle(mle_vec);
                    let indices = mle
                        .mle_indices()
                        .iter()
                        .filter_map(|index| match index {
                            MleIndex::Bound(index) => Some((bind_list[*index].unwrap(), index)),
                            _ => None,
                        })
                        .collect_vec();

                    let start = *indices[0].1;
                    let end = *indices[indices.len() - 1].1;

                    let (indices, _): (Vec<_>, Vec<usize>) = indices.into_iter().unzip();

                    if indices.as_slice() == &challenges[start..=end] {
                        Ok(())
                    } else {
                        Err(anyhow!(ExpressionError::EvaluateBoundIndicesDontMatch))
                    }
                }
                ExpressionNode::Product(mle_vec_indices) => {
                    let mles = mle_vec_indices
                        .iter()
                        .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                        .collect_vec();

                    mles.iter()
                        .map(|mle| {
                            let indices = mle
                                .mle_indices()
                                .iter()
                                .filter_map(|index| match index {
                                    MleIndex::Bound(index) => Some((bind_list[*index].unwrap(), index)),
                                    _ => None,
                                })
                                .collect_vec();

                            let start = *indices[0].1;
                            let end = *indices[indices.len() - 1].1;

                            let (indices, _): (Vec<_>, Vec<usize>) = indices.into_iter().unzip();

                            if indices.as_slice() == &challenges[start..=end] {
                                Ok(())
                            } else {
                                Err(anyhow!(ExpressionError::EvaluateBoundIndicesDontMatch))
                            }
                        })
                        .try_collect()
                }

                _ => Ok(()),
            }
        };
        self.traverse(&mut observer_fn)?;

        // Traverse the expression and pick up all the evals
        self.clone()
            .transform_to_verifier_expression(bind_list)
            .unwrap()
            .evaluate(bind_list)
    }

    #[allow(clippy::too_many_arguments)]
    /// This evaluates a sumcheck message using the beta cascade algorithm by calling it on the root
    /// node of the expression tree. This assumes that there is an independent variable in the
    /// expression, which is the `round_index`.
    pub fn evaluate_sumcheck_beta_cascade(
        &self,
        beta: &[&BetaValues<E>],
        random_coefficients: &[E],
        round_index: usize,
        degree: usize,
        bind_list: &Vec<Option<E>>,
    ) -> SumcheckEvals<E> {
        self.expression_node.evaluate_sumcheck_node_beta_cascade(
            beta,
            &self.mle_vec,
            random_coefficients,
            round_index,
            degree,
            bind_list,
        )
    }

    /// This evaluates a sumcheck message using the beta cascade algorithm, taking the sum
    /// of the expression over all the variables `round_index` and after. For the variables
    /// before, we compute the fully bound beta equality MLE and scale the rest of the sum
    /// by this value.
    pub fn evaluate_sumcheck_node_beta_cascade_sum(
        &self,
        beta_values: &BetaValues<E>,
        round_index: usize,
        degree: usize,
        bind_list: &Vec<Option<E>>,
    ) -> SumcheckEvals<E> {
        self.expression_node
            .evaluate_sumcheck_node_beta_cascade_sum(
                beta_values,
                round_index,
                degree,
                &self.mle_vec,
                bind_list,
            )
    }

    /// Mutate the MLE indices that are [MleIndex::Free] in the expression and
    /// turn them into [MleIndex::Indexed]. Returns the max number of bits
    /// that are indexed.
    pub fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        let (expression_node, mle_vec) = self.deconstruct_mut();
        expression_node.index_mle_indices_node(curr_index, mle_vec)
    }

    /// Gets the number of free variables in an expression.
    pub fn get_expression_num_free_variables(&self) -> usize {
        self.expression_node
            .get_expression_num_free_variables_node(0, &self.mle_vec)
    }

    /// Get the [PostSumcheckLayer] for this expression, which represents the fully bound values of the expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    pub fn get_post_sumcheck_layer(&self, multiplier: E, bind_list: &Vec<Option<E>>) -> PostSumcheckLayer<E, E> {
        self.expression_node
            .get_post_sumcheck_layer_prover(multiplier, &self.mle_vec, bind_list)
    }
}

impl<F: Field> ExpressionNode<F> {
    /// fix the variable at a certain round index, always the most significant index.
    pub fn fix_variable_node<E>(
        &mut self,
        round_index: usize,
        challenge: E,
        mle_vec: &mut [ProverMle<E>], // remove all other cases other than selector, call mle.fix_variable on all mle_vec contents
        bind_list: &mut Vec<Option<E>>,
    )
    where
        E: ExtensionField<BaseField = F>
    {
        match self {
            ExpressionNode::Selector(index, a, b) => {
                if *index == MleIndex::Indexed(round_index) {
                    index.bind_index(challenge, bind_list);
                } else {
                    a.fix_variable_node(round_index, challenge, mle_vec, bind_list);
                    b.fix_variable_node(round_index, challenge, mle_vec, bind_list);
                }
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle_mut(mle_vec);

                if mle.mle_indices().contains(&MleIndex::Indexed(round_index)) {
                    mle.fix_variable(round_index, challenge, bind_list);
                }
            }
            ExpressionNode::Sum(a, b) => {
                a.fix_variable_node(round_index, challenge, mle_vec, bind_list);
                b.fix_variable_node(round_index, challenge, mle_vec, bind_list);
            }
            ExpressionNode::Product(mle_vec_indices) => {
                mle_vec_indices
                    .iter_mut()
                    .map(|mle_vec_index| {
                        let mle = mle_vec_index.get_mle_mut(mle_vec);

                        if mle.mle_indices().contains(&MleIndex::Indexed(round_index)) {
                            mle.fix_variable(round_index, challenge, bind_list);
                        }
                    })
                    .collect_vec();
            }
            ExpressionNode::Scaled(a, _) => {
                a.fix_variable_node(round_index, challenge, mle_vec, bind_list);
            }
            ExpressionNode::Constant(_) => (),
        }
    }

    /// fix the variable at a certain round index, can be arbitrary indices.
    pub fn fix_variable_at_index_node<E>(
        &mut self,
        round_index: usize,
        challenge: E,
        mle_vec: &mut [ProverMle<E>], // remove all other cases other than selector, call mle.fix_variable on all mle_vec contents
        bind_list: &mut Vec<Option<E>>,
    )
    where
        E: ExtensionField<BaseField = F>
    {
        match self {
            ExpressionNode::Selector(index, a, b) => {
                if *index == MleIndex::Indexed(round_index) {
                    index.bind_index(challenge, bind_list);
                } else {
                    a.fix_variable_at_index_node(round_index, challenge, mle_vec, bind_list);
                    b.fix_variable_at_index_node(round_index, challenge, mle_vec, bind_list);
                }
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle_mut(mle_vec);

                if mle.mle_indices().contains(&MleIndex::Indexed(round_index)) {
                    mle.fix_variable_at_index(round_index, challenge, bind_list);
                }
            }
            ExpressionNode::Sum(a, b) => {
                a.fix_variable_at_index_node(round_index, challenge, mle_vec, bind_list);
                b.fix_variable_at_index_node(round_index, challenge, mle_vec, bind_list);
            }
            ExpressionNode::Product(mle_vec_indices) => {
                mle_vec_indices
                    .iter_mut()
                    .map(|mle_vec_index| {
                        let mle = mle_vec_index.get_mle_mut(mle_vec);

                        if mle.mle_indices().contains(&MleIndex::Indexed(round_index)) {
                            mle.fix_variable_at_index(round_index, challenge, bind_list);
                        }
                    })
                    .collect_vec();
            }
            ExpressionNode::Scaled(a, _) => {
                a.fix_variable_at_index_node(round_index, challenge, mle_vec, bind_list);
            }
            ExpressionNode::Constant(_) => (),
        }
    }

    pub fn evaluate_sumcheck_node_beta_cascade_sum<E>(
        &self,
        beta_values: &BetaValues<E>,
        round_index: usize,
        degree: usize,
        mle_vec: &[ProverMle<E>],
        bind_list: &Vec<Option<E>>,
    ) -> SumcheckEvals<E>
    where
        E: ExtensionField<BaseField = F>
    {
        match self {
            ExpressionNode::Constant(constant) => {
                SumcheckEvals(repeat_n((*constant).into(), degree + 1).collect())
            }
            ExpressionNode::Selector(selector_mle_index, lhs, rhs) => {
                let lhs_eval = lhs.evaluate_sumcheck_node_beta_cascade_sum(
                    beta_values,
                    round_index,
                    degree,
                    mle_vec,
                    bind_list,
                );
                let rhs_eval = rhs.evaluate_sumcheck_node_beta_cascade_sum(
                    beta_values,
                    round_index,
                    degree,
                    mle_vec,
                    bind_list,
                );
                match selector_mle_index {
                    MleIndex::Indexed(var_number) => {
                        let index_claim = beta_values.get_unbound_value(*var_number).unwrap();
                        (lhs_eval * (E::ONE - index_claim)) + (rhs_eval * index_claim)
                    }
                    MleIndex::Bound(var_number) => {
                        let bound_value = bind_list[*var_number].unwrap();
                        let identity = E::ONE;
                        let beta_bound = beta_values
                            .get_updated_value(*var_number)
                            .unwrap_or(identity);
                        ((lhs_eval * (E::ONE - bound_value)) + (rhs_eval * bound_value))
                            * beta_bound
                    }
                    _ => panic!("Invalid MLE Index for a selector bit, should be free or indexed"),
                }
            }
            ExpressionNode::Mle(mle_idx) => {
                let mle = mle_idx.get_mle(mle_vec);
                let (unbound, bound) = beta_values.get_relevant_beta_unbound_and_bound(
                    mle.mle_indices(),
                    round_index,
                    false,
                );
                beta_cascade_no_independent_variable(mle.mle.to_vec(), &unbound, &bound, degree)
            }
            ExpressionNode::Sum(lhs, rhs) => {
                let lhs_eval = lhs.evaluate_sumcheck_node_beta_cascade_sum(
                    beta_values,
                    round_index,
                    degree,
                    mle_vec,
                    bind_list,
                );
                let rhs_eval = rhs.evaluate_sumcheck_node_beta_cascade_sum(
                    beta_values,
                    round_index,
                    degree,
                    mle_vec,
                    bind_list,
                );
                lhs_eval + rhs_eval
            }
            ExpressionNode::Product(mle_idx_vec) => {
                let (mles, mles_bookkeeping_tables): (Vec<&ProverMle<E>>, Vec<Vec<E>>) =
                    mle_idx_vec
                        .iter()
                        .map(|mle_vec_index| {
                            let mle = mle_vec_index.get_mle(mle_vec);
                            (mle, mle.mle.to_vec())
                        })
                        .unzip();

                let mut unique_mle_indices = HashSet::new();

                let mle_indices_vec = mles
                    .iter()
                    .flat_map(|mle| mle.mle_indices.clone())
                    .filter(move |mle_index| unique_mle_indices.insert(mle_index.clone()))
                    .collect_vec();

                let (unbound, bound) = beta_values.get_relevant_beta_unbound_and_bound(
                    &mle_indices_vec,
                    round_index,
                    false,
                );
                let evaluated_bookkeeping_tables = evaluate_bookkeeping_tables_given_operation(
                    &mles_bookkeeping_tables,
                    BinaryOperation::Mul,
                );
                beta_cascade_no_independent_variable(
                    evaluated_bookkeeping_tables.to_vec(),
                    &unbound,
                    &bound,
                    degree,
                )
            }
            ExpressionNode::Scaled(expression_node, scale) => {
                expression_node.evaluate_sumcheck_node_beta_cascade_sum(
                    beta_values,
                    round_index,
                    degree,
                    mle_vec,
                    bind_list,
                ) * E::from(*scale)
            }
        }
    }

    /// This is the function to compute a single-round sumcheck message using the
    /// beta cascade algorithm.
    ///
    /// # Arguments
    ///
    /// * `expr`: the Expression `P` defining a GKR layer. The caller is expected to
    ///   have already fixed the variables of previous rounds.
    /// * `round_index`: the MLE index corresponding to the variable that is going
    ///   to be the independent variable for this round. The caller is expected to
    ///   have already fixed variables `1 .. (round_index - 1)` in expression `P` to
    ///   the verifier's challanges.
    /// * `max_degree`: the degree of the polynomial to be exchanged in this round's
    ///   sumcheck message.
    /// * `beta_value`: the `beta` function associated with expression `exp`.  It is
    ///   the caller's responsibility to keep this consistent with `expr`
    ///   before/after each call.
    ///
    /// In particular, if `round_index == k`, and the current GKR layer expression
    /// was originally on `n` variables, `expr` is expected to represent a
    /// polynomial expression on `n - k + 1` variables: `P(r_1, r_2, ..., r_{k-1},
    /// x_k, x_{k+1}, ..., x_n): F^{n - k + 1} -> F`, with the first `k - 1` free
    /// variables already fixed to random challenges `r_1, ..., r_{k-1}`. Similarly,
    /// `beta_values` should represent the polynomial: `\beta(r_1, ..., r_{k-1},
    /// b_k, ..., b_n, g_1, ..., g_n)` whose unbound variables are `b_k, ..., b_n`.
    ///
    /// # Returns
    ///
    /// If successful, this functions returns a representation of the univariate
    /// polynomial:
    /// ```text
    ///     g_{round_index}(x) =
    ///         \sum_{b_{k+1} \in {0, 1}}
    ///         \sum_{b_{k+2} \in {0, 1}}
    ///             ...
    ///         \sum_{b_{n} \in {0, 1}}
    ///             \beta(r_1, ..., r_k, x, b_{k+1}, ..., b_{n}, g_1, ..., g_n)
    ///                 * P(r_1, ..., r_k, x, b_{k+1}, ..., b_n)
    /// ```
    ///
    /// 1. This function should be responsible for mutating `expr` and `beta_values`
    ///    by fixing variables (if any) *after* the sumcheck round. It should
    ///    maintain the invariant that `expr` and `beta_values` are consistent with
    ///    each other!
    /// 2. `max_degree` should NOT be the caller's responsibility to compute. The
    ///    degree should be determined through `expr` and `round_index`.  It is
    ///    error-prone to allow for sumcheck message to go through with an arbitrary
    ///    degree.
    ///
    /// # Beta cascade
    ///
    /// Instead of using a beta table to linearize an expression, we utilize the
    /// fact that for each specific node in an expression tree, we only need exactly
    /// the beta values corresponding to the indices present in that node.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_sumcheck_node_beta_cascade<E>(
        &self,
        beta_vec: &[&BetaValues<E>],
        mle_vec: &[ProverMle<E>],
        random_coefficients: &[E],
        round_index: usize,
        degree: usize,
        bind_list: &Vec<Option<E>>,
    ) -> SumcheckEvals<E>
    where
        E: ExtensionField<BaseField = F>
    {
        match self {
            // Each different type of expression node (constant, selector, product, sum,
            // neg, scaled, mle) is treated differently, so we create closures for each
            // which are then evaluated by the `evaluate_sumcheck_beta_cascade`
            // function.

            // A constant does not have any variables, so we do not need a beta table at
            // all. Therefore we just repeat the constant evaluation for the `degree +
            // 1` number of times as this is how many evaluations we need.
            ExpressionNode::Constant(constant) => {
                let sumcheck_eval_not_scaled_by_constant = beta_vec
                    .iter()
                    .zip(random_coefficients)
                    .map(|(beta_table, random_coeff)| {
                        let folded_updated_vals = beta_table.fold_updated_values();
                        let index_claim = beta_table.get_unbound_value(round_index).unwrap();
                        let one_minus_index_claim = E::ONE - index_claim;
                        let beta_step = index_claim - one_minus_index_claim;
                        let evals =
                            std::iter::successors(Some(one_minus_index_claim), move |item| {
                                Some(*item + beta_step)
                            })
                            .take(degree + 1)
                            .collect_vec();
                        apply_updated_beta_values_to_evals(evals, folded_updated_vals)
                            * random_coeff
                    })
                    .reduce(|acc, elem| acc + elem)
                    .unwrap();
                sumcheck_eval_not_scaled_by_constant * E::from(*constant)
            }

            // the selector is split into three cases:
            // - when the selector bit itself is not the independent variable and hasn't
            //   been bound yet,
            // - when the selector bit is the independent variable
            // - when the selector bit has already been bound we determine which case we
            // are in by comparing the round_index to the selector index which is an
            // argument to the closure.
            ExpressionNode::Selector(index, a, b) => {
                let output = match index {
                    MleIndex::Indexed(indexed_bit) => {
                        let (lhs_evals, rhs_evals) = (
                            beta_vec
                                .iter()
                                .map(|beta| {
                                    a.evaluate_sumcheck_node_beta_cascade_sum(
                                        beta,
                                        round_index,
                                        degree,
                                        mle_vec,
                                        bind_list,
                                    )
                                })
                                .collect_vec(),
                            beta_vec
                                .iter()
                                .map(|beta| {
                                    b.evaluate_sumcheck_node_beta_cascade_sum(
                                        beta,
                                        round_index,
                                        degree,
                                        mle_vec,
                                        bind_list,
                                    )
                                })
                                .collect_vec(),
                        );
                        // because the selector bit itself only has one variable (1 - b_i) *
                        // (a) + b_i * b we only need one value within the beta table in
                        // order to evaluate the selector at this point.
                        match Ord::cmp(&round_index, indexed_bit) {
                            std::cmp::Ordering::Less => {
                                let sumcheck_eval = beta_vec
                                    .iter()
                                    .zip((lhs_evals.iter().zip(rhs_evals.iter())).zip(random_coefficients))
                                    .map(|(beta_table, ((a, b), random_coeff))| {
                                        let index_claim = beta_table.get_unbound_value(*indexed_bit).unwrap();
                                        let a_eval: &SumcheckEvals<E> = a;
                                        let b_eval: &SumcheckEvals<E> = b;
                                        // when the selector bit is not the independent variable and
                                        // has not been bound yet, we are simply summing over
                                        // everything. in order to take the beta values into account
                                        // this means for everything on the "left" side of the
                                        // selector we want to multiply by (1 - g_i) and for
                                        // everything on the "right" side of the selector we want to
                                        // multiply by g_i. we can then add these!
                                        let a_with_sel: SumcheckEvals<E> =
                                            a_eval.clone() * (E::ONE - index_claim);
                                        let b_with_sel: SumcheckEvals<E> = b_eval.clone() * index_claim;
                                        (a_with_sel + b_with_sel) * random_coeff
                                    })
                                    .reduce(|acc, elem| acc + elem)
                                    .unwrap();
                                sumcheck_eval
                            }
                            std::cmp::Ordering::Equal => {
                                // this is when the selector index is the independent
                                // variable! this means the beta value at this index also
                                // has an independent variable.
                                let sumcheck_eval = beta_vec
                                        .iter()
                                        .zip((lhs_evals.iter().zip(rhs_evals)).zip(random_coefficients))
                                        .map(|(beta_table, ((a, b), random_coeff))| {
                                            let SumcheckEvals(first_evals) = a;
                                            let SumcheckEvals(second_evals) = b;
                                            if first_evals.len() == second_evals.len() {
                                                let bound_beta_values = beta_table.fold_updated_values();
                                                let index_claim =
                                                    beta_table.get_unbound_value(*indexed_bit).unwrap();
                                                // therefore we compute the successors of the beta
                                                // values as well, as the successors correspond to
                                                // evaluations at the points 0, 1, ... for the
                                                // independent variable.
                                                let eval_len = first_evals.len();
                                                let one_minus_index_claim = E::ONE - index_claim;
                                                let beta_step = index_claim - one_minus_index_claim;
                                                let beta_evals = std::iter::successors(
                                                    Some(one_minus_index_claim),
                                                    move |item| Some(*item + beta_step),
                                                )
                                                .take(eval_len)
                                                .collect_vec();
                                                // the selector index also has an independent variable
                                                // so we factor this as well as the corresponding beta
                                                // successor at this index.
                                                let first_evals = SumcheckEvals(
                                                    first_evals
                                                        .clone()
                                                        .into_iter()
                                                        .enumerate()
                                                        .map(|(idx, first_eval)| {
                                                            first_eval
                                                                * (E::ONE - E::from(idx as u64))
                                                                * beta_evals[idx]
                                                        })
                                                        .collect(),
                                                );
                                                let second_evals = SumcheckEvals(
                                                    second_evals
                                                        .clone()
                                                        .into_iter()
                                                        .enumerate()
                                                        .map(|(idx, second_eval)| {
                                                            second_eval
                                                            * E::from(idx as u64) * beta_evals[idx]
                                                        })
                                                        .collect(),
                                                );
                                                (first_evals + second_evals) * random_coeff * bound_beta_values
                                            } else {
                                                panic!("Expression returns two evals that do not have the same length on a selector bit")
                                            }
                                        })
                                        .reduce(|acc, elem| acc + elem)
                                        .unwrap();
                                sumcheck_eval
                            }
                            // we cannot have an indexed bit for the selector bit that is
                            // less than the current sumcheck round. therefore this is an
                            // error
                            std::cmp::Ordering::Greater => panic!(
                                "Invalid selector index, cannot be less than the current round index"
                            ),
                        }
                    }
                    // if the selector bit has already been bound, that means the beta value
                    // at this index has also already been bound, if it exists! otherwise we
                    // just treat it as the identity
                    MleIndex::Bound(idx) => {
                        let coeff = bind_list[*idx].unwrap();
                        let (lhs_evals, rhs_evals) = (
                            beta_vec
                                .iter()
                                .map(|beta| {
                                    a.evaluate_sumcheck_node_beta_cascade(
                                        &[*beta],
                                        mle_vec,
                                        &[E::ONE],
                                        round_index,
                                        degree,
                                        bind_list,
                                    )
                                })
                                .collect_vec(),
                            beta_vec
                                .iter()
                                .map(|beta| {
                                    b.evaluate_sumcheck_node_beta_cascade(
                                        &[*beta],
                                        mle_vec,
                                        &[E::ONE],
                                        round_index,
                                        degree,
                                        bind_list,
                                    )
                                })
                                .collect_vec(),
                        );
                        let coeff_neg = E::ONE - coeff;
                        (lhs_evals.iter().zip(rhs_evals))
                            .zip(random_coefficients)
                            .map(|((a, b), random_coeff)| {
                                let a_eval = a;
                                let b_eval = b;
                                ((b_eval.clone() * coeff) + (a_eval.clone() * coeff_neg))
                                    * random_coeff
                            })
                            .reduce(|acc, elem| acc + elem)
                            .unwrap()
                    }
                    _ => panic!("selector index should not be a Free or Fixed bit"),
                };
                output
            }
            // the mle evaluation takes in the mle ref, and the corresponding unbound
            // and bound beta values to pass into the `beta_cascade` function
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle(mle_vec);
                let (unbound_beta_vec, bound_beta_vec): (Vec<Vec<E>>, Vec<Vec<E>>) = beta_vec
                    .iter()
                    .map(|beta| {
                        beta.get_relevant_beta_unbound_and_bound(
                            mle.mle_indices(),
                            round_index,
                            true,
                        )
                    })
                    .unzip();

                beta_cascade(
                    &[&mle.clone()],
                    degree,
                    round_index,
                    &unbound_beta_vec,
                    &bound_beta_vec,
                    random_coefficients,
                )
            }
            // when we have a sum, we can evaluate both parts of the expression
            // separately and just add the evaluations
            ExpressionNode::Sum(a, b) => {
                let a = a.evaluate_sumcheck_node_beta_cascade(
                    beta_vec,
                    mle_vec,
                    random_coefficients,
                    round_index,
                    degree,
                    bind_list,
                );
                let b = b.evaluate_sumcheck_node_beta_cascade(
                    beta_vec,
                    mle_vec,
                    random_coefficients,
                    round_index,
                    degree,
                    bind_list,
                );
                a + b
            }
            // when we have a product, the node can only contain mle refs. therefore
            // this is similar to the mle evaluation, but instead we have a list of mle
            // refs, and the corresponding unbound and bound  beta values for that node.
            ExpressionNode::Product(mle_vec_indices) => {
                let mles = mle_vec_indices
                    .iter()
                    .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                    .collect_vec();

                let mut unique_mle_indices = HashSet::new();

                let mle_indices_vec = mles
                    .iter()
                    .flat_map(|mle| mle.mle_indices.clone())
                    .filter(move |mle_index| unique_mle_indices.insert(mle_index.clone()))
                    .collect_vec();

                let (unbound_beta_vec, bound_beta_vec): (Vec<Vec<E>>, Vec<Vec<E>>) = beta_vec
                    .iter()
                    .map(|beta| {
                        beta.get_relevant_beta_unbound_and_bound(
                            &mle_indices_vec,
                            round_index,
                            true,
                        )
                    })
                    .unzip();

                beta_cascade(
                    &mles,
                    degree,
                    round_index,
                    &unbound_beta_vec,
                    &bound_beta_vec,
                    random_coefficients,
                )
            }

            // when the expression is scaled by a field element, we can scale the
            // evaluations by this element as well
            ExpressionNode::Scaled(a, scale) => {
                let a = a.evaluate_sumcheck_node_beta_cascade(
                    beta_vec,
                    mle_vec,
                    random_coefficients,
                    round_index,
                    degree,
                    bind_list,
                );
                a * E::from(*scale)
            }
        }
    }

    /// Mutate the MLE indices that are [MleIndex::Free] in the expression and
    /// turn them into [MleIndex::Indexed]. Returns the max number of bits
    /// that are indexed.
    pub fn index_mle_indices_node<E>(
        &mut self,
        curr_index: usize,
        mle_vec: &mut [ProverMle<E>],
    ) -> usize
    where 
        E: ExtensionField<BaseField = F>
    {
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                let mut new_index = curr_index;
                match mle_index {
                    MleIndex::Free => {
                        *mle_index = MleIndex::Indexed(curr_index);
                        new_index += 1;
                    }
                    MleIndex::Fixed(_bit) => {}
                    _ => panic!("should not have indexed or bound bits at this point!"),
                };
                let a_bits = a.index_mle_indices_node(new_index, mle_vec);
                let b_bits = b.index_mle_indices_node(new_index, mle_vec);

                max(a_bits, b_bits)
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle_mut(mle_vec);
                mle.index_mle_indices(curr_index)
            }
            ExpressionNode::Sum(a, b) => {
                let a_bits = a.index_mle_indices_node(curr_index, mle_vec);
                let b_bits = b.index_mle_indices_node(curr_index, mle_vec);
                max(a_bits, b_bits)
            }
            ExpressionNode::Product(mle_vec_indices) => mle_vec_indices
                .iter_mut()
                .map(|mle_vec_index| {
                    let mle = mle_vec_index.get_mle_mut(mle_vec);
                    mle.index_mle_indices(curr_index)
                })
                .reduce(max)
                .unwrap_or(curr_index),
            ExpressionNode::Scaled(a, _) => a.index_mle_indices_node(curr_index, mle_vec),
            ExpressionNode::Constant(_) => curr_index,
        }
    }

    /// Gets the number of free variables in an expression.
    pub fn get_expression_num_free_variables_node<E>(
        &self,
        curr_size: usize,
        mle_vec: &[ProverMle<E>],
    ) -> usize
    where 
        E: ExtensionField<BaseField = F>
    {
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                let (a_bits, b_bits) = if matches!(mle_index, &MleIndex::Free) {
                    (
                        a.get_expression_num_free_variables_node(curr_size + 1, mle_vec),
                        b.get_expression_num_free_variables_node(curr_size + 1, mle_vec),
                    )
                } else {
                    (
                        a.get_expression_num_free_variables_node(curr_size, mle_vec),
                        b.get_expression_num_free_variables_node(curr_size, mle_vec),
                    )
                };

                max(a_bits, b_bits)
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle(mle_vec);

                mle.mle_indices()
                    .iter()
                    .filter(|item| matches!(item, &&MleIndex::Free))
                    .collect_vec()
                    .len()
                    + curr_size
            }
            ExpressionNode::Sum(a, b) => {
                let a_bits = a.get_expression_num_free_variables_node(curr_size, mle_vec);
                let b_bits = b.get_expression_num_free_variables_node(curr_size, mle_vec);
                max(a_bits, b_bits)
            }
            ExpressionNode::Product(mle_vec_indices) => {
                let mles = mle_vec_indices
                    .iter()
                    .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                    .collect_vec();

                mles.iter()
                    .map(|mle| {
                        mle.mle_indices()
                            .iter()
                            .filter(|item| matches!(item, &&MleIndex::Free))
                            .collect_vec()
                            .len()
                    })
                    .max()
                    .unwrap_or(0)
                    + curr_size
            }
            ExpressionNode::Scaled(a, _) => {
                a.get_expression_num_free_variables_node(curr_size, mle_vec)
            }
            ExpressionNode::Constant(_) => curr_size,
        }
    }

    /// Recursively get the [PostSumcheckLayer] for an Expression node, which is the fully bound
    /// representation of an expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    pub fn get_post_sumcheck_layer_prover<E>(
        &self,
        multiplier: E,
        mle_vec: &[ProverMle<E>],
        bind_list: &Vec<Option<E>>,
    ) -> PostSumcheckLayer<E, E> 
    where 
        E: ExtensionField<BaseField = F>
    {
        let mut products: Vec<Product<E, E>> = vec![];
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                let left_side_acc = multiplier * (E::ONE - mle_index.val(bind_list).unwrap());
                let right_side_acc = multiplier * (mle_index.val(bind_list).unwrap());
                products.extend(a.get_post_sumcheck_layer_prover(left_side_acc, mle_vec, bind_list).0);
                products.extend(b.get_post_sumcheck_layer_prover(right_side_acc, mle_vec, bind_list).0);
            }
            ExpressionNode::Sum(a, b) => {
                products.extend(a.get_post_sumcheck_layer_prover(multiplier, mle_vec, bind_list).0);
                products.extend(b.get_post_sumcheck_layer_prover(multiplier, mle_vec, bind_list).0);
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle(mle_vec);
                assert!(mle.is_fully_bounded(bind_list));
                products.push(Product::<E, E>::new(&[mle.clone()], multiplier));
            }
            ExpressionNode::Product(mle_vec_indices) => {
                let mles = mle_vec_indices
                    .iter()
                    .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec).clone())
                    .collect_vec();
                let product = Product::<E, E>::new(&mles, multiplier);
                products.push(product);
            }
            ExpressionNode::Scaled(a, scale_factor) => {
                let acc = multiplier * *scale_factor;
                products.extend(a.get_post_sumcheck_layer_prover(acc, mle_vec, bind_list).0);
            }
            ExpressionNode::Constant(constant) => {
                products.push(Product::<E, E>::new(&[], multiplier * *constant));
            }
        }
        PostSumcheckLayer(products)
    }
}
