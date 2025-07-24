//! The prover's view of an [Expression] -- see file-level documentation within
//! [crate::expression] for more details.
//!
//! Conceptually, [ProverExpr] contains the "circuit structure" between the
//! layer whose values are the output of the given expression and those whose
//! values are the inputs to the given expression, i.e. the polynomial
//! relationship between them, as well as (in an ownership sense) the actual
//! data, stored in [DenseMle]s.

use super::{
    expr_errors::ExpressionError,
    generic_expr::{Expression, ExpressionNode, ExpressionType},
    verifier_expr::VerifierExpr,
};
use crate::{
    layer::product::PostSumcheckLayerTree,
    mle::{verifier_mle::VerifierMle, Mle},
};
use crate::{
    mle::{
        betavalues::BetaValues, dense::DenseMle, mle_bookkeeping_table::MleBookkeepingTables,
        mle_combination::MleCombinationSeq, MleIndex,
    },
    sumcheck::{apply_updated_beta_values_to_evals, SumcheckEvals},
};
use itertools::Itertools;
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

use anyhow::{anyhow, Ok, Result};

type BetaValsVecs<F> = (Vec<Option<F>>, (Vec<Vec<F>>, Vec<Vec<F>>));

/// mid-term solution for deduplication of DenseMleRefs
/// basically a wrapper around usize, which denotes the index
/// of the MleRef in an expression's MleRef list/// Generic Expressions
///
/// TODO(ryancao): We should deprecate this and instead just have
/// references to the `DenseMLE<F>`s which are stored in the circuit_map.
#[derive(Serialize, Deserialize, Clone, Debug, Hash)]
pub struct MleVecIndex(usize);

impl MleVecIndex {
    /// create a new MleRefIndex
    pub fn new(index: usize) -> Self {
        MleVecIndex(index)
    }

    /// returns the index
    pub fn index(&self) -> usize {
        self.0
    }

    /// add the index with an increment amount
    pub fn increment(&mut self, offset: usize) {
        self.0 += offset;
    }

    /// return the actual mle in the vec within the prover expression
    pub fn get_mle<'a, F: Field>(&self, mle_vec: &'a [DenseMle<F>]) -> &'a DenseMle<F> {
        &mle_vec[self.0]
    }

    /// return the actual mle in the vec within the prover expression
    pub fn get_mle_mut<'a, F: Field>(&self, mle_vec: &'a mut [DenseMle<F>]) -> &'a mut DenseMle<F> {
        &mut mle_vec[self.0]
    }
}

/// Prover Expression
/// the leaf nodes of the expression tree are DenseMleRefs
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProverExpr;
impl<F: Field> ExpressionType<F> for ProverExpr {
    type MLENodeRepr = MleVecIndex;
    type MleVec = Vec<DenseMle<F>>;
}

/// this is what the prover manipulates to prove the correctness of the computation.
/// Methods here include ones to fix bits, evaluate sumcheck messages, etc.
impl<F: Field> Expression<F, ProverExpr> {
    /// See documentation in [super::circuit_expr::ExprDescription]'s `select()`
    /// function for more details!
    pub fn select(self, mut rhs: Expression<F, ProverExpr>) -> Self {
        let offset = self.num_mle();
        rhs.increment_mle_vec_indices(offset);
        let (lhs_node, lhs_mle_vec) = self.deconstruct();
        let (rhs_node, rhs_mle_vec) = rhs.deconstruct();

        let concat_node =
            ExpressionNode::Selector(MleIndex::Free, Box::new(lhs_node), Box::new(rhs_node));

        let concat_mle_vec = lhs_mle_vec.into_iter().chain(rhs_mle_vec).collect_vec();

        Expression::new(concat_node, concat_mle_vec)
    }

    /// Create a product Expression that raises one expression to a given power
    pub fn pow(pow: usize, base: Expression<F, ProverExpr>) -> Self {
        // lazily construct a linear-depth expression tree
        let mut result = base.clone();
        for _ in 1..pow {
            result = result * base.clone();
        }
        result
    }

    /// Create a product Expression that contains two MLEs
    pub fn products(lhs: Self, mut rhs: Self) -> Self {
        let offset = lhs.num_mle();
        rhs.increment_mle_vec_indices(offset);

        let (lhs_node, lhs_mle_vec) = lhs.deconstruct();
        let (rhs_node, rhs_mle_vec) = rhs.deconstruct();

        let product_node = ExpressionNode::Product(Box::new(lhs_node), Box::new(rhs_node));
        let product_mle_vec = lhs_mle_vec.into_iter().chain(rhs_mle_vec).collect_vec();

        Expression::new(product_node, product_mle_vec)
    }

    /// Create a mle Expression that contains one MLE
    pub fn mle(mle: DenseMle<F>) -> Self {
        let mle_node = ExpressionNode::Mle(MleVecIndex::new(0));

        Expression::new(mle_node, [mle].to_vec())
    }

    /// Create a constant Expression that contains one field element
    pub fn constant(constant: F) -> Self {
        let mle_node = ExpressionNode::Constant(constant);

        Expression::new(mle_node, [].to_vec())
    }

    /// negates an Expression
    pub fn negated(expression: Self) -> Self {
        let (node, mle_vec) = expression.deconstruct();

        let mle_node = ExpressionNode::Scaled(Box::new(node), F::from(1).neg());

        Expression::new(mle_node, mle_vec)
    }

    /// Create a Sum Expression that contains two MLEs
    pub fn sum(lhs: Self, mut rhs: Self) -> Self {
        let offset = lhs.num_mle();
        rhs.increment_mle_vec_indices(offset);

        let (lhs_node, lhs_mle_vec) = lhs.deconstruct();
        let (rhs_node, rhs_mle_vec) = rhs.deconstruct();

        let sum_node = ExpressionNode::Sum(Box::new(lhs_node), Box::new(rhs_node));
        let sum_mle_vec = lhs_mle_vec.into_iter().chain(rhs_mle_vec).collect_vec();

        Expression::new(sum_node, sum_mle_vec)
    }

    /// scales an Expression by a field element
    pub fn scaled(expression: Expression<F, ProverExpr>, scale: F) -> Self {
        let (node, mle_vec) = expression.deconstruct();

        Expression::new(ExpressionNode::Scaled(Box::new(node), scale), mle_vec)
    }

    /// returns the number of MleRefs in the expression
    pub fn num_mle(&self) -> usize {
        self.mle_vec.len()
    }

    /// which increments all the MleVecIndex in the expression by *param* amount
    pub fn increment_mle_vec_indices(&mut self, offset: usize) {
        // define a closure that increments the MleVecIndex by the given amount
        // use traverse_mut
        let mut increment_closure = |expr: &mut ExpressionNode<F, ProverExpr>,
                                     _mle_vec: &mut Vec<DenseMle<F>>|
         -> Result<()> {
            match expr {
                ExpressionNode::Mle(mle_vec_index) => {
                    mle_vec_index.increment(offset);
                    Ok(())
                }
                ExpressionNode::Constant(_)
                | ExpressionNode::Scaled(..)
                | ExpressionNode::Sum(..)
                | ExpressionNode::Product(..)
                | ExpressionNode::Selector(..) => Ok(()),
            }
        };

        self.traverse_mut(&mut increment_closure).unwrap();
    }

    /// Transforms the prover expression to a verifier expression.
    ///
    /// Should only be called when the entire expression is fully bound.
    ///
    /// Traverses the expression and changes the DenseMle to VerifierMle,
    /// by grabbing their bookkeeping table's 1st and only element.
    ///
    /// If the bookkeeping table has more than 1 element, it
    /// throws an ExpressionError::EvaluateNotFullyBoundError
    pub fn transform_to_verifier_expression(self) -> Result<Expression<F, VerifierExpr>> {
        let (mut expression_node, mle_vec) = self.deconstruct();
        Ok(Expression::new(
            expression_node
                .transform_to_verifier_expression_node(&mle_vec)
                .unwrap(),
            (),
        ))
    }

    /// fix the variable at a certain round index, always MSB index
    pub fn fix_variable(&mut self, round_index: usize, challenge: F) {
        let (expression_node, mle_vec) = self.deconstruct_mut();

        expression_node.fix_variable_node(round_index, challenge, mle_vec)
    }

    /// fix the variable at a certain round index, arbitrary index
    pub fn fix_variable_at_index(&mut self, round_index: usize, challenge: F) {
        let (expression_node, mle_vec) = self.deconstruct_mut();

        expression_node.fix_variable_at_index_node(round_index, challenge, mle_vec)
    }

    /// evaluates an expression on the given challenges points, by fixing the variables
    pub fn evaluate_expr(&mut self, challenges: Vec<F>) -> Result<F> {
        // It's as simple as fixing all variables
        challenges
            .iter()
            .enumerate()
            .for_each(|(round_idx, &challenge)| {
                self.fix_variable(round_idx, challenge);
            });

        // ----- this is literally a check -----
        let mut observer_fn = |exp: &ExpressionNode<F, ProverExpr>,
                               mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec|
         -> Result<()> {
            match exp {
                ExpressionNode::Mle(mle_vec_idx) => {
                    let mle = mle_vec_idx.get_mle(mle_vec);
                    let indices = mle
                        .mle_indices()
                        .iter()
                        .filter_map(|index| match index {
                            MleIndex::Bound(chal, index) => Some((*chal, index)),
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
                _ => Ok(()),
            }
        };
        self.traverse(&mut observer_fn)?;

        // Traverse the expression and pick up all the evals
        self.clone()
            .transform_to_verifier_expression()
            .unwrap()
            .evaluate()
    }

    /// This evaluates a sumcheck message using the beta cascade algorithm, taking the sum
    /// of the expression over all the variables, applied with beta_vec
    pub fn evaluate_sumcheck_beta_cascade_sum(
        &self,
        beta_values: &BetaValues<F>,
    ) -> SumcheckEvals<F> {
        // This is equivalent to a degree-0 beta cascade on a round not present in the MLEs
        // First find such a non-existent round
        let dummy_round_index = self.get_all_rounds().len();
        // Then apply beta cascade
        self.expression_node
            .evaluate_sumcheck_node_beta_cascade_bookkeeping_table(
                &[beta_values],
                &self.mle_vec,
                &vec![F::ONE],
                dummy_round_index,
                0,
            )
    }

    #[allow(clippy::too_many_arguments)]
    /// This evaluates a sumcheck message using the beta cascade algorithm by calling it on the root
    /// node of the expression tree. This assumes that there is an independent variable in the
    /// expression, which is the `round_index`.
    pub fn evaluate_sumcheck_beta_cascade(
        &self,
        beta: &[&BetaValues<F>],
        random_coefficients: &[F],
        round_index: usize,
        degree: usize,
    ) -> SumcheckEvals<F> {
        self.expression_node
            .evaluate_sumcheck_node_beta_cascade_bookkeeping_table(
                beta,
                &self.mle_vec,
                random_coefficients,
                round_index,
                degree,
            )
    }

    /// Traverses the expression tree to return all indices within the
    /// expression. Can only be used after indexing the expression.
    pub fn get_all_rounds(&self) -> Vec<usize> {
        let (expression_node, mle_vec) = self.deconstruct_ref();
        let mut all_rounds = expression_node.get_all_rounds(mle_vec);
        all_rounds.sort();
        all_rounds
    }

    /// this traverses the expression tree to get all of the nonlinear rounds. can only be used after indexing the expression.
    /// returns the indices sorted.
    pub fn get_all_nonlinear_rounds(&self) -> Vec<usize> {
        let (expression_node, mle_vec) = self.deconstruct_ref();
        let mut nonlinear_rounds = expression_node.get_all_nonlinear_rounds(mle_vec);
        nonlinear_rounds.sort();
        nonlinear_rounds
    }

    /// this traverses the expression tree to get all of the linear rounds. can only be used after indexing the expression.
    /// returns the indices sorted.
    pub fn get_all_linear_rounds(&self) -> Vec<usize> {
        let (expression_node, mle_vec) = self.deconstruct_ref();
        let mut linear_rounds = expression_node.get_all_linear_rounds(mle_vec);
        linear_rounds.sort();
        linear_rounds
    }

    /// Mutate the MLE indices that are [MleIndex::Free] in the expression and
    /// turn them into [MleIndex::IndexedBit]. Returns the max number of bits
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
    pub fn get_post_sumcheck_layer(&self) -> PostSumcheckLayerTree<F, F> {
        self.expression_node
            .get_post_sumcheck_layer(&self.mle_vec)
    }

    /// Get the maximum degree of any variable in htis expression
    pub fn get_max_degree(&self) -> usize {
        self.expression_node.get_max_degree(&self.mle_vec)
    }
}

impl<F: Field> ExpressionNode<F, ProverExpr> {
    /// Transforms the expression to a verifier expression
    /// should only be called when no variables are bound in the expression.
    /// Traverses the expression and changes the DenseMle to MleDescription.
    pub fn transform_to_verifier_expression_node(
        &mut self,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> Result<ExpressionNode<F, VerifierExpr>> {
        match self {
            ExpressionNode::Constant(scalar) => Ok(ExpressionNode::Constant(*scalar)),
            ExpressionNode::Selector(index, a, b) => Ok(ExpressionNode::Selector(
                index.clone(),
                Box::new(a.transform_to_verifier_expression_node(mle_vec)?),
                Box::new(b.transform_to_verifier_expression_node(mle_vec)?),
            )),
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle(mle_vec);

                if !mle.is_fully_bounded() {
                    return Err(anyhow!(ExpressionError::EvaluateNotFullyBoundError));
                }

                let layer_id = mle.layer_id();
                let mle_indices = mle.mle_indices().to_vec();
                let eval = mle.value();

                Ok(ExpressionNode::Mle(VerifierMle::new(
                    layer_id,
                    mle_indices,
                    eval,
                )))
            }
            ExpressionNode::Sum(a, b) => Ok(ExpressionNode::Sum(
                Box::new(a.transform_to_verifier_expression_node(mle_vec)?),
                Box::new(b.transform_to_verifier_expression_node(mle_vec)?),
            )),
            ExpressionNode::Product(a, b) => Ok(ExpressionNode::Product(
                Box::new(a.transform_to_verifier_expression_node(mle_vec)?),
                Box::new(b.transform_to_verifier_expression_node(mle_vec)?),
            )),
            ExpressionNode::Scaled(mle, scalar) => Ok(ExpressionNode::Scaled(
                Box::new(mle.transform_to_verifier_expression_node(mle_vec)?),
                *scalar,
            )),
        }
    }

    /// fix the variable at a certain round index, always the most significant index.
    pub fn fix_variable_node(
        &mut self,
        round_index: usize,
        challenge: F,
        mle_vec: &mut <ProverExpr as ExpressionType<F>>::MleVec, // remove all other cases other than selector, call mle.fix_variable on all mle_vec contents
    ) {
        match self {
            ExpressionNode::Selector(index, a, b) => {
                if *index == MleIndex::Indexed(round_index) {
                    index.bind_index(challenge);
                } else {
                    a.fix_variable_node(round_index, challenge, mle_vec);
                    b.fix_variable_node(round_index, challenge, mle_vec);
                }
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle_mut(mle_vec);

                if mle.mle_indices().contains(&MleIndex::Indexed(round_index)) {
                    mle.fix_variable(round_index, challenge);
                }
            }
            ExpressionNode::Sum(a, b) => {
                a.fix_variable_node(round_index, challenge, mle_vec);
                b.fix_variable_node(round_index, challenge, mle_vec);
            }
            ExpressionNode::Product(a, b) => {
                a.fix_variable_node(round_index, challenge, mle_vec);
                b.fix_variable_node(round_index, challenge, mle_vec);
            }
            ExpressionNode::Scaled(a, _) => {
                a.fix_variable_node(round_index, challenge, mle_vec);
            }
            ExpressionNode::Constant(_) => (),
        }
    }

    /// fix the variable at a certain round index, can be arbitrary indices.
    pub fn fix_variable_at_index_node(
        &mut self,
        round_index: usize,
        challenge: F,
        mle_vec: &mut <ProverExpr as ExpressionType<F>>::MleVec, // remove all other cases other than selector, call mle.fix_variable on all mle_vec contents
    ) {
        match self {
            ExpressionNode::Selector(index, a, b) => {
                if *index == MleIndex::Indexed(round_index) {
                    index.bind_index(challenge);
                } else {
                    a.fix_variable_at_index_node(round_index, challenge, mle_vec);
                    b.fix_variable_at_index_node(round_index, challenge, mle_vec);
                }
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle_mut(mle_vec);

                if mle.mle_indices().contains(&MleIndex::Indexed(round_index)) {
                    mle.fix_variable_at_index(round_index, challenge);
                }
            }
            ExpressionNode::Sum(a, b) => {
                a.fix_variable_at_index_node(round_index, challenge, mle_vec);
                b.fix_variable_at_index_node(round_index, challenge, mle_vec);
            }
            ExpressionNode::Product(a, b) => {
                a.fix_variable_at_index_node(round_index, challenge, mle_vec);
                b.fix_variable_at_index_node(round_index, challenge, mle_vec);
            }
            ExpressionNode::Scaled(a, _) => {
                a.fix_variable_at_index_node(round_index, challenge, mle_vec);
            }
            ExpressionNode::Constant(_) => (),
        }
    }

    /// Compute a single-round sumcheck message using book keeping table impl.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_sumcheck_node_beta_cascade_bookkeeping_table(
        &self,
        beta_vec: &[&BetaValues<F>],
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
        random_coefficients: &[F],
        round_index: usize,
        degree: usize,
    ) -> SumcheckEvals<F> {
        let mle_bookkeeping_tables: Vec<(bool, Vec<_>)> = mle_vec
            .iter()
            .map(|mle| MleBookkeepingTables::from_mle_evals(mle, degree, round_index))
            .collect();

        let (comb_seq, cnst_tables) =
            MleCombinationSeq::from_expr_and_bind(self, mle_vec.len(), degree, round_index);

        // batch evaluate all tables on each X
        let eval_tables_list: Vec<Vec<&MleBookkeepingTables<F>>> = (0..degree + 1)
            .map(|eval| {
                mle_bookkeeping_tables
                    .iter()
                    .map(|(bounded, tables)| {
                        let j = if *bounded { 0 } else { eval };
                        &tables[j]
                    })
                    .chain(cnst_tables.iter().map(|(bounded, tables)| {
                        let j = if *bounded { 0 } else { eval };
                        &tables[j]
                    }))
                    .collect()
            })
            .collect();

        let comb_tables: Vec<_> = eval_tables_list
            .into_iter()
            .map(|eval_tables| MleBookkeepingTables::comb(eval_tables, &comb_seq))
            .collect();
        // MleBookkeepingTables::comb_batch(eval_tables_list, &comb_seq);

        // all comb_tables are of the same structure, so only
        // need to process beta once on `comb_tables[0]`
        // returns:
        //   - beta_current_val_vec: beta values of the current round
        //   - beta_unbound_vals_vec: beta values yet to be bounded
        //   - beta_updated_vals_vec: beta values already bounded, in the form of a multiple
        let (beta_current_val_vec, (beta_unbound_vals_vec, beta_updated_vals_vec)): BetaValsVecs<
            F,
        > = beta_vec
            .iter()
            .map(|beta| {
                (
                    beta.get_unbound_value(round_index),
                    beta.get_relevant_beta_unbound_and_bound_from_bookkeeping_table(
                        &comb_tables[0],
                    ),
                )
            })
            .unzip();

        let evals_iter = beta_current_val_vec
            .into_iter()
            .zip(beta_unbound_vals_vec)
            .zip(beta_updated_vals_vec)
            .zip(random_coefficients)
            .map(
                |(((beta_cur_val, beta_unbound_vals), beta_updated_vals), random_coeff)| {
                    // bind each table to beta_unbound
                    let folded_mle_successors: Vec<F> = comb_tables
                        .iter()
                        .map(|t| t.beta_cascade(&beta_unbound_vals))
                        .collect();

                    // combine all points with beta_current
                    let evals = if let Some(beta_cur_val) = beta_cur_val {
                        let second_beta_successor = beta_cur_val;
                        let first_beta_successor = F::ONE - second_beta_successor;
                        let step = second_beta_successor - first_beta_successor;
                        let beta_successors =
                            std::iter::successors(Some(first_beta_successor), move |item| {
                                Some(*item + step)
                            });
                        // the length of the mle successor vec before this last step must be
                        // degree + 1. therefore we can just do a zip with the beta
                        // successors to get the final degree + 1 evaluations.
                        beta_successors
                            .zip(folded_mle_successors)
                            .map(|(beta_succ, mle_succ)| beta_succ * mle_succ)
                            .take(degree + 1)
                            .collect_vec()
                    } else {
                        folded_mle_successors
                    };
                    // apply the bound beta values as a scalar factor to each of the
                    // evaluations Multiply by the random coefficient to get the
                    // random linear combination by summing at the end.
                    apply_updated_beta_values_to_evals(evals, beta_updated_vals.iter().product())
                        * random_coeff
                },
            );
        // Combine all the evaluations using a random linear combination. We
        // simply sum because all evaluations are already multiplied by their
        // random coefficient.
        evals_iter.reduce(|acc, elem| acc + elem).unwrap()
    }

    /// Mutate the MLE indices that are [MleIndex::Free] in the expression and
    /// turn them into [MleIndex::IndexedBit]. Returns the max number of bits
    /// that are indexed.
    pub fn index_mle_indices_node(
        &mut self,
        curr_index: usize,
        mle_vec: &mut <ProverExpr as ExpressionType<F>>::MleVec,
    ) -> usize {
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                let mut new_index = curr_index;
                if *mle_index == MleIndex::Free {
                    *mle_index = MleIndex::Indexed(curr_index);
                    new_index += 1;
                }
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
            ExpressionNode::Product(a, b) => {
                let a_bits = a.index_mle_indices_node(curr_index, mle_vec);
                let b_bits = b.index_mle_indices_node(curr_index, mle_vec);
                max(a_bits, b_bits)
            }
            ExpressionNode::Scaled(a, _) => a.index_mle_indices_node(curr_index, mle_vec),
            ExpressionNode::Constant(_) => curr_index,
        }
    }

    /// this traverses the expression to get all of the rounds, in total. requires going through each of the nodes
    /// and collecting the leaf node indices.
    pub(crate) fn get_all_rounds(
        &self,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> Vec<usize> {
        let degree_per_index = self.get_degree_list(mle_vec);
        (0..degree_per_index.len())
            .filter(|&i| degree_per_index[i] > 0)
            .collect()
    }

    /// traverse an expression tree in order to get all of the nonlinear rounds in an expression.
    pub fn get_all_nonlinear_rounds(
        &self,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> Vec<usize> {
        let degree_per_index = self.get_degree_list(mle_vec);
        (0..degree_per_index.len())
            .filter(|&i| degree_per_index[i] > 1)
            .collect()
    }

    /// get all of the linear rounds from an expression tree
    pub fn get_all_linear_rounds(
        &self,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> Vec<usize> {
        let degree_per_index = self.get_degree_list(mle_vec);
        (0..degree_per_index.len())
            .filter(|&i| degree_per_index[i] == 1)
            .collect()
    }

    /// a recursive helper to obtain the degree of every variable
    pub fn get_degree_list(
        &self,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
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
                let a_degree_per_index = a.get_degree_list(mle_vec);
                let b_degree_per_index = b.get_degree_list(mle_vec);
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
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle(mle_vec);
                mle.mle_indices.iter().for_each(|mle_index| {
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
                let a_degree_per_index = a.get_degree_list(mle_vec);
                let b_degree_per_index = b.get_degree_list(mle_vec);
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
                let a_degree_per_index = a.get_degree_list(mle_vec);
                let b_degree_per_index = b.get_degree_list(mle_vec);
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
                degree_per_index = a.get_degree_list(mle_vec);
            }
            // for a constant there are no new indices.
            ExpressionNode::Constant(_) => {}
        }
        degree_per_index
    }

    /// Gets the number of free variables in an expression.
    pub fn get_expression_num_free_variables_node(
        &self,
        curr_size: usize,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> usize {
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
            ExpressionNode::Product(a, b) => {
                let a_bits = a.get_expression_num_free_variables_node(curr_size, mle_vec);
                let b_bits = b.get_expression_num_free_variables_node(curr_size, mle_vec);
                max(a_bits, b_bits)
            }
            ExpressionNode::Scaled(a, _) => {
                a.get_expression_num_free_variables_node(curr_size, mle_vec)
            }
            ExpressionNode::Constant(_) => curr_size,
        }
    }

    /// Recursively get the [PostSumcheckLayerTree] for an Expression node, which is the fully bound
    /// representation of an expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    pub fn get_post_sumcheck_layer(
        &self,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> PostSumcheckLayerTree<F, F> {
        match self {
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle(mle_vec);
                assert!(mle.is_fully_bounded());
                PostSumcheckLayerTree::<F, F>::mle(mle)
            }
            ExpressionNode::Constant(constant) => {
                PostSumcheckLayerTree::constant(
                    *constant
                )
            }
            ExpressionNode::Selector(mle_index, a, b) => {
                let left_prod = PostSumcheckLayerTree::<F, F>::mult(
                    PostSumcheckLayerTree::constant(F::ONE - mle_index.val().unwrap()),
                    a.get_post_sumcheck_layer(mle_vec),
                );
                let right_prod = PostSumcheckLayerTree::<F, F>::mult(
                    PostSumcheckLayerTree::constant(mle_index.val().unwrap()),
                    b.get_post_sumcheck_layer(mle_vec),
                );
                PostSumcheckLayerTree::<F, F>::add(left_prod, right_prod)
            }
            ExpressionNode::Sum(a, b) => {
                PostSumcheckLayerTree::<F, F>::add(
                    a.get_post_sumcheck_layer(mle_vec),
                    b.get_post_sumcheck_layer(mle_vec),
                )
            }
            ExpressionNode::Product(a, b) => {
                PostSumcheckLayerTree::<F, F>::mult(
                    a.get_post_sumcheck_layer(mle_vec),
                    b.get_post_sumcheck_layer(mle_vec),
                )
            }
            ExpressionNode::Scaled(a, scale_factor) => {
                PostSumcheckLayerTree::<F, F>::mult(
                    a.get_post_sumcheck_layer(mle_vec),
                    PostSumcheckLayerTree::constant(*scale_factor),
                )
            }
        }
    }

    /// Get the maximum degree of an ExpressionNode, recursively.
    fn get_max_degree(&self, mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec) -> usize {
        *self.get_degree_list(mle_vec).iter().max().unwrap()
    }
}

impl<F: Field> Neg for Expression<F, ProverExpr> {
    type Output = Expression<F, ProverExpr>;
    fn neg(self) -> Self::Output {
        Expression::<F, ProverExpr>::negated(self)
    }
}

/// implement the Add, Sub, and Mul traits for the Expression
impl<F: Field> Add for Expression<F, ProverExpr> {
    type Output = Expression<F, ProverExpr>;
    fn add(self, rhs: Expression<F, ProverExpr>) -> Expression<F, ProverExpr> {
        Expression::<F, ProverExpr>::sum(self, rhs)
    }
}

impl<F: Field> Sub for Expression<F, ProverExpr> {
    type Output = Expression<F, ProverExpr>;
    fn sub(self, rhs: Expression<F, ProverExpr>) -> Expression<F, ProverExpr> {
        self.add(rhs.neg())
    }
}

impl<F: Field> Mul<F> for Expression<F, ProverExpr> {
    type Output = Expression<F, ProverExpr>;
    fn mul(self, rhs: F) -> Self::Output {
        Expression::<F, ProverExpr>::scaled(self, rhs)
    }
}

impl<F: Field> Mul for Expression<F, ProverExpr> {
    type Output = Expression<F, ProverExpr>;
    fn mul(self, rhs: Expression<F, ProverExpr>) -> Expression<F, ProverExpr> {
        Expression::<F, ProverExpr>::products(self, rhs)
    }
}

// defines how the Expressions are printed and displayed
impl<F: std::fmt::Debug + Field> std::fmt::Debug for Expression<F, ProverExpr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Expression")
            .field("Expression_Node", &self.expression_node)
            .field("MleRef_Vec", &self.mle_vec)
            .finish()
    }
}

// defines how the ExpressionNodes are printed and displayed
impl<F: std::fmt::Debug + Field> std::fmt::Debug for ExpressionNode<F, ProverExpr> {
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
            ExpressionNode::Mle(_mle) => f.debug_struct("Mle").field("mle", _mle).finish(),
            ExpressionNode::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            ExpressionNode::Product(a, b) => f.debug_tuple("Product").field(a).field(b).finish(),
            ExpressionNode::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}
