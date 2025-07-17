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
    generic_expr::{Expression, ExpressionNode, ExpressionType},
    verifier_expr::VerifierExpr,
};
use crate::{
    layer::product::Product,
    mle::{betavalues::BetaValues, dense::DenseMle, MleIndex},
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
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    collections::{HashMap, HashSet},
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

use anyhow::{anyhow, Ok, Result};

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

    /// Create a product Expression that raises one MLE to a given power
    pub fn pow(pow: usize, mle: DenseMle<F>) -> Self {
        let mle_vec_indices = (0..pow).map(|_index| MleVecIndex::new(0)).collect_vec();

        let product_node = ExpressionNode::Product(mle_vec_indices);

        Expression::new(product_node, vec![mle])
    }

    /// Create a product Expression that multiplies many MLEs together
    pub fn products(product_list: <ProverExpr as ExpressionType<F>>::MleVec) -> Self {
        let mle_vec_indices = (0..product_list.len()).map(MleVecIndex::new).collect_vec();

        let product_node = ExpressionNode::Product(mle_vec_indices);

        Expression::new(product_node, product_list)
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
                ExpressionNode::Product(mle_indices) => {
                    for mle_vec_index in mle_indices {
                        mle_vec_index.increment(offset);
                    }
                    Ok(())
                }
                ExpressionNode::Constant(_)
                | ExpressionNode::Scaled(_, _)
                | ExpressionNode::Sum(_, _)
                | ExpressionNode::Selector(_, _, _) => Ok(()),
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
                        })
                        .try_collect()
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
        self.expression_node.evaluate_sumcheck_node_beta_cascade(
            beta,
            &self.mle_vec,
            random_coefficients,
            round_index,
            degree,
        )
    }

    /// This evaluates a sumcheck message using the beta cascade algorithm, taking the sum
    /// of the expression over all the variables `round_index` and after. For the variables
    /// before, we compute the fully bound beta equality MLE and scale the rest of the sum
    /// by this value.
    pub fn evaluate_sumcheck_node_beta_cascade_sum(
        &self,
        beta_values: &BetaValues<F>,
        round_index: usize,
        degree: usize,
    ) -> SumcheckEvals<F> {
        self.expression_node
            .evaluate_sumcheck_node_beta_cascade_sum(
                beta_values,
                round_index,
                degree,
                &self.mle_vec,
            )
    }

    /// Traverses the expression tree to return all indices within the
    /// expression. Can only be used after indexing the expression.
    pub fn get_all_rounds(&self) -> Vec<usize> {
        let (expression_node, mle_vec) = self.deconstruct_ref();
        let mut curr_indices: Vec<usize> = Vec::new();
        let mut all_rounds = expression_node.get_all_rounds(&mut curr_indices, mle_vec);
        all_rounds.sort();
        all_rounds
    }

    /// this traverses the expression tree to get all of the nonlinear rounds. can only be used after indexing the expression.
    /// returns the indices sorted.
    pub fn get_all_nonlinear_rounds(&self) -> Vec<usize> {
        let (expression_node, mle_vec) = self.deconstruct_ref();
        let mut curr_nonlinear_indices: Vec<usize> = Vec::new();
        let mut nonlinear_rounds =
            expression_node.get_all_nonlinear_rounds(&mut curr_nonlinear_indices, mle_vec);
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
    /// turn them into [MleIndex::Indexed]. Returns the max number of bits
    /// that are indexed.
    pub fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        let (expression_node, mle_vec) = self.deconstruct_mut();
        expression_node.index_mle_indices_node(curr_index, mle_vec)
    }

    /// not tested
    /// Gets the size of an expression in terms of the number of rounds of sumcheck
    pub fn get_expression_size(&self) -> usize {
        self.expression_node
            .get_expression_size_node(0, &self.mle_vec)
    }

    /// Gets the number of free variables in an expression.
    pub fn get_expression_num_free_variables(&self) -> usize {
        self.expression_node
            .get_expression_num_free_variables_node(0, &self.mle_vec)
    }

    /// Get the [PostSumcheckLayer] for this expression, which represents the fully bound values of the expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    pub fn get_post_sumcheck_layer(&self, multiplier: F) -> PostSumcheckLayer<F, F> {
        self.expression_node
            .get_post_sumcheck_layer(multiplier, &self.mle_vec)
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
            ExpressionNode::Product(mle_vec_indices) => {
                let mles = mle_vec_indices
                    .iter_mut()
                    .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                    .collect_vec();

                for mle in mles.iter() {
                    if !mle.is_fully_bounded() {
                        return Err(anyhow!(ExpressionError::EvaluateNotFullyBoundError));
                    }
                }

                Ok(ExpressionNode::Product(
                    mles.into_iter()
                        .map(|mle| {
                            VerifierMle::new(
                                mle.layer_id(),
                                mle.mle_indices().to_vec(),
                                mle.value(),
                            )
                        })
                        .collect_vec(),
                ))
            }
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
            ExpressionNode::Product(mle_vec_indices) => {
                mle_vec_indices
                    .iter_mut()
                    .map(|mle_vec_index| {
                        let mle = mle_vec_index.get_mle_mut(mle_vec);

                        if mle.mle_indices().contains(&MleIndex::Indexed(round_index)) {
                            mle.fix_variable(round_index, challenge);
                        }
                    })
                    .collect_vec();
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
            ExpressionNode::Product(mle_vec_indices) => {
                mle_vec_indices
                    .iter_mut()
                    .map(|mle_vec_index| {
                        let mle = mle_vec_index.get_mle_mut(mle_vec);

                        if mle.mle_indices().contains(&MleIndex::Indexed(round_index)) {
                            mle.fix_variable_at_index(round_index, challenge);
                        }
                    })
                    .collect_vec();
            }
            ExpressionNode::Scaled(a, _) => {
                a.fix_variable_at_index_node(round_index, challenge, mle_vec);
            }
            ExpressionNode::Constant(_) => (),
        }
    }

    pub fn evaluate_sumcheck_node_beta_cascade_sum(
        &self,
        beta_values: &BetaValues<F>,
        round_index: usize,
        degree: usize,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> SumcheckEvals<F> {
        match self {
            ExpressionNode::Constant(constant) => {
                SumcheckEvals(repeat_n(*constant, degree + 1).collect())
            }
            ExpressionNode::Selector(selector_mle_index, lhs, rhs) => {
                let lhs_eval = lhs.evaluate_sumcheck_node_beta_cascade_sum(
                    beta_values,
                    round_index,
                    degree,
                    mle_vec,
                );
                let rhs_eval = rhs.evaluate_sumcheck_node_beta_cascade_sum(
                    beta_values,
                    round_index,
                    degree,
                    mle_vec,
                );
                match selector_mle_index {
                    MleIndex::Indexed(var_number) => {
                        let index_claim = beta_values.get_unbound_value(*var_number).unwrap();
                        (lhs_eval * (F::ONE - index_claim)) + (rhs_eval * index_claim)
                    }
                    MleIndex::Bound(bound_value, var_number) => {
                        let identity = F::ONE;
                        let beta_bound = beta_values
                            .get_updated_value(*var_number)
                            .unwrap_or(identity);
                        ((lhs_eval * (F::ONE - bound_value)) + (rhs_eval * bound_value))
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
                );
                let rhs_eval = rhs.evaluate_sumcheck_node_beta_cascade_sum(
                    beta_values,
                    round_index,
                    degree,
                    mle_vec,
                );
                lhs_eval + rhs_eval
            }
            ExpressionNode::Product(mle_idx_vec) => {
                let (mles, mles_bookkeeping_tables): (Vec<&DenseMle<F>>, Vec<Vec<F>>) = mle_idx_vec
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
                ) * scale
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
    pub fn evaluate_sumcheck_node_beta_cascade(
        &self,
        beta_vec: &[&BetaValues<F>],
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
        random_coefficients: &[F],
        round_index: usize,
        degree: usize,
    ) -> SumcheckEvals<F> {
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
                        let one_minus_index_claim = F::ONE - index_claim;
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
                sumcheck_eval_not_scaled_by_constant * constant
            }

            // the selector is split into three cases:
            // - when the selector bit itself is not the independent variable and hasn't
            //   been bound yet,
            // - when the selector bit is the independent variable
            // - when the selector bit has already been bound we determine which case we
            // are in by comparing the round_index to the selector index which is an
            // argument to the closure.
            ExpressionNode::Selector(index, a, b) => {
                match index {
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
                                        let a_eval: &SumcheckEvals<F> = a;
                                        let b_eval: &SumcheckEvals<F> = b;
                                        // when the selector bit is not the independent variable and
                                        // has not been bound yet, we are simply summing over
                                        // everything. in order to take the beta values into account
                                        // this means for everything on the "left" side of the
                                        // selector we want to multiply by (1 - g_i) and for
                                        // everything on the "right" side of the selector we want to
                                        // multiply by g_i. we can then add these!
                                        let a_with_sel: SumcheckEvals<F> =
                                            a_eval.clone() * (F::ONE - index_claim);
                                        let b_with_sel: SumcheckEvals<F> = b_eval.clone() * index_claim;
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
                                                let one_minus_index_claim = F::ONE - index_claim;
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
                                                                * (F::ONE - F::from(idx as u64))
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
                                                            * F::from(idx as u64) * beta_evals[idx]
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
                    MleIndex::Bound(coeff, _) => {
                        let (lhs_evals, rhs_evals) = (
                            beta_vec
                                .iter()
                                .map(|beta| {
                                    a.evaluate_sumcheck_node_beta_cascade(
                                        &[*beta],
                                        mle_vec,
                                        &[F::ONE],
                                        round_index,
                                        degree,
                                    )
                                })
                                .collect_vec(),
                            beta_vec
                                .iter()
                                .map(|beta| {
                                    b.evaluate_sumcheck_node_beta_cascade(
                                        &[*beta],
                                        mle_vec,
                                        &[F::ONE],
                                        round_index,
                                        degree,
                                    )
                                })
                                .collect_vec(),
                        );
                        let coeff_neg = F::ONE - coeff;
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
                }
            }
            // the mle evaluation takes in the mle ref, and the corresponding unbound
            // and bound beta values to pass into the `beta_cascade` function
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle(mle_vec);
                let (unbound_beta_vec, bound_beta_vec): (Vec<Vec<F>>, Vec<Vec<F>>) = beta_vec
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
                );
                let b = b.evaluate_sumcheck_node_beta_cascade(
                    beta_vec,
                    mle_vec,
                    random_coefficients,
                    round_index,
                    degree,
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

                let (unbound_beta_vec, bound_beta_vec): (Vec<Vec<F>>, Vec<Vec<F>>) = beta_vec
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
                );
                a * scale
            }
        }
    }

    /// Mutate the MLE indices that are [MleIndex::Free] in the expression and
    /// turn them into [MleIndex::Indexed]. Returns the max number of bits
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

    /// this traverses the expression to get all of the rounds, in total. requires going through each of the nodes
    /// and collecting the leaf node indices.
    pub(crate) fn get_all_rounds(
        &self,
        curr_indices: &mut Vec<usize>,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> Vec<usize> {
        let indices_in_node = {
            match self {
                // in a product, we need the union of all the indices in each of the individual mle refs.
                ExpressionNode::Product(mle_vec_indices) => {
                    let mles = mle_vec_indices
                        .iter()
                        .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                        .collect_vec();
                    let mut product_indices: HashSet<usize> = HashSet::new();
                    mles.into_iter().for_each(|mle| {
                        mle.mle_indices.iter().for_each(|mle_index| {
                            if let MleIndex::Indexed(i) = mle_index {
                                product_indices.insert(*i);
                            }
                        })
                    });
                    product_indices
                }
                // in an mle, we need all of the mle indices in the mle.
                ExpressionNode::Mle(mle_vec_idx) => {
                    let mle = mle_vec_idx.get_mle(mle_vec);
                    mle.mle_indices
                        .clone()
                        .into_iter()
                        .filter_map(|mle_index| match mle_index {
                            MleIndex::Indexed(i) => Some(i),
                            _ => None,
                        })
                        .collect()
                }
                // in a selector, we traverse each parts of the selector while adding the selector index
                // itself to the total set of all indices in an expression.
                ExpressionNode::Selector(sel_index, a, b) => {
                    let mut sel_indices: HashSet<usize> = HashSet::new();
                    if let MleIndex::Indexed(i) = sel_index {
                        sel_indices.insert(*i);
                    };

                    let a_indices = a.get_all_rounds(curr_indices, mle_vec);
                    let b_indices = b.get_all_rounds(curr_indices, mle_vec);
                    a_indices
                        .into_iter()
                        .zip(b_indices)
                        .for_each(|(a_mle_idx, b_mle_idx)| {
                            sel_indices.insert(a_mle_idx);
                            sel_indices.insert(b_mle_idx);
                        });
                    sel_indices
                }
                // we add the indices in each of the parts of the sum.
                ExpressionNode::Sum(a, b) => {
                    let mut sum_indices: HashSet<usize> = HashSet::new();
                    let a_indices = a.get_all_rounds(curr_indices, mle_vec);
                    let b_indices = b.get_all_rounds(curr_indices, mle_vec);
                    a_indices
                        .into_iter()
                        .zip(b_indices)
                        .for_each(|(a_mle_idx, b_mle_idx)| {
                            sum_indices.insert(a_mle_idx);
                            sum_indices.insert(b_mle_idx);
                        });
                    sum_indices
                }
                // for scaled and negated, we can add all of the indices found in the expression being negated or scaled.
                ExpressionNode::Scaled(a, _) => a
                    .get_all_rounds(curr_indices, mle_vec)
                    .into_iter()
                    .collect(),
                // for a constant there are no new indices.
                ExpressionNode::Constant(_) => HashSet::new(),
            }
        };
        // once all of them have been collected, we can take the union of all of them to grab all of the rounds in an expression.
        indices_in_node.into_iter().for_each(|index| {
            if !curr_indices.contains(&index) {
                curr_indices.push(index);
            }
        });
        curr_indices.clone()
    }

    /// traverse an expression tree in order to get all of the nonlinear rounds in an expression.
    pub fn get_all_nonlinear_rounds(
        &self,
        curr_nonlinear_indices: &mut Vec<usize>,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> Vec<usize> {
        let nonlinear_indices_in_node = {
            match self {
                // the only case where an index is nonlinear is if it is present in multiple mle refs
                // that are part of a product. we iterate through all the indices in the product nodes
                // to look for repeated indices within a single node.
                ExpressionNode::Product(mle_vec_indices) => {
                    let mles = mle_vec_indices
                        .iter()
                        .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                        .collect_vec();
                    let mut product_nonlinear_indices: HashSet<usize> = HashSet::new();
                    let mut product_indices_counts: HashMap<MleIndex<F>, usize> = HashMap::new();
                    mles.into_iter().for_each(|mle| {
                        mle.mle_indices.iter().for_each(|mle_index| {
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
                                if let MleIndex::Indexed(i) = mle_index {
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

    /// get all of the linear rounds from an expression tree
    pub fn get_all_linear_rounds(
        &self,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> Vec<usize> {
        let mut curr_indices: Vec<usize> = Vec::new();
        let mut curr_nonlinear_indices: Vec<usize> = Vec::new();
        // first we get all the rounds of the expression
        let all_indices = &self.get_all_rounds(&mut curr_indices, mle_vec);
        // then we get all of the nonlinear rounds
        let all_nonlinear_indices =
            &self.get_all_nonlinear_rounds(&mut curr_nonlinear_indices, mle_vec);
        let mut all_linear_indices: Vec<usize> = Vec::new();
        // all of the rounds that are in all rounds but not in the nonlinear rounds must be linear rounds.
        all_indices.iter().for_each(|mle_idx| {
            if !all_nonlinear_indices.contains(mle_idx) {
                all_linear_indices.push(*mle_idx);
            }
        });
        all_linear_indices
    }

    /// Gets the size of an expression in terms of the number of rounds of sumcheck
    pub fn get_expression_size_node(
        &self,
        curr_size: usize,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> usize {
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                let (a_bits, b_bits) = if matches!(
                    mle_index,
                    &MleIndex::Free | &MleIndex::Indexed(_) | &MleIndex::Bound(_, _)
                ) {
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
                    .filter(|item| {
                        matches!(
                            item,
                            &&MleIndex::Free | &&MleIndex::Indexed(_) | &&MleIndex::Bound(_, _)
                        )
                    })
                    .collect_vec()
                    .len()
                    + curr_size
            }
            ExpressionNode::Sum(a, b) => {
                let a_bits = a.get_expression_size_node(curr_size, mle_vec);
                let b_bits = b.get_expression_size_node(curr_size, mle_vec);
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
                            .filter(|item| {
                                matches!(
                                    item,
                                    &&MleIndex::Free
                                        | &&MleIndex::Indexed(_)
                                        | &&MleIndex::Bound(_, _)
                                )
                            })
                            .collect_vec()
                            .len()
                    })
                    .max()
                    .unwrap_or(0)
                    + curr_size
            }
            ExpressionNode::Scaled(a, _) => a.get_expression_size_node(curr_size, mle_vec),
            ExpressionNode::Constant(_) => curr_size,
        }
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
    pub fn get_post_sumcheck_layer(
        &self,
        multiplier: F,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
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
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle(mle_vec);
                assert!(mle.is_fully_bounded());
                products.push(Product::<F, F>::new(&[mle.clone()], multiplier));
            }
            ExpressionNode::Product(mle_vec_indices) => {
                let mles = mle_vec_indices
                    .iter()
                    .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec).clone())
                    .collect_vec();
                let product = Product::<F, F>::new(&mles, multiplier);
                products.push(product);
            }
            ExpressionNode::Scaled(a, scale_factor) => {
                let acc = multiplier * scale_factor;
                products.extend(a.get_post_sumcheck_layer(acc, mle_vec).0);
            }
            ExpressionNode::Constant(constant) => {
                products.push(Product::<F, F>::new(&[], *constant * multiplier));
            }
        }
        PostSumcheckLayer(products)
    }

    /// Get the maximum degree of an ExpressionNode, recursively.
    fn get_max_degree(&self, _mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec) -> usize {
        match self {
            ExpressionNode::Selector(_, a, b) | ExpressionNode::Sum(a, b) => {
                let a_degree = a.get_max_degree(_mle_vec);
                let b_degree = b.get_max_degree(_mle_vec);
                max(a_degree, b_degree)
            }
            ExpressionNode::Mle(_) => {
                // 1 for the current MLE
                1
            }
            ExpressionNode::Product(mles) => {
                // max degree is the number of MLEs in a product
                mles.len()
            }
            ExpressionNode::Scaled(a, _) => a.get_max_degree(_mle_vec),
            ExpressionNode::Constant(_) => 1,
        }
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
            ExpressionNode::Product(a) => f.debug_tuple("Product").field(a).finish(),
            ExpressionNode::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}
