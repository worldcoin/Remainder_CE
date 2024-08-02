use super::{
    circuit_expr::{CircuitExpr, CircuitMle},
    expr_errors::ExpressionError,
    generic_expr::{Expression, ExpressionNode, ExpressionType},
    verifier_expr::{VerifierExpr, VerifierMle},
};
use crate::{layer::product::PostSumcheckLayer, mle::Mle};
use crate::{
    layer::product::Product,
    mle::{betavalues::BetaValues, dense::DenseMle, MleIndex},
};
use itertools::Itertools;
use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    collections::{HashMap, HashSet},
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

/// mid-term solution for deduplication of DenseMleRefs
/// basically a wrapper around usize, which denotes the index
/// of the MleRef in an expression's MleRef list/// Generic Expressions
#[derive(Serialize, Deserialize, Clone, Debug)]
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

    /// return the actual mle_ref in the vec within the prover expression
    pub fn get_mle<'a, F: FieldExt>(&self, mle_ref_vec: &'a [DenseMle<F>]) -> &'a DenseMle<F> {
        &mle_ref_vec[self.0]
    }

    /// return the actual mle_ref in the vec within the prover expression
    pub fn get_mle_mut<'a, F: FieldExt>(
        &self,
        mle_ref_vec: &'a mut [DenseMle<F>],
    ) -> &'a mut DenseMle<F> {
        &mut mle_ref_vec[self.0]
    }
}

/// Prover Expression
/// the leaf nodes of the expression tree are DenseMleRefs
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProverExpr;
impl<F: FieldExt> ExpressionType<F> for ProverExpr {
    type MLENodeRepr = MleVecIndex;
    type MleVec = Vec<DenseMle<F>>;
}

/// this is what the prover manipulates to prove the correctness of the computation.
/// Methods here include ones to fix bits, evaluate sumcheck messages, etc.
impl<F: FieldExt> Expression<F, ProverExpr> {
    /// Concatenates two expressions together
    pub fn concat_expr(mut self, lhs: Expression<F, ProverExpr>) -> Self {
        let offset = lhs.num_mle_ref();
        self.increment_mle_vec_indices(offset);

        let (lhs_node, lhs_mle_vec) = lhs.deconstruct();
        let (rhs_node, rhs_mle_vec) = self.deconstruct();

        let concat_node =
            ExpressionNode::Selector(MleIndex::Iterated, Box::new(lhs_node), Box::new(rhs_node));
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

        let mle_node = ExpressionNode::Negated(Box::new(node));

        Expression::new(mle_node, mle_vec)
    }

    /// Create a Sum Expression that contains two MLEs
    pub fn sum(lhs: Self, mut rhs: Self) -> Self {
        let offset = lhs.num_mle_ref();
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
    pub fn num_mle_ref(&self) -> usize {
        self.mle_vec.len()
    }

    /// which increments all the MleVecIndex in the expression by *param* amount
    pub fn increment_mle_vec_indices(&mut self, offset: usize) {
        // define a closure that increments the MleVecIndex by the given amount
        // use traverse_mut
        let mut increment_closure = |expr: &mut ExpressionNode<F, ProverExpr>,
                                     _mle_vec: &mut Vec<DenseMle<F>>|
         -> Result<(), ()> {
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
                | ExpressionNode::Negated(_)
                | ExpressionNode::Selector(_, _, _) => Ok(()),
            }
        };

        self.traverse_mut(&mut increment_closure).unwrap();
    }

    /// Transforms the prover expression to a circuit expression.
    ///
    /// Should only be called for indexed expressions without any bound
    /// variables.
    ///
    /// Traverses the expression and changes the DenseMle to CircuitMle by
    /// ignoring the MLE evaluations and
    ///
    /// If the bookkeeping table has more than 1 element, it
    /// throws an ExpressionError::EvaluateNotFullyBoundError
    pub fn transform_to_circuit_expression(
        &mut self,
    ) -> Result<Expression<F, CircuitExpr>, ExpressionError> {
        self.index_mle_indices(0);

        let (expression_node, mle_vec) = self.deconstruct_mut();

        let expression = Expression::new(
            expression_node
                .transform_to_circuit_expression_node(&mle_vec)
                .unwrap(),
            (),
        );

        Ok(expression)
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
    pub fn transform_to_verifier_expression(
        self,
    ) -> Result<Expression<F, VerifierExpr>, ExpressionError> {
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
    pub fn evaluate_expr(&mut self, challenges: Vec<F>) -> Result<F, ExpressionError> {
        // --- It's as simple as fixing all variables ---
        challenges
            .iter()
            .enumerate()
            .for_each(|(round_idx, &challenge)| {
                self.fix_variable(round_idx, challenge);
            });

        // ----- this is literally a check -----
        let mut observer_fn = |exp: &ExpressionNode<F, ProverExpr>,
                               mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec|
         -> Result<(), ExpressionError> {
            match exp {
                ExpressionNode::Mle(mle_vec_idx) => {
                    let mle_ref = mle_vec_idx.get_mle(mle_vec);

                    let indices = mle_ref
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
                        Err(ExpressionError::EvaluateBoundIndicesDontMatch)
                    }
                }
                ExpressionNode::Product(mle_vec_indices) => {
                    let mle_refs = mle_vec_indices
                        .iter()
                        .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                        .collect_vec();

                    mle_refs
                        .iter()
                        .map(|mle_ref| {
                            let indices = mle_ref
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
                                Err(ExpressionError::EvaluateBoundIndicesDontMatch)
                            }
                        })
                        .try_collect()
                }

                _ => Ok(()),
            }
        };
        self.traverse(&mut observer_fn)?;

        // --- Traverse the expression and pick up all the evals ---
        self.clone()
            .transform_to_verifier_expression()
            .unwrap()
            .evaluate()
    }

    #[allow(clippy::too_many_arguments)]
    /// this evaluates a sumcheck message using the beta cascade algorithm by calling it on the root
    /// node of the expression tree.
    pub fn evaluate_sumcheck_beta_cascade<T>(
        &self,
        constant: &impl Fn(F, &BetaValues<F>) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T, &BetaValues<F>) -> T,
        mle_eval: &impl Fn(&DenseMle<F>, &[F], &[F]) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[&DenseMle<F>], &[F], &[F]) -> T, // changed signature here, note to modify caller's calling code
        scaled: &impl Fn(T, F) -> T,
        beta: &BetaValues<F>,
    ) -> T {
        self.expression_node.evaluate_sumcheck_node_beta_cascade(
            constant,
            selector_column,
            mle_eval,
            negated,
            sum,
            product,
            scaled,
            beta,
            &self.mle_vec,
        )
    }

    /// this traverses the expression tree to get all of the nonlinear rounds. can only be used after indexing the expression.
    /// returns the indices sorted.
    pub fn get_all_nonlinear_rounds(&mut self) -> Vec<usize> {
        let (expression_node, mle_vec) = self.deconstruct_mut();
        let mut curr_nonlinear_indices: Vec<usize> = Vec::new();
        let mut nonlinear_rounds =
            expression_node.get_all_nonlinear_rounds(&mut curr_nonlinear_indices, mle_vec);
        nonlinear_rounds.sort();
        nonlinear_rounds
    }

    /// this traverses the expression tree to get all of the linear rounds. can only be used after indexing the expression.
    /// returns the indices sorted.
    pub fn get_all_linear_rounds(&mut self) -> Vec<usize> {
        let (expression_node, mle_vec) = self.deconstruct_mut();
        let mut linear_rounds = expression_node.get_all_linear_rounds(mle_vec);
        linear_rounds.sort();
        linear_rounds
    }

    /// Mutate the MleIndices that are Iterated in the expression and turn them into IndexedBit
    /// Returns the max number of bits that are indexed
    pub fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        let (expression_node, mle_vec) = self.deconstruct_mut();
        expression_node.index_mle_indices_node(curr_index, mle_vec)
    }

    /// not tested
    /// Gets the size of an expression in terms of the number of rounds of sumcheck
    pub fn get_expression_size(&self, curr_size: usize) -> usize {
        self.expression_node
            .get_expression_size_node(curr_size, &self.mle_vec)
    }

    /// Get the [PostSumcheckLayer] for this expression, which represents the fully bound values of the expression.
    pub fn get_post_sumcheck_layer(&self, multiplier: F) -> PostSumcheckLayer<F, F> {
        self.expression_node
            .get_post_sumcheck_layer(multiplier, &self.mle_vec)
    }

    /// Get the maximum degree of any variable in htis expression
    pub fn get_max_degree(&self) -> usize {
        self.expression_node.get_max_degree(&self.mle_vec)
    }
}

impl<F: FieldExt> ExpressionNode<F, ProverExpr> {
    /// Transforms the expression to a circuit expression
    /// should only be called when no variables are bound in the expression.
    /// Traverses the expression and changes the DenseMle to CircuitMle.
    pub fn transform_to_circuit_expression_node(
        &mut self,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> Result<ExpressionNode<F, CircuitExpr>, ExpressionError> {
        match self {
            ExpressionNode::Constant(scalar) => Ok(ExpressionNode::Constant(*scalar)),
            ExpressionNode::Selector(index, a, b) => Ok(ExpressionNode::Selector(
                index.clone(),
                Box::new(a.transform_to_circuit_expression_node(mle_vec)?),
                Box::new(b.transform_to_circuit_expression_node(mle_vec)?),
            )),
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle(mle_vec);
                Ok(ExpressionNode::Mle(CircuitMle::from_dense_mle(mle)?))
            }
            ExpressionNode::Negated(a) => Ok(ExpressionNode::Negated(Box::new(
                a.transform_to_circuit_expression_node(mle_vec)?,
            ))),
            ExpressionNode::Sum(a, b) => Ok(ExpressionNode::Sum(
                Box::new(a.transform_to_circuit_expression_node(mle_vec)?),
                Box::new(b.transform_to_circuit_expression_node(mle_vec)?),
            )),
            ExpressionNode::Product(mle_vec_indices) => {
                let mles = mle_vec_indices
                    .iter_mut()
                    .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                    .collect_vec();

                Ok(ExpressionNode::Product(
                    mles.into_iter()
                        .map(|mle| CircuitMle::from_dense_mle(mle).unwrap())
                        .collect_vec(),
                ))
            }
            ExpressionNode::Scaled(mle, scalar) => Ok(ExpressionNode::Scaled(
                Box::new(mle.transform_to_circuit_expression_node(mle_vec)?),
                *scalar,
            )),
        }
    }

    /// Transforms the expression to a verifier expression
    /// should only be called when no variables are bound in the expression.
    /// Traverses the expression and changes the DenseMle to CircuitMle.
    pub fn transform_to_verifier_expression_node(
        &mut self,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> Result<ExpressionNode<F, VerifierExpr>, ExpressionError> {
        match self {
            ExpressionNode::Constant(scalar) => Ok(ExpressionNode::Constant(*scalar)),
            ExpressionNode::Selector(index, a, b) => Ok(ExpressionNode::Selector(
                index.clone(),
                Box::new(a.transform_to_verifier_expression_node(mle_vec)?),
                Box::new(b.transform_to_verifier_expression_node(mle_vec)?),
            )),
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle_ref = mle_vec_idx.get_mle(mle_vec);

                if mle_ref.bookkeeping_table().len() != 1 {
                    return Err(ExpressionError::EvaluateNotFullyBoundError);
                }

                let layer_id = mle_ref.get_layer_id();
                let mle_indices = mle_ref.mle_indices().to_vec();
                let eval = mle_ref.bookkeeping_table()[0];

                Ok(ExpressionNode::Mle(VerifierMle::new(
                    layer_id,
                    mle_indices,
                    eval,
                )))
            }
            ExpressionNode::Negated(a) => Ok(ExpressionNode::Negated(Box::new(
                a.transform_to_verifier_expression_node(mle_vec)?,
            ))),
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
                    if mle.bookkeeping_table().len() != 1 {
                        return Err(ExpressionError::EvaluateNotFullyBoundError);
                    }
                }

                Ok(ExpressionNode::Product(
                    mles.into_iter()
                        .map(|mle| {
                            VerifierMle::new(
                                mle.get_layer_id(),
                                mle.mle_indices().to_vec(),
                                mle.bookkeeping_table()[0],
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
                if *index == MleIndex::IndexedBit(round_index) {
                    index.bind_index(challenge);
                } else {
                    a.fix_variable_node(round_index, challenge, mle_vec);
                    b.fix_variable_node(round_index, challenge, mle_vec);
                }
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle_ref = mle_vec_idx.get_mle_mut(mle_vec);

                if mle_ref
                    .mle_indices()
                    .contains(&MleIndex::IndexedBit(round_index))
                {
                    mle_ref.fix_variable(round_index, challenge);
                }
            }
            ExpressionNode::Negated(a) => a.fix_variable_node(round_index, challenge, mle_vec),
            ExpressionNode::Sum(a, b) => {
                a.fix_variable_node(round_index, challenge, mle_vec);
                b.fix_variable_node(round_index, challenge, mle_vec);
            }
            ExpressionNode::Product(mle_vec_indices) => {
                mle_vec_indices
                    .iter_mut()
                    .map(|mle_vec_index| {
                        let mle_ref = mle_vec_index.get_mle_mut(mle_vec);

                        if mle_ref
                            .mle_indices()
                            .contains(&MleIndex::IndexedBit(round_index))
                        {
                            mle_ref.fix_variable(round_index, challenge);
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
                if *index == MleIndex::IndexedBit(round_index) {
                    index.bind_index(challenge);
                } else {
                    a.fix_variable_at_index_node(round_index, challenge, mle_vec);
                    b.fix_variable_at_index_node(round_index, challenge, mle_vec);
                }
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle_ref = mle_vec_idx.get_mle_mut(mle_vec);

                if mle_ref
                    .mle_indices()
                    .contains(&MleIndex::IndexedBit(round_index))
                {
                    mle_ref.fix_variable_at_index(round_index, challenge);
                }
            }
            ExpressionNode::Negated(a) => {
                a.fix_variable_at_index_node(round_index, challenge, mle_vec)
            }
            ExpressionNode::Sum(a, b) => {
                a.fix_variable_at_index_node(round_index, challenge, mle_vec);
                b.fix_variable_at_index_node(round_index, challenge, mle_vec);
            }
            ExpressionNode::Product(mle_vec_indices) => {
                mle_vec_indices
                    .iter_mut()
                    .map(|mle_vec_index| {
                        let mle_ref = mle_vec_index.get_mle_mut(mle_vec);

                        if mle_ref
                            .mle_indices()
                            .contains(&MleIndex::IndexedBit(round_index))
                        {
                            mle_ref.fix_variable_at_index(round_index, challenge);
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

    /// computes the sumcheck message for the given round index, and beta mle
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_sumcheck_node_beta_cascade<T>(
        &self,
        constant: &impl Fn(F, &BetaValues<F>) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T, &BetaValues<F>) -> T,
        mle_eval: &impl Fn(&DenseMle<F>, &[F], &[F]) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[&DenseMle<F>], &[F], &[F]) -> T,
        scaled: &impl Fn(T, F) -> T,
        beta: &BetaValues<F>,
        mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec,
    ) -> T {
        match self {
            ExpressionNode::Constant(scalar) => constant(*scalar, beta),
            ExpressionNode::Selector(index, a, b) => selector_column(
                index,
                a.evaluate_sumcheck_node_beta_cascade(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta,
                    mle_vec,
                ),
                b.evaluate_sumcheck_node_beta_cascade(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta,
                    mle_vec,
                ),
                beta,
            ),
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle_ref = mle_vec_idx.get_mle(mle_vec);
                let (unbound_beta_vec, bound_beta_vec) =
                    beta.get_relevant_beta_unbound_and_bound(mle_ref.mle_indices());
                mle_eval(mle_ref, &unbound_beta_vec, &bound_beta_vec)
            }
            ExpressionNode::Negated(a) => {
                let a = a.evaluate_sumcheck_node_beta_cascade(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta,
                    mle_vec,
                );
                negated(a)
            }
            ExpressionNode::Sum(a, b) => {
                let a = a.evaluate_sumcheck_node_beta_cascade(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta,
                    mle_vec,
                );
                let b = b.evaluate_sumcheck_node_beta_cascade(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta,
                    mle_vec,
                );
                sum(a, b)
            }
            ExpressionNode::Product(mle_vec_indices) => {
                let mle_refs = mle_vec_indices
                    .iter()
                    .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                    .collect_vec();

                let mut unique_mle_indices = HashSet::new();

                let mle_ref_indices_vec = mle_refs
                    .iter()
                    .flat_map(|mle_ref| mle_ref.mle_indices.clone())
                    .filter(move |mle_index| unique_mle_indices.insert(mle_index.clone()))
                    .collect_vec();

                let (unbound_beta_vec, bound_beta_vec) =
                    beta.get_relevant_beta_unbound_and_bound(&mle_ref_indices_vec);

                product(&mle_refs, &unbound_beta_vec, &bound_beta_vec)
            }
            ExpressionNode::Scaled(a, f) => {
                let a = a.evaluate_sumcheck_node_beta_cascade(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta,
                    mle_vec,
                );
                scaled(a, *f)
            }
        }
    }

    /// Mutate the MleIndices that are Iterated in the expression and turn them into IndexedBit
    /// Returns the max number of bits that are indexed
    pub fn index_mle_indices_node(
        &mut self,
        curr_index: usize,
        mle_vec: &mut <ProverExpr as ExpressionType<F>>::MleVec,
    ) -> usize {
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                let mut new_index = curr_index;
                if *mle_index == MleIndex::Iterated {
                    *mle_index = MleIndex::IndexedBit(curr_index);
                    new_index += 1;
                }
                let a_bits = a.index_mle_indices_node(new_index, mle_vec);
                let b_bits = b.index_mle_indices_node(new_index, mle_vec);
                max(a_bits, b_bits)
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle_ref = mle_vec_idx.get_mle_mut(mle_vec);
                mle_ref.index_mle_indices(curr_index)
            }
            ExpressionNode::Sum(a, b) => {
                let a_bits = a.index_mle_indices_node(curr_index, mle_vec);
                let b_bits = b.index_mle_indices_node(curr_index, mle_vec);
                max(a_bits, b_bits)
            }
            ExpressionNode::Product(mle_vec_indices) => mle_vec_indices
                .iter_mut()
                .map(|mle_vec_index| {
                    let mle_ref = mle_vec_index.get_mle_mut(mle_vec);
                    mle_ref.index_mle_indices(curr_index)
                })
                .reduce(max)
                .unwrap_or(curr_index),
            ExpressionNode::Scaled(a, _) => a.index_mle_indices_node(curr_index, mle_vec),
            ExpressionNode::Negated(a) => a.index_mle_indices_node(curr_index, mle_vec),
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
                    let mle_refs = mle_vec_indices
                        .iter()
                        .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                        .collect_vec();
                    let mut product_indices: HashSet<usize> = HashSet::new();
                    mle_refs.into_iter().for_each(|mle_ref| {
                        mle_ref.mle_indices.iter().for_each(|mle_index| {
                            if let MleIndex::IndexedBit(i) = mle_index {
                                product_indices.insert(*i);
                            }
                        })
                    });
                    product_indices
                }
                // in an mle, we need all of the mle indices in the mle.
                ExpressionNode::Mle(mle_vec_idx) => {
                    let mle_ref = mle_vec_idx.get_mle(mle_vec);
                    mle_ref
                        .mle_indices
                        .clone()
                        .into_iter()
                        .filter_map(|mle_index| match mle_index {
                            MleIndex::IndexedBit(i) => Some(i),
                            _ => None,
                        })
                        .collect()
                }
                // in a selector, we traverse each parts of the selector while adding the selector index
                // itself to the total set of all indices in an expression.
                ExpressionNode::Selector(sel_index, a, b) => {
                    let mut sel_indices: HashSet<usize> = HashSet::new();
                    if let MleIndex::IndexedBit(i) = sel_index {
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
                ExpressionNode::Negated(a) => a
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
                    let mle_refs = mle_vec_indices
                        .iter()
                        .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                        .collect_vec();
                    let mut product_nonlinear_indices: HashSet<usize> = HashSet::new();
                    let mut product_indices_counts: HashMap<MleIndex<F>, usize> = HashMap::new();
                    mle_refs.into_iter().for_each(|mle_ref| {
                        mle_ref.mle_indices.iter().for_each(|mle_index| {
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
            ExpressionNode::Selector(_mle_index, a, b) => {
                let a_bits = a.get_expression_size_node(curr_size + 1, mle_vec);
                let b_bits = b.get_expression_size_node(curr_size + 1, mle_vec);
                max(a_bits, b_bits)
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle_ref = mle_vec_idx.get_mle(mle_vec);

                mle_ref
                    .mle_indices()
                    .iter()
                    .filter(|item| {
                        matches!(
                            item,
                            &&MleIndex::Iterated
                                | &&MleIndex::IndexedBit(_)
                                | &&MleIndex::Bound(_, _)
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
                let mle_refs = mle_vec_indices
                    .iter()
                    .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                    .collect_vec();

                mle_refs
                    .iter()
                    .map(|mle_ref| {
                        mle_ref
                            .mle_indices()
                            .iter()
                            .filter(|item| {
                                matches!(
                                    item,
                                    &&MleIndex::Iterated
                                        | &&MleIndex::IndexedBit(_)
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
            ExpressionNode::Negated(a) => a.get_expression_size_node(curr_size, mle_vec),
            ExpressionNode::Constant(_) => curr_size,
        }
    }

    /// Recursively get the [PostSumcheckLayer] for an Expression node, which is the fully bound
    /// representation of an expression.
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
                let mle_ref = mle_vec_idx.get_mle(mle_vec);
                assert_eq!(mle_ref.bookkeeping_table().len(), 1);
                products.push(Product::<F, F>::new(&[mle_ref.clone()], multiplier));
            }
            ExpressionNode::Product(mle_vec_indices) => {
                let mle_refs = mle_vec_indices
                    .iter()
                    .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec).clone())
                    .collect_vec();
                let product = Product::<F, F>::new(&mle_refs, multiplier);
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

    /// Get the maximum degree of an ExpressionNode, recursively.
    fn get_max_degree(&self, mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec) -> usize {
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

impl<F: FieldExt> Neg for Expression<F, ProverExpr> {
    type Output = Expression<F, ProverExpr>;
    fn neg(self) -> Self::Output {
        Expression::<F, ProverExpr>::negated(self)
    }
}

/// implement the Add, Sub, and Mul traits for the Expression
impl<F: FieldExt> Add for Expression<F, ProverExpr> {
    type Output = Expression<F, ProverExpr>;
    fn add(self, rhs: Expression<F, ProverExpr>) -> Expression<F, ProverExpr> {
        Expression::<F, ProverExpr>::sum(self, rhs)
    }
}

impl<F: FieldExt> Sub for Expression<F, ProverExpr> {
    type Output = Expression<F, ProverExpr>;
    fn sub(self, rhs: Expression<F, ProverExpr>) -> Expression<F, ProverExpr> {
        self.add(rhs.neg())
    }
}

impl<F: FieldExt> Mul<F> for Expression<F, ProverExpr> {
    type Output = Expression<F, ProverExpr>;
    fn mul(self, rhs: F) -> Self::Output {
        Expression::<F, ProverExpr>::scaled(self, rhs)
    }
}

// defines how the Expressions are printed and displayed
impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for Expression<F, ProverExpr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Expression")
            .field("Expression_Node", &self.expression_node)
            .field("MleRef_Vec", &self.mle_vec)
            .finish()
    }
}

// defines how the ExpressionNodes are printed and displayed
impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for ExpressionNode<F, ProverExpr> {
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
            ExpressionNode::Mle(_mle_ref) => {
                f.debug_struct("Mle").field("mle_ref", _mle_ref).finish()
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

/// describes the circuit given the expression (includes all the info of the data that the expression is instantiated with)
impl<F: std::fmt::Debug + FieldExt> Expression<F, ProverExpr> {
    pub(crate) fn circuit_description_fmt(&self) -> impl std::fmt::Display + '_ {
        struct CircuitDesc<'a, F: FieldExt>(
            &'a ExpressionNode<F, ProverExpr>,
            &'a <ProverExpr as ExpressionType<F>>::MleVec,
        );

        impl<'a, F: std::fmt::Debug + FieldExt> std::fmt::Display for CircuitDesc<'a, F> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    ExpressionNode::Constant(scalar) => {
                        f.debug_tuple("const").field(scalar).finish()
                    }
                    ExpressionNode::Selector(index, a, b) => f.write_fmt(format_args!(
                        "sel {index:?}; {}; {}",
                        CircuitDesc(a, self.1),
                        CircuitDesc(b, self.1)
                    )),
                    // Skip enum variant and print query struct directly to maintain backwards compatibility.
                    ExpressionNode::Mle(mle_vec_idx) => {
                        let mle_ref = mle_vec_idx.get_mle(self.1);

                        f.debug_struct("mle")
                            .field("layer", &mle_ref.get_layer_id())
                            .field("indices", &mle_ref.mle_indices())
                            .finish()
                    }
                    ExpressionNode::Negated(poly) => {
                        f.write_fmt(format_args!("-{}", CircuitDesc(poly, self.1)))
                    }
                    ExpressionNode::Sum(a, b) => f.write_fmt(format_args!(
                        "+ {}; {}",
                        CircuitDesc(a, self.1),
                        CircuitDesc(b, self.1)
                    )),
                    ExpressionNode::Product(a) => {
                        let str = a
                            .iter()
                            .map(|mle_vec_idx| {
                                let mle = mle_vec_idx.get_mle(self.1);

                                format!("{:?}; {:?}", mle.get_layer_id(), mle.mle_indices())
                            })
                            .reduce(|acc, str| acc + &str)
                            .unwrap();
                        f.write_str(&str)
                    }
                    ExpressionNode::Scaled(poly, scalar) => f.write_fmt(format_args!(
                        "* {}; {:?}",
                        CircuitDesc(poly, self.1),
                        scalar
                    )),
                }
            }
        }

        CircuitDesc(&self.expression_node, &self.mle_vec)
    }
}
