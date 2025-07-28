//! The "pure" polynomial relationship description between the MLE representing
//! a single layer of a "structured" circuit and those representing data from
//! other layers. See documentation in [crate::expression] for more details.

use crate::{
    layer::{gate::BinaryOperation, product::PostSumcheckLayerTree},
    layouter::layouting::CircuitMap,
    mle::{evals::MultilinearExtension, mle_description::MleDescription, MleIndex},
};
use ark_std::log2;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

use remainder_shared_types::{transcript::VerifierTranscript, Field};

use super::{
    expr_errors::ExpressionError,
    generic_expr::{Expression, ExpressionNode, ExpressionType},
    prover_expr::ProverExpr,
    verifier_expr::VerifierExpr,
};

use anyhow::{anyhow, Result};

/// Type for defining [Expression<F, ExprDescription>], the type used
/// for representing expressions in the circuit description.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExprDescription;

// The leaves of an expression of this type contain a [MleDescription], an analogue
// of [crate::mle::dense::DenseMle], storing only metadata related to the MLE,
// without any evaluations.
impl<F: Field> ExpressionType<F> for ExprDescription {
    type MLENodeRepr = MleDescription<F>;
    type MleVec = ();
}

impl<F: Field> Expression<F, ExprDescription> {
    /// Binds the variables of this expression to `point`, and retrieves the
    /// leaf MLE values from the `transcript_reader`.  Returns a `Expression<F,
    /// VerifierExpr>` version of `self`.
    pub fn bind(
        &self,
        point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Expression<F, VerifierExpr>> {
        Ok(Expression::new(
            self.expression_node
                .into_verifier_node(point, transcript_reader)?,
            (),
        ))
    }

    /// Convenience function which creates a trivial [Expression<F, ExprDescription>]
    /// referring to a single MLE.
    pub fn from_mle_desc(mle_desc: MleDescription<F>) -> Self {
        Self {
            expression_node: ExpressionNode::<F, ExprDescription>::Mle(mle_desc),
            mle_vec: (),
        }
    }

    /// Traverses the expression tree to get the indices of all the nonlinear
    /// rounds. Returns a sorted vector of indices.
    pub fn get_all_nonlinear_rounds(&self) -> Vec<usize> {
        self.expression_node.get_all_nonlinear_rounds(&self.mle_vec)
    }

    /// Traverses the expression tree to return all indices within the
    /// expression. Can only be used after indexing the expression.
    pub fn get_all_rounds(&self) -> Vec<usize> {
        self.expression_node.get_all_rounds(&self.mle_vec)
    }

    /// Get the [MleDescription]s for this expression, which are at the leaves of the expression.
    pub fn get_circuit_mles(&self) -> Vec<&MleDescription<F>> {
        let circuit_mles = self.expression_node.get_circuit_mles();
        circuit_mles
    }

    /// Label the free variables in an expression.
    pub fn index_mle_vars(&mut self, start_index: usize) {
        self.expression_node.index_mle_vars(start_index);
    }

    /// Get the [Expression<F, ProverExpr>] corresponding to this [Expression<F, ExprDescription>] using the
    /// associated data in the [CircuitMap].
    pub fn into_prover_expression(&self, circuit_map: &CircuitMap<F>) -> Expression<F, ProverExpr> {
        self.expression_node.into_prover_expression(circuit_map)
    }

    /// Get the [PostSumcheckLayer] for this expression, which represents the fully bound values of the expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    pub fn get_post_sumcheck_layer(&self, challenges: &[F]) -> PostSumcheckLayerTree<F, Option<F>> {
        self.expression_node
            .get_post_sumcheck_layer(challenges, &self.mle_vec)
    }

    /// Get the maximum degree of any variable in this expression.
    pub fn get_max_degree(&self) -> usize {
        self.expression_node.get_max_degree(&self.mle_vec)
    }

    /// Returns the maximum degree of b_{curr_round} within an expression
    /// (and therefore the number of prover messages we need to send)
    pub fn get_round_degree(&self, curr_round: usize) -> usize {
        let mut get_degree_closure = |expr: &ExpressionNode<F, ExprDescription>,
                                      _mle_vec: &<ExprDescription as ExpressionType<F>>::MleVec|
         -> Result<usize> {
            Ok(match expr {
                // If encountering an MLE, determing if b_{curr_round} is a variable
                ExpressionNode::Mle(circuit_mle) => {
                    let mle_indices = circuit_mle.var_indices();
                    if mle_indices.contains(&MleIndex::Indexed(curr_round)) {
                        1
                    } else {
                        0
                    }
                }
                ExpressionNode::Constant(_) => 0,
                _ => unreachable!(),
            })
        };

        // By default, all rounds have degree at least 2 (beta table included)
        let round_degree = max(
            1,
            self.reduce(&mut get_degree_closure, &max, &usize::add)
                .unwrap(),
        );
        // add 1 cuz beta table but idk if we would ever use this without a beta table
        round_degree + 1
    }
}

impl<F: Field> ExpressionNode<F, ExprDescription> {
    /// Turn this expression into a [VerifierExpr] which represents a fully bound expression.
    /// Should only be applicable after a full layer of sumcheck.
    pub fn into_verifier_node(
        &self,
        point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<ExpressionNode<F, VerifierExpr>> {
        match self {
            ExpressionNode::Constant(scalar) => Ok(ExpressionNode::Constant(*scalar)),
            ExpressionNode::Selector(index, lhs, rhs) => match index {
                MleIndex::Indexed(idx) => Ok(ExpressionNode::Selector(
                    MleIndex::Bound(point[*idx], *idx),
                    Box::new(lhs.into_verifier_node(point, transcript_reader)?),
                    Box::new(rhs.into_verifier_node(point, transcript_reader)?),
                )),
                _ => Err(anyhow!(ExpressionError::SelectorBitNotBoundError)),
            },
            ExpressionNode::Mle(circuit_mle) => Ok(ExpressionNode::Mle(
                circuit_mle.into_verifier_mle(point, transcript_reader)?,
            )),
            ExpressionNode::Sum(lhs, rhs) => Ok(ExpressionNode::Sum(
                Box::new(lhs.into_verifier_node(point, transcript_reader)?),
                Box::new(rhs.into_verifier_node(point, transcript_reader)?),
            )),
            ExpressionNode::Product(lhs, rhs) => Ok(ExpressionNode::Product(
                Box::new(lhs.into_verifier_node(point, transcript_reader)?),
                Box::new(rhs.into_verifier_node(point, transcript_reader)?),
            )),
            ExpressionNode::Scaled(circuit_mle, scalar) => Ok(ExpressionNode::Scaled(
                Box::new(circuit_mle.into_verifier_node(point, transcript_reader)?),
                *scalar,
            )),
        }
    }

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
            ExpressionNode::Product(a, b) => {
                let a_bookkeeping_table = a.compute_bookkeeping_table(circuit_map)?;
                let b_bookkeeping_table = b.compute_bookkeeping_table(circuit_map)?;
                Some(evaluate_bookkeeping_tables_given_operation(
                    &[
                        (a_bookkeeping_table.to_vec()),
                        (b_bookkeeping_table.to_vec()),
                    ],
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

    /// Evaluate the polynomial using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce<T>(
        &self,
        constant: &mut impl FnMut(F) -> T,
        selector_column: &mut impl FnMut(&MleIndex<F>, T, T) -> T,
        mle_eval: &mut impl FnMut(&<ExprDescription as ExpressionType<F>>::MLENodeRepr) -> T,
        sum: &mut impl FnMut(T, T) -> T,
        product: &mut impl FnMut(T, T) -> T,
        scaled: &mut impl FnMut(T, F) -> T,
    ) -> T {
        match self {
            ExpressionNode::Constant(scalar) => constant(*scalar),
            ExpressionNode::Selector(index, a, b) => {
                let lhs = a.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                let rhs = b.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                selector_column(index, lhs, rhs)
            }
            ExpressionNode::Mle(query) => mle_eval(query),
            ExpressionNode::Sum(a, b) => {
                let a = a.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                let b = b.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                sum(a, b)
            }
            ExpressionNode::Product(a, b) => {
                let a = a.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                let b = b.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                product(a, b)
            }
            ExpressionNode::Scaled(a, f) => {
                let a = a.reduce(constant, selector_column, mle_eval, sum, product, scaled);
                scaled(a, *f)
            }
        }
    }

    /// this traverses the expression to get all of the rounds, in total. requires going through each of the nodes
    /// and collecting the leaf node indices.
    pub(crate) fn get_all_rounds(
        &self,
        mle_vec: &<ExprDescription as ExpressionType<F>>::MleVec,
    ) -> Vec<usize> {
        let degree_per_index = self.get_degree_list(mle_vec);
        (0..degree_per_index.len())
            .filter(|&i| degree_per_index[i] > 0)
            .collect()
    }

    /// traverse an expression tree in order to get all of the nonlinear rounds in an expression.
    pub fn get_all_nonlinear_rounds(
        &self,
        mle_vec: &<ExprDescription as ExpressionType<F>>::MleVec,
    ) -> Vec<usize> {
        let degree_per_index = self.get_degree_list(mle_vec);
        (0..degree_per_index.len())
            .filter(|&i| degree_per_index[i] > 1)
            .collect()
    }

    /// a recursive helper to obtain the degree of every variable
    pub fn get_degree_list(
        &self,
        mle_vec: &<ExprDescription as ExpressionType<F>>::MleVec,
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
            ExpressionNode::Mle(mle) => {
                mle.var_indices().iter().for_each(|mle_index| {
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

    /// Get all the [MleDescription]s, recursively, for this expression by adding the MLEs in the leaves into the vector of MleDescriptions.
    pub fn get_circuit_mles(&self) -> Vec<&MleDescription<F>> {
        let mut circuit_mles: Vec<&MleDescription<F>> = vec![];
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
            ExpressionNode::Product(a, b) => {
                circuit_mles.extend(a.get_circuit_mles());
                circuit_mles.extend(b.get_circuit_mles());
            }
            ExpressionNode::Scaled(a, _scale_factor) => {
                circuit_mles.extend(a.get_circuit_mles());
            }
            ExpressionNode::Constant(_constant) => {}
        }
        circuit_mles
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
            ExpressionNode::Product(a, b) => {
                a.index_mle_vars(start_index);
                b.index_mle_vars(start_index);
            }
            ExpressionNode::Scaled(a, _scale_factor) => {
                a.index_mle_vars(start_index);
            }
            ExpressionNode::Constant(_constant) => {}
        }
    }

    /// Get the [ExpressionNode<F, ProverExpr>] recursively, for this expression.
    pub fn into_prover_expression(&self, circuit_map: &CircuitMap<F>) -> Expression<F, ProverExpr> {
        match self {
            ExpressionNode::Selector(_mle_index, a, b) => a
                .into_prover_expression(circuit_map)
                .select(b.into_prover_expression(circuit_map)),
            ExpressionNode::Sum(a, b) => {
                a.into_prover_expression(circuit_map) + b.into_prover_expression(circuit_map)
            }
            ExpressionNode::Mle(mle) => {
                let prover_mle = mle.into_dense_mle(circuit_map);
                prover_mle.expression()
            }
            ExpressionNode::Product(a, b) => {
                a.into_prover_expression(circuit_map) * b.into_prover_expression(circuit_map)
            }
            ExpressionNode::Scaled(a, scale_factor) => {
                a.into_prover_expression(circuit_map) * *scale_factor
            }
            ExpressionNode::Constant(constant) => Expression::<F, ProverExpr>::constant(*constant),
        }
    }

    /// Recursively get the [PostSumcheckLayerTree] for an Expression node, which is the fully bound
    /// representation of an expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    pub fn get_post_sumcheck_layer(
        &self,
        challenges: &[F],
        _mle_vec: &<VerifierExpr as ExpressionType<F>>::MleVec,
    ) -> PostSumcheckLayerTree<F, Option<F>> {
        match self {
            ExpressionNode::Mle(mle) => PostSumcheckLayerTree::<F, Option<F>>::mle(mle, challenges),
            ExpressionNode::Constant(constant) => PostSumcheckLayerTree::constant(*constant),
            ExpressionNode::Selector(mle_index, a, b) => {
                let idx_val = match mle_index {
                    MleIndex::Indexed(idx) => challenges[*idx],
                    MleIndex::Bound(chal, _idx) => *chal,
                    // TODO(vishady): actually we should just have an assertion that circuit description only
                    // contains indexed bits
                    _ => panic!("should not have any other index here"),
                };
                let left_prod = PostSumcheckLayerTree::<F, Option<F>>::mult(
                    PostSumcheckLayerTree::constant(F::ONE - idx_val),
                    a.get_post_sumcheck_layer(challenges, _mle_vec),
                );
                let right_prod = PostSumcheckLayerTree::<F, Option<F>>::mult(
                    PostSumcheckLayerTree::constant(idx_val),
                    b.get_post_sumcheck_layer(challenges, _mle_vec),
                );
                PostSumcheckLayerTree::<F, Option<F>>::add(left_prod, right_prod)
            }
            ExpressionNode::Sum(a, b) => PostSumcheckLayerTree::<F, Option<F>>::add(
                a.get_post_sumcheck_layer(challenges, _mle_vec),
                b.get_post_sumcheck_layer(challenges, _mle_vec),
            ),
            ExpressionNode::Product(a, b) => PostSumcheckLayerTree::<F, Option<F>>::mult(
                a.get_post_sumcheck_layer(challenges, _mle_vec),
                b.get_post_sumcheck_layer(challenges, _mle_vec),
            ),
            ExpressionNode::Scaled(a, scale_factor) => PostSumcheckLayerTree::<F, Option<F>>::mult(
                a.get_post_sumcheck_layer(challenges, _mle_vec),
                PostSumcheckLayerTree::constant(*scale_factor),
            ),
        }
    }

    /// Get the maximum degree of an ExpressionNode, recursively.
    fn get_max_degree(&self, _mle_vec: &<ExprDescription as ExpressionType<F>>::MleVec) -> usize {
        *self.get_degree_list(_mle_vec).iter().max().unwrap_or(&1)
    }

    /// Returns the total number of variables (i.e. number of rounds of sumcheck) within
    /// the MLE representing the output "data" of this particular expression.
    ///
    /// Note that unlike within the `AbstractExpr` case, we don't need to return
    /// a `Result` since all MLEs within a `ExprDescription` are instantiated with their
    /// appropriate number of variables.
    fn get_num_vars(&self) -> usize {
        match self {
            ExpressionNode::Constant(_) => 0,
            ExpressionNode::Selector(_, lhs, rhs) => {
                max(lhs.get_num_vars() + 1, rhs.get_num_vars() + 1)
            }
            ExpressionNode::Mle(circuit_mle_desc) => circuit_mle_desc.num_free_vars(),
            ExpressionNode::Sum(lhs, rhs) => max(lhs.get_num_vars(), rhs.get_num_vars()),
            ExpressionNode::Product(lhs, rhs) => max(lhs.get_num_vars(), rhs.get_num_vars()),
            ExpressionNode::Scaled(expr, _) => expr.get_num_vars(),
        }
    }
}

impl<F: Field> Expression<F, ExprDescription> {
    /// Returns the total number of variables (i.e. number of rounds of sumcheck)
    /// within the MLE representing the output "data" of this particular expression.
    ///
    /// Note that unlike within the AbstractExpr case, we don't need to return
    /// a `Result` since all MLEs within a `ExprDescription` are instantiated with their appropriate number of variables.
    pub fn num_vars(&self) -> usize {
        self.expression_node.get_num_vars()
    }

    /// `lhs` * `rhs`, as an "`Expression`"
    pub fn products(lhs: Self, rhs: Self) -> Self {
        let (lhs_node, _) = lhs.deconstruct();
        let (rhs_node, _) = rhs.deconstruct();

        let sum_node = ExpressionNode::Product(Box::new(lhs_node), Box::new(rhs_node));

        Expression::new(sum_node, ())
    }

    /// Create a mle Expression that contains one MLE
    pub fn mle(mle: MleDescription<F>) -> Self {
        let mle_node = ExpressionNode::Mle(mle);

        Expression::new(mle_node, ())
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
    pub fn select(self, rhs: Expression<F, ExprDescription>) -> Self {
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

    /// `constant` as a term, but as an "`Expression`"
    pub fn constant(constant: F) -> Self {
        let mle_node = ExpressionNode::Constant(constant);

        Expression::new(mle_node, ())
    }

    /// `-expression`, as an "`Expression`"
    pub fn negated(expression: Self) -> Self {
        let (node, _) = expression.deconstruct();

        let mle_node = ExpressionNode::Scaled(Box::new(node), F::from(1).neg());

        Expression::new(mle_node, ())
    }

    /// `lhs` + `rhs`, as an "`Expression`"
    pub fn sum(lhs: Self, rhs: Self) -> Self {
        let (lhs_node, _) = lhs.deconstruct();
        let (rhs_node, _) = rhs.deconstruct();

        let sum_node = ExpressionNode::Sum(Box::new(lhs_node), Box::new(rhs_node));

        Expression::new(sum_node, ())
    }

    /// scales an Expression by a field element
    pub fn scaled(expression: Expression<F, ExprDescription>, scale: F) -> Self {
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

impl<F: Field> Neg for Expression<F, ExprDescription> {
    type Output = Expression<F, ExprDescription>;
    fn neg(self) -> Self::Output {
        Expression::<F, ExprDescription>::negated(self)
    }
}

/// implement the Add, Sub, and Mul traits for the Expression
impl<F: Field> Add for Expression<F, ExprDescription> {
    type Output = Expression<F, ExprDescription>;
    fn add(self, rhs: Expression<F, ExprDescription>) -> Expression<F, ExprDescription> {
        Expression::<F, ExprDescription>::sum(self, rhs)
    }
}

impl<F: Field> Sub for Expression<F, ExprDescription> {
    type Output = Expression<F, ExprDescription>;
    fn sub(self, rhs: Expression<F, ExprDescription>) -> Expression<F, ExprDescription> {
        self.add(rhs.neg())
    }
}

impl<F: Field> Mul<F> for Expression<F, ExprDescription> {
    type Output = Expression<F, ExprDescription>;
    fn mul(self, rhs: F) -> Self::Output {
        Expression::<F, ExprDescription>::scaled(self, rhs)
    }
}

impl<F: Field> Mul for Expression<F, ExprDescription> {
    type Output = Expression<F, ExprDescription>;
    fn mul(self, rhs: Expression<F, ExprDescription>) -> Expression<F, ExprDescription> {
        Expression::<F, ExprDescription>::products(self, rhs)
    }
}

impl<F: std::fmt::Debug + Field> std::fmt::Debug for Expression<F, ExprDescription> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Circuit Expression")
            .field("Expression_Node", &self.expression_node)
            .finish()
    }
}

impl<F: std::fmt::Debug + Field> std::fmt::Debug for ExpressionNode<F, ExprDescription> {
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
            ExpressionNode::Mle(mle) => f.debug_struct("Circuit Mle").field("mle", mle).finish(),
            ExpressionNode::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            ExpressionNode::Product(a, b) => f.debug_tuple("Product").field(a).field(b).finish(),
            ExpressionNode::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}
