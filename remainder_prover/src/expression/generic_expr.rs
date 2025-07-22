//! Functionality which is common to all "expression"s (see documentation within
//! [crate::expression]). See documentation in [Expression] for high-level
//! summary.

use crate::mle::{AbstractMle, MleIndex};
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};
use std::{cmp::max, hash::Hash};

use anyhow::{Ok, Result};

/// mid-term solution for deduplication of MleRefs
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
    pub fn get_mle<'a, F: Field, M: AbstractMle<F>>(&self, mle_vec: &'a [M]) -> &'a M {
        &mle_vec[self.0]
    }

    /// return the actual mle in the vec within the prover expression
    pub fn get_mle_mut<'a, F: Field, M: AbstractMle<F>>(&self, mle_vec: &'a mut [M]) -> &'a mut M {
        &mut mle_vec[self.0]
    }
}

/// An [ExpressionType] defines two fields -- the type of MLE representation
/// at the leaf of the expression node tree, and the "global" unique copies
/// of each of the MLEs (this is so that if an expression references the
/// same MLE multiple times, the data stored therein is not duplicated)
pub trait ExpressionType<F: Field>: Serialize + for<'de> Deserialize<'de> {
    /// The type of thing representing an MLE within the leaves of an
    /// expression. Note that for most expression types, this is the
    /// intuitive thing (e.g. for [CircuitExpr] this is an [MleDescription<F>]),
    /// but for [ProverExpr] specifically this is an [MleVecIndex], i.e. the
    /// index within the [MleVec] which contains the unique representation
    /// of the prover's view of each MLE.
    type MLENodeRepr: Clone + Serialize + for<'de> Deserialize<'de> + Hash;

    /// The idea here is that an expression may have many MLEs (or things
    /// representing MLEs) in its description, including duplicates, but
    /// we only wish to store one copy for each instance of a thing
    /// representing an MLE. The [MleVec] represents that list of unique
    /// copies.
    /// For example, this is Vec<DenseMle> for [ProverExpr].
    type MleVec: Serialize + for<'de> Deserialize<'de>;
}

/// [ExpressionNode] can be made up of the following:
/// * [ExpressionNode::Constant], i.e. + c for c \in \mathbb{F}
/// * [ExpressionNode::Mle], i.e. \widetilde{V}_{j > i}(b_1, ..., b_{m \leq n})
/// * [ExpressionNode::Product], i.e. \prod_j \widetilde{V}_{j > i}(b_1, ..., b_{m \leq n})
/// * [ExpressionNode::Selector], i.e. (1 - b_0) * Expr(b_1, ..., b_{m \leq n}) + b_0 * Expr(b_1, ..., b_{m \leq n})
/// * [ExpressionNode::Sum], i.e. \widetilde{V}_{j_1 > i}(b_1, ..., b_{m_1 \leq n}) + \widetilde{V}_{j_2 > i}(b_1, ..., b_{m_2 \leq n})
/// * [ExpressionNode::Scaled], i.e. c * Expr(b_1, ..., b_{m \leq n}) for c \in mathbb{F}
#[derive(Serialize, Deserialize, Clone, PartialEq, Hash, Eq)]
#[serde(bound = "F: Field")]
pub enum ExpressionNode<F: Field, E: ExpressionType<F>> {
    /// See documentation for [ExpressionNode]. Note that
    /// [ExpressionNode::Constant] can be an expression tree's leaf.
    Constant(F),
    /// See documentation for [ExpressionNode].
    Selector(
        MleIndex<F>,
        Box<ExpressionNode<F, E>>,
        Box<ExpressionNode<F, E>>,
    ),
    /// An [ExpressionNode] representing the leaf of an expression tree which
    /// is actually mathematically defined as a multilinear extension.
    Mle(E::MLENodeRepr),
    /// See documentation for [ExpressionNode].
    Sum(Box<ExpressionNode<F, E>>, Box<ExpressionNode<F, E>>),
    /// The product of several multilinear extension functions. This is also
    /// an expression tree's leaf.
    Product(Vec<E::MLENodeRepr>),
    /// See documentation for [ExpressionNode].
    Scaled(Box<ExpressionNode<F, E>>, F),
}

/// The high-level idea is that an [Expression] is generic over [ExpressionType]
/// , and contains within it a single parent [ExpressionNode] as well as an
/// [ExpressionType::MleVec] containing the unique leaf representations for the
/// leaves of the [ExpressionNode] tree.
#[derive(Serialize, Deserialize, Clone, Hash)]
#[serde(bound = "F: Field")]
pub struct Expression<F: Field, E: ExpressionType<F>> {
    /// The root of the expression "tree".
    pub expression_node: ExpressionNode<F, E>,
    /// The unique owned copies of all MLEs which are "leaves" within the
    /// expression "tree".
    pub mle_vec: E::MleVec,
}

/// generic methods shared across all types of expressions
impl<F: Field, E: ExpressionType<F>> Expression<F, E> {
    /// Create a new expression
    pub fn new(expression_node: ExpressionNode<F, E>, mle_vec: E::MleVec) -> Self {
        Self {
            expression_node,
            mle_vec,
        }
    }

    /// Returns a reference to the internal `expression_node` and `mle_vec` fields.
    pub fn deconstruct_ref(&self) -> (&ExpressionNode<F, E>, &E::MleVec) {
        (&self.expression_node, &self.mle_vec)
    }

    /// Returns a mutable reference to the `expression_node` and `mle_vec`
    /// present within the given [Expression].
    pub fn deconstruct_mut(&mut self) -> (&mut ExpressionNode<F, E>, &mut E::MleVec) {
        (&mut self.expression_node, &mut self.mle_vec)
    }

    /// Takes ownership of the [Expression] and returns the owned values to its
    /// internal `expression_node` and `mle_vec`.
    pub fn deconstruct(self) -> (ExpressionNode<F, E>, E::MleVec) {
        (self.expression_node, self.mle_vec)
    }

    /// traverse the expression tree, and applies the observer_fn to all child node
    /// because the expression node has the recursive structure, the traverse_node
    /// helper function is implemented on it, with the mle_vec reference passed in
    pub fn traverse(
        &self,
        observer_fn: &mut impl FnMut(&ExpressionNode<F, E>, &E::MleVec) -> Result<()>,
    ) -> Result<()> {
        self.expression_node
            .traverse_node(observer_fn, &self.mle_vec)
    }

    /// similar to traverse, but allows mutation of self (expression node and mle_vec)
    pub fn traverse_mut(
        &mut self,
        observer_fn: &mut impl FnMut(&mut ExpressionNode<F, E>, &mut E::MleVec) -> Result<()>,
    ) -> Result<()> {
        self.expression_node
            .traverse_node_mut(observer_fn, &mut self.mle_vec)
    }
}

/// Generic helper methods shared across all types of [ExpressionNode]s.
impl<F: Field, E: ExpressionType<F>> ExpressionNode<F, E> {
    /// traverse the expression tree, and applies the observer_fn to all child node / the mle_vec reference
    pub fn traverse_node(
        &self,
        observer_fn: &mut impl FnMut(&ExpressionNode<F, E>, &E::MleVec) -> Result<()>,
        mle_vec: &E::MleVec,
    ) -> Result<()> {
        observer_fn(self, mle_vec)?;
        match self {
            ExpressionNode::Constant(_) | ExpressionNode::Mle(_) | ExpressionNode::Product(_) => {
                Ok(())
            }
            ExpressionNode::Scaled(exp, _) => exp.traverse_node(observer_fn, mle_vec),
            ExpressionNode::Selector(_, lhs, rhs) => {
                lhs.traverse_node(observer_fn, mle_vec)?;
                rhs.traverse_node(observer_fn, mle_vec)
            }
            ExpressionNode::Sum(lhs, rhs) => {
                lhs.traverse_node(observer_fn, mle_vec)?;
                rhs.traverse_node(observer_fn, mle_vec)
            }
        }
    }

    /// similar to traverse, but allows mutation of self (expression node and mle_vec)
    pub fn traverse_node_mut(
        &mut self,
        observer_fn: &mut impl FnMut(&mut ExpressionNode<F, E>, &mut E::MleVec) -> Result<()>,
        mle_vec: &mut E::MleVec,
    ) -> Result<()> {
        observer_fn(self, mle_vec)?;
        match self {
            ExpressionNode::Constant(_) | ExpressionNode::Mle(_) | ExpressionNode::Product(_) => {
                Ok(())
            }
            ExpressionNode::Scaled(exp, _) => exp.traverse_node_mut(observer_fn, mle_vec),
            ExpressionNode::Selector(_, lhs, rhs) => {
                lhs.traverse_node_mut(observer_fn, mle_vec)?;
                rhs.traverse_node_mut(observer_fn, mle_vec)
            }
            ExpressionNode::Sum(lhs, rhs) => {
                lhs.traverse_node_mut(observer_fn, mle_vec)?;
                rhs.traverse_node_mut(observer_fn, mle_vec)
            }
        }
    }

    /// Evaluate the polynomial using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce<T>(
        &self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&E::MLENodeRepr) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[E::MLENodeRepr]) -> T,
        scaled: &impl Fn(T, F) -> T,
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
            ExpressionNode::Product(queries) => product(queries),
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
        mle_vec: &E::MleVec,
    ) -> Vec<usize> {
        let degree_per_index = self.get_rounds_helper(mle_vec);
        (0..degree_per_index.len())
            .filter(|&i| degree_per_index[i] > 0)
            .collect()
    }

    /// traverse an expression tree in order to get all of the nonlinear rounds in an expression.
    pub fn get_all_nonlinear_rounds(
        &self,
        mle_vec: &E::MleVec,
    ) -> Vec<usize> {
        let degree_per_index = self.get_rounds_helper(mle_vec);
        (0..degree_per_index.len())
            .filter(|&i| degree_per_index[i] > 1)
            .collect()
    }

    /// get all of the linear rounds from an expression tree
    pub fn get_all_linear_rounds(
        &self,
        mle_vec: &E::MleVec,
    ) -> Vec<usize> {
        let degree_per_index = self.get_rounds_helper(mle_vec);
        (0..degree_per_index.len())
            .filter(|&i| degree_per_index[i] == 1)
            .collect()
    }

    // a recursive helper for get_all_rounds, get_all_nonlinear_rounds, and get_all_linear_rounds
    fn get_rounds_helper(&self, mle_vec: &E::MleVec) -> Vec<usize> {
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
            ExpressionNode::Product(mle_vec_indices) => {
                let mles = mle_vec_indices
                    .iter()
                    .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                    .collect_vec();
                mles.into_iter().for_each(|mle| {
                    mle.mle_indices.iter().for_each(|mle_index| {
                        if let MleIndex::Indexed(i) = mle_index {
                            add_degree(&mut degree_per_index, i, 1);
                        }
                    })
                });
            }
            // in an mle, we need all of the mle indices in the mle.
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle(mle_vec);
                mle.mle_indices.iter().for_each(|mle_index| {
                    if let MleIndex::Indexed(i) = mle_index {
                        max_degree(&mut degree_per_index, i, 1);
                    }
                });
            }
            // in selector, take the max degree of each children, and add 1 degree to the selector itself
            ExpressionNode::Selector(sel_index, a, b) => {
                if let MleIndex::Indexed(i) = sel_index {
                    add_degree(&mut degree_per_index, *i, 1);
                };
                let a_degree_per_index = a.get_rounds_helper(mle_vec);
                let b_degree_per_index = b.get_rounds_helper(mle_vec);
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
                let a_degree_per_index = a.get_rounds_helper(mle_vec);
                let b_degree_per_index = b.get_rounds_helper(mle_vec);
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
                degree_per_index = a.get_rounds_helper(mle_vec);
            }
            // for a constant there are no new indices.
            ExpressionNode::Constant(_) => {}
        }
        degree_per_index
    }

    /// Get the maximum degree of an ExpressionNode, recursively.
    pub fn get_max_degree(&self, mle_vec: &E::MleVec) -> usize {
        self.get_rounds_helper(mle_vec).into_iter().max().unwrap()
    }
}
