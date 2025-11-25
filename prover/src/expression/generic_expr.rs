//! Functionality which is common to all "expression"s (see documentation within
//! [crate::expression]). See documentation in [Expression] for high-level
//! summary.

use crate::mle::MleIndex;
use serde::{Deserialize, Serialize};
use shared_types::Field;
use std::hash::Hash;

use anyhow::{Ok, Result};

/// An [ExpressionType] defines two fields -- the type of MLE representation
/// at the leaf of the expression node tree, and the "global" unique copies
/// of each of the MLEs (this is so that if an expression references the
/// same MLE multiple times, the data stored therein is not duplicated)
pub trait ExpressionType<F: Field>: Serialize + for<'de> Deserialize<'de> {
    /// The type of thing representing an MLE within the leaves of an
    /// expression. Note that for most expression types, this is the
    /// intuitive thing (e.g. for [crate::expression::circuit_expr::ExprDescription]
    /// this is an [crate::mle::mle_description::MleDescription<F>]),
    /// but for [crate::expression::prover_expr::ProverExpr] specifically this
    /// is an [crate::expression::prover_expr::MleVecIndex], i.e. the
    /// index within the `MleVec` which contains the unique representation
    /// of the prover's view of each MLE.
    type MLENodeRepr: Clone + Serialize + for<'de> Deserialize<'de> + Hash;

    /// The idea here is that an expression may have many MLEs (or things
    /// representing MLEs) in its description, including duplicates, but
    /// we only wish to store one copy for each instance of a thing
    /// representing an MLE. The `MleVec` represents that list of unique
    /// copies.
    /// For example, this is `Vec<DenseMle>` for
    /// [crate::expression::prover_expr::ProverExpr].
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
}
