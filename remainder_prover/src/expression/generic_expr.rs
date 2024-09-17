//! Functionality which is common to all "expression"s (see documentation within
//! [crate::expression]). The high-level idea is that an [Expression] is generic
//! over [ExpressionType], and contains within it a single parent [ExpressionNode]
//! as well as an [ExpressionType::MleVec] containing the unique leaf representations
//! for the leaves of the [ExpressionNode] tree.

use crate::mle::MleIndex;
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

/// Different Expression Types corresponds to different stages in the
/// lift cycle of an expression
pub trait ExpressionType<F: Field>: Serialize + for<'de> Deserialize<'de> {
    /// What the expression is over
    /// for prover expression, it's over DenseMle
    /// for verifier expression, it's over Vec<F>
    /// for abstract expression, it's over [TBD]
    type MLENodeRepr: Clone + Serialize + for<'de> Deserialize<'de>; // either index or F

    /// MleRefs is the optional data array of mle_refs
    /// that can be indexed into by the MleRefIndex defind in the ProverExpr
    /// -- this is either unit type for VerifierExpr or
    /// -- Vec<DenseMle> for ProverExpr
    type MleVec: Serialize + for<'de> Deserialize<'de>;
}

/// Generic Expressions
#[derive(Serialize, Deserialize, Clone, Eq, Hash, PartialEq)]
#[serde(bound = "F: Field")]
pub enum ExpressionNode<F: Field, E: ExpressionType<F>> {
    /// This is a constant polynomial
    Constant(F),
    /// This is a virtual selector
    Selector(
        MleIndex<F>,
        Box<ExpressionNode<F, E>>,
        Box<ExpressionNode<F, E>>,
    ),
    /// This is an MLE node, its repr could be
    /// for prover: MleVecIndex (which index into a vec of DenseMle), or
    /// for verifier: a constant field element
    Mle(E::MLENodeRepr),
    /// This is a negated expression node
    Negated(Box<ExpressionNode<F, E>>),
    /// This is the sum of two expression nodes
    Sum(Box<ExpressionNode<F, E>>, Box<ExpressionNode<F, E>>),
    /// This is the product of some MLE nodes, their repr could be
    /// for prover: a vec of MleVecIndex's (which index into a Vec of DenseMle), or
    /// for verifier: a vec of constant field element
    Product(Vec<E::MLENodeRepr>),
    /// This is a scaled expression node
    Scaled(Box<ExpressionNode<F, E>>, F),
}

/// Generic Expressions
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "F: Field")]
pub struct Expression<F: Field, E: ExpressionType<F>> {
    pub expression_node: ExpressionNode<F, E>,
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

    /// get the expression node and mle_vec, mutable
    pub fn deconstruct_mut(&mut self) -> (&mut ExpressionNode<F, E>, &mut E::MleVec) {
        (&mut self.expression_node, &mut self.mle_vec)
    }

    /// get the expression node and mle_vec, deconstruct the expression node
    pub fn deconstruct(self) -> (ExpressionNode<F, E>, E::MleVec) {
        (self.expression_node, self.mle_vec)
    }

    /// traverse the expression tree, and applies the observer_fn to all child node
    /// because the expression node has the recursive structure, the traverse_node
    /// helper function is implemented on it, with the mle_vec reference passed in
    pub fn traverse<D>(
        &self,
        observer_fn: &mut impl FnMut(&ExpressionNode<F, E>, &E::MleVec) -> Result<(), D>,
    ) -> Result<(), D> {
        self.expression_node
            .traverse_node(observer_fn, &self.mle_vec)
    }

    /// similar to traverse, but allows mutation of self (expression node and mle_vec)
    pub fn traverse_mut<D>(
        &mut self,
        observer_fn: &mut impl FnMut(&mut ExpressionNode<F, E>, &mut E::MleVec) -> Result<(), D>,
    ) -> Result<(), D> {
        self.expression_node
            .traverse_node_mut(observer_fn, &mut self.mle_vec)
    }
}

/// generic methods shared across all types of expressions
impl<F: Field, E: ExpressionType<F>> ExpressionNode<F, E> {
    /// traverse the expression tree, and applies the observer_fn to all child node / the mle_vec reference
    pub fn traverse_node<D>(
        &self,
        observer_fn: &mut impl FnMut(&ExpressionNode<F, E>, &E::MleVec) -> Result<(), D>,
        mle_vec: &E::MleVec,
    ) -> Result<(), D> {
        observer_fn(self, mle_vec)?;
        match self {
            ExpressionNode::Constant(_) | ExpressionNode::Mle(_) | ExpressionNode::Product(_) => {
                Ok(())
            }
            ExpressionNode::Negated(exp) => exp.traverse_node(observer_fn, mle_vec),
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
    pub fn traverse_node_mut<D>(
        &mut self,
        observer_fn: &mut impl FnMut(&mut ExpressionNode<F, E>, &mut E::MleVec) -> Result<(), D>,
        mle_vec: &mut E::MleVec,
    ) -> Result<(), D> {
        observer_fn(self, mle_vec)?;
        match self {
            ExpressionNode::Constant(_) | ExpressionNode::Mle(_) | ExpressionNode::Product(_) => {
                Ok(())
            }
            ExpressionNode::Negated(exp) => exp.traverse_node_mut(observer_fn, mle_vec),
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
