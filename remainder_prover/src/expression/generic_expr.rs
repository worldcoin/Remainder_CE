use std::{
    fmt::Debug, ops::{Add, Mul, Neg, Sub}
};
use serde::{Deserialize, Serialize};
use crate::mle::MleIndex;
use remainder_shared_types::FieldExt;


/// Different Expression Types corresponds to different stages in the
/// lift cycle of an expression
pub trait ExpressionType<F: FieldExt>: Serialize + for<'de> Deserialize<'de> {
    /// What the expression is over
    /// for prover expression, it's over DenseMleRef
    /// for verifier expression, it's over Vec<F>
    /// for abstract expression, it's over [TBD]
    type Container: Clone + Serialize + for<'de> Deserialize<'de>; // either index or F

    /// MleRefs is the optional data array of mle_refs
    /// that can be indexed into by the MleRefIndex defind in the ProverExpression
    type MleVec: Serialize + for<'de> Deserialize<'de>; // -- this is either unit type or Vec<DenseMleRef>
    
    // type thing: Index / F -- this is container
}

/// Generic Expressions
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub enum ExpressionNode<F: FieldExt, E: ExpressionType<F>> {
    /// constant
    Constant(F),
    /// selector
    Selector(
        MleIndex<F>,
        Box<ExpressionNode<F, E>>,
        Box<ExpressionNode<F, E>>,
    ),
    /// Mle
    Mle(E::Container),
    /// negated expression
    Negated(Box<ExpressionNode<F, E>>),
    /// sum of two expressions
    Sum(Box<ExpressionNode<F, E>>, Box<ExpressionNode<F, E>>),
    /// product of multiple Mles
    Product(Vec<E::Container>),
    /// scaled expression
    Scaled(Box<ExpressionNode<F, E>>, F),
}

/// Generic Expressions
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct Expression<F: FieldExt, E: ExpressionType<F>> {
    expression_node: ExpressionNode<F, E>,
    mle_vec: E::MleVec,
}

/// generic methods shared across all types of expressions
impl<F: FieldExt, E: ExpressionType<F>> Expression<F, E> {

    /// Create a new expression 
    pub fn new(expression_node: ExpressionNode<F, E>, mle_vec: E::MleVec) -> Self {
        Self {
            expression_node,
            mle_vec,
        }
    }

    /// get the expression node
    pub fn expression_node(&self) -> &ExpressionNode<F, E> {
        &self.expression_node
    }

    /// get the expression node, mutable
    pub fn expression_node_mut(&mut self) -> &mut ExpressionNode<F, E> {
        &mut self.expression_node
    }

    /// get the mle_vec
    pub fn mle_vec(&self) -> &E::MleVec {
        &self.mle_vec
    }

    /// get the mle_vec, mutable
    pub fn mle_vec_mut(&mut self) -> &mut E::MleVec {
        &mut self.mle_vec
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
        self.expression_node.traverse_node(observer_fn, &self.mle_vec)
    }

    /// similar to traverse, but allows mutation of self (expression node and mle_vec)
    pub fn traverse_mut<D>(
        &mut self,
        observer_fn: &mut impl FnMut(&mut ExpressionNode<F, E>, &mut E::MleVec) -> Result<(), D>,
    ) -> Result<(), D> {
        self.expression_node.traverse_node_mut(observer_fn, &mut self.mle_vec)
    }
}

/// generic methods shared across all types of expressions
impl<F: FieldExt, E: ExpressionType<F>> ExpressionNode<F, E> {

    /// traverse the expression tree, and applies the observer_fn to all child node / the mle_vec reference
    pub fn traverse_node<D>(
        &self,
        observer_fn: &mut impl FnMut(&ExpressionNode<F, E>, &E::MleVec) -> Result<(), D>,
        mle_vec: &E::MleVec,
    ) -> Result<(), D> {
        observer_fn(self, mle_vec)?;
        match self {
            ExpressionNode::Constant(_)
            | ExpressionNode::Mle(_)
            | ExpressionNode::Product(_) => Ok(()),
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
        // dbg!(&self);
        observer_fn(self, mle_vec)?;
        match self {
            ExpressionNode::Constant(_)
            | ExpressionNode::Mle(_)
            | ExpressionNode::Product(_) => Ok(()),
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