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
#[derive(Serialize, Deserialize, Clone, Debug)]
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

    /// get the expression node
    pub fn expression_node_mut(&mut self) -> &mut ExpressionNode<F, E> {
        &mut self.expression_node
    }

    /// get the mle_vec
    pub fn mle_vec(&self) -> &E::MleVec {
        &self.mle_vec
    }

    /// get the mle_vec
    pub fn mle_vec_mut(&mut self) -> &mut E::MleVec {
        &mut self.mle_vec
    }

    /// get the expression node and mle_vec
    pub fn deconstruct_mut(&mut self) -> (&mut ExpressionNode<F, E>, &mut E::MleVec) {
        (&mut self.expression_node, &mut self.mle_vec)
    }

    /// get the expression node and mle_vec
    pub fn deconstruct(self) -> (ExpressionNode<F, E>, E::MleVec) {
        (self.expression_node, self.mle_vec)
    }

    /// needs the caller to modiler the observer_fn's depending on ExpressionType
    /// traverse the expression tree, and applies the observer_fn to all child node
    pub fn traverse<D>(
        &self,
        observer_fn: &mut impl FnMut(&ExpressionNode<F, E>, &E::MleVec) -> Result<(), D>,
    ) -> Result<(), D> {
        self.expression_node.traverse_node(observer_fn, &self.mle_vec)
    }

    /// similar to traverse
    /// similar to traverse, but allows mutation of self
    pub fn traverse_mut<D>(
        &mut self,
        observer_fn: &mut impl FnMut(&mut ExpressionNode<F, E>, &mut E::MleVec) -> Result<(), D>,
    ) -> Result<(), D> {
        self.expression_node.traverse_node_mut(observer_fn, &mut self.mle_vec)
    }
}

/// generic methods shared across all types of expressions
impl<F: FieldExt, E: ExpressionType<F>> ExpressionNode<F, E> {

    /// needs the caller to modiler the observer_fn's depending on ExpressionType
    /// traverse the expression tree, and applies the observer_fn to all child node
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

    /// similar to traverse
    /// similar to traverse, but allows mutation of self
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


/// ----------------------------old stuff below------------------------------------

/// generic methods shared across all types of expressions
impl<F: FieldExt, E: ExpressionType<F>> ExpressionNode<F, E> {

    /// needs the caller to modiler the observer_fn's depending on ExpressionType
    /// traverse the expression tree, and applies the observer_fn to all child node
    pub fn traverse<D>(
        &self,
        observer_fn: &mut impl FnMut(&ExpressionNode<F, E>) -> Result<(), D>,
    ) -> Result<(), D> {
        observer_fn(self)?;
        match self {
            ExpressionNode::Constant(_)
            | ExpressionNode::Mle(_)
            | ExpressionNode::Product(_) => Ok(()),
            ExpressionNode::Negated(exp) => exp.traverse(observer_fn),
            ExpressionNode::Scaled(exp, _) => exp.traverse(observer_fn),
            ExpressionNode::Selector(_, lhs, rhs) => {
                lhs.traverse(observer_fn)?;
                rhs.traverse(observer_fn)
            }
            ExpressionNode::Sum(lhs, rhs) => {
                lhs.traverse(observer_fn)?;
                rhs.traverse(observer_fn)
            }
        }
    }

    /// similar to traverse
    /// similar to traverse, but allows mutation of self
    pub fn traverse_mut<D>(
        &mut self,
        observer_fn: &mut impl FnMut(&mut ExpressionNode<F, E>) -> Result<(), D>,
    ) -> Result<(), D> {
        // dbg!(&self);
        observer_fn(self)?;
        match self {
            ExpressionNode::Constant(_)
            | ExpressionNode::Mle(_)
            | ExpressionNode::Product(_) => Ok(()),
            ExpressionNode::Negated(exp) => exp.traverse_mut(observer_fn),
            ExpressionNode::Scaled(exp, _) => exp.traverse_mut(observer_fn),
            ExpressionNode::Selector(_, lhs, rhs) => {
                lhs.traverse_mut(observer_fn)?;
                rhs.traverse_mut(observer_fn)
            }
            ExpressionNode::Sum(lhs, rhs) => {
                lhs.traverse_mut(observer_fn)?;
                rhs.traverse_mut(observer_fn)
            }
        }
    }

    /// yup needs to be abstract / verifier / prover specific
    /// Concatenates two expressions together
    pub fn concat_expr(self, lhs: ExpressionNode<F, E>) -> ExpressionNode<F, E> {
        ExpressionNode::Selector(MleIndex::Iterated, Box::new(lhs), Box::new(self))
    }

    /// well, if it's prover expression then, the MleRefs field needs to be populated
    /// (not generic)
    /// Create a product Expression that multiplies many MLEs together
    pub fn products(product_list: Vec<E::Container>) -> Self {
        Self::Product(product_list)
    }
}


/// needs to be abstract / verifier / prover specific
// the following implements the basic arithmetic operations for the generic expression
impl<F: FieldExt, E: ExpressionType<F>> Neg for ExpressionNode<F, E> {
    type Output = ExpressionNode<F, E>;
    fn neg(self) -> Self::Output {
        ExpressionNode::Negated(Box::new(self))
    }
}

/// what if the rhs introduces only the same expression,
/// understanding is rn, we just duplicate
impl<F: FieldExt, E: ExpressionType<F>> Add for ExpressionNode<F, E> {
    type Output = ExpressionNode<F, E>;
    fn add(self, rhs: ExpressionNode<F, E>) -> ExpressionNode<F, E> {
        ExpressionNode::Sum(Box::new(self), Box::new(rhs))
    }
}

impl<F: FieldExt, E: ExpressionType<F>> Sub for ExpressionNode<F, E> {
    type Output = ExpressionNode<F, E>;
    fn sub(self, rhs: ExpressionNode<F, E>) -> ExpressionNode<F, E> {
        ExpressionNode::Sum(Box::new(self), Box::new(rhs.neg()))
    }
}

impl<F: FieldExt, E: ExpressionType<F>> Mul<F> for ExpressionNode<F, E> {
    type Output = ExpressionNode<F, E>;
    fn mul(self, rhs: F) -> ExpressionNode<F, E> {
        ExpressionNode::Scaled(Box::new(self), rhs)
    }
}


// /// this also needs to be prover / verifier / abstract specific
// // defines how the Expressions are printed and displayed
// impl<F: std::fmt::Debug + FieldExt, C: Debug, E: ExpressionType<F, Container = C>> std::fmt::Debug for ExpressionNode<F, E> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             ExpressionNode::Constant(scalar) => {
//                 f.debug_tuple("Constant").field(scalar).finish()
//             }
//             ExpressionNode::Selector(index, a, b) => f
//                 .debug_tuple("Selector")
//                 .field(index)
//                 .field(a)
//                 .field(b)
//                 .finish(),
//             // Skip enum variant and print query struct directly to maintain backwards compatibility.
//             ExpressionNode::Mle(_mle_ref) => {
//                 f.debug_struct("Mle").field("mle_ref", _mle_ref).finish()
//             }
//             ExpressionNode::Negated(poly) => f.debug_tuple("Negated").field(poly).finish(),
//             ExpressionNode::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
//             ExpressionNode::Product(a) => f.debug_tuple("Product").field(a).finish(),
//             ExpressionNode::Scaled(poly, scalar) => {
//                 f.debug_tuple("Scaled").field(poly).field(scalar).finish()
//             }
//         }
//     }
// }
