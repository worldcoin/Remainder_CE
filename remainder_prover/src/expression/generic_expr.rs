use std::{
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
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
    type Container: Serialize + for<'de> Deserialize<'de>;
}

/// Generic Expressions
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub enum Expression<F: FieldExt, E: ExpressionType<F>> {
    /// constant
    Constant(F),
    /// selector
    Selector(
        MleIndex<F>,
        Box<Expression<F, E>>,
        Box<Expression<F, E>>,
    ),
    /// Mle
    Mle(E::Container),
    /// negated expression
    Negated(Box<Expression<F, E>>),
    /// sum of two expressions
    Sum(Box<Expression<F, E>>, Box<Expression<F, E>>),
    /// product of multiple Mles
    Product(Vec<E::Container>),
    /// scaled expression
    Scaled(Box<Expression<F, E>>, F),
}

/// generic methods shared across all types of expressions
impl<F: FieldExt, E: ExpressionType<F>> Expression<F, E> {

    /// traverse the expression tree, and applies the observer_fn to all child node
    pub fn traverse<D>(
        &self,
        observer_fn: &mut impl FnMut(&Expression<F, E>) -> Result<(), D>,
    ) -> Result<(), D> {
        observer_fn(self)?;
        match self {
            Expression::Constant(_)
            | Expression::Mle(_)
            | Expression::Product(_) => Ok(()),
            Expression::Negated(exp) => exp.traverse(observer_fn),
            Expression::Scaled(exp, _) => exp.traverse(observer_fn),
            Expression::Selector(_, lhs, rhs) => {
                lhs.traverse(observer_fn)?;
                rhs.traverse(observer_fn)
            }
            Expression::Sum(lhs, rhs) => {
                lhs.traverse(observer_fn)?;
                rhs.traverse(observer_fn)
            }
        }
    }

    /// similar to traverse, but allows mutation of self
    pub fn traverse_mut<D>(
        &mut self,
        observer_fn: &mut impl FnMut(&mut Expression<F, E>) -> Result<(), D>,
    ) -> Result<(), D> {
        // dbg!(&self);
        observer_fn(self)?;
        match self {
            Expression::Constant(_)
            | Expression::Mle(_)
            | Expression::Product(_) => Ok(()),
            Expression::Negated(exp) => exp.traverse_mut(observer_fn),
            Expression::Scaled(exp, _) => exp.traverse_mut(observer_fn),
            Expression::Selector(_, lhs, rhs) => {
                lhs.traverse_mut(observer_fn)?;
                rhs.traverse_mut(observer_fn)
            }
            Expression::Sum(lhs, rhs) => {
                lhs.traverse_mut(observer_fn)?;
                rhs.traverse_mut(observer_fn)
            }
        }
    }

    /// Concatenates two expressions together
    pub fn concat_expr(self, lhs: Expression<F, E>) -> Expression<F, E> {
        Expression::Selector(MleIndex::Iterated, Box::new(lhs), Box::new(self))
    }

    /// Create a product Expression that multiplies many MLEs together
    pub fn products(product_list: Vec<E::Container>) -> Self {
        Self::Product(product_list)
    }
}


// the following implements the basic arithmetic operations for the generic expression
impl<F: FieldExt, E: ExpressionType<F>> Neg for Expression<F, E> {
    type Output = Expression<F, E>;
    fn neg(self) -> Self::Output {
        Expression::Negated(Box::new(self))
    }
}

impl<F: FieldExt, E: ExpressionType<F>> Add for Expression<F, E> {
    type Output = Expression<F, E>;
    fn add(self, rhs: Expression<F, E>) -> Expression<F, E> {
        Expression::Sum(Box::new(self), Box::new(rhs))
    }
}

impl<F: FieldExt, E: ExpressionType<F>> Sub for Expression<F, E> {
    type Output = Expression<F, E>;
    fn sub(self, rhs: Expression<F, E>) -> Expression<F, E> {
        Expression::Sum(Box::new(self), Box::new(rhs.neg()))
    }
}

impl<F: FieldExt, E: ExpressionType<F>> Mul<F> for Expression<F, E> {
    type Output = Expression<F, E>;
    fn mul(self, rhs: F) -> Expression<F, E> {
        Expression::Scaled(Box::new(self), rhs)
    }
}


// defines how the Expressions are printed and displayed
impl<F: std::fmt::Debug + FieldExt, C: Debug, E: ExpressionType<F, Container = C>> std::fmt::Debug for Expression<F, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Constant(scalar) => {
                f.debug_tuple("Constant").field(scalar).finish()
            }
            Expression::Selector(index, a, b) => f
                .debug_tuple("Selector")
                .field(index)
                .field(a)
                .field(b)
                .finish(),
            // Skip enum variant and print query struct directly to maintain backwards compatibility.
            Expression::Mle(_mle_ref) => {
                f.debug_struct("Mle").field("mle_ref", _mle_ref).finish()
            }
            Expression::Negated(poly) => f.debug_tuple("Negated").field(poly).finish(),
            Expression::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            Expression::Product(a) => f.debug_tuple("Product").field(a).finish(),
            Expression::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}
