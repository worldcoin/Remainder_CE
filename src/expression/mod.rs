//! An expression is a type which allows for expressing the definition of a GKR layer

use std::{ops::{Add, Mul, Neg, Sub}, fmt::Debug};

use thiserror::Error;

use crate::{
    mle::{dense::DenseMleRef, MleIndex, MleRef},
    FieldExt,
};

pub trait Expression<F: FieldExt>: Debug + Sized {
    type MleRef: MleRef;

    #[allow(clippy::too_many_arguments)]
    fn evaluate<T>(
        &mut self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(&mut MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(Self::MleRef) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T;

    fn concat(self, lhs: Self) -> Self;
}

#[derive(Error, Debug, Clone)]
///Error for handling the parsing and evaluation of expressions
pub enum ExpressionError {
    #[error("Product can only be used with Mles as children")]
    TopLevelProduct,
    #[error("")]
    InvalidMleIndex,
    #[error("Something went wrong while evaluating: {0}")]
    EvaluationError(&'static str)
}

///TODO!(Genericise this over the MleRef Trait)
///Expression representing the relationship between the current layer and layers claims are being made on
#[derive(Clone)]
pub enum ExpressionStandard<F: FieldExt> {
    /// This is a constant polynomial
    Constant(F),
    /// This is a virtual selector
    Selector(MleIndex<F>, Box<ExpressionStandard<F>>, Box<ExpressionStandard<F>>),
    /// This is an MLE
    Mle(DenseMleRef<F>),
    /// This is a negated polynomial
    Negated(Box<ExpressionStandard<F>>),
    /// This is the sum of two polynomials
    Sum(Box<ExpressionStandard<F>>, Box<ExpressionStandard<F>>),
    /// This is the product of two polynomials
    Product(Box<ExpressionStandard<F>>, Box<ExpressionStandard<F>>),
    /// This is a scaled polynomial
    Scaled(Box<ExpressionStandard<F>>, F),
}

impl<F: FieldExt> Expression<F> for ExpressionStandard<F> {
    type MleRef = DenseMleRef<F>;
    /// Evaluate the polynomial using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    fn evaluate<T>(
        &mut self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(&mut MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(DenseMleRef<F>) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T {
        match self {
            ExpressionStandard::Constant(scalar) => constant(*scalar),
            ExpressionStandard::Selector(index, a, b) => selector_column(
                index,
                a.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                ),
                b.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                ),
            ),
            ExpressionStandard::Mle(query) => mle_eval(query.clone()),
            ExpressionStandard::Negated(a) => {
                let a = a.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                negated(a)
            }
            ExpressionStandard::Sum(a, b) => {
                let a = a.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                let b = b.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                sum(a, b)
            }
            ExpressionStandard::Product(a, b) => {
                let a = a.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                let b = b.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                product(a, b)
            }
            ExpressionStandard::Scaled(a, f) => {
                let a = a.evaluate(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                scaled(a, *f)
            }
        }
    }

    ///Concatonates two expressions together
    fn concat(self, lhs: ExpressionStandard<F>) -> ExpressionStandard<F> {
        ExpressionStandard::Selector(MleIndex::Iterated, Box::new(self), Box::new(lhs))
    }
}

impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for ExpressionStandard<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpressionStandard::Constant(scalar) => f.debug_tuple("Constant").field(scalar).finish(),
            ExpressionStandard::Selector(index, a, b) => f
                .debug_tuple("Selector")
                .field(index)
                .field(a)
                .field(b)
                .finish(),
            // Skip enum variant and print query struct directly to maintain backwards compatibility.
            ExpressionStandard::Mle(_mle_ref) => f.debug_struct("Mle").finish(),
            ExpressionStandard::Negated(poly) => f.debug_tuple("Negated").field(poly).finish(),
            ExpressionStandard::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            ExpressionStandard::Product(a, b) => f.debug_tuple("Product").field(a).field(b).finish(),
            ExpressionStandard::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}

impl<F: FieldExt> Neg for ExpressionStandard<F> {
    type Output = ExpressionStandard<F>;
    fn neg(self) -> Self::Output {
        ExpressionStandard::Negated(Box::new(self))
    }
}

impl<F: FieldExt> Add for ExpressionStandard<F> {
    type Output = ExpressionStandard<F>;
    fn add(self, rhs: ExpressionStandard<F>) -> ExpressionStandard<F> {
        ExpressionStandard::Sum(Box::new(self), Box::new(rhs))
    }
}

impl<F: FieldExt> Sub for ExpressionStandard<F> {
    type Output = ExpressionStandard<F>;
    fn sub(self, rhs: ExpressionStandard<F>) -> ExpressionStandard<F> {
        ExpressionStandard::Sum(Box::new(self), Box::new(rhs.neg()))
    }
}

impl<F: FieldExt> Mul for ExpressionStandard<F> {
    type Output = ExpressionStandard<F>;
    fn mul(self, rhs: ExpressionStandard<F>) -> ExpressionStandard<F> {
        ExpressionStandard::Product(Box::new(self), Box::new(rhs))
    }
}

impl<F: FieldExt> Mul<F> for ExpressionStandard<F> {
    type Output = ExpressionStandard<F>;
    fn mul(self, rhs: F) -> ExpressionStandard<F> {
        ExpressionStandard::Scaled(Box::new(self), rhs)
    }
}

#[cfg(test)]
mod test {
    use crate::mle::{dense::DenseMle, Mle};

    use super::*;
    use ark_bn254::Fr;
    use ark_poly::DenseMultilinearExtension;
    use ark_std::One;

    #[test]
    fn test_expression_operators() {
        let expression1: ExpressionStandard<Fr> = ExpressionStandard::Constant(Fr::one());

        let mle = DenseMle::<_, Fr>::new(vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()]);

        let expression3 = ExpressionStandard::Mle(mle.mle_ref());

        let expression = expression1.clone() + expression3.clone();

        let expression = expression1.clone() * expression;

        let expression = expression1 - expression;

        let expression = expression * Fr::from(2);

        let expression = expression3.concat(expression);

        assert_eq!(format!("{expression:?}"), "Selector(Iterated, Mle, Scaled(Sum(Constant(BigInt([1, 0, 0, 0])), Negated(Product(Constant(BigInt([1, 0, 0, 0])), Sum(Constant(BigInt([1, 0, 0, 0])), Mle)))), BigInt([2, 0, 0, 0])))")
    }
}
