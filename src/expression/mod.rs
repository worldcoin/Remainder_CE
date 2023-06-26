//! An expression is a type which allows for expressing the definition of a GKR layer

use std::{
    ops::{Add, Mul, Neg, Sub},
};

use crate::{
    mle::{dense::DenseMleRef},
    FieldExt,
};

///TODO!(Genericise this over the MleRef Trait)
///Expression representing the relationship between the current layer and layers claims are being made on
#[derive(Clone)]
pub enum Expression<'a, F: FieldExt> {
    /// This is a constant polynomial
    Constant(F),
    /// This is a virtual selector
    Selector(Box<Expression<'a, F>>, Box<Expression<'a, F>>),
    /// This is a fixed column queried at a certain relative location
    Mle(DenseMleRef<'a, F>),
    /// This is a negated polynomial
    Negated(Box<Expression<'a, F>>),
    /// This is the sum of two polynomials
    Sum(Box<Expression<'a, F>>, Box<Expression<'a, F>>),
    /// This is the product of two polynomials
    Product(Box<Expression<'a, F>>, Box<Expression<'a, F>>),
    /// This is a scaled polynomial
    Scaled(Box<Expression<'a, F>>, F),
}

impl<'a, F: FieldExt> Expression<'a, F> {
    /// Evaluate the polynomial using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<T>(
        &self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(T, T) -> T,
        mle_eval: &impl Fn(DenseMleRef<'a, F>) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T {
        match self {
            Expression::Constant(scalar) => constant(*scalar),
            Expression::Selector(a, b) => selector_column(
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
            Expression::Mle(query) => mle_eval(query.clone()),
            Expression::Negated(a) => {
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
            Expression::Sum(a, b) => {
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
            Expression::Product(a, b) => {
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
            Expression::Scaled(a, f) => {
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
    pub fn concat(self, lhs: Expression<'a, F>) -> Expression<'a, F> {
        Expression::Selector(Box::new(self), Box::new(lhs))
    }
}

impl<'a, F: std::fmt::Debug + FieldExt> std::fmt::Debug for Expression<'a, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Constant(scalar) => f.debug_tuple("Constant").field(scalar).finish(),
            Expression::Selector(a, b) => f.debug_tuple("Selector").field(a).field(b).finish(),
            // Skip enum variant and print query struct directly to maintain backwards compatibility.
            Expression::Mle(_mle_ref) => f.debug_struct("Mle").finish(),
            Expression::Negated(poly) => f.debug_tuple("Negated").field(poly).finish(),
            Expression::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            Expression::Product(a, b) => f.debug_tuple("Product").field(a).field(b).finish(),
            Expression::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}

impl<'a, F: FieldExt> Neg for Expression<'a, F> {
    type Output = Expression<'a, F>;
    fn neg(self) -> Self::Output {
        Expression::Negated(Box::new(self))
    }
}

impl<'a, F: FieldExt> Add for Expression<'a, F> {
    type Output = Expression<'a, F>;
    fn add(self, rhs: Expression<'a, F>) -> Expression<'a, F> {
        Expression::Sum(Box::new(self), Box::new(rhs))
    }
}

impl<'a, F: FieldExt> Sub for Expression<'a, F> {
    type Output = Expression<'a, F>;
    fn sub(self, rhs: Expression<'a, F>) -> Expression<'a, F> {
        Expression::Sum(Box::new(self), Box::new(rhs.neg()))
    }
}

impl<'a, F: FieldExt> Mul for Expression<'a, F> {
    type Output = Expression<'a, F>;
    fn mul(self, rhs: Expression<'a, F>) -> Expression<'a, F> {
        Expression::Sum(Box::new(self), Box::new(rhs))
    }
}

impl<'a, F: FieldExt> Mul<F> for Expression<'a, F> {
    type Output = Expression<'a, F>;
    fn mul(self, rhs: F) -> Expression<'a, F> {
        Expression::Scaled(Box::new(self), rhs)
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
        let expression1: Expression<Fr> = Expression::Constant(Fr::one());

        let mle = DenseMle::<_, Fr>::new(DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()],
        ));

        let expression3 = Expression::Mle(mle.mle_ref());

        let expression = expression1.clone() + expression3.clone();

        let expression = expression1.clone() * expression;

        let expression = expression1 - expression;

        let expression = expression * Fr::from(2);

        let expression = expression3.concat(expression);

        assert_eq!(format!("{expression:?}"), "Selector(Mle, Scaled(Sum(Constant(BigInt([1, 0, 0, 0])), Negated(Sum(Constant(BigInt([1, 0, 0, 0])), Sum(Constant(BigInt([1, 0, 0, 0])), Mle)))), BigInt([2, 0, 0, 0])))")
    }
}
