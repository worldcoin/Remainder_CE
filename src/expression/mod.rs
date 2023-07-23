//! An expression is a type which allows for expressing the definition of a GKR layer

use std::{
    cmp::max,
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

use thiserror::Error;

use crate::{
    mle::{dense::DenseMleRef, MleIndex, MleRef, beta::*},
    FieldExt,
    layer::Claim,
};

///trait that defines what an Expression needs to be able to do
///TODO!(Fix to make this more general)
pub trait Expression<F: FieldExt>: Debug + Sized {
    ///The MleRef that this Expression contains
    type MleRef: MleRef<F = F>;

    #[allow(clippy::too_many_arguments)]
    ///Evaluate an expression and return a custom type
    fn evaluate<T>(
        &mut self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&Self::MleRef, Option<&mut BetaTable<F>>) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[Self::MleRef], Option<&mut BetaTable<F>>) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T;

    /// Traverses the expression tree, similarly to `evaluate()`, but with a single
    /// "observer" function which is called at each node. Also takes an immutable reference
    /// to `self` rather than a mutable one (as in `evaluate()`).
    fn traverse<E>(
        &self,
        observer_fn: &mut impl FnMut(&ExpressionStandard<F>) -> Result<(), E>,
    ) -> Result<(), E>;

    ///Add two expressions together
    fn concat(self, lhs: Self) -> Self;

    ///Fix the bit corresponding to `round_index` to `challenge` mutating the MleRefs
    /// so they are accurate as Bookeeping Tables
    fn fix_variable(&mut self, round_index: usize, challenge: F);
}

#[derive(Error, Debug, Clone)]
///Error for handling the parsing and evaluation of expressions
pub enum ExpressionError {
    ///Error for when an InvalidMleIndex is found while evaluating an expression
    /// TODO!(add some diagnoistics here)
    #[error("")]
    InvalidMleIndex,
    ///Error for when Something unlikely goes wrong while evaluating an expression
    /// TODO!(split this up into many error variants)
    #[error("Something went wrong while evaluating: {0}")]
    EvaluationError(&'static str),
    ///Error that wraps an MleError
    /// TODO!(Do we even need this?)
    #[error("Something went wrong while evaluating the MLE")]
    MleError,
    ///Error when there is no beta table!!!!!!
    #[error("No beta table")]
    BetaError,
}

///TODO!(Genericise this over the MleRef Trait)
///Expression representing the relationship between the current layer and layers claims are being made on
#[derive(Clone)]
pub enum ExpressionStandard<F: FieldExt> {
    /// This is a constant polynomial
    Constant(F),
    /// This is a virtual selector
    Selector(
        MleIndex<F>,
        Box<ExpressionStandard<F>>,
        Box<ExpressionStandard<F>>,
    ),
    /// This is an MLE
    Mle(DenseMleRef<F>, Option<BetaTable<F>>),
    /// This is a negated polynomial
    Negated(Box<ExpressionStandard<F>>),
    /// This is the sum of two polynomials
    Sum(Box<ExpressionStandard<F>>, Box<ExpressionStandard<F>>),
    /// This is the product of some polynomials
    Product(Vec<DenseMleRef<F>>, Option<BetaTable<F>>),
    /// This is a scaled polynomial; Optionally a MleIndex to represent a fully bound mle that was this scalar
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
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&DenseMleRef<F>, Option<&mut BetaTable<F>>) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[DenseMleRef<F>], Option<&mut BetaTable<F>>) -> T,
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
            ExpressionStandard::Mle(query, table) => mle_eval(query, table.as_mut()),
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
            ExpressionStandard::Product(queries, table) => {
                product(queries, table.as_mut())
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

    fn traverse<E>(
        &self,
        observer_fn: &mut impl FnMut(&ExpressionStandard<F>) -> Result<(), E>,
    ) -> Result<(), E> {
        match self {
            ExpressionStandard::Constant(_)
            | ExpressionStandard::Mle(_, _) => observer_fn(self),
            ExpressionStandard::Negated(exp) => observer_fn(exp),
            ExpressionStandard::Product(_, _) => observer_fn(self),
            ExpressionStandard::Scaled(exp, _) => exp.traverse(observer_fn),
            ExpressionStandard::Selector(_, lhs, rhs) => {
                observer_fn(self)?;
                lhs.traverse(observer_fn)?;
                rhs.traverse(observer_fn)
            }
            ExpressionStandard::Sum(lhs, rhs) => {
                lhs.traverse(observer_fn)?;
                rhs.traverse(observer_fn)
            }
        }
    }

    ///Concatenates two expressions together
    fn concat(self, lhs: ExpressionStandard<F>) -> ExpressionStandard<F> {
        ExpressionStandard::Selector(MleIndex::Iterated, Box::new(self), Box::new(lhs))
    }

    fn fix_variable(&mut self, round_index: usize, challenge: F) {
        match self {
            ExpressionStandard::Selector(index, a, b) => {
                if *index == MleIndex::IndexedBit(round_index) {
                    *index = MleIndex::Bound(challenge);
                } else {
                    a.fix_variable(round_index, challenge);
                    b.fix_variable(round_index, challenge);
                }
            }
            ExpressionStandard::Mle(mle_ref, betatable) => {
                // update the beta table whenever you fix variable
                let _ = betatable.as_mut().unwrap().beta_update(round_index, challenge);
                if mle_ref
                    .mle_indices()
                    .contains(&MleIndex::IndexedBit(round_index))
                {
                    mle_ref.fix_variable(round_index, challenge);
                }
            }
            ExpressionStandard::Negated(a) => a.fix_variable(round_index, challenge),
            ExpressionStandard::Sum(a, b) => {
                a.fix_variable(round_index, challenge);
                b.fix_variable(round_index, challenge);
            }
            ExpressionStandard::Product(mle_refs, betatable) => {
                // update the beta table every time you fix variable
                let table = betatable.as_mut().unwrap();
                let _ = table.beta_update(round_index, challenge);
                *betatable = Some(table.clone());
                for mle_ref in mle_refs {
                    if mle_ref
                        .mle_indices()
                        .contains(&MleIndex::IndexedBit(round_index))
                    {
                        mle_ref.fix_variable(round_index, challenge);
                    }
                }
            }
            ExpressionStandard::Scaled(a, _) => {
                a.fix_variable(round_index, challenge);
            }
            ExpressionStandard::Constant(_) => (),
        }
    }

}

impl<F: FieldExt> ExpressionStandard<F> {
    ///Create a product Expression that multiplies many MLEs together
    pub fn products(product_list: Vec<DenseMleRef<F>>) -> Self {
        Self::Product(product_list, None)
    }

    /// Initializes all beta tables within the current Expression
    pub fn init_beta_tables(&mut self, layer_claim: Claim<F>) {
        match self {
            ExpressionStandard::Mle(mle_ref, beta_table) => {
                let init_table = Some(BetaTable::new(layer_claim, &[mle_ref.clone()]).unwrap());
                *beta_table = init_table.clone();
            }
            ExpressionStandard::Product(mle_refs, beta_table) => {
                let init_table = Some(BetaTable::new(layer_claim, mle_refs).unwrap());
                *beta_table = init_table;
            }
            ExpressionStandard::Selector(mle_index, a, b) => {
                a.init_beta_tables(layer_claim.clone());
                b.init_beta_tables(layer_claim);
            }
            ExpressionStandard::Sum(a, b) => {
                a.init_beta_tables(layer_claim.clone());
                b.init_beta_tables(layer_claim);
            }
            ExpressionStandard::Scaled(a, _) => a.init_beta_tables(layer_claim),
            ExpressionStandard::Negated(a) => a.init_beta_tables(layer_claim),
            ExpressionStandard::Constant(_) => {},
        }
    }

    ///Mutate the MleIndices that are Iterated in the expression and turn them into IndexedBit
    /// Returns the max number of bits that are indexed
    pub fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        match self {
            ExpressionStandard::Selector(mle_index, a, b) => {
                *mle_index = MleIndex::IndexedBit(curr_index);
                let a_bits = a.index_mle_indices(curr_index + 1);
                let b_bits = b.index_mle_indices(curr_index + 1);
                max(a_bits, b_bits)
            }
            ExpressionStandard::Mle(mle_ref, betatable) => {
                mle_ref.index_mle_indices(curr_index)
            }
            ExpressionStandard::Sum(a, b) => {
                let a_bits = a.index_mle_indices(curr_index);
                let b_bits = b.index_mle_indices(curr_index);
                max(a_bits, b_bits)
            }
            ExpressionStandard::Product(mle_refs, betatable) => {
                mle_refs
                .iter_mut()
                .map(|mle_ref| mle_ref.index_mle_indices(curr_index))
                .reduce(max)
                .unwrap_or(curr_index)
            }
            ExpressionStandard::Scaled(a, _) => a.index_mle_indices(curr_index),
            ExpressionStandard::Negated(a) => a.index_mle_indices(curr_index),
            ExpressionStandard::Constant(_) => curr_index,
        }
    }
}

impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for ExpressionStandard<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpressionStandard::Constant(scalar) => {
                f.debug_tuple("Constant").field(scalar).finish()
            }
            ExpressionStandard::Selector(index, a, b) => f
                .debug_tuple("Selector")
                .field(index)
                .field(a)
                .field(b)
                .finish(),
            // Skip enum variant and print query struct directly to maintain backwards compatibility.
            ExpressionStandard::Mle(_mle_ref, _) => f.debug_struct("Mle").finish(),
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
    use ark_std::One;

    #[test]
    fn test_expression_operators() {
        let expression1: ExpressionStandard<Fr> = ExpressionStandard::Constant(Fr::one());

        let mle =
            DenseMle::<_, Fr>::new(vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()]).mle_ref();

        let expression3 = ExpressionStandard::Mle(mle.clone(), None);

        let expression = expression1.clone() + expression3.clone();

        let expression_product = ExpressionStandard::products(vec![mle.clone(), mle.clone()]);

        let expression = expression_product + expression;

        let expression = expression1 - expression;

        let expression = expression * Fr::from(2);

        let expression = expression3.concat(expression);

        assert_eq!(format!("{expression:?}"), "Selector(Iterated, Mle, Scaled(Sum(Constant(BigInt([1, 0, 0, 0])), Negated(Product(Constant(BigInt([1, 0, 0, 0])), Sum(Constant(BigInt([1, 0, 0, 0])), Mle)))), BigInt([2, 0, 0, 0])))")
    }
}
