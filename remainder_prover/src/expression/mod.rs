//! An expression is a type which allows for expressing the definition of a GKR layer

use std::{
    cmp::max,
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    mle::{beta::*, dense::DenseMleRef, MleIndex, MleRef},
    sumcheck::MleError,
};

use remainder_shared_types::FieldExt;

///trait that defines what an Expression needs to be able to do
///TODO!(Fix to make this more general)
pub trait Expression<F: FieldExt>: Debug + Sized {
    ///The MleRef that this Expression contains
    type MleRef: MleRef<F = F>;

    #[allow(clippy::too_many_arguments)]
    ///Evaluate an expression and return a custom type
    fn evaluate<T>(
        &self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&Self::MleRef) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[Self::MleRef]) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T;

    #[allow(clippy::too_many_arguments)]
    /// Evaluate an expression for sumcheck
    fn evaluate_sumcheck<T>(
        &self,
        constant: &impl Fn(F, &DenseMleRef<F>) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&Self::MleRef, &DenseMleRef<F>) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[Self::MleRef], &DenseMleRef<F>) -> T,
        scaled: &impl Fn(T, F) -> T,
        beta_mle_ref: &DenseMleRef<F>,
        round_index: usize,
    ) -> T;

    /// Traverses the expression tree, similarly to `evaluate()`, but with a single
    /// "observer" function which is called at each node. Also takes an immutable reference
    /// to `self` rather than a mutable one (as in `evaluate()`).
    fn traverse<E>(
        &self,
        observer_fn: &mut impl FnMut(&ExpressionStandard<F>) -> Result<(), E>,
    ) -> Result<(), E>;

    /// Add two expressions together
    fn concat_expr(self, lhs: Self) -> Self;

    /// Fix the bit corresponding to `round_index` to `challenge` mutating the MleRefs
    /// so they are accurate as Bookeeping Tables
    fn fix_variable(&mut self, round_index: usize, challenge: F);

    /// Evaluates the current expression (as a multivariate function) at `challenges`
    ///
    /// If the expression is already bound, this will check that the challenges match the already bound indices
    fn evaluate_expr(&mut self, challenges: Vec<F>) -> Result<F, ExpressionError>;
}

#[derive(Error, Debug, Clone, PartialEq)]
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
    #[error("Something went wrong while evaluating the MLE: {0}")]
    MleError(MleError),
    #[error("Selector bit not bound before final evaluation gather")]
    ///Selector bit not bound before final evaluation gather
    SelectorBitNotBoundError,
    #[error("MLE ref with more than one element in its bookkeeping table")]
    ///MLE ref with more than one element in its bookkeeping table
    EvaluateNotFullyBoundError,
    #[error("The bound indices of this expression don't match the indices passed in")]
    ///The bound indices of this expression don't match the indices passed in
    EvaluateBoundIndicesDontMatch,
}

///TODO!(Genericise this over the MleRef Trait)
///Expression representing the relationship between the current layer and layers claims are being made on
#[derive(Clone, Serialize, Deserialize)]
pub enum ExpressionStandard<F> {
    /// This is a constant polynomial
    Constant(F),
    /// This is a virtual selector
    Selector(
        MleIndex<F>,
        Box<ExpressionStandard<F>>,
        Box<ExpressionStandard<F>>,
    ),
    /// This is an MLE
    Mle(DenseMleRef<F>),
    /// This is a negated polynomial
    Negated(Box<ExpressionStandard<F>>),
    /// This is the sum of two polynomials
    Sum(Box<ExpressionStandard<F>>, Box<ExpressionStandard<F>>),
    /// This is the product of some polynomials
    Product(Vec<DenseMleRef<F>>),
    /// This is a scaled polynomial; Optionally a MleIndex to represent a fully bound mle that was this scalar
    Scaled(Box<ExpressionStandard<F>>, F),
}

impl<F: FieldExt> Expression<F> for ExpressionStandard<F> {
    type MleRef = DenseMleRef<F>;
    /// Evaluate the polynomial using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    fn evaluate<T>(
        &self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&DenseMleRef<F>) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[DenseMleRef<F>]) -> T,
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
            ExpressionStandard::Mle(query) => mle_eval(query),
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
            ExpressionStandard::Product(queries) => product(queries),
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
        observer_fn(self)?;
        match self {
            ExpressionStandard::Constant(_)
            | ExpressionStandard::Mle(_)
            | ExpressionStandard::Product(_) => Ok(()),
            ExpressionStandard::Negated(exp) => exp.traverse(observer_fn),
            ExpressionStandard::Scaled(exp, _) => exp.traverse(observer_fn),
            ExpressionStandard::Selector(_, lhs, rhs) => {
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
    fn concat_expr(self, lhs: ExpressionStandard<F>) -> ExpressionStandard<F> {
        ExpressionStandard::Selector(MleIndex::Iterated, Box::new(lhs), Box::new(self))
    }

    fn fix_variable(&mut self, round_index: usize, challenge: F) {
        match self {
            ExpressionStandard::Selector(index, a, b) => {
                if *index == MleIndex::IndexedBit(round_index) {
                    index.bind_index(challenge);
                } else {
                    a.fix_variable(round_index, challenge);
                    b.fix_variable(round_index, challenge);
                }
            }
            ExpressionStandard::Mle(mle_ref) => {
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
            ExpressionStandard::Product(mle_refs) => {
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

    fn evaluate_expr(&mut self, challenges: Vec<F>) -> Result<F, ExpressionError> {
        // --- It's as simple as fixing all variables ---
        challenges
            .iter()
            .enumerate()
            .for_each(|(round_idx, &challenge)| {
                self.fix_variable(round_idx, challenge);
            });

        let mut observer_fn = |exp: &ExpressionStandard<F>| -> Result<(), ExpressionError> {
            match exp {
                ExpressionStandard::Mle(mle_ref) => {
                    let indices = mle_ref
                        .mle_indices()
                        .iter()
                        .filter_map(|index| match index {
                            MleIndex::Bound(chal, index) => Some((*chal, index)),
                            _ => None,
                        })
                        .collect_vec();

                    let start = *indices[0].1;
                    let end = *indices[indices.len() - 1].1;

                    let (indices, _): (Vec<_>, Vec<usize>) = indices.into_iter().unzip();

                    if indices.as_slice() == &challenges[start..=end] {
                        Ok(())
                    } else {
                        Err(ExpressionError::EvaluateBoundIndicesDontMatch)
                    }
                }
                ExpressionStandard::Product(mle_refs) => mle_refs
                    .iter()
                    .map(|mle_ref| {
                        let indices = mle_ref
                            .mle_indices()
                            .iter()
                            .filter_map(|index| match index {
                                MleIndex::Bound(chal, index) => Some((*chal, index)),
                                _ => None,
                            })
                            .collect_vec();

                        let start = *indices[0].1;
                        let end = *indices[indices.len() - 1].1;

                        let (indices, _): (Vec<_>, Vec<usize>) = indices.into_iter().unzip();

                        if indices.as_slice() == &challenges[start..=end] {
                            Ok(())
                        } else {
                            Err(ExpressionError::EvaluateBoundIndicesDontMatch)
                        }
                    })
                    .try_collect(),

                _ => Ok(()),
            }
        };
        self.traverse(&mut observer_fn)?;

        // --- Traverse the expression and pick up all the evals ---
        gather_combine_all_evals(self)
    }

    ///Similar function to eval, but with minor changes to accomodate sumcheck's peculiarities
    #[allow(clippy::too_many_arguments)]
    fn evaluate_sumcheck<T>(
        &self,
        constant: &impl Fn(F, &DenseMleRef<F>) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&DenseMleRef<F>, &DenseMleRef<F>) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[DenseMleRef<F>], &DenseMleRef<F>) -> T,
        scaled: &impl Fn(T, F) -> T,
        beta_mle_ref: &DenseMleRef<F>,
        round_index: usize,
    ) -> T {
        match self {
            ExpressionStandard::Constant(scalar) => constant(*scalar, beta_mle_ref),
            ExpressionStandard::Selector(index, a, b) => {
                // need to check whether the selector bit is the current independent variable
                if let MleIndex::IndexedBit(idx) = index {
                    match Ord::cmp(&round_index, idx) {
                        // if not, need to split the beta table according to the beta_split function
                        std::cmp::Ordering::Less => {
                            let (beta_mle_first, beta_mle_second) = beta_split(beta_mle_ref);
                            selector_column(
                                index,
                                a.evaluate_sumcheck(
                                    constant,
                                    selector_column,
                                    mle_eval,
                                    negated,
                                    sum,
                                    product,
                                    scaled,
                                    &beta_mle_first,
                                    round_index,
                                ),
                                b.evaluate_sumcheck(
                                    constant,
                                    selector_column,
                                    mle_eval,
                                    negated,
                                    sum,
                                    product,
                                    scaled,
                                    &beta_mle_second,
                                    round_index,
                                ),
                            )
                        }
                        // otherwise, proceed normally
                        std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => selector_column(
                            index,
                            a.evaluate_sumcheck(
                                constant,
                                selector_column,
                                mle_eval,
                                negated,
                                sum,
                                product,
                                scaled,
                                beta_mle_ref,
                                round_index,
                            ),
                            b.evaluate_sumcheck(
                                constant,
                                selector_column,
                                mle_eval,
                                negated,
                                sum,
                                product,
                                scaled,
                                beta_mle_ref,
                                round_index,
                            ),
                        ),
                    }
                } else {
                    selector_column(
                        index,
                        a.evaluate_sumcheck(
                            constant,
                            selector_column,
                            mle_eval,
                            negated,
                            sum,
                            product,
                            scaled,
                            beta_mle_ref,
                            round_index,
                        ),
                        b.evaluate_sumcheck(
                            constant,
                            selector_column,
                            mle_eval,
                            negated,
                            sum,
                            product,
                            scaled,
                            beta_mle_ref,
                            round_index,
                        ),
                    )
                }
            }
            ExpressionStandard::Mle(query) => mle_eval(query, beta_mle_ref),
            ExpressionStandard::Negated(a) => {
                let a = a.evaluate_sumcheck(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta_mle_ref,
                    round_index,
                );
                negated(a)
            }
            ExpressionStandard::Sum(a, b) => {
                let a = a.evaluate_sumcheck(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta_mle_ref,
                    round_index,
                );
                let b = b.evaluate_sumcheck(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta_mle_ref,
                    round_index,
                );
                sum(a, b)
            }
            ExpressionStandard::Product(queries) => product(queries, beta_mle_ref),
            ExpressionStandard::Scaled(a, f) => {
                let a = a.evaluate_sumcheck(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta_mle_ref,
                    round_index,
                );
                scaled(a, *f)
            }
        }
    }
}

/// Helper function for `evaluate_expr` to traverse the expression and simply
/// gather all of the evaluations, combining them as appropriate.
/// Strictly speaking this doesn't need to be `&mut` but we call `self.evaluate()`
/// within.
pub fn gather_combine_all_evals<F: FieldExt, Exp: Expression<F>>(
    expr: &Exp,
) -> Result<F, ExpressionError> {
    let constant = |c| Ok(c);
    let selector_column =
        |idx: &MleIndex<F>, lhs: Result<F, ExpressionError>, rhs: Result<F, ExpressionError>| {
            // --- Selector bit must be bound ---
            if let MleIndex::Bound(val, _) = idx {
                return Ok(*val * rhs? + (F::one() - val) * lhs?);
            }
            Err(ExpressionError::SelectorBitNotBoundError)
        };
    let mle_eval = for<'a> |mle_ref: &'a Exp::MleRef| -> Result<F, ExpressionError> {
        if mle_ref.bookkeeping_table().len() != 1 {
            return Err(ExpressionError::EvaluateNotFullyBoundError);
        }
        Ok(mle_ref.bookkeeping_table()[0])
    };
    let negated = |a: Result<F, ExpressionError>| match a {
        Err(e) => Err(e),
        Ok(val) => Ok(val.neg()),
    };
    let sum = |lhs: Result<F, ExpressionError>, rhs: Result<F, ExpressionError>| Ok(lhs? + rhs?);
    let product = for<'a, 'b> |mle_refs: &'a [Exp::MleRef]| -> Result<F, ExpressionError> {
        mle_refs.iter().try_fold(F::one(), |acc, new_mle_ref| {
            // --- Accumulate either errors or multiply ---
            if new_mle_ref.bookkeeping_table().len() != 1 {
                return Err(ExpressionError::EvaluateNotFullyBoundError);
            }
            Ok(acc * new_mle_ref.bookkeeping_table()[0])
        })
    };
    let scaled = |a: Result<F, ExpressionError>, scalar: F| Ok(a? * scalar);
    expr.evaluate(
        &constant,
        &selector_column,
        &mle_eval,
        &negated,
        &sum,
        &product,
        &scaled,
    )
}

impl<F: FieldExt> ExpressionStandard<F> {
    ///Create a product Expression that multiplies many MLEs together
    pub fn products(product_list: Vec<DenseMleRef<F>>) -> Self {
        Self::Product(product_list)
    }

    /// Mutate the MleIndices that are Iterated in the expression and turn them into IndexedBit
    ///
    /// Returns the max number of bits that are indexed
    pub fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        match self {
            ExpressionStandard::Selector(mle_index, a, b) => {
                *mle_index = MleIndex::IndexedBit(curr_index);
                let a_bits = a.index_mle_indices(curr_index + 1);
                let b_bits = b.index_mle_indices(curr_index + 1);
                max(a_bits, b_bits)
            }
            ExpressionStandard::Mle(mle_ref) => mle_ref.index_mle_indices(curr_index),
            ExpressionStandard::Sum(a, b) => {
                let a_bits = a.index_mle_indices(curr_index);
                let b_bits = b.index_mle_indices(curr_index);
                max(a_bits, b_bits)
            }
            ExpressionStandard::Product(mle_refs) => mle_refs
                .iter_mut()
                .map(|mle_ref| mle_ref.index_mle_indices(curr_index))
                .reduce(max)
                .unwrap_or(curr_index),
            ExpressionStandard::Scaled(a, _) => a.index_mle_indices(curr_index),
            ExpressionStandard::Negated(a) => a.index_mle_indices(curr_index),
            ExpressionStandard::Constant(_) => curr_index,
        }
    }

    ///traverse an expression mutably changing it's contents
    pub fn traverse_mut<E>(
        &mut self,
        observer_fn: &mut impl FnMut(&mut ExpressionStandard<F>) -> Result<(), E>,
    ) -> Result<(), E> {
        // dbg!(&self);
        observer_fn(self)?;
        match self {
            ExpressionStandard::Constant(_)
            | ExpressionStandard::Mle(_)
            | ExpressionStandard::Product(_) => Ok(()),
            ExpressionStandard::Negated(exp) => exp.traverse_mut(observer_fn),
            ExpressionStandard::Scaled(exp, _) => exp.traverse_mut(observer_fn),
            ExpressionStandard::Selector(_, lhs, rhs) => {
                lhs.traverse_mut(observer_fn)?;
                rhs.traverse_mut(observer_fn)
            }
            ExpressionStandard::Sum(lhs, rhs) => {
                lhs.traverse_mut(observer_fn)?;
                rhs.traverse_mut(observer_fn)
            }
        }
    }

    ///Gets the size of an expression in terms of the number of rounds of sumcheck
    pub fn get_expression_size(&self, curr_size: usize) -> usize {
        match self {
            ExpressionStandard::Selector(_mle_index, a, b) => {
                let a_bits = a.get_expression_size(curr_size + 1);
                let b_bits = b.get_expression_size(curr_size + 1);
                max(a_bits, b_bits)
            }
            ExpressionStandard::Mle(mle_ref) => {
                mle_ref
                    .mle_indices()
                    .iter()
                    .filter(|item| {
                        matches!(
                            item,
                            &&MleIndex::Iterated
                                | &&MleIndex::IndexedBit(_)
                                | &&MleIndex::Bound(_, _)
                        )
                    })
                    .collect_vec()
                    .len()
                    + curr_size
            }
            ExpressionStandard::Sum(a, b) => {
                let a_bits = a.get_expression_size(curr_size);
                let b_bits = b.get_expression_size(curr_size);
                max(a_bits, b_bits)
            }
            ExpressionStandard::Product(mle_refs) => {
                mle_refs
                    .iter()
                    .map(|mle_ref| {
                        mle_ref
                            .mle_indices()
                            .iter()
                            .filter(|item| {
                                matches!(
                                    item,
                                    &&MleIndex::Iterated
                                        | &&MleIndex::IndexedBit(_)
                                        | &&MleIndex::Bound(_, _)
                                )
                            })
                            .collect_vec()
                            .len()
                    })
                    .max()
                    .unwrap_or(0)
                    + curr_size
            }
            ExpressionStandard::Scaled(a, _) => a.get_expression_size(curr_size),
            ExpressionStandard::Negated(a) => a.get_expression_size(curr_size),
            ExpressionStandard::Constant(_) => curr_size,
        }
    }
}

impl<F: std::fmt::Debug> std::fmt::Debug for ExpressionStandard<F> {
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
            ExpressionStandard::Mle(_mle_ref) => {
                f.debug_struct("Mle").field("mle_ref", _mle_ref).finish()
            }
            ExpressionStandard::Negated(poly) => f.debug_tuple("Negated").field(poly).finish(),
            ExpressionStandard::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            ExpressionStandard::Product(a) => f.debug_tuple("Product").field(a).finish(),
            ExpressionStandard::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}

impl<F: std::fmt::Debug + FieldExt> ExpressionStandard<F> {
    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> impl std::fmt::Display + 'a {
        struct CircuitDesc<'a, F>(&'a ExpressionStandard<F>);
        impl<'a, F: std::fmt::Debug + FieldExt> std::fmt::Display for CircuitDesc<'a, F> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    ExpressionStandard::Constant(scalar) => {
                        f.debug_tuple("const").field(scalar).finish()
                    }
                    ExpressionStandard::Selector(index, a, b) => f.write_fmt(format_args!("sel {index:?}; {}; {}", CircuitDesc(a), CircuitDesc(b))),
                    // Skip enum variant and print query struct directly to maintain backwards compatibility.
                    ExpressionStandard::Mle(mle_ref) => {
                        f.debug_struct("mle").field("layer", &mle_ref.get_layer_id()).field("indices", &mle_ref.mle_indices()).finish()
                    }
                    ExpressionStandard::Negated(poly) => f.write_fmt(format_args!("-{}", CircuitDesc(poly))),
                    ExpressionStandard::Sum(a, b) => f.write_fmt(format_args!("+ {}; {}", CircuitDesc(a), CircuitDesc(b))),
                    ExpressionStandard::Product(a) => {
                        let str = a.iter().map(|mle| {
                            format!("{:?}; {:?}", mle.get_layer_id(), mle.mle_indices())
                        }).reduce(|acc, str| acc + &str).unwrap();
                        f.write_str(&str)
                    },
                    ExpressionStandard::Scaled(poly, scalar) => {
                        f.write_fmt(format_args!("* {}; {:?}", CircuitDesc(poly), scalar))
                    }
                }
            }
        }

        CircuitDesc(self)
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
mod tests {
    use crate::{layer::LayerId, mle::dense::DenseMle};

    use super::*;
    use ark_std::One;
    use remainder_shared_types::Fr;

    // #[test]
    // fn test_expression_operators() {
    //     let expression1: ExpressionStandard<Fr> = ExpressionStandard::Constant(Fr::one());

    //     let mle = DenseMle::new_from_raw(
    //         vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()],
    //         LayerId::Input(0),
    //         None,
    //     )
    //     .mle_ref();

    //     let expression3 = ExpressionStandard::Mle(mle.clone());

    //     let expression = expression1.clone() + expression3.clone();

    //     let expression_product = ExpressionStandard::products(vec![mle.clone(), mle]);

    //     let expression = expression_product + expression;

    //     let expression = expression1 - expression;

    //     let expression = expression * Fr::from(2);

    //     let expression = expression3.concat(expression);

    //     let dense_mle_print = "DenseMleRef { bookkeeping_table: [BigInt([1, 0, 0, 0]), BigInt([1, 0, 0, 0]), BigInt([1, 0, 0, 0]), BigInt([1, 0, 0, 0])], mle_indices: [Iterated, Iterated], num_vars: 2, layer_id: None }";

    //     assert_eq!(format!("{expression:?}"), format!("Selector(Iterated, Mle, Scaled(Sum(Constant(BigInt([1, 0, 0, 0])), Negated(Sum(Product([{dense_mle_print}, {dense_mle_print}]), Sum(Constant(BigInt([1, 0, 0, 0])), Mle)))), BigInt([2, 0, 0, 0])))"));
    // }

    #[test]
    fn test_constants_eval() {
        let expression1: ExpressionStandard<Fr> = ExpressionStandard::Constant(Fr::one());

        let expression2: ExpressionStandard<Fr> = ExpressionStandard::Constant(Fr::from(2));

        let expression3 = expression1.clone() + expression2.clone();

        let mut expression = (expression1 - expression2) * Fr::from(2);
        let mut expression_another = expression.clone() + expression3;

        let challenge = vec![Fr::one()];
        let eval = expression.evaluate_expr(challenge.clone());
        assert_eq!(eval.unwrap(), Fr::from(2).neg());

        let eval_another = expression_another.evaluate_expr(challenge);
        assert_eq!(eval_another.unwrap(), Fr::from(1));
    }

    #[test]
    fn test_mle_eval_two_variable() {
        let mle = DenseMle::new_from_raw(
            vec![Fr::from(4), Fr::from(2), Fr::from(5), Fr::from(7)],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let mut expression = ExpressionStandard::Mle(mle);
        let num_indices = expression.index_mle_indices(0);
        assert_eq!(num_indices, 2);

        let challenge = vec![Fr::from(2).neg(), Fr::from(9)];
        let eval = expression.evaluate_expr(challenge);
        assert_eq!(eval.unwrap(), Fr::from(55).neg());
    }

    #[test]
    fn test_mle_eval_three_variable() {
        let mle = DenseMle::new_from_raw(
            vec![
                Fr::from(4),
                Fr::from(2),
                Fr::from(5),
                Fr::from(7),
                Fr::from(2),
                Fr::from(4),
                Fr::from(9),
                Fr::from(6),
            ],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let mut expression = ExpressionStandard::Mle(mle);
        let num_indices = expression.index_mle_indices(0);
        assert_eq!(num_indices, 3);

        let challenge = vec![Fr::from(2).neg(), Fr::from(3), Fr::from(5)];
        let eval = expression.evaluate_expr(challenge);
        assert_eq!(eval.unwrap(), Fr::from(297));
    }

    #[test]
    fn test_mle_eval_sum_w_constant_then_scale() {
        let mle = DenseMle::new_from_raw(
            vec![Fr::from(4), Fr::from(2), Fr::from(1), Fr::from(7)],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let expression = ExpressionStandard::Mle(mle);
        let mut expression = (expression + ExpressionStandard::Constant(Fr::from(5))) * Fr::from(2);
        let num_indices = expression.index_mle_indices(0);
        assert_eq!(num_indices, 2);

        let challenge = vec![Fr::from(1).neg(), Fr::from(7)];
        let eval = expression.evaluate_expr(challenge);
        assert_eq!(eval.unwrap(), Fr::from(132).neg());
    }

    #[test]
    fn test_mle_eval_selector() {
        let mle_1 = DenseMle::new_from_raw(
            vec![Fr::from(4), Fr::from(2), Fr::from(1), Fr::from(7)],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let expression_1 = ExpressionStandard::Mle(mle_1);

        let mle_2 = DenseMle::new_from_raw(
            vec![Fr::from(1), Fr::from(9), Fr::from(8), Fr::from(2)],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let expression_2 = ExpressionStandard::Mle(mle_2);

        let mut expression = expression_1.concat_expr(expression_2);

        let num_indices = expression.index_mle_indices(0);
        assert_eq!(num_indices, 3);

        let challenge = vec![Fr::from(2), Fr::from(7), Fr::from(3)];
        let eval = expression.evaluate_expr(challenge);

        let mle_concat = DenseMle::new_from_raw(
            vec![
                Fr::from(1),
                Fr::from(9),
                Fr::from(8),
                Fr::from(2),
                Fr::from(4),
                Fr::from(2),
                Fr::from(1),
                Fr::from(7),
            ],
            LayerId::Input(0),
            None,
        )
        .mle_ref(); // cancat actually prepends

        let challenge_concat = vec![Fr::from(7), Fr::from(3), Fr::from(2)]; // move the first challenge towards the end

        let mut expression_concat = ExpressionStandard::Mle(mle_concat);

        let num_indices_concat = expression_concat.index_mle_indices(0);
        assert_eq!(num_indices_concat, 3);

        let eval_concat = expression_concat.evaluate_expr(challenge_concat);

        assert_eq!(eval.unwrap(), eval_concat.unwrap());
    }

    #[test]
    fn test_mle_eval_selector_w_constant() {
        let mle_1 = DenseMle::new_from_raw(
            vec![Fr::from(4), Fr::from(2), Fr::from(1), Fr::from(7)],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let expression_1 = ExpressionStandard::Mle(mle_1);

        let mut expression = expression_1.concat_expr(ExpressionStandard::Constant(Fr::from(5)));

        let num_indices = expression.index_mle_indices(0);
        assert_eq!(num_indices, 3);

        let challenge = vec![Fr::from(1).neg(), Fr::from(7), Fr::from(3)];
        let eval = expression.evaluate_expr(challenge);

        // -149 + (1 - (-1)) * 5
        assert_eq!(eval.unwrap(), Fr::from(139).neg());
    }

    #[test]
    fn test_mle_refs_eval() {
        let challenge = vec![Fr::from(2), Fr::from(3)];

        let mle_1 = DenseMle::new_from_raw(
            vec![Fr::from(2), Fr::from(2), Fr::from(1), Fr::from(3)],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let mut expression_1 = ExpressionStandard::Mle(mle_1.clone());
        let _ = expression_1.index_mle_indices(0);
        let eval_1 = expression_1.evaluate_expr(challenge.clone()).unwrap();

        let mle_2 = DenseMle::new_from_raw(
            vec![Fr::from(1), Fr::from(4), Fr::from(5), Fr::from(2)],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let mut expression_2 = ExpressionStandard::Mle(mle_2.clone());
        let _ = expression_2.index_mle_indices(0);
        let eval_2 = expression_2.evaluate_expr(challenge.clone()).unwrap();

        let mut expression_product = ExpressionStandard::products(vec![mle_1, mle_2]);
        let num_indices = expression_product.index_mle_indices(0);
        assert_eq!(num_indices, 2);

        let eval_prod = expression_product.evaluate_expr(challenge).unwrap();

        assert_eq!(eval_prod, (eval_1 * eval_2));
        // 11 * -17
        assert_eq!(eval_prod, Fr::from(187).neg());
    }

    #[test]
    fn test_mle_different_length_eval() {
        let challenge = vec![Fr::from(2), Fr::from(3), Fr::from(5)];

        let mle_1 = DenseMle::new_from_raw(
            vec![Fr::from(2), Fr::from(2), Fr::from(1), Fr::from(3)],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let expression_1 = ExpressionStandard::Mle(mle_1);

        let mle_2 = DenseMle::new_from_raw(
            vec![
                Fr::from(1),
                Fr::from(4),
                Fr::from(5),
                Fr::from(2),
                Fr::from(1),
                Fr::from(9),
                Fr::from(8),
                Fr::from(2),
            ],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let expression_2 = ExpressionStandard::Mle(mle_2);

        let mut expression = expression_1 + expression_2;
        let num_indices = expression.index_mle_indices(0);
        assert_eq!(num_indices, 3);

        let eval_prod = expression.evaluate_expr(challenge).unwrap();

        // -230 + 68 + 11
        assert_eq!(eval_prod, Fr::from(151).neg());
    }

    #[test]
    fn test_mle_different_length_prod() {
        let challenge = vec![Fr::from(2), Fr::from(3), Fr::from(5)];

        let mle_1 = DenseMle::new_from_raw(
            vec![Fr::from(2), Fr::from(2), Fr::from(1), Fr::from(3)],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let mle_2 = DenseMle::new_from_raw(
            vec![
                Fr::from(1),
                Fr::from(4),
                Fr::from(5),
                Fr::from(2),
                Fr::from(1),
                Fr::from(9),
                Fr::from(8),
                Fr::from(2),
            ],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let mut expression_product = ExpressionStandard::products(vec![mle_1, mle_2]);
        let num_indices = expression_product.index_mle_indices(0);
        assert_eq!(num_indices, 3);

        let eval_prod = expression_product.evaluate_expr(challenge).unwrap();

        // (-230 + 68) * 11
        assert_eq!(eval_prod, Fr::from(1782).neg());
    }

    // #[test]
    // fn test_not_fully_bounded_eval() {
    //     let mle = DenseMle::<_, Fr>::new(vec![
    //         Fr::from(4),
    //         Fr::from(2),
    //         Fr::from(5),
    //         Fr::from(7),
    //         Fr::from(2),
    //         Fr::from(4),
    //         Fr::from(9),
    //         Fr::from(6),
    //     ])
    //     .mle_ref();

    //     let mut expression = ExpressionStandard::Mle(mle);
    //     let _ = expression.index_mle_indices(3);

    //     let challenge = vec![Fr::from(-2), Fr::from(3), Fr::from(5)];
    //     let eval = expression.evaluate_expr(challenge);
    //     assert_eq!(eval, Err(ExpressionError::EvaluateNotFullyBoundError));
    // }

    #[test]
    fn big_test_eval() {
        let expression1: ExpressionStandard<Fr> = ExpressionStandard::Constant(Fr::one());

        let mle = DenseMle::new_from_raw(
            vec![Fr::one(), Fr::from(2), Fr::from(3), Fr::one()],
            LayerId::Input(0),
            None,
        )
        .mle_ref();

        let expression3 = ExpressionStandard::Mle(mle.clone());

        let expression = expression1.clone() + expression3.clone();

        let expression_product = ExpressionStandard::products(vec![mle.clone(), mle]);

        let expression = expression_product + expression;

        let expression = expression1 - expression;

        let expression = expression * Fr::from(2);

        let mut expression = expression3.concat_expr(expression);
        let num_indices = expression.index_mle_indices(0);
        assert_eq!(num_indices, 3);

        let challenge = vec![Fr::from(2), Fr::from(3), Fr::from(4)];
        let eval = expression.evaluate_expr(challenge).unwrap();

        // -((1 - (24 * 24 - 23)) * 2) - 24 * 2
        assert_eq!(eval, Fr::from(1056));
    }
}
