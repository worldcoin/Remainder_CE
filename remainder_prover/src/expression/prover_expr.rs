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

use super::{expr_errors::ExpressionError, generic_expr::{Expression, ExpressionType}, verifier_expr::{gather_combine_all_evals_verifier, VerifierExpression}};

/// Prover Expression
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProverExpression;
impl<F: FieldExt> ExpressionType<F> for ProverExpression {
    type Container = DenseMleRef<F>;
}


/// specific methods for prover expressions
impl<F: FieldExt> Expression<F, ProverExpression> {

    /// transforms the expression to a verifier expression
    /// should only be called when the entire expression is fully bound
    /// traverses the expression and changes the DenseMleRef to F,
    /// by grabbing their bookkeeping table's 1st and only element,
    /// if the bookkeeping table has more than 1 element, it 
    /// throws an ExpressionError::EvaluateNotFullyBoundError
    pub fn transform_to_verifier_expression(&mut self) -> Result<Expression<F, VerifierExpression>, ExpressionError>{

        match self {
            Expression::Constant(scalar) => Ok(Expression::Constant(*scalar)),
            Expression::Selector(index, a, b) => {
                Ok(
                    Expression::Selector(
                        index.clone(),
                        Box::new(a.transform_to_verifier_expression()?),
                        Box::new(b.transform_to_verifier_expression()?)
                    )
                )
            }
            Expression::Mle(mle_ref) => {

                if mle_ref.bookkeeping_table().len() != 1 {
                    return Err(ExpressionError::EvaluateNotFullyBoundError);
                }
                Ok(Expression::Mle(mle_ref.bookkeeping_table()[0]))
            }
            Expression::Negated(a) => Ok(
                Expression::Negated(Box::new(a.transform_to_verifier_expression()?))
            ),
            Expression::Sum(a, b) => {

                Ok(
                    Expression::Sum(
                        Box::new(a.transform_to_verifier_expression()?),
                        Box::new(b.transform_to_verifier_expression()?)
                    )
                )
            }
            Expression::Product(mle_refs) => {
                for mle_ref in mle_refs.iter() {
                    if mle_ref.bookkeeping_table().len() != 1 {
                        return Err(ExpressionError::EvaluateNotFullyBoundError);
                    }
                }
                Ok(
                    Expression::Product(
                        mle_refs.into_iter().map(|mle_ref| mle_ref.bookkeeping_table()[0].clone()).collect_vec()
                    )
                )
                
            }
            Expression::Scaled(mle, scalar) => Ok(
                Expression::Scaled(Box::new(mle.transform_to_verifier_expression()?), *scalar)
            ),
        }
    }

    // prover
    /// fix the variable at a certain round index
    pub fn fix_variable(&mut self, round_index: usize, challenge: F) {
        match self {
            Expression::Selector(index, a, b) => {
                if *index == MleIndex::IndexedBit(round_index) {
                    index.bind_index(challenge);
                } else {
                    a.fix_variable(round_index, challenge);
                    b.fix_variable(round_index, challenge);
                }
            }
            Expression::Mle(mle_ref) => {
                if mle_ref
                    .mle_indices()
                    .contains(&MleIndex::IndexedBit(round_index))
                {
                    mle_ref.fix_variable(round_index, challenge);
                }
            }
            Expression::Negated(a) => a.fix_variable(round_index, challenge),
            Expression::Sum(a, b) => {
                a.fix_variable(round_index, challenge);
                b.fix_variable(round_index, challenge);
            }
            Expression::Product(mle_refs) => {
                for mle_ref in mle_refs {
                    if mle_ref
                        .mle_indices()
                        .contains(&MleIndex::IndexedBit(round_index))
                    {
                        mle_ref.fix_variable(round_index, challenge);
                    }
                }
            }
            Expression::Scaled(a, _) => {
                a.fix_variable(round_index, challenge);
            }
            Expression::Constant(_) => (),
        }
    }

    /// prover
    pub fn evaluate_expr(&mut self, challenges: Vec<F>) -> Result<F, ExpressionError> {
        // --- It's as simple as fixing all variables ---
        challenges
            .iter()
            .enumerate()
            .for_each(|(round_idx, &challenge)| {
                self.fix_variable(round_idx, challenge);
            });

        // ----- this is literally a check -----
        let mut observer_fn = |exp: &Expression<F, ProverExpression>| -> Result<(), ExpressionError> {
            match exp {
                Expression::Mle(mle_ref) => {
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
                Expression::Product(mle_refs) => mle_refs
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
        // ----- this is literally a check -----

        // --- Traverse the expression and pick up all the evals ---
        gather_combine_all_evals_verifier(&self.transform_to_verifier_expression().unwrap())
    }

    // prover

    ///Similar function to eval, but with minor changes to accomodate sumcheck's peculiarities
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_sumcheck<T>(
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
            Expression::Constant(scalar) => constant(*scalar, beta_mle_ref),
            Expression::Selector(index, a, b) => {
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
            Expression::Mle(query) => mle_eval(query, beta_mle_ref),
            Expression::Negated(a) => {
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
            Expression::Sum(a, b) => {
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
            Expression::Product(queries) => product(queries, beta_mle_ref),
            Expression::Scaled(a, f) => {
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

    // prover
    /// Mutate the MleIndices that are Iterated in the expression and turn them into IndexedBit
    ///
    /// Returns the max number of bits that are indexed
    pub fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        match self {
            Expression::Selector(mle_index, a, b) => {
                *mle_index = MleIndex::IndexedBit(curr_index);
                let a_bits = a.index_mle_indices(curr_index + 1);
                let b_bits = b.index_mle_indices(curr_index + 1);
                max(a_bits, b_bits)
            }
            Expression::Mle(mle_ref) => mle_ref.index_mle_indices(curr_index),
            Expression::Sum(a, b) => {
                let a_bits = a.index_mle_indices(curr_index);
                let b_bits = b.index_mle_indices(curr_index);
                max(a_bits, b_bits)
            }
            Expression::Product(mle_refs) => mle_refs
                .iter_mut()
                .map(|mle_ref| mle_ref.index_mle_indices(curr_index))
                .reduce(max)
                .unwrap_or(curr_index),
            Expression::Scaled(a, _) => a.index_mle_indices(curr_index),
            Expression::Negated(a) => a.index_mle_indices(curr_index),
            Expression::Constant(_) => curr_index,
        }
    }

    // prover
    ///Gets the size of an expression in terms of the number of rounds of sumcheck
    pub fn get_expression_size(&self, curr_size: usize) -> usize {
        match self {
            Expression::Selector(_mle_index, a, b) => {
                let a_bits = a.get_expression_size(curr_size + 1);
                let b_bits = b.get_expression_size(curr_size + 1);
                max(a_bits, b_bits)
            }
            Expression::Mle(mle_ref) => {
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
            Expression::Sum(a, b) => {
                let a_bits = a.get_expression_size(curr_size);
                let b_bits = b.get_expression_size(curr_size);
                max(a_bits, b_bits)
            }
            Expression::Product(mle_refs) => {
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
            Expression::Scaled(a, _) => a.get_expression_size(curr_size),
            Expression::Negated(a) => a.get_expression_size(curr_size),
            Expression::Constant(_) => curr_size,
        }
    }
}


// prover
impl<F: std::fmt::Debug + FieldExt> Expression<F, ProverExpression> {
    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> impl std::fmt::Display + 'a {
        struct CircuitDesc<'a, F: FieldExt>(&'a Expression<F, ProverExpression>);
        impl<'a, F: std::fmt::Debug + FieldExt> std::fmt::Display for CircuitDesc<'a, F> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    Expression::Constant(scalar) => {
                        f.debug_tuple("const").field(scalar).finish()
                    }
                    Expression::Selector(index, a, b) => f.write_fmt(format_args!("sel {index:?}; {}; {}", CircuitDesc(a), CircuitDesc(b))),
                    // Skip enum variant and print query struct directly to maintain backwards compatibility.
                    Expression::Mle(mle_ref) => {
                        f.debug_struct("mle").field("layer", &mle_ref.get_layer_id()).field("indices", &mle_ref.mle_indices()).finish()
                    }
                    Expression::Negated(poly) => f.write_fmt(format_args!("-{}", CircuitDesc(poly))),
                    Expression::Sum(a, b) => f.write_fmt(format_args!("+ {}; {}", CircuitDesc(a), CircuitDesc(b))),
                    Expression::Product(a) => {
                        let str = a.iter().map(|mle| {
                            format!("{:?}; {:?}", mle.get_layer_id(), mle.mle_indices())
                        }).reduce(|acc, str| acc + &str).unwrap();
                        f.write_str(&str)
                    },
                    Expression::Scaled(poly, scalar) => {
                        f.write_fmt(format_args!("* {}; {:?}", CircuitDesc(poly), scalar))
                    }
                }
            }
        }

        CircuitDesc(self)
    }
}