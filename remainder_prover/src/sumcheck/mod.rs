//! Contains cryptographic algorithms for going through the sumcheck protocol

use std::{
    iter::{repeat, Successors},
    ops::{Add, Mul, Neg},
};

pub mod tests;

use ark_std::cfg_into_iter;
use itertools::Itertools;
use rayon::prelude::ParallelSlice;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use thiserror::Error;

use crate::{
    expression::{
        expr_errors::ExpressionError,
        generic_expr::{Expression, ExpressionNode, ExpressionType},
        prover_expr::ProverExpr,
    },
    mle::{
        beta::BetaTable,
        dense::{DenseMle, DenseMleRef},
        newbeta::NewBeta,
        MleIndex, MleRef,
    },
};
use remainder_shared_types::FieldExt;

#[derive(Error, Debug, Clone, PartialEq)]
///Errors to do with the evaluation of MleRefs
pub enum MleError {
    #[error("Passed list of Mles is empty")]
    ///Passed list of Mles is empty
    EmptyMleList,
    #[error("Beta table not yet initialized for Mle")]
    ///Beta table not yet initialized for Mle
    NoBetaTable,
    #[error("Layer does not have claims yet")]
    ///Layer does not have claims yet
    NoClaim,
    #[error("Unable to eval beta")]
    ///Unable to eval beta
    BetaEvalError,
    #[error("Cannot compute sumcheck message on un-indexed MLE")]
    ///Cannot compute sumcheck message on un-indexed MLE
    NotIndexedError,
}

#[derive(Error, Debug, Clone)]
///Verification error
pub enum VerifyError {
    #[error("Failed sumcheck round")]
    ///Failed sumcheck round
    SumcheckBad,
}
#[derive(Error, Debug, Clone)]
///Error when Interpolating a univariate polynomial
pub enum InterpError {
    #[error("Too few evaluation points")]
    ///Too few evaluation points
    EvalLessThanDegree,
    #[error("No possible polynomial")]
    ///No possible polynomial
    NoInverse,
}

/// A type representing the univariate message g_i(x) which the prover
/// sends to the verifier in each round of sumcheck. Note that the prover
/// in our case always sends evaluations g_i(0), ..., g_i(d) to the verifier,
/// and thus the struct is called `Evals`.
#[derive(PartialEq, Debug, Clone)]
pub struct Evals<F: FieldExt>(pub Vec<F>);

impl<F: FieldExt> Neg for Evals<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        // --- Negation for a bunch of eval points is just element-wise negation ---
        Evals(self.0.into_iter().map(|eval| eval.neg()).collect_vec())
    }
}

impl<F: FieldExt> Add for Evals<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Evals(
            self.0
                .into_iter()
                .zip(rhs.0)
                .map(|(lhs, rhs)| lhs + rhs)
                .collect_vec(),
        )
    }
}

impl<F: FieldExt> Mul<F> for Evals<F> {
    type Output = Self;
    fn mul(self, rhs: F) -> Self {
        Evals(
            self.0
                .into_iter()
                .zip(repeat(rhs))
                .map(|(lhs, rhs)| lhs * rhs)
                .collect_vec(),
        )
    }
}

impl<F: FieldExt> Mul<&F> for Evals<F> {
    type Output = Self;
    fn mul(self, rhs: &F) -> Self {
        Evals(
            self.0
                .into_iter()
                .zip(repeat(rhs))
                .map(|(lhs, rhs)| lhs * rhs)
                .collect_vec(),
        )
    }
}

/// Gives us an evaluation for an entire expression. Returns either a single
/// value (e.g. if all variables are bound and/or the expression is just over
/// a constant), or a vector of evals at 0, ..., deg - 1 for an expression
/// where there are iterated variables.
///
/// # Arguments
/// * `expr` - The actual expression to evaluate
/// * `round_index` - The sumcheck round index, I think??
/// * `max_degree` - The maximum degree of the `round_index`th variable
///
/// # Errors
/// - Error::BetaError when the beta table has not been initialized
/// - TODO!(ryancao || vishady) -- Error::NotIndexedError when ANY MLE is not
///     fully indexed.
pub fn compute_sumcheck_message<F: FieldExt>(
    expr: &Expression<F, ProverExpr>,
    round_index: usize,
    max_degree: usize,
    beta_table: &BetaTable<F>,
) -> Result<Evals<F>, ExpressionError> {
    // --- TODO!(ende): REMEMBER TO REMOVE ALL THE PRINTLN STATEMENTS ---

    // --- TODO!(ryancao): (From Zhenfei): So we can probably cache this beta table evaluation somehow
    // and then use those evals many times and not have to do this over and over again
    // Zhenfei's idea: Just memoize

    // --- Constant evaluation is just Sum(k) ---
    let constant = |constant, beta_mle_ref: &DenseMleRef<F>| {
        // need to actually treat this like a 'scaled' because there is a beta table
        let beta_bt = beta_mle_ref.bookkeeping_table();
        // just scale the beta table by the constant
        let first = beta_bt
            .iter()
            .step_by(2)
            .fold(F::zero(), |elem, acc| elem + acc);
        let second = beta_bt
            .iter()
            .skip(1)
            .step_by(2)
            .fold(F::zero(), |elem, acc| elem + acc);
        let evals =
            (1..max_degree + 1).map(|index| first + (second - first) * F::from(index as u64));
        let beta_eval = Evals(std::iter::once(first).chain(evals).collect_vec());
        Ok(beta_eval * constant)
    };

    // V_i(b_1, ..., b_n) = ((1 - b1) * (V_{i + 1}(0, 0, b2, ..., bn) + V_{i + 1}(0, 0, b2, ..., bn))) +
    // b1 * (V_{i + 1}(0, 0, b2, ..., bn) * V_{i + 1}(0, 0, b2, ..., bn)))
    let selector = |index: &MleIndex<F>, a, b| match index {
        MleIndex::IndexedBit(indexed_bit) => {
            match Ord::cmp(&round_index, indexed_bit) {
                // --- We haven't gotten to the indexed bit yet: just "combine" the two MLEs ---
                std::cmp::Ordering::Less => {
                    let a = a?;
                    let b = b?;
                    Ok(a + b)
                }
                // --- We are exactly looking at the indexed bit: the two MLEs we're summing ---
                // --- over should just be values. The result is that if you plug in 0, you get ---
                // --- the first value, and if you plug in 1 you get the second ---
                std::cmp::Ordering::Equal => {
                    let first = a?;
                    let second: Evals<F> = b?;

                    let (Evals(first_evals), Evals(second_evals)) = (first, second);
                    if first_evals.len() == second_evals.len() {
                        // we need to combine the evals by doing (1-x) * first eval + x * second eval
                        let first_evals = Evals(
                            first_evals
                                .into_iter()
                                .enumerate()
                                .map(|(idx, first_eval)| {
                                    first_eval * (F::one() - F::from(idx as u64))
                                })
                                .collect(),
                        );

                        let second_evals = Evals(
                            second_evals
                                .into_iter()
                                .enumerate()
                                .map(|(idx, second_eval)| second_eval * F::from(idx as u64))
                                .collect(),
                        );

                        Ok(first_evals + second_evals)
                    } else {
                        Err(ExpressionError::EvaluationError("Expression returns two evals that do not have length 3 on a selector bit"))
                    }
                }
                // --- If we're past the evaluation round, we should not have an unbound selector ---
                std::cmp::Ordering::Greater => Err(ExpressionError::InvalidMleIndex),
            }
        }
        MleIndex::Bound(coeff, _) => {
            let coeff_neg = F::one() - coeff;
            let a: Evals<F> = a?;
            let b: Evals<F> = b?;

            // --- Just r * V[2i + 1] + (1 - r) * V[2i] ---
            // --- (I.e. the selector formulation after the selector bit is bound to `r` above) ---
            Ok((b * coeff) + (a * coeff_neg))
        }
        _ => Err(ExpressionError::InvalidMleIndex),
    };

    let mle_eval = for<'a, 'b> |mle_ref: &'a DenseMleRef<F>,
                                beta_mle_ref: &'b DenseMleRef<F>|
                 -> Result<Evals<F>, ExpressionError> {
        // --- Just take the "independent variable" thing into account when we're evaluating the MLE reference as a product ---
        evaluate_mle_ref_product_with_beta(
            &[&mle_ref.clone()],
            round_index,
            max_degree,
            beta_mle_ref.clone(),
        )
        .map_err(ExpressionError::MleError)
    };

    // --- Just invert ---
    let negated = |a: Result<_, _>| a.map(|a: Evals<F>| a.neg());

    // --- Use the distributed/element-wise addition impl from earlier ---
    let sum = |a, b| {
        let a: Evals<F> = a?;
        let b: Evals<F> = b?;

        Ok(a + b)
    };

    // --- First see whether there are any iterated variables we should go over ---
    // --- Then just call the `evaluate_mle_ref_product` function ---
    let product = for<'a, 'b, 'c> |mle_refs: &'a [&'b DenseMleRef<F>],
                                   beta_mle_ref: &'c DenseMleRef<F>|
                     -> Result<Evals<F>, ExpressionError> {
        // have to include the beta table and evaluate as a product
        evaluate_mle_ref_product_with_beta(mle_refs, round_index, max_degree, beta_mle_ref.clone())
            .map_err(ExpressionError::MleError)
    };

    // --- Scalar is just distributed mult as defined earlier ---
    let scaled = |a, scalar| {
        let a = a?;
        Ok(a * scalar)
    };

    expr.evaluate_sumcheck(
        &constant,
        &selector,
        &mle_eval,
        &negated,
        &sum,
        &product,
        &scaled,
        &beta_table.table,
        round_index,
    )
}

pub fn compute_sumcheck_message_beta_cascade<F: FieldExt>(
    expr: &Expression<F, ProverExpr>,
    round_index: usize,
    max_degree: usize,
    beta_table: &NewBeta<F>,
) -> Result<Evals<F>, ExpressionError> {
    // --- Constant evaluation is just Sum(k) ---
    let constant = |constant| Ok(Evals((1..max_degree + 1).map(|_| constant).collect_vec()));

    // V_i(b_1, ..., b_n) = ((1 - b1) * (V_{i + 1}(0, 0, b2, ..., bn) + V_{i + 1}(0, 0, b2, ..., bn))) +
    // b1 * (V_{i + 1}(0, 0, b2, ..., bn) * V_{i + 1}(0, 0, b2, ..., bn)))
    let selector = |index: &MleIndex<F>, a, b, beta_table: &NewBeta<F>| {
        match index {
            MleIndex::IndexedBit(indexed_bit) => {
                let index_claim = beta_table.unbound_values.get(indexed_bit).unwrap();
                match Ord::cmp(&round_index, indexed_bit) {
                    // --- We haven't gotten to the indexed bit yet: just "combine" the two MLEs ---
                    std::cmp::Ordering::Less => {
                        let a_with_sel = a? * (F::one() - index_claim);
                        let b_with_sel = b? * index_claim;
                        Ok(a_with_sel + b_with_sel)
                    }
                    // --- We are exactly looking at the indexed bit: the two MLEs we're summing ---
                    // --- over should just be values. The result is that if you plug in 0, you get ---
                    // --- the first value, and if you plug in 1 you get the second ---
                    std::cmp::Ordering::Equal => {
                        let first = a?;
                        let second: Evals<F> = b?;

                        let (Evals(first_evals), Evals(second_evals)) = (first, second);
                        if first_evals.len() == second_evals.len() {
                            let eval_len = first_evals.len();
                            let one_minus_index_claim = F::one() - index_claim;
                            let beta_step = *index_claim - one_minus_index_claim;
                            let beta_evals =
                                std::iter::successors(Some(one_minus_index_claim), move |item| {
                                    Some(*item + beta_step)
                                })
                                .take(eval_len)
                                .collect_vec();

                            // we need to combine the evals by doing (1-x) * first eval + x * second eval
                            let first_evals = Evals(
                                first_evals
                                    .into_iter()
                                    .enumerate()
                                    .map(|(idx, first_eval)| {
                                        first_eval
                                            * (F::one() - F::from(idx as u64))
                                            * beta_evals[idx]
                                    })
                                    .collect(),
                            );

                            let second_evals = Evals(
                                second_evals
                                    .into_iter()
                                    .enumerate()
                                    .map(|(idx, second_eval)| {
                                        second_eval * F::from(idx as u64) * beta_evals[idx]
                                    })
                                    .collect(),
                            );

                            Ok(first_evals + second_evals)
                        } else {
                            Err(ExpressionError::EvaluationError("Expression returns two evals that do not have length 3 on a selector bit"))
                        }
                    }
                    // --- If we're past the evaluation round, we should not have an unbound selector ---
                    std::cmp::Ordering::Greater => Err(ExpressionError::InvalidMleIndex),
                }
            }
            MleIndex::Bound(coeff, bound_round_idx) => {
                let coeff_neg = F::one() - coeff;
                let one = &F::one();
                let beta_bound_val = beta_table
                    .updated_values
                    .get(bound_round_idx)
                    .unwrap_or(one);
                let a: Evals<F> = a?;
                let b: Evals<F> = b?;

                // --- Just r * V[2i + 1] + (1 - r) * V[2i] ---
                // --- (I.e. the selector formulation after the selector bit is bound to `r` above) ---
                Ok(((b * coeff) + (a * coeff_neg)) * *beta_bound_val)
            }
            _ => Err(ExpressionError::InvalidMleIndex),
        }
    };

    let mle_eval = for<'a, 'b, 'c> |mle_ref: &'a DenseMleRef<F>,
                                    unbound_beta_vals: &'b [F],
                                    bound_beta_vals: &'c [F]|
                     -> Result<Evals<F>, ExpressionError> {
        // --- Just take the "independent variable" thing into account when we're evaluating the MLE reference as a product ---
        Ok(beta_cascade(
            &[&mle_ref.clone()],
            max_degree,
            round_index,
            unbound_beta_vals,
            bound_beta_vals,
        ))
    };

    // --- Just invert ---
    let negated = |a: Result<_, _>| a.map(|a: Evals<F>| a.neg());

    // --- Use the distributed/element-wise addition impl from earlier ---
    let sum = |a, b| {
        let a: Evals<F> = a?;
        let b: Evals<F> = b?;

        Ok(a + b)
    };

    // --- First see whether there are any iterated variables we should go over ---
    // --- Then just call the `evaluate_mle_ref_product` function ---
    let product = for<'a, 'b, 'c, 'd> |mle_refs: &'a [&'b DenseMleRef<F>],
                                       unbound_beta_vals: &'c [F],
                                       bound_beta_vals: &'d [F]|
                         -> Result<Evals<F>, ExpressionError> {
        // have to include the beta table and evaluate as a product
        let hello = beta_cascade(
            mle_refs,
            max_degree,
            round_index,
            unbound_beta_vals,
            bound_beta_vals,
        );
        dbg!(&hello);
        Ok(beta_cascade(
            mle_refs,
            max_degree,
            round_index,
            unbound_beta_vals,
            bound_beta_vals,
        ))
    };

    // --- Scalar is just distributed mult as defined earlier ---
    let scaled = |a, scalar| {
        let a = a?;
        Ok(a * scalar)
    };

    expr.evaluate_sumcheck_beta_cascade(
        &constant,
        &selector,
        &mle_eval,
        &negated,
        &sum,
        &product,
        &scaled,
        &beta_table,
        round_index,
    )
}

pub fn get_relevant_beta_unbound_and_bound<F: FieldExt>(
    mle_indices: &[MleIndex<F>],
    newbeta: &NewBeta<F>,
) -> (Vec<F>, Vec<F>) {
    let bound_betas = mle_indices
        .iter()
        .filter_map(|index| match index {
            MleIndex::Bound(_, round_idx) => newbeta
                .updated_values
                .get(round_idx)
                .map(|&item| item.clone()),
            _ => None,
        })
        .collect_vec();

    let unbound_betas = mle_indices
        .iter()
        .filter_map(|index| match index {
            MleIndex::IndexedBit(round_idx) => {
                let item = newbeta.unbound_values[round_idx];
                Some(item)
            }
            _ => None,
        })
        .collect_vec();

    (unbound_betas, bound_betas)
}

/// evaluate mle refs when there's no independent variable, i.e.
/// sum_{x_1, ..., x_n} V_1(x_1, ..., x_n) * V_2(x_1, ..., x_n),
/// with x_1, ..., x_n over the boolean hypercube
/// the evaluation will be one single field element
pub fn evaluate_mle_ref_product_no_inde_var<F: FieldExt>(
    mle_refs: &[&impl MleRef<F = F>],
) -> Result<F, MleError> {
    // --- Gets the total number of iterated variables across all MLEs within this product ---
    let max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    // eval_count = 1 because there's no independent variable
    let eval_count = 1;

    //iterate across all pairs of evaluations
    let evals = cfg_into_iter!((0..1 << max_num_vars)).fold(
        #[cfg(feature = "parallel")]
        || vec![F::zero(); eval_count],
        #[cfg(not(feature = "parallel"))]
        vec![F::zero(); eval_count],
        |mut acc, index| {
            //get the product of all evaluations over 0/1/..degree
            let evals = mle_refs
                .iter()
                .map(|mle_ref| {
                    let zero = F::zero();
                    let index = if mle_ref.num_vars() < max_num_vars {
                        let max = 1 << mle_ref.num_vars();
                        index % max
                    } else {
                        index
                    };
                    let first = *mle_ref.bookkeeping_table().get(index).unwrap_or(&zero);

                    std::iter::once(first)
                })
                .map(|item| -> Box<dyn Iterator<Item = F>> { Box::new(item) })
                .reduce(|acc, evals| Box::new(acc.zip(evals).map(|(acc, eval)| acc * eval)))
                .unwrap();

            acc.iter_mut()
                .zip(evals)
                .for_each(|(acc, eval)| *acc += eval);
            acc
        },
    );

    #[cfg(feature = "parallel")]
    let evals = evals.reduce(
        || vec![F::zero(); eval_count],
        |mut acc, partial| {
            acc.iter_mut()
                .zip(partial)
                .for_each(|(acc, partial)| *acc += partial);
            acc
        },
    );

    assert_eq!(evals.len(), 1);

    Ok(evals[0])
}

/// evaluates the product of multiple mle refs (in the evalutaion form),
/// (the mles could be beta tables as well)
/// the returned results can be the following expresssion:
/// sum_{x_2, ..., x_n} V_1(X, x_2, ..., x_n) * V_2(X, x_2, ..., x_n) * V_2(X, x_2, x_3),
/// evaluated at X = 0, 1, ..., degree
/// note that when one of the mle_refs have less variables, there's a wrap around: % max
pub fn evaluate_mle_ref_product<F: FieldExt>(
    mle_refs: &[&impl MleRef<F = F>],
    degree: usize,
) -> Result<Evals<F>, MleError> {
    // --- Gets the total number of iterated variables across all MLEs within this product ---
    let max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    //There is an independent variable, and we must extract `degree` evaluations of it, over `0..degree`
    let eval_count = degree + 1;

    //iterate across all pairs of evaluations
    let evals = cfg_into_iter!((0..1 << (max_num_vars - 1))).fold(
        #[cfg(feature = "parallel")]
        || vec![F::zero(); eval_count],
        #[cfg(not(feature = "parallel"))]
        vec![F::zero(); eval_count],
        |mut acc, index| {
            //get the product of all evaluations over 0/1/..degree
            let evals = mle_refs
                .iter()
                .map(|mle_ref| {
                    let zero = F::zero();
                    let index = if mle_ref.num_vars() < max_num_vars {
                        let max = 1 << mle_ref.num_vars();
                        (index * 2) % max
                    } else {
                        index * 2
                    };
                    let first = *mle_ref.bookkeeping_table().get(index).unwrap_or(&zero);

                    let second = if mle_ref.num_vars() != 0 {
                        *mle_ref.bookkeeping_table().get(index + 1).unwrap_or(&zero)
                    } else {
                        first
                    };

                    // let second = *mle_ref.mle().get(index + 1).unwrap_or(&zero);
                    let step = second - first;
                    let successors =
                        std::iter::successors(Some(second), move |item| Some(*item + step));
                    //iterator that represents all evaluations of the MLE extended to arbitrarily many linear extrapolations on the line of 0/1
                    std::iter::once(first).chain(successors)
                })
                .map(|item| -> Box<dyn Iterator<Item = F>> { Box::new(item) })
                .reduce(|acc, evals| Box::new(acc.zip(evals).map(|(acc, eval)| acc * eval)))
                .unwrap();

            acc.iter_mut()
                .zip(evals)
                .for_each(|(acc, eval)| *acc += eval);
            acc
        },
    );

    #[cfg(feature = "parallel")]
    let evals = evals.reduce(
        || vec![F::zero(); eval_count],
        |mut acc, partial| {
            acc.iter_mut()
                .zip(partial)
                .for_each(|(acc, partial)| *acc += partial);
            acc
        },
    );

    Ok(Evals(evals))
}

/// BETA CASCADE
pub fn successors_from_mle_ref_product<F: FieldExt>(
    mle_refs: &[&impl MleRef<F = F>],
    degree: usize,
) -> Result<Vec<F>, MleError> {
    // --- Gets the total number of iterated variables across all MLEs within this product ---
    let max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    //There is an independent variable, and we must extract `degree` evaluations of it, over `0..degree`
    let eval_count = degree + 1;

    //iterate across all pairs of evaluations
    let evals = cfg_into_iter!((0..1 << (max_num_vars - 1))).flat_map(|index| {
        //get the product of all evaluations over 0/1/..degree
        let successors_product = mle_refs
            .iter()
            .map(|mle_ref| {
                let zero = F::zero();
                let index = if mle_ref.num_vars() < max_num_vars {
                    let max = 1 << mle_ref.num_vars();
                    (index * 2) % max
                } else {
                    index * 2
                };
                let first = *mle_ref.bookkeeping_table().get(index).unwrap_or(&zero);

                let second = if mle_ref.num_vars() != 0 {
                    *mle_ref.bookkeeping_table().get(index + 1).unwrap_or(&zero)
                } else {
                    first
                };

                // let second = *mle_ref.mle().get(index + 1).unwrap_or(&zero);
                let step = second - first;
                let successors =
                    std::iter::successors(Some(second), move |item| Some(*item + step));
                //iterator that represents all evaluations of the MLE extended to arbitrarily many linear extrapolations on the line of 0/1
                std::iter::once(first).chain(successors)
            })
            .map(|item| -> Box<dyn Iterator<Item = F>> { Box::new(item) })
            .reduce(|acc, evals| Box::new(acc.zip(evals).map(|(acc, eval)| acc * eval)))
            .unwrap();

        let eval_elements = successors_product.take(eval_count).collect_vec();

        eval_elements
    });

    let evals_vec: Vec<F> = evals.collect();

    Ok(evals_vec)
}

pub fn beta_cascade_step<F: FieldExt>(mle_successor_vec: &mut Vec<F>, beta_val: F) -> Vec<F> {
    let (one_minus_beta_val, beta_val) = (F::one() - beta_val, beta_val);
    let half_vec_len = mle_successor_vec.len() / 2;
    let new_successor = cfg_into_iter!((0..half_vec_len)).map(|idx| {
        (mle_successor_vec[idx] * one_minus_beta_val) + (mle_successor_vec[2 * idx] * beta_val)
    });
    let new_successor_vec = new_successor.collect();
    new_successor_vec
}

pub fn apply_updated_beta_values_to_evals<F: FieldExt>(
    evals: Vec<F>,
    beta_updated_vals: &[F],
) -> Evals<F> {
    let beta_total_updated_product = beta_updated_vals
        .iter()
        .fold(F::one(), |acc, elem| acc * elem);
    evals
        .iter()
        .map(|elem| beta_total_updated_product * elem)
        .collect_vec();

    Evals(evals)
}

pub fn beta_cascade_no_independent_variable<F: FieldExt>(
    mle_refs: &[&impl MleRef<F = F>],
    degree: usize,
    beta_vals: &[F],
    beta_updated_vals: &[F],
) -> Evals<F> {
    let mut mle_successor_vec = successors_from_mle_ref_product(mle_refs, degree).unwrap();
    beta_vals.iter().for_each(|beta_val| {
        let (one_minus_beta_val, beta_val) = (F::one() - beta_val, beta_val);
        let mle_successor_vec_iter = mle_successor_vec
            .par_chunks(2)
            .map(|bits| bits[0] * one_minus_beta_val + bits[1] * beta_val);
        mle_successor_vec = mle_successor_vec_iter.collect();
    });

    assert_eq!(mle_successor_vec.len(), 1);

    apply_updated_beta_values_to_evals(mle_successor_vec, beta_updated_vals)
}

pub fn beta_cascade<F: FieldExt>(
    mle_refs: &[&impl MleRef<F = F>],
    degree: usize,
    round_index: usize,
    beta_vals: &[F],
    beta_updated_vals: &[F],
) -> Evals<F> {
    let mut mle_successor_vec = successors_from_mle_ref_product(mle_refs, degree).unwrap();

    let mles_have_independent_variable = mle_refs
        .iter()
        .map(|mle_ref| {
            mle_ref
                .mle_indices()
                .contains(&MleIndex::IndexedBit(round_index))
        })
        .reduce(|acc, item| acc | item)
        .unwrap();

    if mles_have_independent_variable {
        beta_vals.iter().skip(1).rev().for_each(|val| {
            mle_successor_vec = beta_cascade_step(&mut mle_successor_vec, *val);
        });
        let evals = if beta_vals.len() >= 1 {
            let second_beta_successor = beta_vals[0];
            let first_beta_successor = F::one() - second_beta_successor;
            let step = second_beta_successor - first_beta_successor;
            let beta_successors =
                std::iter::successors(Some(first_beta_successor), move |item| Some(*item + step));
            dbg!(mle_successor_vec.len());
            dbg!(degree);
            beta_successors
                .zip(mle_successor_vec)
                .map(|(beta_succ, mle_succ)| beta_succ * mle_succ)
                .collect_vec()
        } else {
            vec![F::one()]
        };
        apply_updated_beta_values_to_evals(evals, beta_updated_vals)
    } else {
        beta_cascade_no_independent_variable(mle_refs, degree, beta_vals, beta_updated_vals)
    }
}

/// @param mle_refs: The list of MLEs which are being multiplied together
/// @param round_index: Which round of sumcheck it currently is
/// @param degree: The degree of the indexed variable
/// @param beta_ref: The beta table which is being multiplied with the MLEs
///
/// # Errors:
/// - MleError::EmptyMleList -- when there are zero MLEs within the list
/// - MleError::NotIndexedError -- when any MLE within the list is not indexed
pub fn evaluate_mle_ref_product_with_beta<F: FieldExt>(
    mle_refs: &[&DenseMleRef<F>],
    round_index: usize,
    degree: usize,
    beta_ref: DenseMleRef<F>,
) -> Result<Evals<F>, MleError> {
    for mle_ref in mle_refs {
        if !mle_ref.indexed() {
            return Err(MleError::NotIndexedError);
        }
    }
    if !beta_ref.indexed() {
        return Err(MleError::NotIndexedError);
    }

    let mles_have_independent_variable = mle_refs
        .iter()
        .map(|mle_ref| {
            mle_ref
                .mle_indices()
                .contains(&MleIndex::IndexedBit(round_index))
        })
        .reduce(|acc, item| acc | item)
        .ok_or(MleError::EmptyMleList)?;

    if mles_have_independent_variable {
        let mut mle_refs = mle_refs.to_vec();
        mle_refs.push(&beta_ref);
        evaluate_mle_ref_product(&mle_refs, degree)
    } else {
        // say we have some expression like sum_{x_1, x_2} beta(x_0, x_1, x_2) \times V(x_1, x_2)^2
        //     (we assume there's only one extra variable in the beginning of beta mle,
        //     otherwise, we need to use beta split)
        // First,
        // we fix x_0 = 0, and want to get sum_{x_1, x_2} beta(0, x_1, x_2) \times V(x_1, x_2)^2
        let beta_first_half: DenseMleRef<F> = DenseMle::new_from_raw(
            beta_ref
                .bookkeeping_table()
                .iter()
                .step_by(2)
                .cloned()
                .collect_vec(),
            beta_ref.get_layer_id(),
            None,
        )
        .mle_ref();

        let mut mle_ref_first_half = mle_refs.to_vec().clone();
        mle_ref_first_half.push(&beta_first_half);
        let beta_at_0 = evaluate_mle_ref_product_no_inde_var(&mle_ref_first_half)?;

        // Then,
        // we do the same for the second half of the beta table, i.e. fixing x_0 = 1,
        // to get sum_{x_1, x_2} beta(1, x_1, x_2) \times V(x_1, x_2)^2
        let beta_second_half: DenseMleRef<F> = DenseMle::new_from_raw(
            beta_ref
                .bookkeeping_table()
                .iter()
                .skip(1)
                .step_by(2)
                .cloned()
                .collect_vec(),
            beta_ref.get_layer_id(),
            None,
        )
        .mle_ref();

        let mut mle_ref_second_half = mle_refs.to_vec().clone();
        mle_ref_second_half.push(&beta_second_half);
        let beta_at_1 = evaluate_mle_ref_product_no_inde_var(&mle_ref_second_half)?;

        // partials have two elements (beta is always linear)
        // 1. sum_{x_1, x_2} beta(x_0 = 0, x_1, x_2) \times V(x_1, x_2)^2
        // 2. sum_{x_1, x_2} beta(x_0 = 1, x_1, x_2) \times V(x_1, x_2)^2
        // for however many more evaluations, we just extrapolate
        let partials = [beta_at_0, beta_at_1];

        let eval_count = degree + 1;

        let step: F = partials[1] - partials[0];
        let mut counter = 2;
        let evals = std::iter::once(partials[0])
            .chain(std::iter::successors(Some(partials[1]), move |item| {
                if counter < eval_count {
                    counter += 1;
                    Some(*item + step)
                } else {
                    None
                }
            }))
            .collect_vec();

        Ok(Evals(evals))
    }
}

/// Returns the maximum degree of b_{curr_round} within an expression
/// (and therefore the number of prover messages we need to send)
pub(crate) fn get_round_degree<F: FieldExt>(
    expr: &Expression<F, ProverExpr>,
    curr_round: usize,
) -> usize {
    // --- By default, all rounds have degree at least 2 (beta table included) ---
    let mut round_degree = 1;

    let mut get_degree_closure =
        for<'a, 'b> |expr: &'a ExpressionNode<F, ProverExpr>,
                     mle_vec: &'b <ProverExpr as ExpressionType<F>>::MleVec|
                     -> Result<(), ()> {
            let round_degree = &mut round_degree;

            // --- The only exception is within a product of MLEs ---
            if let ExpressionNode::Product(mle_vec_indices) = expr {
                let mut product_round_degree: usize = 0;
                for mle_vec_index in mle_vec_indices {
                    let mle_ref = mle_vec_index.get_mle(mle_vec);

                    let mle_indices = mle_ref.mle_indices();
                    for mle_index in mle_indices {
                        if *mle_index == MleIndex::IndexedBit(curr_round) {
                            product_round_degree += 1;
                            break;
                        }
                    }
                }
                if *round_degree < product_round_degree {
                    *round_degree = product_round_degree;
                }
            }
            Ok(())
        };

    expr.traverse(&mut get_degree_closure).unwrap();
    // add 1 cuz beta table but idk if we would ever use this without a beta table
    round_degree + 1
}

/// Use degree + 1 evaluations to figure out the evaluation at some arbitrary point
pub(crate) fn evaluate_at_a_point<F: FieldExt>(
    given_evals: &Vec<F>,
    point: F,
) -> Result<F, InterpError> {
    // Need degree + 1 evaluations to interpolate
    let eval = (0..given_evals.len())
        .map(
            // Create an iterator of everything except current value
            |x| {
                (0..x)
                    .chain(x + 1..given_evals.len())
                    .map(|x| F::from(x as u64))
                    .fold(
                        // Compute vector of (numerator, denominator)
                        (F::one(), F::one()),
                        |(num, denom), val| {
                            (num * (point - val), denom * (F::from(x as u64) - val))
                        },
                    )
            },
        )
        .enumerate()
        .map(
            // Add up barycentric weight * current eval at point
            |(x, (num, denom))| given_evals[x] * num * denom.invert().unwrap(),
        )
        .reduce(|x, y| x + y);
    eval.ok_or(InterpError::NoInverse)
}
