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
        betavalues::BetaValues,
        dense::{DenseMle, DenseMleRef},
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

/// this is the function to compute a sumcheck message using the beta cascade algorithm. instead
/// of using a beta table to linearize an expression, we utilize the fact that for each specific
/// node in an expression tree, we only need exactly the beta values corresponding to the
/// indices present in that node.
/// each different type of expression node (constant, selector, product, sum, neg, scaled, mle) is
/// treated differently, so we create closures for each which are then evaluated by the
/// `evaluate_sumcheck_beta_cascade` function.
pub fn compute_sumcheck_message_beta_cascade<F: FieldExt>(
    expr: &Expression<F, ProverExpr>,
    round_index: usize,
    max_degree: usize,
    beta_values: &BetaValues<F>,
) -> Result<Evals<F>, ExpressionError> {
    // a constant does not have any variables, so we do not need a beta table at all. therefore we just repeat
    // the constant evaluation for the degree+1 number of times as this is how many evaluations we need.
    let constant = |constant| Ok(Evals((0..max_degree + 1).map(|_| constant).collect_vec()));

    // the selector is split into three cases:
    let selector = |index: &MleIndex<F>, a, b, beta_table: &BetaValues<F>| match index {
        MleIndex::IndexedBit(indexed_bit) => {
            let index_claim = beta_table.unbound_values.get(indexed_bit).unwrap();
            match Ord::cmp(&round_index, indexed_bit) {
                std::cmp::Ordering::Less => {
                    let a_with_sel = a? * (F::one() - index_claim);
                    let b_with_sel = b? * index_claim;
                    Ok(a_with_sel + b_with_sel)
                }
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

                        let first_evals = Evals(
                            first_evals
                                .into_iter()
                                .enumerate()
                                .map(|(idx, first_eval)| {
                                    first_eval * (F::one() - F::from(idx as u64)) * beta_evals[idx]
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

            Ok(((b * coeff) + (a * coeff_neg)) * *beta_bound_val)
        }
        _ => Err(ExpressionError::InvalidMleIndex),
    };

    let mle_eval = for<'a, 'b, 'c> |mle_ref: &'a DenseMleRef<F>,
                                    unbound_beta_vals: &'b [F],
                                    bound_beta_vals: &'c [F]|
                     -> Result<Evals<F>, ExpressionError> {
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

    let sum = |a, b| {
        let a: Evals<F> = a?;
        let b: Evals<F> = b?;
        Ok(a + b)
    };

    let product = for<'a, 'b, 'c, 'd> |mle_refs: &'a [&'b DenseMleRef<F>],
                                       unbound_beta_vals: &'c [F],
                                       bound_beta_vals: &'d [F]|
                         -> Result<Evals<F>, ExpressionError> {
        Ok(beta_cascade(
            mle_refs,
            max_degree,
            round_index,
            unbound_beta_vals,
            bound_beta_vals,
        ))
    };

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
        &beta_values,
        round_index,
    )
}

/// BETA CASCADE
fn successors_from_mle_ref_product<F: FieldExt>(
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

fn successors_from_mle_ref_product_no_ind_var<F: FieldExt>(
    mle_refs: &[&impl MleRef<F = F>],
) -> Result<Vec<F>, MleError> {
    // --- Gets the total number of iterated variables across all MLEs within this product ---
    let max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    let evals_vec = if max_num_vars > 0 {
        let evals = cfg_into_iter!((0..1 << (max_num_vars - 1))).map(|index| {
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
                    *mle_ref.bookkeeping_table().get(index).unwrap_or(&zero)
                })
                .reduce(|acc, eval| acc * eval)
                .unwrap();
            successors_product
        });
        evals.collect()
    } else {
        let val = mle_refs.into_iter().fold(F::one(), |acc, mle_ref| {
            assert_eq!(mle_ref.bookkeeping_table().len(), 1);
            acc * mle_ref.bookkeeping_table()[0]
        });
        vec![val]
    };

    Ok(evals_vec)
}

fn beta_cascade_step<F: FieldExt>(mle_successor_vec: &mut Vec<F>, beta_val: F) -> Vec<F> {
    let (one_minus_beta_val, beta_val) = (F::one() - beta_val, beta_val);
    let half_vec_len = mle_successor_vec.len() / 2;
    let new_successor = cfg_into_iter!((0..half_vec_len)).map(|idx| {
        (mle_successor_vec[idx] * one_minus_beta_val)
            + (mle_successor_vec[idx + half_vec_len] * beta_val)
    });
    let new_successor_vec = new_successor.collect();
    new_successor_vec
}

fn apply_updated_beta_values_to_evals<F: FieldExt>(
    evals: Vec<F>,
    beta_updated_vals: &[F],
) -> Evals<F> {
    let beta_total_updated_product = beta_updated_vals
        .iter()
        .fold(F::one(), |acc, elem| acc * elem);
    let evals = evals
        .iter()
        .map(|elem| beta_total_updated_product * elem)
        .collect_vec();

    Evals(evals)
}

fn beta_cascade_no_independent_variable<F: FieldExt>(
    mle_refs: &[&impl MleRef<F = F>],
    beta_vals: &[F],
    beta_updated_vals: &[F],
) -> Evals<F> {
    let mut mle_successor_vec = successors_from_mle_ref_product_no_ind_var(mle_refs).unwrap();
    if mle_successor_vec.len() > 1 {
        beta_vals.iter().for_each(|beta_val| {
            let (one_minus_beta_val, beta_val) = (F::one() - beta_val, beta_val);
            let mle_successor_vec_iter = mle_successor_vec
                .par_chunks(2)
                .map(|bits| bits[0] * one_minus_beta_val + bits[1] * beta_val);
            mle_successor_vec = mle_successor_vec_iter.collect();
        });
    }

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
        let mut mle_successor_vec = successors_from_mle_ref_product(mle_refs, degree).unwrap();
        beta_vals.iter().skip(1).rev().for_each(|val| {
            mle_successor_vec = beta_cascade_step(&mut mle_successor_vec, *val);
        });
        let evals = if beta_vals.len() >= 1 {
            let second_beta_successor = beta_vals[0];
            let first_beta_successor = F::one() - second_beta_successor;
            let step = second_beta_successor - first_beta_successor;
            let beta_successors =
                std::iter::successors(Some(first_beta_successor), move |item| Some(*item + step));
            beta_successors
                .zip(mle_successor_vec)
                .map(|(beta_succ, mle_succ)| beta_succ * mle_succ)
                .collect_vec()
        } else {
            vec![F::one()]
        };
        apply_updated_beta_values_to_evals(evals, beta_updated_vals)
    } else {
        beta_cascade_no_independent_variable(mle_refs, beta_vals, beta_updated_vals)
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
