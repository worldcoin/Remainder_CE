//! Contains cryptographic algorithms for going through the sumcheck protocol in
//! the context of a GKR prover.
//!
//! Let `P: F^n -> F` denote the polynomial [Expression] used to define some GKR
//! Layer. This means that the value at a certain index `b \in {0, 1}^n` of the
//! layer is given by `P(b)`. Denote by `V: {0, 1}^n -> F` the restriction of
//! `P` on the hypercube.
//!
//! As part of the GKR protocol, the prover needs to assert the following
//! statement about the multilinear extention `\tilde{V}: F^n -> F` of `V`:
//! ```text
//!     \tilde{V}(g_1, ..., g_n) = r \in F`,
//!         for some challenges g_1, ..., g_n \in F                    (1)
//! ```
//! (Note that, in general, `P` and `\tilde{V}` are different functions. They
//!  are both extensions of `V`, but `\tilde{V}` is a linear polynomial on each
//!  of it's variables).
//!
//! The left-hand side of (1) can be expressed as a sum over the hypercube as
//! follows:
//! ```text
//!     \sum_{b_1 \in {0, 1}}
//!     \sum_{b_2 \in {0, 1}}
//!         ...
//!     \sum_{b_n \in {0, 1}}
//!        \beta(b_1, ..., b_n, g_1, ..., g_n) * P(b_1, b_2, ..., b_n) = r  (2)
//! ```
//! where `\beta` is the following polynomial extending the equality predicate:
//! ```text
//!     \beta(b_1, ..., b_n, g_1, ..., g_n) =
//!         \prod_{i = 1}^n [ b_i * g_i + (1 - b_i) * (1 - g_i) ]
//! ```
//!
//! The functions in this module run the sumcheck protocol on expressions of the
//! form described in equation (2). See the documentation of
//! `compute_sumcheck_message_beta_cascade` for more information.

use std::{
    iter::{repeat, successors},
    ops::{Add, Mul, Neg},
};

/// Tests for sumcheck with various expressions.
#[cfg(test)]
pub mod tests;

use anyhow::{anyhow, Result};
use ark_std::{cfg_chunks, cfg_into_iter};
use itertools::{repeat_n, Itertools};
use thiserror::Error;

use crate::{
    expression::{
        generic_expr::{Expression, ExpressionNode, ExpressionType},
        prover_expr::ProverExpr,
    },
    mle::{Mle, MleIndex},
};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
#[cfg(feature = "parallel")]
use rayon::prelude::ParallelSlice;

use remainder_shared_types::Field;

/// Errors to do with the evaluation of MleRefs.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MleError {
    /// Passed list of Mles is empty.
    #[error("Passed list of Mles is empty")]
    EmptyMleList,

    /// Beta table not yet initialized for Mle.
    #[error("Beta table not yet initialized for Mle")]
    NoBetaTable,

    /// Layer does not have claims yet.
    #[error("Layer does not have claims yet")]
    NoClaim,

    /// Unable to eval beta.
    #[error("Unable to eval beta")]
    BetaEvalError,

    /// Cannot compute sumcheck message on un-indexed MLE.
    #[error("Cannot compute sumcheck message on un-indexed MLE")]
    NotIndexedError,
}

/// Verification error.
#[derive(Error, Debug, Clone)]
pub enum VerifyError {
    /// Failed sumcheck round.
    #[error("Failed sumcheck round")]
    SumcheckBad,
}

/// Error when Interpolating a univariate polynomial.
#[derive(Error, Debug, Clone)]
pub enum InterpError {
    /// Too few evaluation points.
    #[error("Too few evaluation points")]
    EvalLessThanDegree,

    /// No possible polynomial.
    #[error("No possible polynomial")]
    NoInverse,
}

/// A type representing the univariate polynomial `g_i: F -> F` which the prover
/// sends to the verifier in each round of sumcheck. Note that we are using an
/// evaluation representation of polynomials, which means this type just holds
/// the evaluations: `[g_i(0), g_i(1), ..., g_i(d)]`, where `d` is the degree of
/// `g_i`.
#[derive(PartialEq, Debug, Clone)]
pub struct SumcheckEvals<F: Field>(pub Vec<F>);

impl<F: Field> Neg for SumcheckEvals<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        // Negation for a bunch of eval points is just element-wise negation
        SumcheckEvals(self.0.into_iter().map(|eval| eval.neg()).collect_vec())
    }
}

impl<F: Field> Add for SumcheckEvals<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        SumcheckEvals(
            self.0
                .into_iter()
                .zip(rhs.0)
                .map(|(lhs, rhs)| lhs + rhs)
                .collect_vec(),
        )
    }
}

impl<F: Field> Mul<F> for SumcheckEvals<F> {
    type Output = Self;
    fn mul(self, rhs: F) -> Self {
        SumcheckEvals(
            self.0
                .into_iter()
                .zip(repeat(rhs))
                .map(|(lhs, rhs)| lhs * rhs)
                .collect_vec(),
        )
    }
}

impl<F: Field> Mul<&F> for SumcheckEvals<F> {
    type Output = Self;
    fn mul(self, rhs: &F) -> Self {
        SumcheckEvals(
            self.0
                .into_iter()
                .zip(repeat(rhs))
                .map(|(lhs, rhs)| lhs * rhs)
                .collect_vec(),
        )
    }
}

/// this function will take a list of mle refs, and compute the element-wise
/// product of all of their bookkeeping tables along with the "successors."
///
/// for example, if we have two bookkeeping tables [a_1, a_2, a_3, a_4] and
/// [c_1, c_2, c_3, c_4] and the degree of our expression at this index is 3, we
/// need 4 evaluations for a unique curve. therefore first we will compute [a_1,
/// a_2, (1-2)a_1 + 2a_2, (1-3)a_1 + 3a_2, a_3, a_4, (1-2)a_3 + 2a_4, (1-3)a_3 +
/// 3a_4] and the same thing for the other mle and element-wise multiply both
/// results. the resulting vector will always be size (degree + 1) * (2 ^
/// (max_num_vars - 1))
///
/// this function assumes that the first variable is an independent variable.
pub fn successors_from_mle_product<F: Field>(
    mles: &[&impl Mle<F>],
    degree: usize,
    round_index: usize,
) -> Result<Vec<Vec<F>>> {
    // Gets the total number of free variables across all MLEs within this
    // product
    let mut max_num_vars = mles
        .iter()
        .map(|mle| mle.num_free_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    let mles_have_independent_variable = mles
        .iter()
        .map(|mle| mle.mle_indices().contains(&MleIndex::Indexed(round_index)))
        .reduce(|acc, item| acc | item)
        .unwrap();

    // We add 1 to the max number of variables if there is no independent variable
    // to account for the independent variable contained within the beta.
    if !mles_have_independent_variable {
        max_num_vars += 1;
    }

    let successors_vec = cfg_into_iter!((0..1 << (max_num_vars - 1)))
        .map(|mle_index| {
            mles.iter()
                .map(|mle| {
                    let num_coefficients_in_mle = 1 << mle.num_free_vars();

                    // Over here, we perform the wrap-around functionality if we
                    // are multiplying two mles with different number of
                    // variables. for example if we are multiplying V(b_1, b_2)
                    // * V(b_1), and summing over b_2, then the overall sum is
                    // V(b_1, 0) * V(b_1) + V(b_1, 1) * V(b_1). it can be seen
                    // that the "smaller" mle (the one over less variables) has
                    // to repeat itself an according number of times when the
                    // sum is over a variable it does not contain. the
                    // appropriate index is therefore determined as follows.
                    let mle_index = if mle.num_free_vars() < max_num_vars {
                        // If we have less than the max number of variables,
                        // then we perform this variable-repeat functionality by
                        // first rounding to the nearest power of 2, and then
                        // taking the floor of the index divided by the difference in
                        // power of 2.
                        let multiple = (1 << max_num_vars) / num_coefficients_in_mle;
                        mle_index / multiple
                    } else {
                        mle_index
                    };
                    // Over here, we get the elements in the pair so when index
                    // = 0, it's [0] and [1], if index = 1, it's [2] and [3],
                    // etc. because we are extending a function that was
                    // originally defined over the hypercube, each pair
                    // corresponds to two points on a line. we grab these two
                    // points here
                    let first = mle.get(mle_index).unwrap_or(F::ZERO);
                    let second = if mle.num_free_vars() != 0 {
                        mle.get(mle_index + (num_coefficients_in_mle / 2))
                            .unwrap_or(F::ZERO)
                    } else {
                        first
                    };
                    let step = second - first;

                    // creating the successors representing the evaluations for
                    // \pi_{i = 1}^n f_i(X, b_2, ..., b_n) across X = 0, 1, 2,
                    // ... for a specific set of b_2, ..., b_n.
                    Box::new(successors(Some(first), move |item| Some(*item + step)))
                        as Box<dyn Iterator<Item = F> + Send>
                })
                .reduce(|mut a, mut b| {
                    Box::new(
                        successors(Some((a.next().unwrap(), b.next().unwrap())), move |_| {
                            Some((a.next().unwrap(), b.next().unwrap()))
                        })
                        .map(|(a_val, b_val)| a_val * b_val),
                    ) as Box<dyn Iterator<Item = F> + Send>
                })
                .unwrap()
                .take(degree + 1)
                .collect()
        })
        .collect();
    Ok(successors_vec)
}

/// This is one step of the beta cascade algorithm, performing `(1 - beta_val) *
/// mle[index] + beta_val * mle[index + 1]`
pub(crate) fn beta_cascade_step<F: Field>(
    mle_successor_vec: &[Vec<F>],
    beta_val: F,
) -> Vec<Vec<F>> {
    let (one_minus_beta_val, beta_val) = (F::ONE - beta_val, beta_val);

    cfg_chunks!(mle_successor_vec, 2)
        .map(|successor_pair| {
            let first_evals = &successor_pair[0];
            let second_evals = &successor_pair[1];
            let mut inner_result = Vec::with_capacity(first_evals.len());
            inner_result.extend(
                first_evals
                    .iter()
                    .zip(second_evals)
                    .map(|(fold_a, fold_b)| one_minus_beta_val * fold_a + beta_val * fold_b),
            );
            inner_result
        })
        .collect()
}

/// This is the final step of beta cascade, where we take all the "bound" beta
/// values and scale all of the evaluations by the product of all of these
/// values.
pub(crate) fn apply_updated_beta_values_to_evals<F: Field>(
    evals: Vec<F>,
    beta_updated_vals: &[F],
) -> SumcheckEvals<F> {
    let beta_total_updated_product = beta_updated_vals
        .iter()
        .fold(F::ONE, |acc, elem| acc * elem);
    let evals = evals
        .iter()
        .map(|elem| beta_total_updated_product * elem)
        .collect_vec();

    SumcheckEvals(evals)
}

/// this is how we compute the evaluations of a product of mle refs along with a
/// beta table. rather than using the full expanded version of a beta table, we
/// instead just use the beta values vectors (which are unbound beta values, and
/// the bound beta values) there are (degree + 1) evaluations that are returned
/// which are the evaluations of the univariate polynomial where the
/// "round_index"-th bit is the independent variable.
pub fn beta_cascade<F: Field>(
    mles: &[&impl Mle<F>],
    degree: usize,
    round_index: usize,
    beta_vals_vec: &[Vec<F>],
    beta_updated_vals_vec: &[Vec<F>],
    random_coefficients: &[F],
) -> SumcheckEvals<F> {
    // Check that the number of beta values that we have is equal to the number
    // of random coefficients, which must be the same because these are the
    // number of claims we are aggregating over.
    assert_eq!(beta_vals_vec.len(), beta_updated_vals_vec.len());
    assert_eq!(beta_vals_vec.len(), random_coefficients.len());

    let mle_successor_vec = successors_from_mle_product(mles, degree, round_index).unwrap();

    // We compute the sumcheck evaluations using beta cascade for the same
    // set of MLE successors, but different beta values. All of these are
    // stored in the iterator.
    let evals_iter = (beta_vals_vec.iter().zip(beta_updated_vals_vec))
        .zip(random_coefficients)
        .map(|((beta_vals, beta_updated_vals), random_coeff)| {
            // Apply beta cascade steps, reducing `mle_successor_vec` size
            // progressively.
            let final_successor_vec = if beta_vals.len() > 1 {
                let mut current_successor_vec =
                    beta_cascade_step(&mle_successor_vec, *beta_vals.last().unwrap());
                // All the skips, a really gross way of making sure we
                // don't clone all of mle_successor_vec each time.
                for val in beta_vals.iter().skip(1).rev().skip(1) {
                    // Apply beta cascade step and return the new vector, replacing
                    // the previous one
                    current_successor_vec = beta_cascade_step(&current_successor_vec, *val);
                }
                current_successor_vec
            } else {
                // Only clone if this is going to be the final one we fold to get evaluations.
                mle_successor_vec.clone()
            };
            // Check that mle_successor_vec now contains only one element after
            // cascading
            assert_eq!(final_successor_vec.len(), 1);

            // Extract the remaining iterator from mle_successor_vec by popping it
            let folded_mle_successors = &final_successor_vec[0];

            // for the MSB of the beta value, this must be
            // the independent variable. otherwise it would already be bound.
            // therefore we need to compute the successors of this value in order to
            // get its evaluations.
            let evals = if !beta_vals.is_empty() {
                let second_beta_successor = beta_vals[0];
                let first_beta_successor = F::ONE - second_beta_successor;
                let step = second_beta_successor - first_beta_successor;
                let beta_successors =
                    std::iter::successors(Some(first_beta_successor), move |item| {
                        Some(*item + step)
                    });
                // the length of the mle successor vec before this last step must be
                // degree + 1. therefore we can just do a zip with the beta
                // successors to get the final degree + 1 evaluations.
                beta_successors
                    .zip(folded_mle_successors)
                    .map(|(beta_succ, mle_succ)| beta_succ * mle_succ)
                    .take(degree + 1)
                    .collect_vec()
            } else {
                vec![F::ONE]
            };
            // apply the bound beta values as a scalar factor to each of the
            // evaluations Multiply by the random coefficient to get the
            // random linear combination by summing at the end.
            apply_updated_beta_values_to_evals(evals, beta_updated_vals) * random_coeff
        });
    // Combine all the evaluations using a random linear combination. We
    // simply sum because all evaluations are already multiplied by their
    // random coefficient.
    evals_iter.reduce(|acc, elem| acc + elem).unwrap()
}

/// Similar to [beta_cascade], but does not compute any evaluations and
/// simply multiplies the appropriate beta values by the evaluated bookkeeping table.
pub fn beta_cascade_no_independent_variable<F: Field>(
    mut evaluated_bookkeeping_table: Vec<F>,
    beta_vals: &[F],
    beta_updated_vals: &[F],
    degree: usize,
) -> SumcheckEvals<F> {
    if evaluated_bookkeeping_table.len() > 1 {
        beta_vals.iter().rev().for_each(|beta_val| {
            let (one_minus_beta_val, beta_val) = (F::ONE - beta_val, beta_val);
            evaluated_bookkeeping_table = evaluated_bookkeeping_table
                .chunks(2)
                .map(|bits| bits[0] * one_minus_beta_val + bits[1] * beta_val)
                .collect_vec();
        });
    }

    assert_eq!(evaluated_bookkeeping_table.len(), 1);
    let eval_vec: Vec<F> = repeat_n(evaluated_bookkeeping_table[0], degree + 1).collect();

    apply_updated_beta_values_to_evals(eval_vec, beta_updated_vals)
}

/// Returns the maximum degree of b_{curr_round} within an expression (and
/// therefore the number of prover messages we need to send)
pub(crate) fn get_round_degree<F: Field>(
    expr: &Expression<F, ProverExpr>,
    curr_round: usize,
) -> usize {
    // By default, all rounds have degree at least 2 (beta table included)
    let mut round_degree = 1;

    let mut get_degree_closure = |expr: &ExpressionNode<F, ProverExpr>,
                                  mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec|
     -> Result<()> {
        let round_degree = &mut round_degree;

        // The only exception is within a product of MLEs
        if let ExpressionNode::Product(mle_vec_indices) = expr {
            let mut product_round_degree: usize = 0;
            for mle_vec_index in mle_vec_indices {
                let mle = mle_vec_index.get_mle(mle_vec);

                let mle_indices = mle.mle_indices();
                for mle_index in mle_indices {
                    if *mle_index == MleIndex::Indexed(curr_round) {
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
    // add 1 cuz beta table but idk if we would ever use this without a beta
    // table
    round_degree + 1
}

/// Use degree + 1 evaluations to figure out the evaluation at some arbitrary
/// point
pub fn evaluate_at_a_point<F: Field>(given_evals: &[F], point: F) -> Result<F> {
    // Special case for the constant polynomial.
    if given_evals.len() == 1 {
        return Ok(given_evals[0]);
    }

    debug_assert!(given_evals.len() > 1);

    // Special cases for `point == 0` and `point == 1`.
    if point == F::ZERO {
        return Ok(given_evals[0]);
    }
    if point == F::ONE {
        return Ok(*given_evals.get(1).unwrap_or(&given_evals[0]));
    }

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
                        (F::ONE, F::ONE),
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
    eval.ok_or(anyhow!("Interpretation Error: No Inverse"))
}
