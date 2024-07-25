//! Contains cryptographic algorithms for going through the sumcheck protocol

use std::{
    iter::repeat,
    ops::{Add, Mul, Neg},
};

/// tests for sumcheck with various expressions
#[cfg(test)]
pub mod tests;

use ark_std::cfg_into_iter;
use itertools::{repeat_n, Itertools};
use rayon::prelude::ParallelSlice;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use thiserror::Error;

use crate::{
    expression::{
        expr_errors::ExpressionError,
        generic_expr::{Expression, ExpressionNode, ExpressionType},
        prover_expr::ProverExpr,
    },
    mle::{betavalues::BetaValues, dense::DenseMle, Mle, MleIndex},
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
    let constant = |constant, beta_table: &BetaValues<F>| {
        let constant_updated_vals = beta_values.updated_values.values().copied().collect_vec();
        let index_claim = beta_table.unbound_values.get(&round_index).unwrap();
        let one_minus_index_claim = F::ONE - index_claim;
        let beta_step = *index_claim - one_minus_index_claim;
        let evals = std::iter::successors(Some(one_minus_index_claim), move |item| {
            Some(*item + beta_step)
        })
        .take(max_degree + 1)
        .map(|elem| constant * elem)
        .collect_vec();
        let updated_evals = apply_updated_beta_values_to_evals(evals, &constant_updated_vals);
        Ok(updated_evals)
    };

    // the selector is split into three cases:
    // - when the selector bit itself is not the independent variable and hasn't been bound yet,
    // - when the selector bit is the independent variable
    // - when the selector bit has already been bound
    // we determine which case we are in by comparing the round_index to the selector index which is an
    // argument to the closure.
    let selector = |index: &MleIndex<F>, a, b, beta_table: &BetaValues<F>| match index {
        MleIndex::IndexedBit(indexed_bit) => {
            // because the selector bit itself only has one variable (1 - b_i) * (a) + b_i * b
            // we only need one value within the beta table in order to evaluate the selector
            // at this point.
            let index_claim = beta_table.unbound_values.get(indexed_bit).unwrap();
            match Ord::cmp(&round_index, indexed_bit) {
                std::cmp::Ordering::Less => {
                    // when the selector bit is not the independent variable and has not been bound yet, we
                    // are simply summing over everything. in order to take the beta values into account
                    // this means for everything on the "left" side of the selector we want to multiply
                    // by (1 - g_i) and for everything on the "right" side of the selector we want to
                    // multiply by g_i. we can then add these!
                    let a_with_sel = a? * (F::ONE - index_claim);
                    let b_with_sel = b? * index_claim;
                    Ok(a_with_sel + b_with_sel)
                }
                std::cmp::Ordering::Equal => {
                    // this is when the selector index is the independent variable! this means the beta
                    // value at this index also has an independent variable.
                    let first = a?;
                    let second: Evals<F> = b?;

                    let (Evals(first_evals), Evals(second_evals)) = (first, second);
                    if first_evals.len() == second_evals.len() {
                        // therefore we compute the successors of the beta values as well, as the successors
                        // correspond to evaluations at the points 0, 1, ... for the independent variable.
                        let eval_len = first_evals.len();
                        let one_minus_index_claim = F::ONE - index_claim;
                        let beta_step = *index_claim - one_minus_index_claim;
                        let beta_evals =
                            std::iter::successors(Some(one_minus_index_claim), move |item| {
                                Some(*item + beta_step)
                            })
                            .take(eval_len)
                            .collect_vec();

                        // the selector index also has an independent variable so we factor this as well
                        // as the corresponding beta successor at this index.
                        let first_evals = Evals(
                            first_evals
                                .into_iter()
                                .enumerate()
                                .map(|(idx, first_eval)| {
                                    first_eval * (F::ONE - F::from(idx as u64)) * beta_evals[idx]
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
                        Err(ExpressionError::EvaluationError("Expression returns two evals that do not have the same length on a selector bit"))
                    }
                }
                // we cannot have an indexed bit for the selector bit that is less than the current
                // sumcheck round. therefore this is an error
                std::cmp::Ordering::Greater => Err(ExpressionError::InvalidMleIndex),
            }
        }
        // if the selector bit has already been bound, that means the beta value at this index has also
        // already been bound, if it exists! otherwise we just treat it as the identity
        MleIndex::Bound(coeff, bound_round_idx) => {
            let coeff_neg = F::ONE - coeff;
            let one = &F::ONE;
            let beta_bound_val = beta_table
                .updated_values
                .get(bound_round_idx)
                .unwrap_or(one);
            let a: Evals<F> = a?;
            let b: Evals<F> = b?;

            // the evaluation is just scaled by the beta bound value
            Ok(((b * coeff) + (a * coeff_neg)) * *beta_bound_val)
        }
        _ => Err(ExpressionError::InvalidMleIndex),
    };

    // the mle evaluation takes in the mle ref, and the corresponding unbound and bound beta values
    // to pass into the `beta_cascade` function
    let mle_eval = |mle_ref: &DenseMle<F>,
                    unbound_beta_vals: &[F],
                    bound_beta_vals: &[F]|
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

    // when we have a sum, we can evaluate both parts of the expression separately and just add the evaluations
    let sum = |a, b| {
        let a: Evals<F> = a?;
        let b: Evals<F> = b?;
        Ok(a + b)
    };

    // when we have a product, the node can only contain mle refs. therefore this is similar to the mle
    // evaluation, but instead we have a list of mle refs, and the corresponding unbound and bound
    // beta values for that node.
    let product = |mle_refs: &[&DenseMle<F>],
                   unbound_beta_vals: &[F],
                   bound_beta_vals: &[F]|
     -> Result<Evals<F>, ExpressionError> {
        Ok(beta_cascade(
            mle_refs,
            max_degree,
            round_index,
            unbound_beta_vals,
            bound_beta_vals,
        ))
    };

    // when the expression is scaled by a field element, we can scale the evaluations by this element as well
    let scaled = |a, scalar| {
        let a = a?;
        Ok(a * scalar)
    };

    // pass these closures into the `evaluate_sumcheck_beta_cascade` function!
    expr.evaluate_sumcheck_beta_cascade(
        &constant,
        &selector,
        &mle_eval,
        &negated,
        &sum,
        &product,
        &scaled,
        beta_values,
    )
}

/// this function will take a list of mle refs, and compute the element-wise product of all of their
/// bookkeeping tables along with the "successors."
///
/// for example, if we have two bookkeeping tables [a_1, a_2, a_3, a_4] and [c_1, c_2, c_3, c_4] and
/// the degree of our expression at this index is 3, we need 4 evaluations for a unique curve.
/// therefore first we will compute
/// [a_1, a_2, (1-2)a_1 + 2a_2, (1-3)a_1 + 3a_2, a_3, a_4, (1-2)a_3 + 2a_4, (1-3)a_3 + 3a_4]
/// and the same thing for the other mle and element-wise multiply both results.
/// the resulting vector will always be size (degree + 1) * (2 ^ (max_num_vars - 1))
///
/// this function assumes that the first variable is an independent variable.
pub fn successors_from_mle_ref_product<F: FieldExt>(
    mle_refs: &[&impl Mle<F>],
    degree: usize,
) -> Result<Vec<F>, MleError> {
    // --- Gets the total number of iterated variables across all MLEs within this product ---
    let max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_iterated_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    // because there is an independent variable, we need degree + 1 evaluations in order to determine
    // a unique curve over the evaluations.
    let eval_count = degree + 1;

    let evals = cfg_into_iter!((0..eval_count * (1 << (max_num_vars - 1))))
        .map(|index| {
            mle_refs
                .iter()
                .map(|mle_ref| {
                    let zero = F::ZERO;

                    // The relevant index into the mle bookkeeping table.
                    let mle_index = index / eval_count;
                    // We're computing `eval_count` evaluations of the MLE product.
                    // This is the `eval_index`-th one.
                    let eval_index = index % eval_count;

                    // over here, we perform the wrap-around functionality if we are multiplying
                    // two mle_refs with different number of variables.
                    // for example if we are multiplying V(b_1, b_2) * V(b_1), and summing over
                    // b_2, then the overall sum is V(b_1, 0) * V(b_1) + V(b_1, 1) * V(b_1).
                    // it can be seen that the "smaller" mle_ref (the one over less variables) has
                    // to repeat itself an according number of times when the sum is over a variable
                    // it does not contain. the appropriate index is therefore
                    // determined as follows.
                    let mle_index = if mle_ref.num_iterated_vars() < max_num_vars {
                        // if we have less than the max number of variables, then we perform this wrap-around
                        // functionality by first rounding to the nearest power of 2, and then taking the mod
                        // of this index as we implicitly pad for powers of 2.
                        let max = 1 << mle_ref.num_iterated_vars();
                        (mle_index * 2) % max
                    } else {
                        mle_index * 2
                    };
                    // over here, we get the elements in the pair so when index = 0, it's [0] and [1], if index = 1,
                    // it's [2] and [3], etc. because we are extending a function that was originally defined
                    // over the hypercube, each pair corresponds to two points on a line. we grab these two points here
                    let first = *mle_ref.bookkeeping_table().get(mle_index).unwrap_or(&zero);
                    let second = if mle_ref.num_iterated_vars() != 0 {
                        *mle_ref
                            .bookkeeping_table()
                            .get(mle_index + 1)
                            .unwrap_or(&zero)
                    } else {
                        first
                    };
                    let step = second - first;

                    // and then we use the difference between the points in order to
                    // generate the `eval_index`-th successor.
                    first + step * F::from(eval_index as u64)
                })
                .fold(F::ONE, |acc, eval: F| acc * eval)
        })
        .collect();

    Ok(evals)
}

/// this function performs the same funcionality as the above, except it is when the mle refs we
/// are working with have no independent variable. therefore, we are actually just taking the
/// sum over all of the variables and do not need evaluations.
pub(crate) fn successors_from_mle_ref_product_no_ind_var<F: FieldExt>(
    mle_refs: &[&impl Mle<F>],
) -> Result<Vec<F>, MleError> {
    let max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_iterated_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    let evals_vec = if max_num_vars > 0 {
        let evals = cfg_into_iter!((0..1 << (max_num_vars))).map(|index| {
            // we take the element-wise product at each of the points instead of looking at successors.
            let successors_product = mle_refs
                .iter()
                .map(|mle_ref| {
                    let zero = F::ZERO;
                    let index = if mle_ref.num_iterated_vars() < max_num_vars {
                        let max = 1 << mle_ref.num_iterated_vars();
                        (index) % max
                    } else {
                        index
                    };
                    *mle_ref.bookkeeping_table().get(index).unwrap_or(&zero)
                })
                .reduce(|acc, eval| acc * eval)
                .unwrap();
            successors_product
        });
        evals.collect()
    } else {
        let val = mle_refs.iter().fold(F::ONE, |acc, mle_ref| {
            assert_eq!(mle_ref.bookkeeping_table().len(), 1);
            acc * mle_ref.bookkeeping_table()[0]
        });
        vec![val]
    };

    Ok(evals_vec)
}

/// this is one step of the beta cascade algorithm. essentially we are doing
/// (1 - beta_val) * mle[index] + beta_val * mle[index + half_vec_len] (big-endian version of fix variable)
pub(crate) fn beta_cascade_step<F: FieldExt>(mle_successor_vec: &mut [F], beta_val: F) -> Vec<F> {
    let (one_minus_beta_val, beta_val) = (F::ONE - beta_val, beta_val);
    let half_vec_len = mle_successor_vec.len() / 2;
    let new_successor = cfg_into_iter!((0..half_vec_len)).map(|idx| {
        (mle_successor_vec[idx] * one_minus_beta_val)
            + (mle_successor_vec[idx + half_vec_len] * beta_val)
    });

    new_successor.collect()
}

/// this is the final step of beta cascade, where we take all the "bound" beta values
/// and scale all of the evaluations by the product of all of these values
fn apply_updated_beta_values_to_evals<F: FieldExt>(
    evals: Vec<F>,
    beta_updated_vals: &[F],
) -> Evals<F> {
    let beta_total_updated_product = beta_updated_vals
        .iter()
        .fold(F::ONE, |acc, elem| acc * elem);
    let evals = evals
        .iter()
        .map(|elem| beta_total_updated_product * elem)
        .collect_vec();

    Evals(evals)
}

/// if there is no independent variable, beta cascade can be done the little-endian fix variable way
/// where we go from the most significant --> least significant bit to compute the single evaluation.
///
/// regardless of the degree, since there is no independent variable this evaluates to a constant,
/// so we just repeat this (degree + 1) times.
fn beta_cascade_no_independent_variable<F: FieldExt>(
    mle_refs: &[&impl Mle<F>],
    beta_vals: &[F],
    degree: usize,
    beta_updated_vals: &[F],
) -> Evals<F> {
    let mut mle_successor_vec = successors_from_mle_ref_product_no_ind_var(mle_refs).unwrap();
    if mle_successor_vec.len() > 1 {
        beta_vals.iter().for_each(|beta_val| {
            let (one_minus_beta_val, beta_val) = (F::ONE - beta_val, beta_val);
            let mle_successor_vec_iter = mle_successor_vec
                .par_chunks(2)
                .map(|bits| bits[0] * one_minus_beta_val + bits[1] * beta_val);
            mle_successor_vec = mle_successor_vec_iter.collect();
        });
    }

    assert_eq!(mle_successor_vec.len(), 1);
    let eval_vec: Vec<F> = repeat_n(mle_successor_vec[0], degree + 1).collect();

    apply_updated_beta_values_to_evals(eval_vec, beta_updated_vals)
}

/// this is how we compute the evaluations of a product of mle refs along with a beta table.
/// rather than using the full expanded version of a beta table, we instead just use the beta values
/// vectors (which are unbound beta values, and the bound beta values)
/// there are (degree + 1) evaluations that are returned which are the evaluations of the univariate
/// polynomial where the "round_index"-th bit is the independent variable.
pub fn beta_cascade<F: FieldExt>(
    mle_refs: &[&impl Mle<F>],
    degree: usize,
    round_index: usize,
    beta_vals: &[F],
    beta_updated_vals: &[F],
) -> Evals<F> {
    // determine whether there is an independent variable within these mle refs by iterating through
    // all of their indices and determining whether there is an indexed bit at the round index.
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
        // we go from the LSB --> MSB for the beta values, and do the big-endian fix variable for each one,
        // reducing the size of `mle_successor_vec` by half.
        beta_vals.iter().skip(1).rev().for_each(|val| {
            mle_successor_vec = beta_cascade_step(&mut mle_successor_vec, *val);
        });
        // for the MSB of the beta value, this must be the independent variable. otherwise it would already be
        // bound. therefore we need to compute the successors of this value in order to get its evaluations.
        let evals = if !beta_vals.is_empty() {
            let second_beta_successor = beta_vals[0];
            let first_beta_successor = F::ONE - second_beta_successor;
            let step = second_beta_successor - first_beta_successor;
            let beta_successors =
                std::iter::successors(Some(first_beta_successor), move |item| Some(*item + step));
            // the length of the mle successor vec before this last step must be degree + 1. therefore we can
            // just do a zip with the beta successors to get the final degree + 1 evaluations.
            beta_successors
                .zip(mle_successor_vec)
                .map(|(beta_succ, mle_succ)| beta_succ * mle_succ)
                .collect_vec()
        } else {
            vec![F::ONE]
        };
        // apply the bound beta values as a scalar factor to each of the evaluations
        apply_updated_beta_values_to_evals(evals, beta_updated_vals)
    } else {
        beta_cascade_no_independent_variable(mle_refs, beta_vals, degree, beta_updated_vals)
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

    let mut get_degree_closure = |expr: &ExpressionNode<F, ProverExpr>,
                                  mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec|
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
pub fn evaluate_at_a_point<F: FieldExt>(given_evals: &[F], point: F) -> Result<F, InterpError> {
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
    eval.ok_or(InterpError::NoInverse)
}

/// evaluates the product of multiple mle refs (in the evalutaion form),
/// (the mles could be beta tables as well)
/// the returned results can be the following expresssion:
/// sum_{x_2, ..., x_n} V_1(X, x_2, ..., x_n) * V_2(X, x_2, ..., x_n) * V_2(X, x_2, x_3),
/// evaluated at X = 0, 1, ..., degree
/// note that when one of the mle_refs have less variables, there's a wrap around: % max
pub fn evaluate_mle_ref_product<F: FieldExt>(
    mle_refs: &[&impl Mle<F>],
    degree: usize,
) -> Result<Evals<F>, MleError> {
    // --- Gets the total number of iterated variables across all MLEs within this product ---
    let max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_iterated_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    //There is an independent variable, and we must extract `degree` evaluations of it, over `0..degree`
    let eval_count = degree + 1;

    //iterate across all pairs of evaluations
    let evals = cfg_into_iter!((0..1 << (max_num_vars - 1))).fold(
        #[cfg(feature = "parallel")]
        || vec![F::ZERO; eval_count],
        #[cfg(not(feature = "parallel"))]
        vec![F::ZERO; eval_count],
        |mut acc, index| {
            //get the product of all evaluations over 0/1/..degree
            let evals = mle_refs
                .iter()
                .map(|mle_ref| {
                    let zero = F::ZERO;
                    let index = if mle_ref.num_iterated_vars() < max_num_vars {
                        let max = 1 << mle_ref.num_iterated_vars();
                        (index * 2) % max
                    } else {
                        index * 2
                    };
                    let first = *mle_ref.bookkeeping_table().get(index).unwrap_or(&zero);

                    let second = if mle_ref.num_iterated_vars() != 0 {
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
        || vec![F::ZERO; eval_count],
        |mut acc, partial| {
            acc.iter_mut()
                .zip(partial)
                .for_each(|(acc, partial)| *acc += partial);
            acc
        },
    );

    Ok(Evals(evals))
}
