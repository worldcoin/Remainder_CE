//! Contains cryptographic algorithms for going through the sumcheck protocol

use std::{
    iter::repeat,
    ops::{Add, Mul, Neg},
};

#[cfg(test)]
pub(crate) mod tests;

use ark_std::cfg_into_iter;
use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use thiserror::Error;

use crate::{
    expression::{Expression, ExpressionError, ExpressionStandard},
    mle::{beta::BetaTable, dense::DenseMleRef, MleIndex, MleRef},
    FieldExt,
};

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

///A newtype that makes it easier to do arithmatic on sets of evaluations of the summed expression
#[derive(PartialEq, Debug, Clone)]
pub(crate) struct Evals<F: FieldExt>(pub(crate) Vec<F>);

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
pub(crate) fn compute_sumcheck_message<
    F: FieldExt,
    Exp: Expression<F, MleRef = Mle>,
    Mle: MleRef<F = F> + Clone,
>(
    expr: &Exp,
    round_index: usize,
    max_degree: usize,
    beta_table: &BetaTable<F>,
) -> Result<Evals<F>, ExpressionError> {
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
                    let first = b?;
                    let second: Evals<F> = a?;

                    let (Evals(first_evals), Evals(second_evals)) = (first, second);
                    if (first_evals.len() == second_evals.len()) && first_evals.len() == 3 {
                        if max_degree == 2 {
                            // we need to combine the evals by doing (1-x) * first eval + x * second eval
                            let first_evals = Evals(
                                (0..3)
                                    .map(|idx| first_evals[idx] * (F::one() - F::from(idx as u64)))
                                    .collect(),
                            );

                            let second_evals = Evals(
                                (0..3)
                                    .map(|idx| second_evals[idx] * F::from(idx as u64))
                                    .collect(),
                            );

                            Ok(first_evals + second_evals)
                        } else {
                            Err(ExpressionError::EvaluationError(
                                "Expression has a degree > 2 when the round is on a selector bit",
                            ))
                        }
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
            Ok((a * coeff) + (b * coeff_neg))
        }
        _ => Err(ExpressionError::InvalidMleIndex),
    };

    let mle_eval = for<'a, 'b> |mle_ref: &'a Exp::MleRef,
                                beta_mle_ref: &'b DenseMleRef<F>|
                 -> Result<Evals<F>, ExpressionError> {
        let mle_indicies = mle_ref.mle_indices();
        let independent_variable = mle_indicies.contains(&MleIndex::IndexedBit(round_index));
        // --- Just take the "independent variable" thing into account when we're evaluating the MLE reference as a product ---
        evaluate_mle_ref_product(
            &[mle_ref.clone()],
            independent_variable,
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
    let product =
        // for<'a, 'b> |mle_refs: &'a [Exp::MleRef], beta_table: Option<&'b mut BetaTable<F>>| -> Result<PartialSum<F>, ExpressionError> {
        for<'a, 'b> |mle_refs: &'a [Exp::MleRef], beta_mle_ref: &'b DenseMleRef<F>| -> Result<Evals<F>, ExpressionError> {
            let independent_variable = mle_refs
                .iter()
                .map(|mle_ref| {
                    mle_ref
                        .mle_indices()
                        .contains(&MleIndex::IndexedBit(round_index))
                })
                .reduce(|acc, item| acc | item)
                .ok_or(ExpressionError::MleError(MleError::EmptyMleList))?;
            // have to include the beta table and evaluate as a product
            evaluate_mle_ref_product(mle_refs, independent_variable, max_degree, beta_mle_ref.clone())
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

/// Evaluates a product in the form factor V_1(x_1, ..., x_n) * V_2(y_1, ..., y_m) * ...
/// @param mle_refs: The list of MLEs which are being multiplied together
/// @param independent_variable: Whether there is an iterated variable (for this)
///     sumcheck round) which we need to take into account
///
/// # Errors:
/// - MleError::EmptyMleList -- when there are zero MLEs within the list
/// - TODO!(ryancao || vishady): MleError::NotIndexedError -- when ANY MLE is not fully indexed
fn evaluate_mle_ref_product<F: FieldExt>(
    mle_refs: &[impl MleRef<F = F>],
    independent_variable: bool,
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

    // --- Gets the total number of iterated variables across all MLEs within this product ---
    let mut max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    max_num_vars = std::cmp::max(max_num_vars, beta_ref.num_vars());

    if independent_variable {
        //There is an independent variable, and we must extract `degree` evaluations of it, over `0..degree`
        let eval_count = degree + 1;

        //iterate across all pairs of evaluations
        let evals = cfg_into_iter!((0..1 << (max_num_vars - 1))).fold(
            #[cfg(feature = "parallel")]
            || vec![F::zero(); eval_count],
            #[cfg(not(feature = "parallel"))]
            vec![F::zero(); eval_count],
            |mut acc, index| {
                // compute the beta successors the same way it's done for each mle. do it outside the loop
                // because it only needs to be done once per product of mles
                let zero = F::zero();
                let idx = if beta_ref.num_vars() < max_num_vars {
                    let max = 1 << beta_ref.num_vars();
                    (index * 2) % max
                } else {
                    index * 2
                };
                let first = *beta_ref.bookkeeping_table().get(idx).unwrap_or(&zero);
                let second = if beta_ref.num_vars() != 0 {
                    *beta_ref.bookkeeping_table().get(idx + 1).unwrap_or(&zero)
                } else {
                    first
                };
                let step = second - first;

                let beta_successors_snd =
                    std::iter::successors(Some(second), move |item| Some(*item + step));
                //iterator that represents all evaluations of the MLE extended to arbitrarily many linear extrapolations on the line of 0/1
                let beta_successors = std::iter::once(first).chain(beta_successors_snd);
                let beta_iter: Box<dyn Iterator<Item = F>> = Box::new(beta_successors);

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
                    // chain the beta successors
                    .chain(std::iter::once(beta_iter))
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

        //     dbg!(&evals, real_num_vars);
        Ok(Evals(evals))
    } else {
        // beta table still has an independent variable so we have a line
        let eval_count = 2;

        let partials = cfg_into_iter!((0..1 << (max_num_vars))).fold(
            #[cfg(feature = "parallel")]
            || vec![F::zero(); eval_count],
            #[cfg(not(feature = "parallel"))]
            vec![F::zero(); eval_count],
            |mut acc, index| {
                let beta_idx_0 = if beta_ref.num_vars() < max_num_vars {
                    let max = 1 << beta_ref.num_vars();
                    (index * 2) % max
                } else {
                    index * 2
                };

                let beta_idx_1 = beta_idx_0 + 1;

                // get the index of the beta table at the binary string (0, ...) and (1, ...) by doing 2*index and 2*index + 1
                let beta_at_0 = beta_ref
                    .bookkeeping_table()
                    .get(beta_idx_0)
                    .cloned()
                    .unwrap_or(F::zero());
                let beta_at_1 = beta_ref
                    .bookkeeping_table()
                    .get(beta_idx_1)
                    .cloned()
                    .unwrap_or(F::zero());

                // Go through each MLE within the product
                let product = mle_refs
                    .iter()
                    // Result of this `map()`: A list of evaluations of the MLEs at `index`
                    .map(|mle_ref| {
                        let index = if mle_ref.num_vars() < max_num_vars {
                            // max = 2^{num_vars}; index := index % 2^{num_vars}
                            let max = 1 << mle_ref.num_vars();
                            index % max
                        } else {
                            index
                        };
                        // --- Access the MLE at that index. Pad with zeros ---
                        mle_ref
                            .bookkeeping_table()
                            .get(index)
                            .cloned()
                            .unwrap_or(F::zero())
                    })
                    .reduce(|acc, eval| acc * eval)
                    .unwrap();

                let beta_evals = [beta_at_0, beta_at_1];
                // multiply the beta evals by the product of the resulting mles
                acc.iter_mut()
                    .zip(beta_evals.iter())
                    .for_each(|(acc, eval)| *acc += product * eval);
                acc
            },
        );

        #[cfg(feature = "parallel")]
        let partials = partials.reduce(
            || vec![F::zero(); eval_count],
            |mut acc, partial| {
                acc.iter_mut()
                    .zip(partial)
                    .for_each(|(acc, partial)| *acc += partial);
                acc
            },
        );

        let evals = vec![
            partials[0],
            partials[1],
            F::from(2_u64) * partials[1] - partials[0],
        ];

        Ok(Evals(evals))
    }
}

/// Returns the maximum degree of b_{curr_round} within an expression
/// (and therefore the number of prover messages we need to send)
pub(crate) fn get_round_degree<F: FieldExt>(
    expr: &ExpressionStandard<F>,
    curr_round: usize,
) -> usize {
    // --- By default, all rounds have degree at least 2 (beta table included) ---
    let mut round_degree = 1;

    let mut traverse = for<'a> |expr: &'a ExpressionStandard<F>| -> Result<(), ()> {
        let round_degree = &mut round_degree;

        // --- The only exception is within a product of MLEs ---
        if let ExpressionStandard::Product(mle_refs) = expr {
            let mut product_round_degree: usize = 0;
            for mle_ref in mle_refs {
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

    expr.traverse(&mut traverse).unwrap();
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
                        vec![F::one(), F::one()],
                        |acc, val| vec![acc[0] * (point - val), acc[1] * (F::from(x as u64) - val)],
                    )
            },
        )
        .enumerate()
        .map(
            // Add up barycentric weight * current eval at point
            |(x, y)| given_evals[x] * y[0] * y[1].inverse().unwrap(),
        )
        .reduce(|x, y| x + y);
    eval.ok_or(InterpError::NoInverse)
}
