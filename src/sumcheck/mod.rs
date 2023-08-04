//! Contains cryptographic algorithms for going through the sumcheck protocol

use std::{
    cmp::Ordering,
    iter::repeat,
    ops::{Add, Mul, Neg, Sub},
};

use ark_poly::MultilinearExtension;
use ark_std::{cfg_into_iter, cfg_iter, rand::Rng};
use itertools::Itertools;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use thiserror::Error;

use crate::{
    expression::{Expression, ExpressionError, ExpressionStandard},
    layer::Claim,
    mle::{
        beta::{BetaError, BetaTable, evaluate_beta},
        dense::{DenseMle, DenseMleRef},
        MleIndex, MleRef,
    },
    FieldExt,
};

#[derive(Error, Debug, Clone)]
enum MleError {
    #[error("Passed list of Mles is empty")]
    EmptyMleList,
    #[error("Beta table not yet initialized for Mle")]
    NoBetaTable,
    #[error("Layer does not have claims yet")]
    NoClaim,
    #[error("Unable to eval beta")]
    BetaEvalError,
    #[error("Cannot compute sumcheck message on un-indexed MLE")]
    NotIndexedError,
}

#[derive(Error, Debug, Clone)]
pub enum VerifyError {
    #[error("Failed sumcheck round")]
    SumcheckBad,
}
#[derive(Error, Debug, Clone)]
pub enum InterpError {
    #[error("Too few evaluation points")]
    EvalLessThanDegree,
    #[error("No possible polynomial")]
    NoInverse,
}

/// Apparently this is for "going up the tree" -- Ryan
/// I guess the idea is that either we're taking a summation over
/// the single value stored in Sum(F), or that we have a bunch of
/// eval points (i.e. stored in Evals(Vec<F>)) giving us e.g.
/// g(0), g(1), g(2), ...
#[derive(PartialEq, Debug, Clone)]
pub(crate) enum SumOrEvals<F: FieldExt> {
    Sum(F),
    Evals(Vec<F>),
}

#[derive(Debug, Clone)]
pub(crate) struct PartialSum<F: FieldExt> {
    sum_or_eval: SumOrEvals<F>,
    max_num_vars: usize,
}

impl<F: FieldExt> Neg for SumOrEvals<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        match self {
            // --- Negation for a constant is just its negation ---
            SumOrEvals::Sum(sum) => SumOrEvals::Sum(sum.neg()),
            // --- Negation for a bunch of eval points is just element-wise negation ---
            SumOrEvals::Evals(evals) => {
                SumOrEvals::Evals(evals.into_iter().map(|eval| eval.neg()).collect_vec())
            }
        }
    }
}

impl<F: FieldExt> Add for SumOrEvals<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match self {
            SumOrEvals::Sum(sum) => match rhs {
                // --- Sum(F) + Sum(F) --> Just add them ---
                SumOrEvals::Sum(rhs) => SumOrEvals::Sum(sum + rhs),
                // --- Sum(F) + Evals(Vec<F>) --> Distributed addition ---
                SumOrEvals::Evals(rhs) => SumOrEvals::Evals(
                    repeat(sum)
                        .zip(rhs)
                        .map(|(lhs, rhs)| lhs + rhs)
                        .collect_vec(),
                ),
            },
            SumOrEvals::Evals(evals) => match rhs {
                // --- Evals(Vec<F>) + Sum(F) --> Distributed addition ---
                SumOrEvals::Sum(rhs) => SumOrEvals::Evals(
                    evals
                        .into_iter()
                        .zip(repeat(rhs))
                        .map(|(lhs, rhs)| lhs + rhs)
                        .collect_vec(),
                ),
                // --- Evals(Vec<F>) + Evals(Vec<F>) --> Zipped addition ---
                SumOrEvals::Evals(rhs) => SumOrEvals::Evals(
                    evals
                        .into_iter()
                        .zip(rhs)
                        .map(|(lhs, rhs)| lhs + rhs)
                        .collect_vec(),
                ),
            },
        }
    }
}

impl<F: FieldExt> Mul<F> for SumOrEvals<F> {
    type Output = Self;
    fn mul(self, rhs: F) -> Self {
        match self {
            SumOrEvals::Sum(sum) => SumOrEvals::Sum(sum * rhs),
            SumOrEvals::Evals(evals) => SumOrEvals::Evals(
                evals
                    .into_iter()
                    .zip(repeat(rhs))
                    .map(|(lhs, rhs)| lhs * rhs)
                    .collect_vec(),
            ),
        }
    }
}

impl<F: FieldExt> Mul<&F> for SumOrEvals<F> {
    type Output = Self;
    fn mul(self, rhs: &F) -> Self {
        match self {
            SumOrEvals::Sum(sum) => SumOrEvals::Sum(sum * rhs),
            SumOrEvals::Evals(evals) => SumOrEvals::Evals(
                evals
                    .into_iter()
                    .zip(repeat(rhs))
                    .map(|(lhs, rhs)| lhs * rhs)
                    .collect_vec(),
            ),
        }
    }
}

impl<F: FieldExt> Neg for PartialSum<F> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        self.sum_or_eval = self.sum_or_eval.neg();
        self
    }
}

impl<F: FieldExt> Add for PartialSum<F> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        let (larger, smaller) = {
            match Ord::cmp(&self.max_num_vars, &rhs.max_num_vars) {
                Ordering::Less => (rhs, self),
                Ordering::Equal => {
                    self.sum_or_eval = self.sum_or_eval + rhs.sum_or_eval;
                    return self;
                }
                Ordering::Greater => (self, rhs),
            }
        };

        //let diff = larger.max_num_vars - smaller.max_num_vars;
        let diff = 0;

        //this is probably more efficient than F::pow for small exponents
        let mult_factor = (0..diff).fold(F::one(), |acc, _| acc * F::from(2_u64));
        let smaller = smaller * mult_factor;

        PartialSum {
            sum_or_eval: larger.sum_or_eval + smaller.sum_or_eval,
            max_num_vars: larger.max_num_vars,
        }
    }
}

impl<F: FieldExt> Mul<F> for PartialSum<F> {
    type Output = Self;
    fn mul(mut self, rhs: F) -> Self {
        self.sum_or_eval = self.sum_or_eval * rhs;
        self
    }
}

impl<F: FieldExt> Mul<&F> for PartialSum<F> {
    type Output = Self;
    fn mul(mut self, rhs: &F) -> Self {
        self.sum_or_eval = self.sum_or_eval * rhs;
        self
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
pub(crate) fn compute_sumcheck_message<F: FieldExt, Exp: Expression<F, MleRef = Mle>, Mle: MleRef<F = F> + Clone>(
    expr: &mut Exp,
    round_index: usize,
    max_degree: usize,
    beta_table: &BetaTable<F>,
) -> Result<SumOrEvals<F>, ExpressionError> {

    // --- Constant evaluation is just Sum(k) ---
    let constant = |constant, beta_mle_ref: &DenseMleRef<F>| {

        // need to actually treat this like a 'scaled' because there is a beta table
        let beta_bt = beta_mle_ref.bookkeeping_table();
        let independent_variable = beta_mle_ref.mle_indices().contains(&MleIndex::IndexedBit(round_index));
        // just scale the beta table by the constant
        if independent_variable {
            let first = beta_bt.iter().step_by(2).fold(F::zero(), |elem, acc| elem + acc);
            let second = beta_bt.iter().skip(1).step_by(2).fold(F::zero(), |elem, acc| elem + acc);
            let evals = (1..max_degree+1).map(
                |index| first + (second - first) * F::from(index as u64)
            );
            let beta_eval = PartialSum { sum_or_eval: SumOrEvals::Evals(std::iter::once(first).chain(evals).collect_vec()), max_num_vars: beta_mle_ref.num_vars() };
            Ok(beta_eval * constant)
        }
        else {
            let eval = beta_bt.iter().fold(F::zero(), |elem, acc| elem + acc);
            let beta_eval = PartialSum { sum_or_eval: SumOrEvals::Sum(eval), max_num_vars: 0 };
            Ok(beta_eval * constant)
        }
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
                    let second = a?;

                    if let (
                        PartialSum {
                            sum_or_eval: SumOrEvals::Evals(first_evals),
                            max_num_vars: first_num_vars,
                        },
                        PartialSum {
                            sum_or_eval: SumOrEvals::Evals(second_evals),
                            max_num_vars: second_num_vars,
                        },
                    ) = (first, second)
                    {
                        if (first_evals.len() == second_evals.len()) && first_evals.len() == 3 {
                            if max_degree == 2 {

                                // we need to combine the evals by doing (1-x) * first eval + x * second eval
                                let first_evals = SumOrEvals::Evals(
                                    (0..3).map(
                                    |idx| first_evals[idx] * (F::one() - F::from(idx as u64))
                                ).collect());

                                let second_evals = SumOrEvals::Evals(
                                    (0..3).map(
                                    |idx| second_evals[idx] * F::from(idx as u64)
                                ).collect());

                                Ok(PartialSum {
                                     sum_or_eval: first_evals + second_evals,
                                     // max num vars should not matter anymore
                                     max_num_vars: 0,
                                 })

                            } else {
                                Err(ExpressionError::EvaluationError(
                                    "Expression has a degree > 2 when the round is on a selector bit",
                                ))
                            }
                        }

                        else {
                            Err(ExpressionError::EvaluationError("Expression returns two evals that do not have length 3 on a selector bit"))
                        }
                        
                    } else {

                        // if let SumOrEvals::Evals(first) = first {
                        //     first_sum = SumOrEvals::Sum(first.iter().reduce(|item, acc| item + acc))
                        // }
                        Err(ExpressionError::EvaluationError("Expression returns a Sum variant when the round is on a selector bit"))
                    }
                }
                // --- If we're past the evaluation round, we should not have an unbound selector ---
                std::cmp::Ordering::Greater => Err(ExpressionError::InvalidMleIndex),
            }
        }
        MleIndex::Bound(coeff) => {
            let coeff_neg = F::one() - coeff;
            let a: PartialSum<F> = a?;
            let b: PartialSum<F> = b?;

            // --- Just r * V[2i + 1] + (1 - r) * V[2i] ---
            // --- (I.e. the selector formulation after the selector bit is bound to `r` above) ---
            Ok((a * coeff) + (b * coeff_neg))
        }
        _ => Err(ExpressionError::InvalidMleIndex),
    };

    let mle_eval = 
        // for<'a, 'b> |mle_ref: &'a Exp::MleRef, beta_table: Option<&'b mut BetaTable<F>>| -> Result<PartialSum<F>, ExpressionError> {
        for<'a, 'b> |mle_ref: &'a Exp::MleRef, beta_mle_ref: &'b DenseMleRef<F>| -> Result<PartialSum<F>, ExpressionError> {
        //let beta_mle_ref = beta_table.table;
        let mle_indicies = mle_ref.mle_indices();
        let independent_variable = mle_indicies.contains(&MleIndex::IndexedBit(round_index));
        // include beta table
        // let betatable = beta_table.as_ref().ok_or(ExpressionError::BetaError)?;
        // dbg!(betatable);
        // --- Just take the "independent variable" thing into account when we're evaluating the MLE reference as a product ---
        evaluate_mle_ref_product(&[mle_ref.clone()], independent_variable, max_degree, beta_mle_ref.clone(), false)
            .map_err(|_| ExpressionError::MleError)
    };

    // --- Just invert ---
    let negated = |a: Result<_, _>| a.map(|a: PartialSum<F>| a.neg());

    // --- Use the distributed/element-wise addition impl from earlier ---
    let sum = |a, b| {
        let a: PartialSum<F> = a?;
        let b: PartialSum<F> = b?;

        Ok(a + b)
    };

    // --- First see whether there are any iterated variables we should go over ---
    // --- Then just call the `evaluate_mle_ref_product` function ---
    let product =
        // for<'a, 'b> |mle_refs: &'a [Exp::MleRef], beta_table: Option<&'b mut BetaTable<F>>| -> Result<PartialSum<F>, ExpressionError> {
        for<'a, 'b> |mle_refs: &'a [Exp::MleRef], beta_mle_ref: &'b DenseMleRef<F>| -> Result<PartialSum<F>, ExpressionError> {
            let independent_variable = mle_refs
                .iter()
                .map(|mle_ref| {
                    mle_ref
                        .mle_indices()
                        .contains(&MleIndex::IndexedBit(round_index))
                })
                .reduce(|acc, item| acc | item)
                .ok_or(ExpressionError::MleError)?;
            // have to include the beta table and evaluate as a product
            evaluate_mle_ref_product(mle_refs, independent_variable, max_degree, beta_mle_ref.clone(), false)
                .map_err(|_| ExpressionError::MleError)
        };

    // --- Scalar is just distributed mult as defined earlier ---
    let scaled = |a, scalar| {
        let a = a?;
        Ok(a * scalar)
    };

    Ok(expr
        .evaluate_sumcheck(
            &constant, &selector, &mle_eval, &negated, &sum, &product, &scaled, &beta_table.table, round_index
        )?
        .sum_or_eval)
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
    beta_split: bool,
) -> Result<PartialSum<F>, MleError> {

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
    
    // let real_num_vars = if independent_variable {
    //     max_num_vars 
    // } else {
    //     max_num_vars
    // };

    let real_num_vars = if max_num_vars != 0 {
        max_num_vars - 1
    } else {
        max_num_vars
    };

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
                    .for_each(|(acc, eval)| {
                        *acc += eval
                });
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
        Ok(PartialSum {
            sum_or_eval: SumOrEvals::Evals(evals),
            max_num_vars: real_num_vars,
        })
    } 
    
    
    else {

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
                let beta_at_0 = beta_ref.bookkeeping_table().get(beta_idx_0).cloned().unwrap_or(F::zero());
                let beta_at_1 = beta_ref.bookkeeping_table().get(beta_idx_1).cloned().unwrap_or(F::zero());



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


                let beta_evals = vec![beta_at_0, beta_at_1,];
                // multiply the beta evals by the product of the resulting mles
                acc.iter_mut()
                    .zip(beta_evals.iter())
                    .for_each(|(acc, eval)| {
                        *acc += product*eval
                });
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

        let evals = vec![partials[0], partials[1], F::from(2_u64)*partials[1] - partials[0]];

        Ok(PartialSum {
            sum_or_eval: SumOrEvals::Evals(evals),
            max_num_vars: real_num_vars,
        })
    }
}


/// Returns the maximum degree of b_{curr_round} within an expression
/// (and therefore the number of prover messages we need to send)
pub fn get_round_degree<F: FieldExt>(expr: &ExpressionStandard<F>, curr_round: usize) -> usize {
    // --- By default, all rounds have degree at least 2 (beta table included) ---
    let mut round_degree = 1;

    let mut traverse = for<'a> |expr: &'a ExpressionStandard<F>| -> Result<(), ()> {
        let round_degree = &mut round_degree;
        match expr {
            // --- The only exception is within a product of MLEs ---
            ExpressionStandard::Product(mle_refs) => {
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
            _ => {}
        }
        Ok(())
    };

    expr.traverse(&mut traverse).unwrap();
    // add 1 cuz beta table but idk if we would ever use this without a beta table
    round_degree + 1
}

/// Does a dummy version of sumcheck with a testing RNG
pub fn dummy_sumcheck<F: FieldExt>(
    expr: &mut ExpressionStandard<F>,
    rng: &mut impl Rng,
    layer_claim: Claim<F>,
) -> Vec<(Vec<F>, Option<F>)> {
    let mut beta_table = BetaTable::new(layer_claim).unwrap();

    // --- Does the bit indexing ---
    let max_round = std::cmp::max(expr.index_mle_indices(0), beta_table.table.index_mle_indices(0));

    // --- The prover messages to the verifier...? ---
    let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
    let mut challenge: Option<F> = None;

    for round_index in 0..max_round {
        // --- First fix the variable representing the challenge from the last round ---
        // (This doesn't happen for the first round)
        if let Some(challenge) = challenge {
            expr.fix_variable(round_index - 1, challenge);
            beta_table.beta_update(round_index - 1, challenge).unwrap();
        }

        // --- Grabs the degree of univariate polynomial we are sending over ---
        let degree = get_round_degree(&expr, round_index);

        // --- Gives back the evaluations g(0), g(1), ..., g(d - 1) ---
        let eval = compute_sumcheck_message(expr, round_index, degree, &mut beta_table);

        if let Ok(SumOrEvals::Evals(evaluations)) = eval {
            // dbg!(&evaluations);
            messages.push((evaluations, challenge))
            
        } else {
            // dbg!(&eval);
            panic!();
        };

        challenge = Some(F::rand(rng));
        // challenge = Some(F::one());
    }

    // expr.fix_variable(max_round - 1, challenge.unwrap());
    // beta_table.beta_update(max_round - 1, challenge.unwrap()).unwrap();

    messages

}

/// Returns the curr random challenge if verified correctly, otherwise verify error
/// can change this to take prev round random challenge, and then compute the new random challenge
/// TODO!(ryancao): Change this to take in the expression as well and do the final sumcheck check
pub fn verify_sumcheck_messages<F: FieldExt>(
    messages: Vec<(Vec<F>, Option<F>)>,
    mut expression: ExpressionStandard<F>,
    layer_claim: Claim<F>,
    rng: &mut impl Rng,
) -> Result<F, VerifyError> {
    let mut prev_evals = &messages[0].0;
    let mut chal = F::zero();

    // Thaler book page 34
    // First round:
    // For the first two evals, i.e. g_1(0), g_1(1), add them together.
    // This is the claimed sumcheck result from the prover.
    // TODO!(ende): degree check?
    let claimed_val = messages[0].0[0] + messages[0].0[1];
    if claimed_val != layer_claim.1 {
        return Err(VerifyError::SumcheckBad);
    }
    let mut challenges = vec![];

    // --- Go through sumcheck messages + (FS-generated) challenges ---
    // Round j, 1 < j < v
    for (evals, challenge) in messages.iter().skip(1) {
        let curr_evals = evals;
        chal = (*challenge).unwrap();
        // --- Evaluate the previous round's polynomial at the random challenge point, i.e. g_{i - 1}(r_i) ---
        let prev_at_r = evaluate_at_a_point(prev_evals, challenge.unwrap())
            .expect("could not evaluate at challenge point");

        // dbg!(&prev_evals);
        // dbg!(-prev_evals[0]);
        // dbg!(prev_at_r);
        // dbg!(curr_evals[0], curr_evals[1], curr_evals[0] + curr_evals[1]);
        // --- g_{i - 1}(r) should equal g_i(0) + g_i(1) ---
        if prev_at_r != curr_evals[0] + curr_evals[1] {
            return Err(VerifyError::SumcheckBad);
        };
        prev_evals = curr_evals;
        challenges.push(chal);
    }

    // Round v, again Thaler book page 34
    let final_chal = F::rand(rng);
    // let final_chal = F::one();
    challenges.push(final_chal);

    // uses the expression to make one single oracle query
    let mut beta = BetaTable::new(layer_claim).unwrap();
    // let _ = expression.index_mle_indices(0);
    let mle_bound = expression.evaluate_expr(challenges.clone()).unwrap();
    let beta_bound = evaluate_beta(&mut beta, challenges).unwrap();
    let oracle_query = mle_bound * beta_bound;

    dbg!(oracle_query);

    let prev_at_r = evaluate_at_a_point(prev_evals, final_chal)
            .expect("could not evaluate at challenge point");
    if oracle_query != prev_at_r {
        dbg!("oh oh");
        dbg!(&oracle_query);
        dbg!(&prev_at_r);
        return Err(VerifyError::SumcheckBad);
    }

    Ok(chal)
}

/// Use degree + 1 evaluations to figure out the evaluation at some arbitrary point
pub fn evaluate_at_a_point<F: FieldExt>(given_evals: &Vec<F>, point: F) -> Result<F, InterpError> {
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
    if eval.is_none() {
        Err(InterpError::NoInverse)
    } else {
        Ok(eval.unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        expression::ExpressionStandard,
        mle::{dense::{DenseMle, Tuple2}, Mle}, layer::claims::aggregate_claims,
    };
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use ark_std::{One, Zero};
    use ark_std::UniformRand;

    fn get_dummy_claim<F: FieldExt>(mle_ref: DenseMleRef<F>, rng: &mut impl Rng, challenges: Option<Vec<F>>) -> Claim<F> {
        let mut expression = mle_ref.expression();
        let num_vars = expression.index_mle_indices(0);
        let challenges = if let Some(challenges) = challenges {assert_eq!(challenges.len(), num_vars); challenges} else {(0..num_vars).map(|_| F::rand(rng)).collect_vec()};
        let eval = expression.evaluate_expr(challenges.clone()).unwrap();
        let claim = match expression {
            ExpressionStandard::Mle(mle) => {
                mle.mle_indices.iter().map(|item| match item {
                    MleIndex::Fixed(bit) => if *bit {F::one()} else {F::zero()},
                    MleIndex::Bound(chal) => *chal,
                    _ => panic!(),
                }).collect_vec()
            },
            _ => panic!()
        };
        (claim, eval)
    }

    /// Test regular numerical evaluation, last round type beat
    #[test]
    fn eval_expr_nums() {
        let layer_claim: Claim<Fr> = (vec![Fr::from(2), Fr::from(3)], Fr::one());
        let mut beta_table = BetaTable::new(layer_claim).unwrap();
        beta_table.table.index_mle_indices(0);
        let mut expression1: ExpressionStandard<Fr> = ExpressionStandard::Constant(Fr::from(6));
        let res = compute_sumcheck_message(&mut expression1, 0, 1, &mut beta_table);
        let exp = SumOrEvals::Evals(vec![Fr::from(-6), Fr::from(12)]);
        assert_eq!(res.unwrap(), exp);
    }

    /// Test the evaluation at an arbitrary point, all positives
    #[test]
    fn eval_at_point_pos() {
        //poly = 3x^2 + 5x + 9
        let evals = vec![Fr::from(9), Fr::from(17), Fr::from(31)];
        let point = Fr::from(3);
        let evald = evaluate_at_a_point(&evals, point);
        assert_eq!(
            evald.unwrap(),
            Fr::from(3) * point * point + Fr::from(5) * point + Fr::from(9)
        );
    }

    /// Test the evaluation at an arbitrary point, neg numbers
    #[test]
    fn eval_at_point_neg() {
        // poly = 2x^2 - 6x + 3
        let evals = vec![Fr::from(3), Fr::from(-1), Fr::from(-1)];
        let degree = 2;
        let point = Fr::from(3);
        let evald = evaluate_at_a_point(&evals, point);
        assert_eq!(
            evald.unwrap(),
            Fr::from(2) * point * point - Fr::from(6) * point + Fr::from(3)
        );
    }

    /// Test the evaluation at an arbitrary point, more evals than degree
    #[test]
    fn eval_at_point_more_than_degree() {
        // poly = 3 + 10x
        let evals = vec![Fr::from(3), Fr::from(13), Fr::from(23)];
        let point = Fr::from(3);
        let evald = evaluate_at_a_point(&evals, point);
        assert_eq!(evald.unwrap(), Fr::from(3) + Fr::from(10) * point);
    }

    /// Test whether evaluate_mle_ref correctly computes the evaluations for a single MLE
    #[test]
    fn test_linear_sum() {
        let layer_claim = (vec![Fr::from(2), Fr::from(4)], Fr::one());
        let mut beta_table = BetaTable::new(layer_claim).unwrap();
        let mle_v1 = vec![
            Fr::from(3),
            Fr::from(2),
            Fr::from(2),
            Fr::from(5),
        ];
        let mle1: DenseMleRef<Fr> = DenseMle::new(mle_v1).mle_ref();
        let mut mleexpr = ExpressionStandard::Mle(mle1);
        mleexpr.index_mle_indices(0);
        beta_table.table.index_mle_indices(0);
        // mleexpr.init_beta_tables(layer_claim);

        let res = compute_sumcheck_message(&mut mleexpr, 1, 2, &mut beta_table);
        let exp = SumOrEvals::Evals(vec![Fr::from(1), Fr::from(28), Fr::from(145)]);
        assert_eq!(res.unwrap(), exp);
    }

    /// Test whether evaluate_mle_ref correctly computes the evaluations for a product of MLEs
    #[test]
    fn test_quadratic_sum() {
        let layer_claim = (vec![Fr::from(2), Fr::from(4)], Fr::one());
        let mut beta_table = BetaTable::new(layer_claim).unwrap();

        let mle_v1 = vec![Fr::from(1), Fr::from(0), Fr::from(2), Fr::from(3)];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(5)];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mut expression = ExpressionStandard::Product(vec![mle1.mle_ref(), mle2.mle_ref()]);
        expression.index_mle_indices(0);
        beta_table.table.index_mle_indices(0);
        // expression.init_beta_tables(layer_claim);

        let res = compute_sumcheck_message(&mut expression, 1, 3, &mut beta_table);
        let exp = SumOrEvals::Evals(vec![Fr::from(-2), Fr::from(120), Fr::from(780), Fr::from(2320)]);
        assert_eq!(res.unwrap(), exp);
    }

    /// test whether evaluate_mle_ref correctly computes the evalutaions for a product of MLEs
    /// where one of the MLEs is a log size step smaller than the other (e.g. V(b_1, b_2)*V(b_1))
    #[test]
    fn test_quadratic_sum_differently_sized_mles2() {
        let layer_claim = (vec![Fr::from(2), Fr::from(4), Fr::from(3)], Fr::one());
        let mut beta_table = BetaTable::new(layer_claim).unwrap();

        let mle_v1 = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(5)];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mut expression = ExpressionStandard::Product(vec![mle1.mle_ref(), mle2.mle_ref()]);
        expression.index_mle_indices(0);
        beta_table.table.index_mle_indices(0);

        let res = compute_sumcheck_message(&mut expression, 1, 3, &mut beta_table);
        let exp = SumOrEvals::Evals(vec![Fr::from(-12), Fr::from(230), Fr::from(1740), Fr::from(5688)]);
        assert_eq!(res.unwrap(), exp);
    }

    /// test dummy sumcheck against verifier for product of the same mle
    #[test]
    fn test_dummy_sumcheck_1() {
        // let layer_claims = (vec![Fr::from(1), Fr::from(-1)], Fr::from(2));
        let mut rng = test_rng();
        let mle_vec = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(2)];

        let mle_new: DenseMle<Fr, Fr> = DenseMle::new(mle_vec);
        let mle_v2 = vec![Fr::from(1), Fr::from(5), Fr::from(1), Fr::from(5)];
        let mle_2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mle_output = mle_new.clone().into_iter().zip(mle_2.clone().into_iter()).map(|(first, second)| first * second).collect::<DenseMle<Fr, Fr>>();
        let layer_claims = get_dummy_claim(mle_output.mle_ref(), &mut rng, None);

        // dbg!(claimed_claim);

        let mle_ref_1 = mle_new.mle_ref();
        let mle_ref_2 = mle_2.mle_ref();

        let mut expression = ExpressionStandard::Product(vec![mle_ref_1, mle_ref_2]);
        let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
        let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }

    /// test dummy sumcheck against product of two diff mles
    #[test]
    fn test_dummy_sumcheck_2() {
        // let layer_claims = (vec![Fr::from(3), Fr::from(4), Fr::from(2)], Fr::one());
        let mut rng = test_rng();
        let mle_v1 = vec![Fr::from(1), Fr::from(0), Fr::from(2), Fr::from(3)];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(5)];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mle_output = mle1.clone().into_iter().zip(mle2.clone().into_iter()).map(|(first, second)| first * second).collect::<DenseMle<Fr, Fr>>();
        let layer_claims = get_dummy_claim(mle_output.mle_ref(), &mut rng, None);


        let mle_ref_1 = mle1.mle_ref();
        let mle_ref_2 = mle2.mle_ref();

        let mut expression = ExpressionStandard::Product(vec![mle_ref_1, mle_ref_2]);
        let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
        let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }

    /// test dummy sumcheck against product of two mles diff sizes
    #[test]
    fn test_dummy_sumcheck_3() {
        let mut rng = test_rng();
        let mle_v1 = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(5)];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mle_output = mle1.clone().into_iter().zip(mle2.clone().into_iter().chain(mle2.clone().into_iter())).map(|(first, second)| first * second).collect::<DenseMle<Fr, Fr>>();
        let layer_claims = get_dummy_claim(mle_output.mle_ref(), &mut rng, None);

        let mut mle_ref_1 = mle1.mle_ref();
        let mut mle_ref_2 = mle2.mle_ref();

        let mut expression = ExpressionStandard::Product(vec![mle_ref_1, mle_ref_2]);
        let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
        let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }

    /// test dummy sumcheck against sum of two mles
    #[test]
    fn test_dummy_sumcheck_sum_small() {
        let mut rng = test_rng();
        let mle_v1 = vec![Fr::from(1), Fr::from(0), Fr::from(1), Fr::from(2)];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(5)];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mle_output = mle1.clone().into_iter().zip(mle2.clone().into_iter()).map(|(first, second)| first + second).collect::<DenseMle<Fr, Fr>>();
        let layer_claims = get_dummy_claim(mle_output.mle_ref(), &mut rng, None);

        let mut mle_ref_1 = mle1.mle_ref();
        let mut mle_ref_2 = mle2.mle_ref();

        let mut expression = mle_ref_1.expression() + mle_ref_2.expression();

        let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
        let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }

    /// test dummy sumcheck for concatenated expr SMALL
    #[test]
    fn test_dummy_sumcheck_concat() {
        // let layer_claims = (vec![Fr::from(3), Fr::from(1), Fr::from(2)], Fr::one());
        let mut rng = test_rng();
        let mle_v1 = vec![Fr::from(5), Fr::from(2)];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![Fr::from(2), Fr::from(3)];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let output: DenseMle<Fr, Fr> = DenseMle::new(vec![Fr::from(5), Fr::from(2), Fr::from(2), Fr::from(3)]);

        let layer_claims = get_dummy_claim(output.mle_ref(), &mut rng, None);

        let mle_ref_1 = mle1.mle_ref();
        let mle_ref_2 = mle2.mle_ref();

        let expression = ExpressionStandard::Mle(mle_ref_1);
        let expr2 = ExpressionStandard::Mle(mle_ref_2);

        let mut expression = expr2.concat(expression);
        let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
        let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }

    /// test dummy sumcheck for concatenated expr SMALL BUT LESS SMALL
    #[test]
    fn test_dummy_sumcheck_concat_2() {
        // let layer_claims = (
        //     vec![Fr::from(2), Fr::from(4), Fr::from(2), Fr::from(3)],
        //     Fr::one(),
        // );
        let mut rng = test_rng();
        let mle_v1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![Fr::from(1), Fr::from(3), Fr::from(1), Fr::from(6)];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mle_out = DenseMle::<Fr, Fr>::new(vec![Fr::from(1), Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(3), Fr::from(1), Fr::from(4), Fr::from(6)]);
        let layer_claims = get_dummy_claim(mle_out.mle_ref(), &mut rng, None);

        let mle_ref_1 = mle1.mle_ref();
        let mle_ref_2 = mle2.mle_ref();

        let expression = ExpressionStandard::Mle(mle_ref_1);
        let expr2 = ExpressionStandard::Mle(mle_ref_2);

        let mut expression = expr2.concat(expression);
        let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
        let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }

    /// test dummy sumcheck for concatenated expr
    #[test]
    fn test_dummy_sumcheck_concat_aggro() {
        // let layer_claims = (
        //     vec![Fr::from(2), Fr::from(4), Fr::from(2), Fr::from(3)],
        //     Fr::one(),
        // );
        let mut rng = test_rng();
        let mle_v1 = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(-23),
            Fr::from(-47),
            Fr::from(5),
            Fr::from(22),
            Fr::from(31),
            Fr::from(-4),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![Fr::from(1), Fr::from(3), Fr::from(1), Fr::from(6)];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mle_ref_1 = mle1.mle_ref();
        let mle_ref_2 = mle2.mle_ref();

        let mle_output = mle1.clone().into_iter().zip(mle2.clone().into_iter().chain(mle2.clone().into_iter())).map(|(first, second)| first * second).flat_map(|x| vec![x.clone(), x]).collect::<DenseMle<Fr, Fr>>();
        let layer_claims = get_dummy_claim(mle_output.mle_ref(), &mut rng, None);

        let expression = ExpressionStandard::Product(vec![mle_ref_1, mle_ref_2.clone()]);
        let expr2 = expression.clone();

        let mut expression = expr2.concat(expression);
        let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());

        let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }
   
    #[test]
    fn test_dummy_sumcheck_concat_aggro_aggro() {

        let layer_claims = (vec![Fr::from(12190), Fr::from(28912), Fr::from(1),], Fr::one());
        let mut rng = test_rng();
        let mle_v1 = vec![
            Fr::from(1),
            Fr::from(2),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![Fr::from(5), Fr::from(1291),];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mle_ref_1 = mle1.mle_ref();
        let mle_ref_2 = mle2.mle_ref();

        let expression = ExpressionStandard::Mle(mle_ref_1);
        let expr2 = ExpressionStandard::Mle(mle_ref_2);

        let expression = expr2.clone().concat(expression);
        let expression_aggro = expression.clone().concat(expr2);
        let res_messages = dummy_sumcheck( &mut expression_aggro.clone(), &mut rng, layer_claims.clone());
        let verifyres = verify_sumcheck_messages(res_messages, expression_aggro, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }

    /// test dummy sumcheck for concatenated expr 
    #[test]
    fn test_dummy_sumcheck_concat_aggro_aggro_aggro() {


        let layer_claims = (vec![Fr::from(12190), Fr::from(28912), Fr::from(1), Fr::from(12901), Fr::from(65309)], Fr::one());
        let mut rng = test_rng();
        let mle_v1 = vec![
            Fr::from(1390),
            Fr::from(222104),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![Fr::from(5), Fr::from(1291),];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mle_ref_1 = mle1.mle_ref();
        let mle_ref_2 = mle2.mle_ref();

        let expression = ExpressionStandard::Mle(mle_ref_1);
        let expr2 = ExpressionStandard::Mle(mle_ref_2);

        let expression = expr2.clone().concat(expression);
        let expression_aggro = expression.clone().concat(expr2.clone());
        let expression_aggro_aggro = expression_aggro.clone().concat(expr2);
        let res_messages = dummy_sumcheck( &mut expression_aggro_aggro.clone(), &mut rng, layer_claims.clone());
        let verifyres = verify_sumcheck_messages(res_messages, expression_aggro_aggro, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }
   
    #[test]
    fn test_dummy_sumcheck_sum() {
        let layer_claims = (vec![Fr::from(2), Fr::from(1), Fr::from(10),], Fr::one());
        let mut rng = test_rng();
        let mle_v1 = vec![
            Fr::from(0),
            Fr::from(2),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![Fr::from(5), Fr::from(1291),];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mle_ref_1 = mle1.mle_ref();
        let mle_ref_2 = mle2.mle_ref();

        let expression = ExpressionStandard::Mle(mle_ref_1);
        let expr2 = ExpressionStandard::Mle(mle_ref_2);

        let expression = expr2.clone().concat(expression);
        let expression_aggro = expression.clone().concat(expr2);
        let res_messages = dummy_sumcheck( &mut expression_aggro.clone(), &mut rng, layer_claims.clone());
        let verifyres = verify_sumcheck_messages(res_messages, expression_aggro, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }

    #[test]
    fn test_dummy_sumcheck_subtract() {
        // let layer_claims = (vec![Fr::from(2), Fr::from(1), Fr::from(10),], Fr::one());
        let mut rng = test_rng();
        let mle_v1 = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(64),
            Fr::from(129),
            // Fr::from(23),
            // Fr::from(383),
            // Fr::from(30489),
            // Fr::from(382),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![
            Fr::from(5), Fr::from(1291), Fr::from(8), Fr::from(30), 
            // Fr::from(292), Fr::from(733), Fr::from(2281), Fr::from(485)
        ];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mle_out = mle1.clone().into_iter().zip(mle2.clone().into_iter()).map(|(first, second)| first - second).collect::<DenseMle<Fr, Fr>>().split();

        let first_claim = get_dummy_claim(mle_out.first(), &mut rng, None);
        let second_claim = get_dummy_claim(mle_out.second(), &mut rng, Some(first_claim.0[1..].to_vec()));

        let mle_ref_1 = mle1.mle_ref();
        let mle_ref_2 = mle2.mle_ref();

        let mut expression = mle_ref_1.expression() - mle_ref_2.expression();

        let (layer_claims, _) = aggregate_claims(&[first_claim, second_claim], &expression, Fr::rand(&mut rng)).unwrap();

        let res_messages = dummy_sumcheck( &mut expression, &mut rng, layer_claims.clone());
        let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }

    #[test]
    fn test_dummy_sumcheck_product_and_claim_aggregate() {
        // let layer_claims = (vec![Fr::from(2), Fr::from(1), Fr::from(10),], Fr::one());
        let mut rng = test_rng();
        let mle_v1 = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(64),
            Fr::from(129),
            Fr::from(23),
            Fr::from(383),
            Fr::from(30489),
            Fr::from(382),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);

        let mle_v2 = vec![
            Fr::from(5), Fr::from(1291), Fr::from(8), Fr::from(30), 
            Fr::from(292), Fr::from(733), Fr::from(2281), Fr::from(485)
        ];
        let mle2: DenseMle<Fr, Fr> = DenseMle::new(mle_v2);

        let mle_out = mle1.clone().into_iter().zip(mle2.clone().into_iter()).map(|(first, second)| first * second).collect::<DenseMle<Fr, Fr>>().split();

        let first_claim = get_dummy_claim(mle_out.first(), &mut rng, None);
        let second_claim = get_dummy_claim(mle_out.second(), &mut rng, Some(first_claim.0[1..].to_vec()));

        let mle_ref_1 = mle1.mle_ref();
        let mle_ref_2 = mle2.mle_ref();

        let mut expression = ExpressionStandard::products(vec![mle_ref_1, mle_ref_2]);

        let (layer_claims, _) = aggregate_claims(&[first_claim, second_claim], &expression, Fr::rand(&mut rng)).unwrap();

        let res_messages = dummy_sumcheck( &mut expression, &mut rng, layer_claims.clone());
        let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }

    #[test]
    fn test_dummy_sumcheck_example() {
        let mut rng = test_rng();
        let mle: DenseMle<Fr, Tuple2<Fr>> = vec![(Fr::from(2), Fr::from(8)), (Fr::from(7), Fr::from(3))].into_iter().map(|x| x.into()).collect();
        let mle_2: DenseMle<Fr, Tuple2<Fr>> = vec![(Fr::from(9), Fr::from(2)), (Fr::from(12), Fr::from(1))].into_iter().map(|x| x.into()).collect();

        let binding = mle.clone();
        let mut output_mle_1 = binding.into_iter().map(|(first, second)| first + second).collect::<DenseMle<Fr, Fr>>();
        let binding: DenseMle<Fr, Tuple2<Fr>> = mle_2.clone();
        let mut output_mle_2 = binding.into_iter().map(|(first, second)| first + second).collect::<DenseMle<Fr, Fr>>();

        output_mle_1.add_prefix_bits(Some(vec![MleIndex::Fixed(true)]));
        output_mle_2.add_prefix_bits(Some(vec![MleIndex::Fixed(false)]));

        let output = output_mle_2.clone().into_iter().interleave(output_mle_1.clone().into_iter()).collect::<DenseMle<Fr, Fr>>();

        let first_claim = get_dummy_claim(output_mle_1.mle_ref(), &mut rng, None);
        let second_claim = get_dummy_claim(output_mle_2.mle_ref(), &mut rng, Some(first_claim.0[1..].to_vec()));

        // let expr_1 = ExpressionStandard::products(vec![mle.first(), mle.second()]);
        let expr_1 = mle.first().expression() + mle.second().expression();
        let expr_2 = mle_2.first().expression() + mle_2.second().expression();
        let mut expression = expr_1.concat(expr_2);

        let (layer_claims, _) = aggregate_claims(&[first_claim, second_claim], &expression, Fr::rand(&mut rng)).unwrap();

        let layer_claims_real = get_dummy_claim(output.mle_ref(), &mut rng, Some(layer_claims.0.clone()));

        assert_eq!(layer_claims_real, layer_claims);

        let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
        let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
        assert!(verifyres.is_ok());
    }

}
