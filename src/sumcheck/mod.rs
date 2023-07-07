//! Contains cryptographic algorithms for going through the sumcheck protocol

use std::{f32::NEG_INFINITY, iter::repeat, ops::{Neg, Add, Sub, Mul}};

use ark_poly::MultilinearExtension;
use ark_std::{cfg_into_iter, rand::Rng};
use itertools::{repeat_n, Itertools};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator, IntoParallelRefIterator,
};
use thiserror::Error;

use crate::{expression::{ExpressionStandard, ExpressionError, Expression}, mle::{MleRef, MleIndex, dense::DenseMleRef}, FieldExt};

#[derive(Error, Debug, Clone)]
enum MleError {
    #[error("Passed list of Mles is empty")]
    EmptyMleList,
}

pub(crate) enum SumOrEvals<F: FieldExt> {
    Sum(F),
    Evals(Vec<F>),
}

impl<F: FieldExt> Neg for SumOrEvals<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        match self {
            SumOrEvals::Sum(sum) => SumOrEvals::Sum(sum.neg()),
            SumOrEvals::Evals(evals) => SumOrEvals::Evals(evals.into_iter().map(|eval| eval.neg()).collect_vec())
        }
    }
}

impl<F: FieldExt> Add for SumOrEvals<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match self {
            SumOrEvals::Sum(sum) => {
                match rhs {
                    SumOrEvals::Sum(rhs) => SumOrEvals::Sum(sum + rhs),
                    SumOrEvals::Evals(rhs) => SumOrEvals::Evals(repeat(sum).zip(rhs.into_iter()).map(|(lhs, rhs)| lhs + rhs).collect_vec()),
                }
            },
            SumOrEvals::Evals(evals) => {
                match rhs {
                    SumOrEvals::Sum(rhs) => SumOrEvals::Evals(evals.into_iter().zip(repeat(rhs)).map(|(lhs, rhs)| lhs + rhs).collect_vec()),
                    SumOrEvals::Evals(rhs) => SumOrEvals::Evals(evals.into_iter().zip(rhs.into_iter()).map(|(lhs, rhs)| lhs + rhs).collect_vec()),
                }
            }
        }
    }
}

impl<F: FieldExt> Mul for SumOrEvals<F> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        match self {
            SumOrEvals::Sum(sum) => {
                match rhs {
                    SumOrEvals::Sum(rhs) => SumOrEvals::Sum(sum * rhs),
                    SumOrEvals::Evals(rhs) => SumOrEvals::Evals(repeat(sum).zip(rhs.into_iter()).map(|(lhs, rhs)| lhs * rhs).collect_vec()),
                }
            },
            SumOrEvals::Evals(evals) => {
                match rhs {
                    SumOrEvals::Sum(rhs) => SumOrEvals::Evals(evals.into_iter().zip(repeat(rhs)).map(|(lhs, rhs)| lhs * rhs).collect_vec()),
                    SumOrEvals::Evals(rhs) => SumOrEvals::Evals(evals.into_iter().zip(rhs.into_iter()).map(|(lhs, rhs)| lhs * rhs).collect_vec()),
                }
            }
        }
    }
}

pub(crate) fn evaluate_expr<F: FieldExt, Exp: Expression<F>>(mut expr: &mut Exp, round_index: usize, max_degree: usize) -> Result<SumOrEvals<F>, ExpressionError> {
    let constant = |constant| {
        Ok(SumOrEvals::Sum(constant))
    };

    let selector = |index: &mut MleIndex<F>, a, b| {
        match index {
            MleIndex::IndexedBit(indexed_bit) => {
                match Ord::cmp(&round_index, &indexed_bit) {
                    std::cmp::Ordering::Less => {
                        let a = a?;
                        let b = b?;
                        Ok(a + b)
                    },
                    std::cmp::Ordering::Equal => {
                        let first = b?;
                        let second = a?;
                        if let (SumOrEvals::Sum(first), SumOrEvals::Sum(second)) = (first, second) {
                            if max_degree == 1 {
                                Ok(SumOrEvals::Evals(vec![first, second]))
                            } else {
                                Err(ExpressionError::EvaluationError("Expression has a degree > 1 when the round is on a selector bit"))
                            }  
                        } else {
                            Err(ExpressionError::EvaluationError("Expression returns an Evals variant when the round is on a selector bit"))
                        }
                    },
                    std::cmp::Ordering::Greater => Err(ExpressionError::InvalidMleIndex),
                }
            },
            MleIndex::Bound(coeff_raw) => {
                let coeff_raw = coeff_raw.clone();
                let coeff = SumOrEvals::Sum(coeff_raw);
                let coeff_neg = SumOrEvals::Sum(F::one()-coeff_raw);
                let a: SumOrEvals<F> = a?;
                let b: SumOrEvals<F> = b?;

                Ok((a * coeff) + (b * coeff_neg))
            },
            _ => Err(ExpressionError::InvalidMleIndex)
        }
    };

    let mle_eval = |mle_ref: &mut Exp::MleRef| {
        let mle_indicies = mle_ref.mle_indices();
        let independent_variable = mle_indicies.contains(&MleIndex::IndexedBit(round_index));
        evaluate_mle_ref(&[mle_ref.clone()], independent_variable, max_degree).map_err(|_| ExpressionError::MleError)
    };

    let negated = |a: Result<_, _>| {
        a.map(|a: SumOrEvals<F>| a.neg())
    };

    let sum = |a, b| {
        let a = a?;
        let b = b?;

        Ok(a + b)
    };

    let product = for <'a> |mle_refs: &'a mut [Exp::MleRef]| -> Result<SumOrEvals<F>, ExpressionError> {
        let independent_variable = mle_refs.iter().map(|mle_ref| mle_ref.mle_indices().contains(&MleIndex::IndexedBit(round_index))).reduce(|acc, item| acc & item).ok_or(ExpressionError::MleError)?;

        evaluate_mle_ref(mle_refs, independent_variable, max_degree).map_err(|_| ExpressionError::MleError)
    };

    let scaled = |a, scalar| {
        let a = a?;
        let scalar = SumOrEvals::Sum(scalar);

        Ok(a * scalar)
    };

    expr.evaluate(&constant, &selector, &mle_eval, &negated, &sum, &product, &scaled)

}

fn evaluate_mle_ref<F: FieldExt>(
    mle_refs: &[impl MleRef<F = F>],
    independent_variable: bool,
    degree: usize,
) -> Result<SumOrEvals<F>, MleError> {
    let max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

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
                        let first = *mle_ref.mle().get(index).unwrap_or(&zero);
                        let second = *mle_ref.mle().get(index + 1).unwrap_or(&zero);

                        let step = second - first;

                        let successors =
                            std::iter::successors(Some(second), move |item| Some(*item + step));
                        //iterator that represents all evaluations of the MLE extended to arbitrarily many linear extrapolations on the line of 0/1
                        std::iter::once(first)
                            .chain(successors)
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
                    .zip(partial.into_iter())
                    .for_each(|(acc, partial)| *acc += partial);
                acc
            },
        );

        Ok(SumOrEvals::Evals(evals))
    } else {
        //There is no independent variable and we can sum over everything
        let partials = cfg_into_iter!((0..1 << (max_num_vars))).fold(
            #[cfg(feature = "parallel")]
            || F::zero(),
            #[cfg(not(feature = "parallel"))]
            F::zero(),
            |acc, index| {
                //get the product of all evaluations over 0/1/..degree
                let product = mle_refs
                    .iter()
                    .map(|mle_ref| {
                        let index = if mle_ref.num_vars() < max_num_vars {
                            let max = 1 << mle_ref.num_vars();
                            index % max
                        } else {
                            index
                        };
                        mle_ref.mle().get(index).cloned().unwrap_or(F::zero())
                    })
                    .reduce(|acc, eval| acc * eval)
                    .unwrap();

                acc + product
            },
        );

        #[cfg(feature = "parallel")]
        let sum = partials.sum();
        Ok(SumOrEvals::Sum(sum))
    }
}

fn dummy_sumcheck<F: FieldExt>(mut expr: ExpressionStandard<F>, max_degree: usize, rng: &mut impl Rng) {
    let max_round = expr.index_mle_indices(0);
    let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
    let mut challenge: Option<F> = None;

    for round_index in 0..max_round{
        if let Some(challenge) = challenge {
            expr.fix_variable(round_index - 1, challenge)
        }

        if let Ok(SumOrEvals::Evals(evaluations)) = evaluate_expr(&mut expr, round_index, max_degree){
            dbg!(&evaluations);
            dbg!(challenge);
            messages.push((evaluations, challenge.clone()))
        } else {
            panic!();
        };

        challenge = Some(F::rand(rng));
    }

    expr.fix_variable(max_round - 1, challenge.unwrap());

    if let Ok(SumOrEvals::Sum(final_sum)) = evaluate_expr(&mut expr, max_round, max_degree) {
        dbg!(final_sum);
    } else {
        panic!();
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::test_rng;

    use crate::{mle::{dense::DenseMle, Mle}, expression::ExpressionStandard};

    use super::dummy_sumcheck;

    ///test whether evaluate_mle_ref correctly computes the evaluations for a single MLE
    #[test]
    fn test_linear_sum() {
        todo!()
    }

    ///test whether evaluate_mle_ref correctly computes the evaluations for a product of MLEs
    #[test]
    fn test_quadratic_sum() {
        todo!()
    }

    ///test whether evaluate_mle_ref correctly computes the evalutaions for a product of MLEs
    /// where one of the MLEs is a log size step smaller than the other (e.g. V(b_1, b_2)*V(b_1))
    #[test]
    fn test_quadratic_sum_differently_sized_mles() {
        todo!()
    }

    #[test]
    fn test_dummy_sumcheck() {
        let mut rng = test_rng();
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
            Fr::from(7),
        ];

        let mle_new: DenseMle<Fr, Fr> = DenseMle::new(mle_vec);
        let mle_2 = mle_new.clone();

        let mle_ref_1 = mle_new.mle_ref();
        let mle_ref_2 = mle_2.mle_ref();

        let expression = ExpressionStandard::Product(vec![mle_ref_1, mle_ref_2]);
        let expression = expression * Fr::from(5);
        dummy_sumcheck(expression, 2, &mut rng)
    }
}
