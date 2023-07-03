//! Contains cryptographic algorithms for going through the sumcheck protocol

use std::f32::NEG_INFINITY;

use ark_poly::MultilinearExtension;
use ark_std::cfg_into_iter;
use itertools::{repeat_n, Itertools};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator, IntoParallelRefIterator};
use thiserror::Error;

use crate::{expression::Expression, mle::MleRef, FieldExt};

#[derive(Error, Debug, Clone)]
enum MleError {
    #[error("Passed list of Mles is empty")]
    EmptyMleList
}

pub(crate) fn fix_variables<F: FieldExt>(mle_ref: &mut impl MleRef, challenges: &[F]) {
    todo!()
}

pub(crate) fn prove_round<F: FieldExt>(expr: Expression<F>) -> Vec<F> {
    todo!()
}

//FUCK this needs to be re-written for high degree cases
fn evaluate_mle_ref<F: FieldExt, Mle: MleRef<F = F, Mle = Vec<F>>>(
    mle_refs: (Mle, Mle),
    degree: usize,
) -> Result<Vec<F>, MleError> {
    let eval_count = degree + 1;

    // This MLE has been mutated so that it represents the evaluations where the first round bits are bound to random values.
    // In other words it is the "bookkeeping table" referenced in theory

    let (mle_ref_1, mle_ref_2) = mle_refs;

    let (mle_1, mle_2) = { (mle_ref_1.mle_owned(), mle_ref_2.mle_owned()) };

    let (difference, (bigger, bigger_num_vars), (smaller, smaller_num_vars)): (usize, (rayon::vec::IntoIter<F>, usize), (rayon::slice::Iter<F>, usize)) = if mle_1.len() > mle_2.len() {
        (
            mle_1.len() - mle_2.len(),
            (mle_1.into_par_iter(), mle_ref_1.num_vars()),
            (mle_2.par_iter(), mle_ref_2.num_vars()),
        )
    } else {
        (
            mle_2.len() - mle_1.len(),
            (mle_2.into_par_iter(), mle_ref_2.num_vars()),
            (mle_1.par_iter(), mle_ref_1.num_vars()),
        )
    };

    // let zero = F::zero();

    // if bigger_num_vars != smaller_num_vars {
    //     let difference_num_vars = bigger_num_vars - smaller_num_vars;
    //     let smaller = smaller.chain(rayon::iter::repeatn(&zero, (difference as u32/2_u32.pow(difference_num_vars as u32)) as usize).into_par_iter());
    //     let smaller = smaller.clone().chain(smaller);
    // }

    // let iter = cfg_into_iter!(mle_refs.0.mle_owned()).zip(cfg_into_iter!(mle_refs.1.mle_owned()));

    // take all sets of two evaluations (get all evaluations across the boolean hypercube of V(0/1, b_1..b_i))
    #[cfg(feature = "parallel")]
    let evaluations = bigger
        .chunks(2).enumerate()
        .map(|(index, evals)| {
            let extra = if eval_count > 2 {
                let first = &evals[0];
                let second = evals.get(1).cloned().unwrap_or(F::zero());

                let step = second - first;
                //Iterator that represent extrapolations on the linear evaluations of X
                Some(std::iter::successors(Some(second + step), move |item| {
                    Some(step + item)
                }))
            } else {
                None
            };

            evals.into_iter().chain(extra.into_iter().flatten())
        })
        .map(|evals| -> Box<dyn Iterator<Item = F> + Send> { Box::new(evals) })
        .reduce(
            || Box::new(repeat_n(F::zero(), eval_count)),
            |sum, evals| Box::new(sum.zip(evals).map(|(sum, eval)| sum + eval)),
        )
        .collect_vec();

    #[cfg(not(feature = "parallel"))]
    todo!("Implement not!parallel for mle_evaluation");
    //TODO!(not parallel feature)

    todo!()
}
