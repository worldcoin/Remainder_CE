//! Contains cryptographic algorithms for going through the sumcheck protocol

use std::f32::NEG_INFINITY;

use ark_poly::MultilinearExtension;
use ark_std::cfg_into_iter;
use itertools::{repeat_n, Itertools};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use thiserror::Error;

use crate::{expression::Expression, mle::MleRef, FieldExt};

#[derive(Error, Debug, Clone)]
enum MleError {
    #[error("Passed list of Mles is empty")]
    EmptyMleList,
}

pub(crate) fn fix_variables<F: FieldExt>(mle_ref: &mut impl MleRef, challenges: &[F]) {
    todo!()
}

pub(crate) fn prove_round<F: FieldExt>(expr: Expression<F>) -> Vec<F> {
    todo!()
}

//TODO!(This should be able to handle the case where there is no independent variable (e.g. we're iterating over a variable that doesn't effect this set of mle products))
fn evaluate_mle_ref<F: FieldExt>(
    mle_refs: &[impl MleRef<F = F>],
    degree: usize,
) -> Result<Vec<F>, MleError> {
    let eval_count = degree + 1;
    let max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    
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
                        std::iter::successors(Some(second + step), move |item| Some(*item + step));
                    //iterator that represents all evaluations of the MLE extended to arbitrarily many linear extrapolations on the line of 0/1
                    std::iter::once(first)
                        .chain(std::iter::once(second))
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

    todo!()
}
