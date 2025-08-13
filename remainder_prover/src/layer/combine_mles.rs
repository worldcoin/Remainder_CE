//! This module contains the code used to combine parts of a layer and combining
//! them to determine the evaluation of a layer's layerwise bookkeeping table at
//! a point.

use crate::{
    mle::{
        dense::DenseMle, evals::{Evaluations, MultilinearExtension}, mle_enum::LiftTo, Mle, MleIndex
    },
    utils::mle::evaluate_mle_at_a_point_gray_codes,
};
use ark_std::{cfg_iter, cfg_iter_mut};

use itertools::Itertools;

use remainder_shared_types::{field::ExtensionField, Field};
use thiserror::Error;

use anyhow::{anyhow, Ok, Result};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

/// Type alias for an MLE evaluation and its prefix bits, as this type is used
/// throughout the combining code.
type MleEvaluationAndPrefixBits<F> = (F, Vec<bool>);

/// Error handling for gate mle construction
#[derive(Error, Debug, Clone)]
pub enum CombineMleRefError {
    #[error("we have not fully combined all the mles because the list size is > 1")]
    /// We have not fully combined all the mles because the list size is > 1.
    NotFullyCombined,
    #[error("we have an mle that is not fully fixed even after fixing on the challenge point")]
    /// We have an mle that is not fully fixed even after fixing on the
    /// challenge point.
    MleRefNotFullyFixed,
}

/// This fixes mles with shared points in the claims so that we don't repeatedly
/// do so.
pub fn pre_fix_mles<F: Field, E: ExtensionField<F>>(mles: &[DenseMle<F>], chal_point: &[E], common_idx: Vec<usize>) -> Vec<DenseMle<E>> {
    cfg_iter!(mles).map(|mle| {
        // TODO (Benny): this is certainly not the best way to do this
        let mut ext_mle = mle.lift();
        common_idx.iter().for_each(|chal_idx| {
            if let MleIndex::Indexed(idx_bit_num) = ext_mle.mle_indices()[*chal_idx] {
                ext_mle.fix_variable_at_index(idx_bit_num, chal_point[*chal_idx]);
            }
        });
        ext_mle
    }).collect()
}

/// Function that prepares all the mles to be combined. We simply index all the
/// MLEs that are to be fixed and combined, and ensure that all fixed bits are
/// always contiguous.
pub fn get_indexed_layer_mles_to_combine<F: Field>(mles: Vec<DenseMle<F>>) -> Vec<DenseMle<F>> {
    // We split all the mles with a free bit within the fixed bits. This is in
    // order to ensure that all the fixed bits are truly "prefix" bits.
    let mut mles_split = collapse_mles_with_free_in_prefix(mles);

    // Index all the MLEs as they will be fixed throughout the combining
    // process.
    cfg_iter_mut!(mles_split).for_each(|mle| {
        mle.index_mle_indices(0);
    });
    mles_split
}

/// This function takes in a list of mles, a challenge point we want to combine
/// them under, and returns the final value in the bookkeeping table of the
/// combined mle. This is equivalent to combining all of these mles according to
/// their prefix bits, and then fixing variable on this combined mle (which is
/// the layerwise bookkeeping table). Instead, we fix variable as we combine as
/// this keeps the bookkeeping table sizes at one.
pub fn combine_mles_with_aggregate<F: Field, E: ExtensionField<F>>(mles: &[DenseMle<F>], chal_point: &[E]) -> Result<E> {
    // We go through all of the mles and fix variable in all of them given at
    // the correct indices so that they are fully bound.
    let fix_var_mles = mles
        .iter()
        .map(|mle| {
            let point_to_bind = mle
                .mle_indices
                .iter()
                .enumerate()
                .filter_map(|(idx, mle_idx)| {
                    if let MleIndex::Indexed(_idx_num) = mle_idx {
                        Some(chal_point[idx])
                    } else {
                        None
                    }
                })
                .collect_vec();

            // Fully evaluate the MLE at a point using the gray codes algorithm.
            let mle_evaluation = evaluate_mle_at_a_point_gray_codes(&mle.mle, &point_to_bind);
            let prefix_bits = mle
                .mle_indices
                .iter()
                .filter_map(|mle_index| {
                    if let MleIndex::Fixed(prefix_bool) = mle_index {
                        Some(*prefix_bool)
                    } else {
                        None
                    }
                })
                .collect_vec();

            (mle_evaluation, prefix_bits)
        })
        .collect_vec();

    // Mutable variable that is overwritten every time we combine mles.
    let mut updated_list = fix_var_mles;

    // A loop that breaks when all the mles no longer have any fixed bits and
    // only have free bits. This means we have fully combined the MLEs to form
    // the layerwise bookkeeping table (but fully bound to a point).
    loop {
        // We first get the lsb fixed bit and the evaluation of the MLE that
        // contributes to it.
        let (mle_evaluation, mle_prefix_bits) = get_lsb_fixed_var(&updated_list);

        // There are only 0 prefix bits for the MLE contributing to the lsb
        // fixed bit if the MLEs have been fully combined.
        if mle_prefix_bits.is_empty() {
            break;
        }

        // Otherwise, overwrite updated_list to contain the combined MLE instead
        // of the two MLEs contributing to the lsb fixed bit.
        updated_list =
            find_pair_and_combine(&updated_list, mle_prefix_bits, *mle_evaluation, chal_point);
    }

    // The list now should only have one combined mle, and its bookkeeping table
    // should only have one value in it since we were binding variables as we
    // were combining.
    if updated_list.len() > 1 {
        return Err(anyhow!(CombineMleRefError::NotFullyCombined));
    }
    let (full_eval, prefix_bits) = &updated_list[0];
    assert_eq!(
        prefix_bits.len(),
        0,
        "there should be no more prefix bits left after fully combining!"
    );

    Ok(*full_eval)
}

/// This function takes an MLE that has a free variable in between fixed
/// variables, and it splits it into two MLEs, one where the free variable is
/// replaced with `Fixed(false)`, and the other where it is replaced with
/// `Fixed(true)`. This ensures that all the fixed bits are contiguous. NOTE we
/// assume that this function is called on an mle that has a free variable
/// within a bunch of fixed variables (note how it is used in the
/// `collapse_mles_with_free_in_prefix` function)
fn split_mle<F: Field>(mle: &DenseMle<F>) -> Vec<DenseMle<F>> {
    // Get the index of the first free bit in the mle.
    let first_free_idx: usize = mle.mle_indices().iter().enumerate().fold(
        mle.mle_indices().len(),
        |acc, (idx, mle_idx)| {
            if let MleIndex::Free = mle_idx {
                std::cmp::min(acc, idx)
            } else {
                acc
            }
        },
    );

    // Compute the correct indices, we have the first one be false, the second
    // one as true instead of the free bit.
    let first_indices = mle.mle_indices()[0..first_free_idx]
        .iter()
        .cloned()
        .chain(std::iter::once(MleIndex::Fixed(false)))
        .chain(mle.mle_indices()[first_free_idx + 1..].iter().cloned())
        .collect_vec();
    let second_indices = mle.mle_indices()[0..first_free_idx]
        .iter()
        .cloned()
        .chain(std::iter::once(MleIndex::Fixed(true)))
        .chain(mle.mle_indices()[first_free_idx + 1..].iter().cloned())
        .collect_vec();

    // Construct the first MLE in the pair.
    let first_mle = DenseMle {
        mle: MultilinearExtension::new_from_evals(Evaluations::<F>::new(
            mle.num_free_vars() - 1,
            mle.mle.iter().step_by(2).collect_vec(),
        )),
        mle_indices: first_indices,
        layer_id: mle.layer_id,
    };

    // Second mle in the pair.
    let second_mle = DenseMle {
        mle: MultilinearExtension::new_from_evals(Evaluations::<F>::new(
            mle.num_free_vars() - 1,
            mle.mle.iter().skip(1).step_by(2).collect_vec(),
        )),
        mle_indices: second_indices,
        layer_id: mle.layer_id,
    };

    vec![first_mle, second_mle]
}

/// This function will take a list of MLEs and updates the list to contain MLEs
/// where all fixed bits are contiguous
fn collapse_mles_with_free_in_prefix<F: Field>(mles: Vec<DenseMle<F>>) -> Vec<DenseMle<F>> {
    mles.into_iter()
        .flat_map(|mle| {
            // This iterates through the mle indices to check whether there is a
            // free bit within the fixed bits.
            let (_, contains_free_in_fixed) = mle.mle_indices().iter().fold(
                (false, false),
                |(free_seen_so_far, fixed_after_free_so_far), mle_idx| match mle_idx {
                    MleIndex::Free => (true, fixed_after_free_so_far),
                    MleIndex::Fixed(_) => (free_seen_so_far, free_seen_so_far),
                    _ => (free_seen_so_far, fixed_after_free_so_far),
                },
            );
            // If true, we split, otherwise, we don't.
            if contains_free_in_fixed {
                split_mle(&mle)
            } else {
                vec![mle]
            }
        })
        .collect()
}

/// Gets the index of the least significant bit (lsb) of the fixed bits out of a
/// vector of MLEs.
///
/// In other words, this is the MLE evaluation pertaining to the MLE with the
/// most fixed bits.
fn get_lsb_fixed_var<F: Field>(
    mles: &[MleEvaluationAndPrefixBits<F>],
) -> &MleEvaluationAndPrefixBits<F> {
    mles.iter()
        .max_by_key(|(_mle_evaluation, prefix_bits)| prefix_bits.len())
        .unwrap()
}

/// Given an MLE evaluation, and an option of a second MLE evaluation pair, this
/// combines the two together this assumes that the first MLE evaluation and the
/// second MLE evaluation are pairs, if the second MLE evaluation is a Some()
///
/// A pair consists of two MLE evaluations that match in every fixed bit except
/// for the least significant one. This is because we combine in the reverse
/// order that we split in terms of selectors, and we split in terms of
/// selectors by doing huffman (most significant bit).
///
/// Example: if mle_evaluation_first has fixed bits true, true, false, its pair
/// would have fixed bits true, true, true. When we combine them, the combined
/// MLE has fixed bits true, true. We also simultaneously update the combined
/// evaluation to use the challenge according to the index we combined at.
///
/// If there is no pair, then this is assumed to be an mle with all 0s.
fn combine_pair<F: Field>(
    mle_evaluation_first: F,
    maybe_mle_evaluation_second: Option<F>,
    prefix_vars_first: &[bool],
    chal_point: &[F],
) -> MleEvaluationAndPrefixBits<F> {
    // If the second mle is None, we assume its bookkeeping table is all zeros.
    // We are dealing with fully fixed mles, so we just use F::ZERO.
    let mle_evaluation_second = maybe_mle_evaluation_second.unwrap_or(F::ZERO);

    // Depending on whether the lsb fixed bit was true or false, we bind it to
    // the correct challenge point at this index this is either the challenge
    // point at the index, or one minus this value.
    let bound_coord = if !prefix_vars_first.last().unwrap() {
        F::ONE - chal_point[prefix_vars_first.len() - 1]
    } else {
        chal_point[prefix_vars_first.len() - 1]
    };

    // We compute the combined evaluation using the according index challenge
    // point.
    let new_eval =
        bound_coord * mle_evaluation_first + (F::ONE - bound_coord) * mle_evaluation_second;

    (
        new_eval,
        prefix_vars_first[..prefix_vars_first.len() - 1].to_vec(),
    )
}

/// Given a list of mles, the lsb fixed var index, and the MLE evaluation that
/// contributes to it, this will go through all of them and find its pair (if
/// none exists, we assume it is 0) and combine the two it will then update the
/// original list of MLEs to contain the combined MLE evaluation and remove the
/// original ones that were paired.
fn find_pair_and_combine<F: Field>(
    all_refs: &[MleEvaluationAndPrefixBits<F>],
    prefix_indices: &[bool],
    mle_evaluation: F,
    chal_point: &[F],
) -> Vec<MleEvaluationAndPrefixBits<F>> {
    // We want to compare all fixed bits except the one at the least significant
    // bit index.
    let indices_to_compare = &prefix_indices[0..prefix_indices.len() - 1];
    let mut mle_eval_pair = None;
    let mut all_refs_updated = Vec::new();

    for (mle_eval, mle_indices) in all_refs {
        let max_slice_idx = mle_indices.len();
        let compare_indices =
            &mle_indices[0..std::cmp::min(prefix_indices.len() - 1, max_slice_idx)];
        // We want to make sure we aren't combining an mle with itself!
        if (compare_indices == indices_to_compare) && (mle_indices != prefix_indices) {
            mle_eval_pair = Some(*mle_eval);
        } else if mle_indices != prefix_indices {
            all_refs_updated.push((*mle_eval, mle_indices.to_vec()));
        }
    }

    // Add the paired mle to the list and return this new updated list.
    let new_mle_to_add = combine_pair(mle_evaluation, mle_eval_pair, prefix_indices, chal_point);
    all_refs_updated.push(new_mle_to_add);
    all_refs_updated
}
