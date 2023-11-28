use ark_std::{cfg_iter_mut, cfg_into_iter};
use rayon::{iter::{IntoParallelRefMutIterator, IntoParallelIterator}, prelude::{ParallelIterator, IndexedParallelIterator}};
use itertools::{Itertools, repeat_n};
use remainder_shared_types::FieldExt;
use thiserror::Error;
use crate::mle::{mle_enum::MleEnum, MleRef, MleIndex, dense::{DenseMleRef, DenseMle}};
use ark_std::log2;

use super::LayerId;


/// Error handling for gate mle construction
#[derive(Error, Debug, Clone)]
pub enum CombineMleRefError {
    #[error("we have not fully combined all the mle refs because the list size is > 1")]
    NotFullyCombined,
    #[error("we have an mle ref that is not fully fixed even after fixing on the challenge point")]
    MleRefNotFullyFixed,
}

/// this function takes an mle ref that has an iterated bit in between a bunch of fixed bits
/// and it splits it into two mle refs, one where the iterated bit is replaced with
/// fixed(false), and the other where it is replaced with fixed(true). this ensures that all
/// the fixed bits are contiguous
/// NOTE we assume that this function is called on an mle ref that has an iterated bit within 
/// a bunch of fixed bits (note how it is used in the `collapse_mles_with_iterated_in_prefix` 
/// function)
fn split_mle_ref<F: FieldExt>(
    mle_ref: MleEnum<F>
) -> Vec<MleEnum<F>> {

    // get the index of the first iterated bit in the mle ref 
    let first_iterated_idx: usize = 
        mle_ref.original_mle_indices().iter().enumerate().fold(
            mle_ref.original_mle_indices().len(),
            |acc, (idx, mle_idx)| {
                if let MleIndex::Iterated = mle_idx {
                    std::cmp::min(acc, idx)
                }
                else {
                    acc
                }
            }
        );

    // compute the correct original indices, we have the first one be false, the second one as true instead of the iterated bit
    let first_og_indices = mle_ref.original_mle_indices()[0..first_iterated_idx].into_iter().cloned().chain(
        std::iter::once(MleIndex::Fixed(false))).chain(
            mle_ref.original_mle_indices()[first_iterated_idx + 1..].into_iter().cloned()).collect_vec();
    let second_og_indices = mle_ref.original_mle_indices()[0..first_iterated_idx].into_iter().cloned().chain(
        std::iter::once(MleIndex::Fixed(true))).chain(
            mle_ref.original_mle_indices()[first_iterated_idx + 1..].into_iter().cloned()).collect_vec();


    // depending on whether this is a zero mle ref or dense mle ref, construct the first mle_ref in the pair
    let first_mle_ref = {
        match mle_ref.clone() {
            MleEnum::Dense(dense_mle_ref) => {
                MleEnum::Dense(
                    DenseMleRef {
                    bookkeeping_table: dense_mle_ref.bookkeeping_table.clone(),
                    original_bookkeeping_table: dense_mle_ref.original_bookkeeping_table.clone().into_iter().step_by(2).collect_vec(),
                    mle_indices: dense_mle_ref.mle_indices.clone(),
                    original_mle_indices: first_og_indices,
                    num_vars: dense_mle_ref.num_vars,
                    original_num_vars: dense_mle_ref.original_num_vars,
                    layer_id: dense_mle_ref.layer_id,
                    indexed: false,
                })
            }
            MleEnum::Zero(mut zero_mle_ref) => { 
                zero_mle_ref.original_mle_indices = first_og_indices;
                MleEnum::Zero(
                    zero_mle_ref
                )
            }
        }
        
    };

    // second mle ref in the pair
    let second_mle_ref = {
        match mle_ref {
            MleEnum::Dense(dense_mle_ref) => {
                MleEnum::Dense(
                    DenseMleRef {
                        bookkeeping_table: dense_mle_ref.bookkeeping_table,
                        original_bookkeeping_table: dense_mle_ref.original_bookkeeping_table.into_iter().skip(1).step_by(2).collect_vec(),
                        mle_indices: dense_mle_ref.mle_indices,
                        original_mle_indices: second_og_indices,
                        num_vars: dense_mle_ref.num_vars,
                        original_num_vars: dense_mle_ref.original_num_vars,
                        layer_id: dense_mle_ref.layer_id,
                        indexed: false,
                })
            }
            MleEnum::Zero(mut zero_mle_ref) => { 
                zero_mle_ref.original_mle_indices = second_og_indices;
                MleEnum::Zero(
                    zero_mle_ref
                )
            }
        }
        
    };

    vec![first_mle_ref, second_mle_ref]

}

/// this function will take a list of mle refs and update the list to contain mle_refs where all fixed bits are contiguous
fn collapse_mles_with_iterated_in_prefix<F: FieldExt> (
    mle_refs: &Vec<MleEnum<F>>,
) -> Vec<MleEnum<F>> {
    mle_refs.into_iter().flat_map(
        |mle_ref| {
            // this iterates through the mle indices to check whether there is an iterated bit within the fixed bits
            let (_, check_iterated_within_fixed) = mle_ref.original_mle_indices().iter().fold((false, false), 
                |(iterated_seen_so_far, fixed_after_iterated_so_far), mle_idx| {
                    match mle_idx {
                        MleIndex::Iterated => (true, fixed_after_iterated_so_far),
                        MleIndex::Fixed(_) => (iterated_seen_so_far, iterated_seen_so_far),
                        _ => (iterated_seen_so_far, fixed_after_iterated_so_far),
                    }
                }
            );
            // if true, we split, otherwise, we don't
            if check_iterated_within_fixed {
                split_mle_ref(mle_ref.clone())
            }
            else {
                vec![mle_ref.clone()]
            }
        }
    ).collect_vec()
}

/// gets the index of the least significant bit (lsb) of the fixed bits out of a vector of mle refs
/// in other words, this computes the lsb fixed bit in each mle ref, and then returns the max of those
/// 
/// returns a tuple of an option of the index of the least significant bit and an option of the mle
/// ref that contributes to this lsb
/// if there are no fixed bits in any of the mle refs, it returns a `(None, None)` tuple
fn get_lsb_fixed_var<F: FieldExt>(
    mle_refs: &Vec<MleEnum<F>>
) -> (Option<usize>, Option<MleEnum<F>>) {
    mle_refs.into_iter().fold(
        (None, None),
        |(acc_idx, acc_mle), mle_ref| {

            // this grabs the least significant bit of the fixed bits within each mle
            let lsb_within_mle = mle_ref.original_mle_indices().iter().enumerate().fold(
                None, 
                |acc, (idx_num, mle_idx)| {
                    if let MleIndex::Fixed(_) = mle_idx {
                        Some(idx_num)
                    } else {
                        acc
                    }
                }
            );

            // this computes the maximum of those lsb's, along with keeping track of the mle_ref that contributes to it
            if let Some(idx_num) = acc_idx {
                if let Some(max_within_mle) = lsb_within_mle {
                    if max_within_mle > idx_num {
                        (lsb_within_mle, Some(mle_ref.clone()))
                    }
                    else {
                        (acc_idx, acc_mle)
                    }
                }
                else {
                    (Some(idx_num), acc_mle)
                }
            }
            else {
                (lsb_within_mle, Some(mle_ref.clone()))
            }
        }
    )
}

/// given an mle ref, and an option of a second mle ref, this combines the two together
/// this assumes that the first mle ref and the second mle ref are pairs, if the second
/// mle ref is a Some()
/// 
/// a pair consists of two mle refs that match in every fixed bit except for the least
/// significant one. this is because we combine in the reverse order that we split in 
/// terms of selectors, and we split in terms of selectors by doing huffman (most significant bit)
/// 
/// example: if mle_ref_first has fixed bits true, true, false, its pair would have fixed bits
/// true, true, true. when we combine them, the combined mle ref has fixed bits true, true, and then 
/// the third fixed it is replaced with a bound bit, since we simultaneously fix that bit
/// to the correct index in the challenge point.
/// 
/// if there is no pair, then this is assumed to be an mle_ref with all 0s. 
fn combine_pair<F: FieldExt>(
    mle_ref_first: MleEnum<F>,
    mle_ref_second: Option<MleEnum<F>>,
    lsb_idx: usize,
    chal_point: &Vec<F>,
) -> DenseMleRef<F> {
    
    let mle_ref_first_bt = mle_ref_first.bookkeeping_table().to_vec();

    // if the second mle ref is None, we assume its bookkeeping table is all zeros. we are dealing with 
    // fully fixed mle_refs, so this bookkeeping table size is just 1
    let mle_ref_second_bt = {
        if mle_ref_second.clone().is_none() {
            vec![F::zero()]
        }
        else {
            mle_ref_second.clone().unwrap().bookkeeping_table().to_vec()
        }
    };

    // recomputes the mle indices, which now reflect that that we are binding the bit in the least significant bit fixed bit index
    let interleaved_mle_indices = mle_ref_first.mle_indices()[0..lsb_idx].into_iter().cloned().chain(
        std::iter::once(MleIndex::Bound(chal_point[lsb_idx], lsb_idx))).chain(
            mle_ref_first.mle_indices()[lsb_idx + 1..].into_iter().cloned()).collect_vec();

    let interleaved_mle_indices_og = mle_ref_first.original_mle_indices()[0..lsb_idx].into_iter().cloned().chain(
        std::iter::once(MleIndex::Iterated)).chain(
            mle_ref_first.original_mle_indices()[lsb_idx + 1..].into_iter().cloned()).collect_vec();

    // depending on whether the lsb fixed bit was true or false, we bind it to the correct challenge point at this index 
    // this is either the challenge point at the index, or one minus this value

    let bound_coord = if let MleIndex::Fixed(false) = mle_ref_first.original_mle_indices()[lsb_idx] {
        F::one() - chal_point[lsb_idx]
    }
    else {
        chal_point[lsb_idx]
    };

    // the new bookkeeping table also only has size one, but now reflects that we have another bound index
    let new_bt = vec![bound_coord * mle_ref_first_bt[0] + (F::one() - bound_coord) * mle_ref_second_bt[0]];

    // construct the dense mle ref that we return. note that even if we are pairing zero mle refs, we just return a dense mle ref here
    //
    // TODO!(vishady) also this is factually incorrect info lol because the original bookkeeping table is just wrong but 
    // it is kind of dumb to recompute it because we don't use it anymore. ideally these would be stored somewhere else so we don't 
    // have to keep catering to the fields we don't need ?
    let res = DenseMleRef {
        bookkeeping_table: new_bt.clone(),
        original_bookkeeping_table: new_bt,
        mle_indices: interleaved_mle_indices,
        original_mle_indices: interleaved_mle_indices_og,
        num_vars: mle_ref_first.num_vars(),
        original_num_vars: mle_ref_first.original_num_vars(),
        layer_id: mle_ref_first.get_layer_id(),
        indexed: false,
    };
    res

}

/// given a list of mle refs, the lsb fixed var index, and the mle ref that contributes to it, this will go through all of them
/// and find its pair (if none exists, that's fine) and combine the two
/// it will then update the original list of mle refs to contain the combined mle ref and remove the original ones that were paired
fn find_pair_and_combine<F: FieldExt> (
    all_refs: &Vec<MleEnum<F>>,
    lsb_idx: usize,
    mle_ref_of_lsb: MleEnum<F>,
    chal_point: &Vec<F>,
) -> Vec<MleEnum<F>> {

    // we want to compare all fixed bits except the one at the least significant bit index
    let indices_to_compare = mle_ref_of_lsb.original_mle_indices()[0..lsb_idx].to_vec();
    let og_indices = &mle_ref_of_lsb.original_mle_indices();
    let mut mle_ref_pair = None;
    let mut all_refs_updated = Vec::new();

    for mle_ref in all_refs {
        let max_slice_idx = mle_ref.original_mle_indices().len();
        let compare_indices = mle_ref.original_mle_indices()[0..std::cmp::min(lsb_idx, max_slice_idx)].to_vec();
        // we want to make sure we aren't combining an mle_ref with itself!
        if (compare_indices == indices_to_compare) && (&mle_ref.original_mle_indices() != og_indices) {
            mle_ref_pair = Some(mle_ref.clone());
        }
        // we also want to check whether we should add it back to this new updated list, it will always be added here unless
        // the mle_ref contributes to the pair
        if &mle_ref.original_mle_indices() != og_indices {
            if mle_ref_pair.is_some() {
                if mle_ref.original_mle_indices() != mle_ref_pair.clone().unwrap().original_mle_indices() {
                    all_refs_updated.push(mle_ref.clone());
                }
            }
            else {
                all_refs_updated.push(mle_ref.clone());
            }
        }
    }

    // add the paired mle ref to the list and return this new updated list
    let new_mle_ref_to_add = combine_pair(mle_ref_of_lsb, mle_ref_pair, lsb_idx, chal_point);
    all_refs_updated.push(MleEnum::Dense(new_mle_ref_to_add));
    all_refs_updated
}

/// this fixes mle refs with shared points in the claims so that we don't have to keep doing them
pub fn pre_fix_mle_refs<F: FieldExt>(
    mle_refs: &mut Vec<MleEnum<F>>,
    chal_point: &Vec<F>,
    common_idx: Vec<usize>,
) {
    cfg_iter_mut!(mle_refs).for_each(
        |mle_ref| {
            common_idx.iter().for_each(
                |chal_idx| {
                    if let MleIndex::IndexedBit(idx_bit_num) = mle_ref.mle_indices()[*chal_idx] {
                        mle_ref.fix_variable_at_index(idx_bit_num, chal_point[*chal_idx]);
        }});
    });
}


/// function that prepares all the mle refs to be fixed, then combined. this involves filtering out for
/// unique original mle indices, then splitting the mles with iterated bits within prefix bits, then 
/// indexing them so that their mutable bookkeeping table is the original bookkeeping table.
pub fn get_og_mle_refs<F: FieldExt>(
    mle_refs: Vec<MleEnum<F>>,
) -> Vec<MleEnum<F>> {

    // first we want to filter out for mle_refs that are duplicates. we look at their original indices
    // instead of their bookkeeping tables because sometimes two mle_refs can have the same original_bookkeeping_table
    // but have different prefix bits. if they have the same prefix bits, they must be duplicates.
    let mle_refs = mle_refs.into_iter().unique_by(|mle_ref| {
        match mle_ref {
            MleEnum::Dense(dense_mle_ref) => { dense_mle_ref.original_mle_indices.clone() }
            MleEnum::Zero(zero_mle_ref) => { zero_mle_ref.original_mle_indices.clone() }
        }
    }).collect_vec();

    // then, we split all the mle_refs with an iterated bit within the prefix bits
    let mle_refs_split = collapse_mles_with_iterated_in_prefix(&mle_refs);

    // go through and create mle refs that have original bookkeeping tables, and index them so that they can be fixed later
    let mle_ref_fix = cfg_into_iter!(mle_refs_split).map(
        |mle_ref| {
            match mle_ref {
                MleEnum::Dense(dense_mle_ref) => {
                    let mut mle_ref_og = DenseMleRef {
                        bookkeeping_table: dense_mle_ref.original_bookkeeping_table.clone(),
                        original_bookkeeping_table: dense_mle_ref.original_bookkeeping_table.clone(),
                        mle_indices: dense_mle_ref.original_mle_indices.clone(),
                        original_mle_indices: dense_mle_ref.original_mle_indices.clone(),
                        num_vars: dense_mle_ref.original_num_vars,
                        original_num_vars: dense_mle_ref.original_num_vars,
                        layer_id: dense_mle_ref.get_layer_id(),
                        indexed: false,
                    };
                    mle_ref_og.index_mle_indices(0);
                    MleEnum::Dense(mle_ref_og)
                }
                zero => zero
            }
        }
    );

    let mut ret_mles: Vec<MleEnum<F>> = vec![];
    mle_ref_fix.collect_into_vec(&mut ret_mles);
    ret_mles

}

/// this function takes in a list of mle refs, a challenge point we want to combine them under, and returns 
/// the final value in the bookkeeping table of the combined mle_ref. 
/// this is equivalent to combining all of these mle refs according to their prefix bits, and then fixing
/// variable on this combined mle ref using the challenge point
/// instead, we fix variable as we combine as this keeps the bookkeeping table sizes at one and is faster to compute
pub fn combine_mle_refs_with_aggregate<F: FieldExt>(
    mle_refs: &Vec<MleEnum<F>>,
    chal_point: &Vec<F>,
) -> Result<F, CombineMleRefError> {

    // we go through all of the mle_refs and fix variable in all of them given the indexed indices they already have
    // so that they are fully bound. 
    let fix_var_mle_refs = mle_refs.into_iter().map(
        |mle_ref| {
            match mle_ref.clone() {
                MleEnum::Dense(mut dense_mle_ref) => {
                    dense_mle_ref.mle_indices.clone().into_iter().enumerate().for_each(
                        |(idx, mle_idx)| {
                            if let MleIndex::IndexedBit(idx_num) = mle_idx {
                                dense_mle_ref.fix_variable(idx_num, chal_point[idx]);
                            }
                        }
                    );
                    MleEnum::Dense(dense_mle_ref)
                }
                zero => zero
            }
        }
    ).collect_vec();

    // mutable variable that is overwritten every time we combine mle refs
    let mut updated_list = fix_var_mle_refs;

    // an infinite loop that breaks when all the mle refs no longer have any fixed bits and only have iterated bits
    loop {
        // we first get the lsb fixed bit and the mle_ref that contributes to it
        let (lsb_fixed_var_opt, mle_ref_to_pair_opt) = get_lsb_fixed_var(&updated_list); 
        
        // this is only none of all the bits in all of the mle refs are iterated, so we are done with the combining process
        if lsb_fixed_var_opt.is_none() {
            break
        }

        // now we know they are not none, so unwrap and overwrite updated_list with the combined mle_ref
        let (lsb_fixed_var, mle_ref_to_pair) = (lsb_fixed_var_opt.unwrap(), mle_ref_to_pair_opt.unwrap()); // change to error
        updated_list = find_pair_and_combine(&updated_list, lsb_fixed_var, mle_ref_to_pair, chal_point);
    }

    // the list now should only have one combined mle ref because we removed duplicates, and its bookkeeping table should only have one value

    if updated_list.len() > 1 {
        return Err(CombineMleRefError::NotFullyCombined);
    }
    if updated_list[0].bookkeeping_table().len() != 1 {
        return Err(CombineMleRefError::MleRefNotFullyFixed);
    }

    Ok(updated_list[0].bookkeeping_table()[0])

}


/// for input layer stuff, combining refs together
/// Takes the individual bookkeeping tables from the MleRefs within an MLE
/// and merges them with padding, using a little-endian representation
/// merge strategy. Assumes that ALL MleRefs are the same size.
pub fn combine_mle_refs<F: FieldExt>(items: Vec<DenseMleRef<F>>) -> DenseMle<F, F> {

    let num_fields = items.len();

    // --- All the items within should be the same size ---
    let max_size = items
        .iter()
        .map(|mle_ref| mle_ref.bookkeeping_table.len())
        .max().unwrap();

    let part_size = 1 << log2(max_size);
    let part_count = 2_u32.pow(log2(num_fields)) as usize;

    // --- Number of "part" slots which need to filled with padding ---
    let padding_count = part_count - num_fields;
    let total_size = part_size * part_count;
    let total_padding: usize = total_size - max_size * part_count;

    let result = (0..max_size)
        .flat_map(|index| {
            items
                .iter()
                .map(move |item| *item.bookkeeping_table().get(index).unwrap_or(&F::zero()))
                .chain(repeat_n(F::zero(), padding_count))
        })
        .chain(repeat_n(F::zero(), total_padding))
        .collect_vec();

    DenseMle::new_from_raw(result, LayerId::Input(0), None)
}