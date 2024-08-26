//! Module for generating and manipulating mles, also includes a function to
//! generate the description of circuits.

use std::{fs, iter::repeat_with};

use ark_std::{log2, test_rng};
use itertools::{repeat_n, Itertools};
use rand::Rng;
use remainder_shared_types::{FieldExt, Fr, Poseidon};

use crate::{
    layer::{layer_enum::LayerEnum, LayerId},
    layouter::nodes::{
        circuit_inputs::{InputLayerNode, InputShred},
        Context,
    },
    mle::{
        dense::DenseMle,
        evals::{Evaluations, MultilinearExtension},
        MleIndex,
    },
    prover::layers::Layers,
};


pub mod arithmetic;

#[cfg(test)]
pub(crate) mod test_utils;

/// Returns an [InputShred] with the appropriate [MultilinearExtension], but given as input an mle_vec
pub fn get_input_shred_from_vec<F: FieldExt>(
    mle_vec: Vec<F>,
    ctx: &Context,
    input_node: &InputLayerNode<F>,
) -> InputShred<F> {
    assert!(mle_vec.len().is_power_of_two());
    InputShred::new(
        ctx,
        MultilinearExtension::new_from_evals(Evaluations::new(
            log2(mle_vec.len()) as usize,
            mle_vec,
        )),
        input_node,
    )
}

/// Returns a zero-padded version of `coeffs` with length padded
/// to the nearest power of two.
///
/// ## Arguments
///
/// * `coeffs` - The coefficients to be padded
///
/// ## Returns
///
/// * `padded_coeffs` - The coeffients, zero-padded to the nearest power of two
///   (in length)
pub fn pad_to_nearest_power_of_two<F: FieldExt>(coeffs: Vec<F>) -> Vec<F> {
    // --- No need to duplicate things if we're already a power of two! ---
    if coeffs.len().is_power_of_two() {
        return coeffs;
    }

    let num_padding = coeffs.len().checked_next_power_of_two().unwrap() - coeffs.len();
    coeffs
        .into_iter()
        .chain(repeat_with(|| F::ZERO).take(num_padding))
        .collect_vec()
}

/// Returns the argsort (i.e. indices) of the given vector slice.
///
/// Thanks ChatGPT!!!
pub fn argsort<T: Ord>(slice: &[T], invert: bool) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..slice.len()).collect();

    indices.sort_by(|&i, &j| {
        if invert {
            slice[j].cmp(&slice[i])
        } else {
            slice[i].cmp(&slice[j])
        }
    });

    indices
}

/// Helper function to create random MLE with specific number of vars
// pub fn get_random_mle<F: FieldExt>(num_vars: usize, rng: &mut impl Rng) ->
// DenseMle<F,> {
pub fn get_random_mle<F: FieldExt>(num_vars: usize, rng: &mut impl Rng) -> DenseMle<F> {
    let capacity = 2_u32.pow(num_vars as u32);
    let bookkeeping_table = repeat_with(|| F::from(rng.gen::<u64>()))
        .take(capacity as usize)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input(0))
}

/// Helper function to create random MLE with specific number of vars
pub fn get_range_mle<F: FieldExt>(num_vars: usize) -> DenseMle<F> {
    // let mut rng = test_rng();
    let capacity = 2_u32.pow(num_vars as u32);
    let bookkeeping_table = (0..capacity)
        .map(|idx| F::from(idx as u64 + 1))
        .take(capacity as usize)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input(0))
}

/// Helper function to create random MLE with specific length
pub fn get_random_mle_with_capacity<F: FieldExt>(capacity: usize) -> DenseMle<F> {
    let mut rng = test_rng();
    let bookkeeping_table = repeat_with(|| F::from(rng.gen::<u64>()))
        .take(capacity)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input(0))
}

/// Returns a vector of MLEs for dataparallel testing according to the number of variables and
/// number of dataparallel bits.
pub fn get_dummy_random_mle_vec<F: FieldExt>(
    num_vars: usize,
    num_dataparallel_bits: usize,
    rng: &mut impl Rng,
) -> Vec<DenseMle<F>> {
    (0..(1 << num_dataparallel_bits))
        .map(|_| {
            let mle_vec = (0..(1 << num_vars))
                .map(|_| F::from(rng.gen::<u64>()))
                .collect_vec();
            DenseMle::new_from_raw(mle_vec, LayerId::Input(0))
        })
        .collect_vec()
}

///returns an iterator that wil give permutations of binary bits of size
/// num_bits
///
/// 0,0,0 -> 0,0,1 -> 0,1,0 -> 0,1,1 -> 1,0,0 -> 1,0,1 -> 1,1,0 -> 1,1,1
pub(crate) fn bits_iter<F: FieldExt>(num_bits: usize) -> impl Iterator<Item = Vec<MleIndex<F>>> {
    std::iter::successors(
        Some(vec![MleIndex::<F>::Fixed(false); num_bits]),
        move |prev| {
            let mut prev = prev.clone();
            let mut removed_bits = 0;
            for index in (0..num_bits).rev() {
                let curr = prev.remove(index);
                if curr == MleIndex::Fixed(false) {
                    prev.push(MleIndex::Fixed(true));
                    break;
                } else {
                    removed_bits += 1;
                }
            }
            if removed_bits == num_bits {
                None
            } else {
                Some(
                    prev.into_iter()
                        .chain(repeat_n(MleIndex::Fixed(false), removed_bits))
                        .collect_vec(),
                )
            }
        },
    )
}

/// Returns the specific bit decomp for a given index,
/// using `num_bits` bits. Note that this returns the
/// decomposition in BIG ENDIAN!
pub fn get_mle_idx_decomp_for_idx<F: FieldExt>(idx: usize, num_bits: usize) -> Vec<MleIndex<F>> {
    (0..(num_bits))
        .rev()
        .map(|cur_num_bits| {
            let is_one =
                (idx % 2_usize.pow(cur_num_bits as u32 + 1)) >= 2_usize.pow(cur_num_bits as u32);
            MleIndex::Fixed(is_one)
        })
        .collect_vec()
}

/// Returns whether a particular file exists in the filesystem
///
/// TODO!(ryancao): Shucks does this check a relative path...?
pub fn file_exists(file_path: &String) -> bool {
    match fs::metadata(file_path) {
        Ok(file_metadata) => file_metadata.is_file(),
        Err(_) => false,
    }
}

/// Hashes the layers of a GKR circuit by calling their circuit descriptions
/// Returns one single Field element
pub fn hash_layers<F: FieldExt>(layers: &Layers<F, LayerEnum<F>>) -> F {
    let mut sponge: Poseidon<F, 3, 2> = Poseidon::new(8, 57);

    layers.layers.iter().for_each(|layer| {
        let item = format!("{}", layer.circuit_description_fmt());
        let bytes = item.as_bytes();
        let elements: Vec<F> = bytes
            .chunks(62)
            .map(|bytes| {
                let base = F::from(8);
                let first = bytes[0];
                bytes
                    .iter()
                    .skip(1)
                    .fold((F::from(first as u64), base), |(accum, power), byte| {
                        let accum = accum + (F::from(*byte as u64) * power);
                        let power = power * base;
                        (accum, power)
                    })
                    .0
            })
            .collect_vec();

        sponge.update(&elements);
    });

    sponge.squeeze()
}
