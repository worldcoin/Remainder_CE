//! Module for generating and manipulating mles.

use std::{fs, iter::repeat_with};

use ark_std::{log2, test_rng};
use itertools::{repeat_n, Itertools};
use rand::Rng;
use remainder_shared_types::{Field, Fr, Poseidon};

use crate::{
    layer::{layer_enum::LayerEnum, LayerId},
    layouter::nodes::{
        circuit_inputs::{InputLayerNode, InputShred, InputShredData},
        Context,
    },
    mle::{
        dense::DenseMle,
        evals::{Evaluations, MultilinearExtension},
        MleIndex,
    },
    prover::layers::Layers,
};

/// Return a vector containing a padded version of the input data, with the padding value at the end
/// of the vector, such that the length is `data.len().next_power_of_two()`.  This is a no-op if the
/// length is already a power of two.
/// # Examples:
/// ```
/// use remainder::utils::pad_with;
/// let data = vec![1, 2, 3];
/// let padded_data = pad_with(0, &data);
/// assert_eq!(padded_data, vec![1, 2, 3, 0]);
/// assert_eq!(pad_with(0, &padded_data), vec![1, 2, 3, 0]); // length is already a power of two.
/// ```
pub fn pad_with<F: Clone>(padding_value: F, data: &[F]) -> Vec<F> {
    let padded_length = data.len().checked_next_power_of_two().unwrap();
    let mut padded_data = Vec::with_capacity(padded_length);
    padded_data.extend_from_slice(data);
    padded_data.extend(std::iter::repeat(padding_value).take(padded_length - data.len()));
    padded_data
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
pub fn pad_to_nearest_power_of_two<F: Field>(coeffs: &[F]) -> Vec<F> {
    pad_with(F::ZERO, coeffs)
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
// pub fn get_random_mle<F: Field>(num_vars: usize, rng: &mut impl Rng) ->
// DenseMle<F,> {
pub fn get_random_mle<F: Field>(num_vars: usize, rng: &mut impl Rng) -> DenseMle<F> {
    let capacity = 2_u32.pow(num_vars as u32);
    let bookkeeping_table = repeat_with(|| F::from(rng.gen::<u64>()))
        .take(capacity as usize)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input(0))
}

/// Helper function to create random MLE with specific number of vars
pub fn get_range_mle<F: Field>(num_vars: usize) -> DenseMle<F> {
    // let mut rng = test_rng();
    let capacity = 2_u32.pow(num_vars as u32);
    let bookkeeping_table = (0..capacity)
        .map(|idx| F::from(idx as u64 + 1))
        .take(capacity as usize)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input(0))
}

/// Helper function to create random MLE with specific length
pub fn get_random_mle_with_capacity<F: Field>(capacity: usize) -> DenseMle<F> {
    let mut rng = test_rng();
    let bookkeeping_table = repeat_with(|| F::from(rng.gen::<u64>()))
        .take(capacity)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input(0))
}

/// Returns a vector of MLEs for dataparallel testing according to the number of variables and
/// number of dataparallel bits.
pub fn get_dummy_random_mle_vec<F: Field>(
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
pub(crate) fn bits_iter<F: Field>(num_bits: usize) -> impl Iterator<Item = Vec<MleIndex<F>>> {
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
pub fn get_mle_idx_decomp_for_idx<F: Field>(idx: usize, num_bits: usize) -> Vec<MleIndex<F>> {
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
