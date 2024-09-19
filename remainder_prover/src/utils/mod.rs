//! Module for useful functions
/// Helpful arithmetic functions.
pub mod arithmetic;
/// Helpful functions for manipulating ndarray Array objects (e.g. padding)
pub mod array;
/// Helpful functions for debugging.
pub mod debug;
/// Helpful functions for manipulating MLEs (e.g. padding).
pub mod mle;

#[cfg(test)]
/// Utilities that are only useful for tests
pub(crate) mod test_utils;

use std::{fs, iter::repeat_with};

use ark_std::test_rng;
use itertools::{repeat_n, Itertools};
use rand::Rng;
/// FIXME the functions below are uncategorized and probably should be moved to a more appropriate
/// module or submodule.
use remainder_shared_types::{Field, Poseidon};

use crate::{
    layer::{layer_enum::LayerEnum, LayerId},
    layouter::nodes::{
        circuit_inputs::{InputLayerNode, InputShred, InputShredData},
        CircuitNode, Context,
    },
    mle::{dense::DenseMle, evals::MultilinearExtension, MleIndex},
    prover::layers::Layers,
};

/// Using the number of variables, get an input shred that represents
/// this information.
pub fn get_input_shred_from_num_vars(
    num_vars: usize,
    ctx: &Context,
    input_node: &InputLayerNode,
) -> InputShred {
    InputShred::new(ctx, num_vars, input_node)
}

/// Using a data vector, get an [InputShred] which represents its
/// shape, along with [InputShredData] which represents the
/// corresponding data.
pub fn get_input_shred_and_data<F: Field>(
    mle_vec: Vec<F>,
    ctx: &Context,
    input_node: &InputLayerNode,
) -> (InputShred, InputShredData<F>) {
    assert!(mle_vec.len().is_power_of_two());
    let data = MultilinearExtension::new(mle_vec);
    let input_shred = InputShred::new(ctx, data.num_vars(), input_node);
    let input_shred_data = InputShredData::new(input_shred.id(), data);
    (input_shred, input_shred_data)
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
pub fn pad_to_nearest_power_of_two<F: Field>(coeffs: Vec<F>) -> Vec<F> {
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
/// # Example:
/// ```
/// use remainder::utils::argsort;
/// let data = vec![3, 1, 4, 1, 5, 9, 2];
///
/// // Ascending order
/// let indices = argsort(&data, false);
/// assert_eq!(indices, vec![1, 3, 6, 0, 2, 4, 5]);
/// ```
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

/// Helper function to create random MLE with specific number of variables.
pub fn get_random_mle<F: Field>(num_vars: usize, rng: &mut impl Rng) -> DenseMle<F> {
    let capacity = 2_u32.pow(num_vars as u32);
    let bookkeeping_table = repeat_with(|| F::from(rng.gen::<u64>()))
        .take(capacity as usize)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input(0))
}

/// Helper function to create random MLE with specific number of variables.
pub fn get_range_mle<F: Field>(num_vars: usize) -> DenseMle<F> {
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

/// Returns the total MLE indices given a Vec<bool>
/// for the prefix bits and then the number of iterated
/// bits after.
pub fn get_total_mle_indices<F: Field>(
    prefix_bits: &[bool],
    num_iterated_bits: usize,
) -> Vec<MleIndex<F>> {
    prefix_bits
        .iter()
        .map(|bit| MleIndex::Fixed(*bit))
        .chain(repeat_n(MleIndex::Iterated, num_iterated_bits))
        .collect()
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

/// Hashes the layers of a GKR circuit by calling their circuit descriptions
/// Returns one single Field element
pub fn hash_layers<F: Field>(layers: &Layers<F, LayerEnum<F>>) -> F {
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
            .collect::<Vec<_>>();

        sponge.update(&elements);
    });

    sponge.squeeze()
}
