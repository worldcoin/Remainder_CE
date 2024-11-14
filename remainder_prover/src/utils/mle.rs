//! Module for generating and manipulating mles.

use std::iter::repeat_with;

use itertools::{repeat_n, Itertools};
use rand::Rng;
use remainder_shared_types::Field;

use crate::{
    layer::LayerId,
    mle::{dense::DenseMle, evals::MultilinearExtension, MleIndex},
};

/// Return a vector containing a padded version of the input data, with the padding value at the end
/// of the vector, such that the length is `data.len().next_power_of_two()`.  This is a no-op if the
/// length is already a power of two.
/// # Examples:
/// ```
/// use remainder::utils::mle::pad_with;
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

/// Returns the argsort (i.e. indices) of the given vector slice.
///
/// ## Example:
/// ```
/// use remainder::utils::mle::argsort;
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

/// Helper function to create random MLE with specific number of vars
pub fn get_random_mle<F: Field>(num_vars: usize, rng: &mut impl Rng) -> DenseMle<F> {
    let capacity = 2_u32.pow(num_vars as u32);
    let bookkeeping_table = repeat_with(|| F::from(rng.gen::<u64>()) * F::from(rng.gen::<u64>()))
        .take(capacity as usize)
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

/// Returns the total MLE indices given a `Vec<bool>`.
/// for the prefix bits and then the number of free
/// bits after.
pub fn get_total_mle_indices<F: Field>(
    prefix_bits: &[bool],
    num_free_bits: usize,
) -> Vec<MleIndex<F>> {
    prefix_bits
        .iter()
        .map(|bit| MleIndex::Fixed(*bit))
        .chain(repeat_n(MleIndex::Free, num_free_bits))
        .collect()
}

/// Construct a parent MLE for the given MLEs and prefix bits, where the prefix bits of each MLE specify how it should be inserted into the parent.
/// Entries left unspecified are filled with `F::ZERO`.
/// # Requires:
/// * that the number of variables in each MLE, plus the number of its prefix bits, is the same across all pairs; this will be the number of variables in the returned MLE.
/// * the slice is non-empty.
///
/// # Example:
/// ```
/// use remainder::utils::mle::build_composite_mle;
/// use remainder::mle::evals::MultilinearExtension;
/// use remainder_shared_types::Fr;
/// use itertools::{Itertools};
/// let mle1 = MultilinearExtension::new(vec![Fr::from(2)]);
/// let mle2 = MultilinearExtension::new(vec![Fr::from(1), Fr::from(3)]);
/// let result = build_composite_mle(&[(&mle1, &vec![false, true]), (&mle2, &vec![true])]);
/// assert_eq!(*result.f.iter().collect_vec().clone(), vec![Fr::from(0), Fr::from(2), Fr::from(1), Fr::from(3)]);
/// ```
pub fn build_composite_mle<F: Field>(
    mles: &[(&MultilinearExtension<F>, &Vec<bool>)],
) -> MultilinearExtension<F> {
    assert!(!mles.is_empty());
    let out_num_vars = mles[0].0.num_vars() + mles[0].1.len();
    // Check that all (MLE, prefix bit) pairs require the same total number of variables.

    mles.iter().for_each(|(mle, prefix_bits)| {
        assert_eq!(mle.num_vars() + prefix_bits.len(), out_num_vars);
    });
    let mut out = vec![F::ZERO; 1 << out_num_vars];
    for (mle, prefix_bits) in mles {
        let mut current_window = 1 << out_num_vars;
        let starting_index = prefix_bits.iter().fold(0, |acc_index, bit| {
            let starting_index_acc = if *bit {
                acc_index + current_window / 2
            } else {
                acc_index
            };
            current_window /= 2;
            starting_index_acc
        });
        assert_eq!(current_window, mle.len());
        (starting_index..(starting_index + current_window))
            .enumerate()
            .for_each(|(mle_idx, out_idx)| {
                out[out_idx] = mle.get(mle_idx).unwrap_or(F::ZERO);
            });
    }
    MultilinearExtension::new(out)
}
