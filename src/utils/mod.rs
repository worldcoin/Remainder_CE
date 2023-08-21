use std::iter::repeat_with;

use ark_std::test_rng;
use itertools::Itertools;
use lcpc_2d::FieldExt;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::{mle::dense::DenseMle, layer::LayerId};

/// Returns a zero-padded version of `coeffs` with length padded
/// to the nearest power of two.
/// 
/// ## Arguments
/// 
/// * `coeffs` - The coefficients to be padded
/// 
/// ## Returns
/// 
/// * `padded_coeffs` - The coeffients, zero-padded to the nearest power of two (in length)
pub fn pad_to_nearest_power_of_two<F: FieldExt>(coeffs: Vec<F>) -> Vec<F> {

    // --- No need to duplicate things if we're already a power of two! ---
    if coeffs.len().is_power_of_two() {
        return coeffs;
    }

    let num_padding = coeffs.len().checked_next_power_of_two().unwrap() - coeffs.len();
    coeffs.into_iter().chain(
        repeat_with(|| F::zero()).take(num_padding)
    ).collect_vec()
}

/// Returns the argsort (i.e. indices) of the given vector slice.
/// 
/// Thanks ChatGPT!!!
pub fn argsort<T: Ord>(slice: &[T], invert: bool) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..slice.len()).collect();

    indices.sort_by(|&i, &j| {
        if invert {slice[j].cmp(&slice[i])} else {slice[i].cmp(&slice[j])}
    });

    indices
}

/// Helper function to create random MLE with specific number of vars
pub fn get_random_mle<F: FieldExt>(num_vars: usize) -> DenseMle<F, F>
where
    Standard: Distribution<F>,
{
    let mut rng = test_rng();
    let capacity = 2_u32.pow(num_vars as u32);
    let bookkeeping_table = repeat_with(|| rng.gen::<F>())
        .take(capacity as usize)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input, None)
}

/// Helper function to create random MLE with specific number of vars
pub fn get_range_mle<F: FieldExt>(num_vars: usize) -> DenseMle<F, F>
where
    Standard: Distribution<F>,
{
    // let mut rng = test_rng();
    let capacity = 2_u32.pow(num_vars as u32);
    let bookkeeping_table = (0..capacity).map(|idx| F::from(idx + 1))
        .take(capacity as usize)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input, None)
}

/// Helper function to create random MLE with specific length
pub fn get_random_mle_with_capacity<F: FieldExt>(capacity: usize) -> DenseMle<F, F>
where
    Standard: Distribution<F>,
{
    let mut rng = test_rng();
    let bookkeeping_table = repeat_with(|| rng.gen::<F>())
        .take(capacity as usize)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input, None)
}