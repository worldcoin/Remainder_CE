use std::iter::repeat_with;

use itertools::Itertools;
use lcpc_2d::FieldExt;

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