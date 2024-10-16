//! Module for generating and manipulating mles.

use std::iter::repeat_with;

use ark_std::test_rng;
use itertools::{repeat_n, Itertools};
use rand::Rng;
use remainder_shared_types::Field;

use crate::{
    layer::LayerId,
    mle::{betavalues::BetaValues, dense::DenseMle, evals::MultilinearExtension, MleIndex},
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

/// Returns the total MLE indices given a Vec<bool>
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
/// let mle1 = MultilinearExtension::new(vec![Fr::from(2)]);
/// let mle2 = MultilinearExtension::new(vec![Fr::from(1), Fr::from(3)]);
/// let result = build_composite_mle(&[(&mle1, &vec![false, true]), (&mle2, &vec![true])]);
/// assert_eq!(result.get_evals_vector(), &vec![Fr::from(0), Fr::from(1), Fr::from(2), Fr::from(3)]);
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
        mle.get_evals_vector()
            .iter()
            .enumerate()
            .for_each(|(idx, eval)| {
                let mut out_idx = 0;
                for (i, bit) in prefix_bits.iter().enumerate() {
                    out_idx |= (*bit as usize) << i;
                }
                out_idx |= idx << prefix_bits.len();
                out[out_idx] = *eval;
            });
    }
    MultilinearExtension::new(out)
}

/// A struct representing an iterator that iterates through
/// the range (0..2^{`num_bits`}) but in the ordering of a
/// Gray Code, which means that the edit distance between
/// the bit representation of any consecutive indices is only 1.
///
/// The iterator is of the type (u32, (u32, bool))
/// which represents:
/// (index, (index of the bit that was flipped, the previous value of the flipped bit.))
pub struct GrayCode {
    num_bits: usize,
    current_val: u32,
}

impl GrayCode {
    fn new(num_bits: usize) -> Self {
        GrayCode {
            num_bits,
            current_val: 0,
        }
    }
}

impl Iterator for GrayCode {
    type Item = (u32, (u32, bool));

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_val >= ((1 << self.num_bits) - 1) {
            return None;
        }

        // Mask current value to ensure we only get num_bits
        // number of bits per result.
        let mask = (1 << self.num_bits) - 1;

        // Because we don't store the previous value, just calculate it
        // using the current_val that's stored.
        let prev_gray = (self.current_val ^ (self.current_val >> 1)) & mask;
        // The next value is simply XOR of the counter incremented by 1 and itself
        // right-shifted.
        // (source: algorithm in Wikipedia and ChatGPT hehe)
        self.current_val += 1;
        let new_gray = (self.current_val ^ (self.current_val >> 1)) & mask;

        // Compute which bit was flipped.
        let flipped_bit = (prev_gray ^ new_gray).trailing_zeros();
        let previous_value = (prev_gray & (1 << flipped_bit)) != 0;

        // The new gray code index is already in little endian, so for our
        // current purposes, this is okay. For big endian, we need to reverse
        // the bits.
        Some((new_gray, (flipped_bit, previous_value)))
    }
}

/// This function non-destructively evaluates an MLE at a given point using
/// the gray codes iterator.
pub fn evaluate_mle_at_a_point<F: Field>(mle: &MultilinearExtension<F>, point: &[F]) -> F {
    let mle_num_vars = mle.num_vars();
    let mle_coefficients = mle.get_evals_vector();
    assert_eq!(point.len(), mle_num_vars);
    // The gray codes start at index 1, so we start with the first value which
    // is \widetilde{\beta}(\vec{0}, point).
    let starting_beta_value =
        BetaValues::compute_beta_over_two_challenges(&vec![F::ZERO; mle_num_vars], point);
    // This is the value that gets multiplied to the first MLE coefficient,
    // which is (1 - r_1) * (1 - r_2) * ... * (1 - r_n) where (r_1, ..., r_n) is the point.
    let starting_evaluation_acc = starting_beta_value * mle_coefficients[0];
    let gray_code = GrayCode::new(mle_num_vars);
    // We simply compute the correct inverse and new multiplicative term
    // for each bit that is flipped in the beta value, and accumulate these
    // by doing an element-wise multiplication with the correct index
    // of the MLE coefficients.
    let (_final_beta_value, evaluation) = gray_code.fold(
        (starting_beta_value, starting_evaluation_acc),
        |(prev_beta_value, evaluation_acc), (index, (flipped_bit_index, flipped_bit_value))| {
            // For every bit i that is flipped, if it used to be a 1,
            // then we multiply by r_i^{-1} and multiply by
            // (1 - r_i) to account for this bit flip.
            let next_beta_value = if flipped_bit_value {
                prev_beta_value
                    * point[flipped_bit_index as usize].invert().unwrap()
                    * (F::ONE - point[flipped_bit_index as usize])
            }
            // For every bit i that is flipped, if it used to be a 0,
            // then we multiply by (1 - r_i)^{-1} and multiply by
            // r_i to account for this bit flip.
            else {
                prev_beta_value
                    * (F::ONE - point[flipped_bit_index as usize])
                        .invert()
                        .unwrap()
                    * point[flipped_bit_index as usize]
            };
            // Multiply this by the appropriate MLE coefficient.
            let next_evaluation_acc = next_beta_value * mle_coefficients[index as usize];
            (next_beta_value, evaluation_acc + next_evaluation_acc)
        },
    );
    evaluation
}

#[cfg(test)]
mod tests {
    use ark_std::test_rng;
    use itertools::Itertools;
    use remainder_shared_types::{ff_field, Fr};

    use crate::mle::evals::MultilinearExtension;

    use super::evaluate_mle_at_a_point;

    fn evaluate_mle_destructive(mle: &mut MultilinearExtension<Fr>, point: &[Fr]) -> Fr {
        point.iter().for_each(|challenge| {
            mle.fix_variable(*challenge);
        });
        assert_eq!(mle.get_evals_vector().len(), 1);
        mle.get_evals_vector()[0]
    }

    #[test]
    fn test_evaluate_mle_at_a_point_1_variable() {
        let mut mle = MultilinearExtension::new(vec![Fr::ONE, Fr::from(2)]);
        let point = &[Fr::from(2)];
        let computed_evaluation = evaluate_mle_at_a_point(&mle, point);
        let expected_evaluation = evaluate_mle_destructive(&mut mle, point);
        assert_eq!(computed_evaluation, expected_evaluation);
    }

    #[test]
    fn test_evaluate_mle_at_a_point_2_variable() {
        let mut mle = MultilinearExtension::new(vec![Fr::ONE, Fr::from(2), Fr::ONE, Fr::from(2)]);
        let point = &[Fr::from(2), Fr::from(3)];
        let computed_evaluation = evaluate_mle_at_a_point(&mle, point);
        let expected_evaluation = evaluate_mle_destructive(&mut mle, point);
        assert_eq!(computed_evaluation, expected_evaluation);
    }

    #[test]
    fn test_evaluate_mle_at_a_point_3_variable_random() {
        let mut rng = test_rng();
        let mut mle = MultilinearExtension::new((0..8).map(|_| Fr::random(&mut rng)).collect());
        let point = &(0..3).map(|_| Fr::random(&mut rng)).collect_vec();
        let computed_evaluation = evaluate_mle_at_a_point(&mle, point);
        let expected_evaluation = evaluate_mle_destructive(&mut mle, point);
        assert_eq!(computed_evaluation, expected_evaluation);
    }
}
