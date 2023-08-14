use std::iter::repeat_with;

use ark_std::log2;
use itertools::Itertools;

use crate::{
    mle::{dense::DenseMle, Mle, MleIndex},
};

use lcpc_2d::FieldExt;

use super::LayerId;

/// Input layer struct containing the information we need to 
/// 
/// a) Aggregate the input layer MLEs into a single large DenseMle<F, F>
/// 
/// b) Aggregate claims on the input layer
/// 
/// c) Evaluate the final claim on the input layer
pub struct InputLayer<F: FieldExt> {
    combined_dense_mle: DenseMle<F, F>,
}

impl<F: FieldExt> InputLayer<F> {

    /// Creates a new InputLayer from a bunch of MLEs which belong
    /// in the input layer by merging them.
    pub fn new_from_mles(input_mles: &mut Vec<Box<&mut dyn Mle<F>>>) -> Self {
        Self {
            combined_dense_mle: combine_input_mles(input_mles)
        }
    }

    /// Creates an empty InputLayer
    pub fn new() -> Self {
        Self {
            combined_dense_mle: DenseMle::new_from_raw(vec![], LayerId::Input, None)
        }
    }

    /// Getter for the DenseMLE making up the input layer
    pub fn get_combined_mle(&self) -> &DenseMle<F, F> {
        &self.combined_dense_mle
    }
}

/// Exactly what it says
fn round_to_next_largest_power_of_2(x: usize) -> u32 {
    2_u32.pow(log2(x))
}

/// Returns the padded bookkeeping table of the given MLE
fn get_padded_bookkeeping_table<F: FieldExt>(mle: &DenseMle<F, F>) -> Vec<F> {
    // --- Amount of zeros we need to add ---
    let padding_amt = 2_usize.pow(mle.num_vars() as u32) - mle.mle.len();

    mle.mle
        .clone()
        .into_iter()
        .chain(repeat_with(|| F::zero()).take(padding_amt))
        .collect_vec()
}

/// Returns the vector of prefix bits corresponding to a particular capacity
/// Thanks ChatGPT! -- Ryan
fn get_prefix_bits_from_capacity<F: FieldExt>(
    capacity: u32,
    total_num_bits: usize,
    num_iterated_bits: usize,
) -> Vec<MleIndex<F>> {
    (0..total_num_bits - num_iterated_bits)
        .map(|bit_position| {
            let bit_val = (capacity >> (total_num_bits - bit_position - 1)) & 1;
            MleIndex::Fixed(bit_val == 1)
        })
        .collect()
}

/// Takes in a list of input MLEs to be combined into a single
/// MLE via a greedy algorithm. Additionally, modifies the list
/// of `input_mles` with prefix bits such that each is appropriately
/// prefixed within the single `DenseMle<F, F>` which is returned
/// representing the combined MLE of the layer.
///
/// The algorithm used here is a simple greedy algorithm.
/// TODO!(ryancao): Ensure that this is optimal
/// TODO!(ryancao): Do we need to take in a mutable reference to `input_mles`???
///
/// # Arguments
///
/// * `input_mles` - The list of disjointed input chunks to combine.
///
/// # Examples
/// ```
/// 
/// ```
pub fn combine_input_mles<F: FieldExt>(
    input_mles: &mut Vec<Box<&mut dyn Mle<F>>>,
) -> DenseMle<F, F> {
    // --- First, just sort the `input_mles` by number of variables (but inverted) ---
    input_mles.sort_by(|a, b| b.num_vars().partial_cmp(&a.num_vars()).unwrap());

    // --- Get the total needed capacity by rounding the raw capacity up to the nearest power of 2 ---
    let raw_needed_capacity = input_mles.into_iter().fold(0, |prev, input_mle| {
        prev + 2_usize.pow(input_mle.num_vars() as u32)
    });
    let padded_needed_capacity = round_to_next_largest_power_of_2(raw_needed_capacity);
    let total_num_vars = log2(padded_needed_capacity as usize) as usize;

    // --- Next, grab their bookkeeping tables and combine them ---
    let initial_vec: Vec<F> = vec![];
    let mut current_padded_usage: u32 = 0;
    let final_bookkeeping_table =
        input_mles
            .into_iter()
            .fold(initial_vec, |current_bookkeeping_table, input_mle| {
                // --- Grab the prefix bits and add them to the individual MLEs ---
                let prefix_bits: Vec<MleIndex<F>> = get_prefix_bits_from_capacity(
                    current_padded_usage,
                    total_num_vars,
                    input_mle.num_vars(),
                );
                input_mle.add_prefix_bits(Some(prefix_bits));
                current_padded_usage += 2_u32.pow(input_mle.num_vars() as u32);

                // --- Fold the new (padded) bookkeeping table with the old ones ---
                let padded_bookkeeping_table = input_mle.get_padded_evaluations();
                current_bookkeeping_table
                    .into_iter()
                    .chain(padded_bookkeeping_table.into_iter())
                    .collect()
            });

    // --- Return the combined bookkeeping table ---
    DenseMle::new_from_raw(final_bookkeeping_table, LayerId::Input, None)
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::{test_rng, Zero};
    use itertools::Itertools;
    use rand::{distributions::Standard, prelude::Distribution, Rng};
    use std::iter::repeat_with;

    use crate::{
        layer::LayerId,
        mle::{dense::DenseMle, Mle, MleIndex},
    };

    use lcpc_2d::FieldExt;

    use super::combine_input_mles;

    /// Helper function to create random MLE with specific number of vars
    fn get_random_mle<F: FieldExt>(num_vars: usize) -> DenseMle<F, F>
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

    /// Helper function to create random MLE with specific length
    fn get_random_mle_with_capacity<F: FieldExt>(capacity: usize) -> DenseMle<F, F>
    where
        Standard: Distribution<F>,
    {
        let mut rng = test_rng();
        let bookkeeping_table = repeat_with(|| rng.gen::<F>())
            .take(capacity as usize)
            .collect_vec();
        DenseMle::new_from_raw(bookkeeping_table, LayerId::Input, None)
    }

    #[test]
    fn simple_test() {
        // --- Create MLEs of size 2^5, 2^5, 2^4 ---
        let mut mle_1 = get_random_mle::<Fr>(5);
        let mut mle_2 = get_random_mle::<Fr>(5);
        let mut mle_3 = get_random_mle::<Fr>(4);
        let mut mle_list: Vec<Box<&mut dyn Mle<Fr>>> = vec![Box::new(&mut mle_1), Box::new(&mut mle_3), Box::new(&mut mle_2)];

        let combined_mle = combine_input_mles(&mut mle_list);

        // --- The padded combined version should have size 2^7 ---
        assert_eq!(combined_mle.num_vars(), 7);

        // --- The prefix bits should be (0, 0), (0, 1), (1, 0, 0) ---
        assert_eq!(
            mle_1.prefix_bits,
            Some(vec![MleIndex::Fixed(false), MleIndex::Fixed(false)])
        );
        assert_eq!(
            mle_2.prefix_bits,
            Some(vec![MleIndex::Fixed(false), MleIndex::Fixed(true)])
        );
        assert_eq!(
            mle_3.prefix_bits,
            Some(vec![
                MleIndex::Fixed(true),
                MleIndex::Fixed(false),
                MleIndex::Fixed(false)
            ])
        );
    }

    #[test]
    fn test_with_padding() {
        // --- Create MLEs of size < 2^5, < 2^7, 2^6 ---
        let mut mle_1 = get_random_mle_with_capacity::<Fr>(31);
        let mut mle_2 = get_random_mle_with_capacity::<Fr>(115);
        let mut mle_3 = get_random_mle::<Fr>(6);
        let mut mle_list: Vec<Box<&mut dyn Mle<Fr>>> = vec![Box::new(&mut mle_1), Box::new(&mut mle_3), Box::new(&mut mle_2)];

        let combined_mle = combine_input_mles(&mut mle_list);

        // --- The padded combined version should have size 2^8 ---
        assert_eq!(combined_mle.num_vars(), 8);

        // --- The prefix bits should be (1, 1, 0), (0,), (1, 0) ---
        assert_eq!(
            mle_1.prefix_bits,
            Some(vec![
                MleIndex::Fixed(true),
                MleIndex::Fixed(true),
                MleIndex::Fixed(false)
            ])
        );
        assert_eq!(mle_2.prefix_bits, Some(vec![MleIndex::Fixed(false)]));
        assert_eq!(
            mle_3.prefix_bits,
            Some(vec![MleIndex::Fixed(true), MleIndex::Fixed(false)])
        );

        // --- Finally, let's check the bookkeeping tables ---
        // First up we should have mle_2's bookkeeping table
        assert_eq!(combined_mle.mle[0..115], mle_2.mle);
        assert_eq!(combined_mle.mle[115..128], vec![Fr::zero(); 128 - 115]);

        // Next, mle_3's bookkeeping table
        assert_eq!(combined_mle.mle[128..(128 + 64)], mle_3.mle);

        // Finally, mle_1's bookkeeping table
        assert_eq!(combined_mle.mle[(128 + 64)..(128 + 64 + 31)], mle_1.mle);

        // Padding
        assert_eq!(
            combined_mle.mle[(128 + 64 + 31)..(128 + 64 + 32)],
            vec![Fr::zero(); 1]
        );
    }
}
