use std::iter::repeat_with;

use ark_std::log2;
use itertools::Itertools;

use crate::{mle::{dense::DenseMle, Mle, MleIndex}, utils::argsort};

use lcpc_2d::FieldExt;

use super::LayerId;

/*
Idea for handling output-input layer thingy:
I. Lazy initialization of the final MLE bookkeeping table
II. Initially, just do prefix bits and stuff
III. Keep track of the sorted list internally (or rather the sorted indices)
IV. When the caller passes in the actual output layer, then merge everything
 */

/// Input layer struct containing the information we need to
/// * Aggregate the input layer MLEs into a single large DenseMle<F, F>
/// * Aggregate claims on the input layer
/// * Evaluate the final claim on the input layer
pub struct InputLayer<F: FieldExt> {
    /// Stop
    pub combined_dense_mle: Option<DenseMle<F, F>>,
    /// Stop
    pub mle_combine_indices: Vec<usize>,
    /// Stop
    pub maybe_output_input_mle_num_vars: Option<usize>,
    /// Stop
    pub total_num_vars: usize,
    /// Stop
    pub maybe_output_input_mle_prefix_indices: Option<Vec<MleIndex<F>>>,
}

impl<F: FieldExt> InputLayer<F> {

    /// Creates a new `InputLayer` from a bunch of MLEs which belong
    /// in the input layer by merging them.
    ///
    /// ## Arguments
    ///
    /// * `input_mles` - MLEs in the input layer to be merged
    /// * `maybe_output_input_mle_num_vars` - An output MLE to be zero-checked against,
    ///     but currently unpopulated
    pub fn new_from_mles(input_mles: &mut Vec<Box<&mut dyn Mle<F>>>, maybe_output_input_mle_num_vars: Option<usize>) -> Self {
        let mut ret = Self {
            combined_dense_mle: None,
            mle_combine_indices: vec![],
            maybe_output_input_mle_num_vars,
            total_num_vars: 0,
            maybe_output_input_mle_prefix_indices: None,
        };
        ret.index_input_mles(input_mles, maybe_output_input_mle_num_vars);
        ret
    }

    /// Creates an empty InputLayer
    pub fn new() -> Self {
        Self {
            combined_dense_mle: None,
            mle_combine_indices: vec![],
            maybe_output_input_mle_num_vars: None,
            total_num_vars: 0,
            maybe_output_input_mle_prefix_indices: None,
        }
    }

    /// Getter for the DenseMLE making up the input layer
    /// * TODO!(ryancao): Return the DenseMle by reference!
    pub fn get_combined_mle(&self) -> Option<DenseMle<F, F>> {
        self.combined_dense_mle.clone()
    }

    /// Takes in the same list of `input_mles` as earlier (i.e. when the `InputLayer`
    /// was constructed) and actually performs the combining, i.e. sets `combined_dense_mle`
    /// to `Some()` rather than `None`.
    ///
    /// ## Arguments
    ///
    /// * `input_mles` - The list of disjointed input chunks to combine. Note that
    ///     this should now contain the populated input-output MLE, if specified earlier!
    /// * `input_output_mle` - The actual input-output MLE specified earlier, if any
    /// * `total_num_vars` - Total number of variables, as computed earlier, within the
    ///     combined input MLE
    ///
    /// ## Examples
    /// ```
    ///
    /// ```
    pub fn combine_input_mles<'a>(
        &mut self,
        input_mles: &Vec<Box<&'a mut dyn Mle<F>>>,
        maybe_input_output_mle: Option<Box<&'a mut (dyn Mle<F> + 'a)>>,
    ) {

        // --- Create dummy input-output MLE in case there is no real input-output MLE ---
        let mut dummy_input_output_mle_ref = DenseMle::<F, F>::new_from_raw(vec![], LayerId::Input, None);
        let maybe_input_output_mle_ref = &maybe_input_output_mle.unwrap_or_else(|| {
            Box::new(&mut dummy_input_output_mle_ref)
        });

        // --- Next, grab their bookkeeping tables (in sorted order by length!) and combine them ---
        let initial_vec: Vec<F> = vec![];
        let final_bookkeeping_table = self.mle_combine_indices.clone()
            .into_iter()
            .fold(initial_vec, |current_bookkeeping_table, input_mle_idx| {

                // --- Grab from the list of input MLEs OR the input-output MLE if the index calls for it ---
                let input_mle = if input_mle_idx < input_mles.len() 
                {
                    &input_mles[input_mle_idx]
                } else {
                    maybe_input_output_mle_ref
                };

                // --- Fold the new (padded) bookkeeping table with the old ones ---
                let padded_bookkeeping_table = input_mle.get_padded_evaluations();
                current_bookkeeping_table
                    .into_iter()
                    .chain(padded_bookkeeping_table.into_iter())
                    .collect()
            });

        // --- Return the combined bookkeeping table ---
        self.combined_dense_mle = Some(DenseMle::new_from_raw(final_bookkeeping_table, LayerId::Input, None));

    }

    /// Adds prefix bits to the given input-output MLE.
    pub fn index_input_output_mle(
        &self,
        input_output_mle: &mut Box<&mut dyn Mle<F>>,
    ) {
        input_output_mle.add_prefix_bits(self.maybe_output_input_mle_prefix_indices.clone());
    }

    /// Takes in a list of input MLEs to be combined into a single
    /// MLE via a greedy algorithm. Additionally, modifies the list
    /// of `input_mles` with prefix bits such that each is appropriately
    /// prefixed within the single `DenseMle<F, F>` representing the
    /// combined MLE of the layer. (Note that this `DenseMle` is not
    /// actually returned, but rather lazily initialized via the sorted
    /// indices!)
    ///
    /// The algorithm used here is a simple greedy algorithm.
    /// * TODO!(ryancao): Ensure that this is optimal
    ///
    /// ## Arguments
    ///
    /// * `input_mles` - The list of disjointed input chunks to combine.
    /// * `maybe_output_input_mle_num_vars` - An output MLE to be zero-checked against,
    ///     but currently unpopulated
    ///
    /// ## Examples
    /// ```
    ///
    /// ```
    pub fn index_input_mles(
        &mut self,
        input_mles: &mut Vec<Box<&mut dyn Mle<F>>>,
        maybe_output_input_mle_num_vars: Option<usize>
    ) {

        // --- Grab sorted indices of the MLEs ---
        let mut input_mle_num_vars = input_mles
            .into_iter()
            .map(|input_mle| input_mle.num_vars())
            .collect_vec();
        // --- Add input-output MLE length if needed ---
        if let Some(output_input_mle_num_vars) = maybe_output_input_mle_num_vars {
            input_mle_num_vars.push(output_input_mle_num_vars);
        }
        let mle_combine_indices = argsort(&input_mle_num_vars, true);

        // --- Get the total needed capacity by rounding the raw capacity up to the nearest power of 2 ---
        let raw_needed_capacity = input_mle_num_vars.into_iter().fold(0, |prev, input_mle_num_vars| {
            prev + 2_usize.pow(input_mle_num_vars as u32)
        });
        let padded_needed_capacity = round_to_next_largest_power_of_2(raw_needed_capacity) as usize;
        let total_num_vars = log2(padded_needed_capacity as usize) as usize;

        // --- Go through individual MLEs and add prefix bits ---
        let mut current_padded_usage: u32 = 0;
        mle_combine_indices
            .clone()
            .into_iter()
            .for_each(|input_mle_idx| {

                // --- Only add prefix bits to the non-input-output MLEs ---
                if input_mle_idx < input_mles.len() {
                    let input_mle = &mut input_mles[input_mle_idx];

                    // --- Grab the prefix bits and add them to the individual MLEs ---
                    let prefix_bits: Vec<MleIndex<F>> = get_prefix_bits_from_capacity(
                        current_padded_usage as u32,
                        total_num_vars,
                        input_mle.num_vars(),
                    );
                    input_mle.add_prefix_bits(Some(prefix_bits));
                    current_padded_usage += 2_u32.pow(input_mle.num_vars() as u32);
                } else {
                    // --- Grab the prefix bits for the dummy padded MLE (this should ONLY happen if we have a dummy padded MLE) ---
                    let prefix_bits: Vec<MleIndex<F>> = get_prefix_bits_from_capacity(
                        current_padded_usage as u32,
                        total_num_vars,
                        maybe_output_input_mle_num_vars.unwrap(),
                    );
                    self.maybe_output_input_mle_prefix_indices = Some(prefix_bits);
                    current_padded_usage += 2_u32.pow(maybe_output_input_mle_num_vars.unwrap() as u32);
                }
            });

        // --- Sets self state for later ---
        self.mle_combine_indices = mle_combine_indices;
        self.total_num_vars = total_num_vars;

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

/// Returns the vector of prefix bits corresponding to a particular capacity.
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
    use super::InputLayer;

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
        let mut mle_list: Vec<Box<&mut dyn Mle<Fr>>> = vec![Box::new(&mut mle_1), Box::new(&mut mle_2), Box::new(&mut mle_3)];

        let mut dummy_input_layer: InputLayer<ark_ff::Fp<ark_ff::MontBackend<ark_bn254::FrConfig, 4>, 4>> = InputLayer::new_from_mles(&mut mle_list, None);
        dummy_input_layer.combine_input_mles(&mle_list, None);

        // --- The padded combined version should have size 2^7 (but only 2^5 + 2^5 + 2^4 = 80 unpadded elems) ---
        assert_eq!(dummy_input_layer.combined_dense_mle.clone().unwrap().num_vars(), 7);
        assert_eq!(dummy_input_layer.combined_dense_mle.unwrap().mle.len(), 32 + 32 + 16 as usize);

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
        let mut mle_list: Vec<Box<&mut dyn Mle<Fr>>> = vec![Box::new(&mut mle_1), Box::new(&mut mle_2), Box::new(&mut mle_3)];

        let mut dummy_input_layer: InputLayer<ark_ff::Fp<ark_ff::MontBackend<ark_bn254::FrConfig, 4>, 4>> = InputLayer::new_from_mles(&mut mle_list, None);
        dummy_input_layer.combine_input_mles(&mle_list, None);

        // --- The padded combined version should have size 2^8 ---
        assert_eq!(dummy_input_layer.combined_dense_mle.clone().unwrap().num_vars(), 8);

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
        assert_eq!(dummy_input_layer.combined_dense_mle.clone().unwrap().mle[0..115], mle_2.mle);
        assert_eq!(dummy_input_layer.combined_dense_mle.clone().unwrap().mle[115..128], vec![Fr::zero(); 128 - 115]);

        // Next, mle_3's bookkeeping table
        assert_eq!(dummy_input_layer.combined_dense_mle.clone().unwrap().mle[128..(128 + 64)], mle_3.mle);

        // Finally, mle_1's bookkeeping table
        assert_eq!(dummy_input_layer.combined_dense_mle.clone().unwrap().mle[(128 + 64)..(128 + 64 + 31)], mle_1.mle);

        // Padding
        assert_eq!(
            dummy_input_layer.combined_dense_mle.clone().unwrap().mle[(128 + 64 + 31)..(128 + 64 + 32)],
            vec![Fr::zero(); 1]
        );
    }

    #[test]
    fn test_with_padding_with_input_output_mle() {

        // --- Create MLEs of size < 2^5, < 2^7, 2^6 (just like the last test) ---
        let mut mle_1 = get_random_mle_with_capacity::<Fr>(31);
        let mut mle_2 = get_random_mle_with_capacity::<Fr>(115);
        let mut mle_3 = get_random_mle::<Fr>(6);
        let mut mle_list: Vec<Box<&mut dyn Mle<Fr>>> = vec![Box::new(&mut mle_1), Box::new(&mut mle_2), Box::new(&mut mle_3)];

        // --- Also create an input-output layer of size 2^8 ---
        let mut dummy_input_layer = InputLayer::new_from_mles(&mut mle_list, Some(8));
        let mut input_output_mle = get_random_mle::<Fr>(8);
        dummy_input_layer.combine_input_mles(&mle_list, Some(Box::new(&mut input_output_mle)));
        dummy_input_layer.index_input_output_mle(&mut Box::new(&mut input_output_mle));

        // --- The padded combined version should have size 2^9 ---
        assert_eq!(dummy_input_layer.combined_dense_mle.clone().unwrap().num_vars(), 9);

        dbg!(dummy_input_layer.mle_combine_indices);

        // --- The prefix bits should be (1, 1, 1, 0), (1, 0,), (1, 1, 0), (0,) ---
        assert_eq!(
            mle_1.prefix_bits,
            Some(vec![
                MleIndex::Fixed(true),
                MleIndex::Fixed(true),
                MleIndex::Fixed(true),
                MleIndex::Fixed(false)
            ])
        );
        assert_eq!(mle_2.prefix_bits, Some(vec![MleIndex::Fixed(true), MleIndex::Fixed(false)]));
        assert_eq!(
            mle_3.prefix_bits,
            Some(vec![MleIndex::Fixed(true), MleIndex::Fixed(true), MleIndex::Fixed(false)])
        );
        assert_eq!(
            input_output_mle.prefix_bits,
            Some(vec![MleIndex::Fixed(false)])
        );

        // --- Finally, let's check the bookkeeping tables ---

        // First up we should have input_output_mle's bookkeeping table
        assert_eq!(dummy_input_layer.combined_dense_mle.clone().unwrap().mle[0..256], input_output_mle.mle);

        // Next, we should have mle_2's bookkeeping table
        assert_eq!(dummy_input_layer.combined_dense_mle.clone().unwrap().mle[256..(256 + 115)], mle_2.mle);
        assert_eq!(dummy_input_layer.combined_dense_mle.clone().unwrap().mle[(256 + 115)..(256 + 128)], vec![Fr::zero(); 128 - 115]);

        // Next, mle_3's bookkeeping table
        assert_eq!(dummy_input_layer.combined_dense_mle.clone().unwrap().mle[256 + 128..(256 + 128 + 64)], mle_3.mle);

        // Finally, mle_1's bookkeeping table
        assert_eq!(dummy_input_layer.combined_dense_mle.clone().unwrap().mle[(256 + 128 + 64)..(256 + 128 + 64 + 31)], mle_1.mle);

        // Padding
        assert_eq!(
            dummy_input_layer.combined_dense_mle.clone().unwrap().mle[(256 + 128 + 64 + 31)..(256 + 128 + 64 + 32)],
            vec![Fr::zero(); 1]
        );

    }
}
