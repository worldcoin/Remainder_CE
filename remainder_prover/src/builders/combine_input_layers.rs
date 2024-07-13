//! Combine multiple MLEs into one input layer.

use ark_std::log2;
use itertools::Itertools;
use remainder_ligero::{
    ligero_structs::LigeroAuxInfo, poseidon_ligero::PoseidonSpongeHasher, LcCommit, LcRoot,
};
use remainder_shared_types::FieldExt;

use crate::{
    layer::LayerId,
    mle::{dense::DenseMle, Mle, MleIndex},
    utils::{argsort, pad_to_nearest_power_of_two},
};

use crate::input_layer::{ligero_input_layer::LigeroInputLayer, MleInputLayer};

/// Function which returns a vector of `MleIndex::Fixed` for prefix bits according to which
/// position we are in the range from 0 to `total_num_bits` - `num_iterated_bits`.
fn get_prefix_bits_from_capacity<F: FieldExt>(
    capacity: u32,
    total_num_bits: usize,
    num_iterated_bits: usize,
) -> Vec<MleIndex<F>> {
    (0..total_num_bits - num_iterated_bits)
        .map(|bit_position| {
            // Divide capacity by 2**(total_num_bits - bit_position - 1) and see whether the last bit is 1
            let bit_val = (capacity >> (total_num_bits - bit_position - 1)) & 1;
            MleIndex::Fixed(bit_val == 1)
        })
        .collect()
}

/// Takes an MLE bookkeeping table interpreted as (big/little)-endian,
/// and converts it into a bookkeeping table interpreted as (little/big)-endian.
///
/// ## Arguments
/// * `bookkeeping_table` - Original MLE bookkeeping table
///
/// ## Returns
/// * `opposite_endian_bookkeeping_table` - MLE bookkeeping table, which, when
///     indexed (b_n, ..., b_1) rather than (b_1, ..., b_n), yields the same
///     result.
fn invert_mle_bookkeeping_table<F: FieldExt>(bookkeeping_table: Vec<F>) -> Vec<F> {
    // --- This should only happen the first time!!! ---
    let padded_bookkeeping_table = pad_to_nearest_power_of_two(bookkeeping_table);

    // --- 2 or fewer elements: No-op ---
    if padded_bookkeeping_table.len() <= 2 {
        return padded_bookkeeping_table;
    }

    // --- Grab the table by pairs, and create iterators over each half ---
    let tuples: (Vec<F>, Vec<F>) = padded_bookkeeping_table
        .chunks(2)
        .map(|pair| (pair[0], pair[1]))
        .unzip();

    // --- Recursively flip each half ---
    let inverted_first_half = invert_mle_bookkeeping_table(tuples.0);
    let inverted_second_half = invert_mle_bookkeeping_table(tuples.1);

    // --- Return the concatenation of the two ---
    inverted_first_half
        .into_iter()
        .chain(inverted_second_half)
        .collect()
}

/// An interface for defining the set of MLEs you want to combine into a single InputLayer.
pub struct InputLayerBuilder<F> {
    mles: Vec<Box<dyn Mle<F>>>,
    extra_mle_indices: Option<Vec<Vec<MleIndex<F>>>>,
    layer_id: LayerId,
}

impl<F: FieldExt> InputLayerBuilder<F> {
    /// Creates a new InputLayerBuilder that will yield an InputLayer from many MLEs
    ///
    /// Note that `extra_mle_num_vars` refers to the length of any MLE you want to be a part of this
    /// input_layer, but haven't yet generated the data for
    pub fn new(
        mut input_mles: Vec<&mut (dyn Mle<F> + 'static)>,
        extra_mle_num_vars: Option<Vec<usize>>,
        layer_id: LayerId,
    ) -> Self {
        let extra_mle_indices =
            InputLayerBuilder::index_input_mles(&mut input_mles, extra_mle_num_vars);
        let input_mles = input_mles
            .into_iter()
            .map(|mle| {
                assert_eq!(mle.layer_id(), layer_id);
                dyn_clone::clone_box(mle)
            })
            .collect_vec();
        Self {
            mles: input_mles,
            extra_mle_indices,
            layer_id,
        }
    }

    fn index_input_mles(
        input_mles: &mut [&mut (dyn Mle<F> + 'static)],
        extra_mle_num_vars: Option<Vec<usize>>,
    ) -> Option<Vec<Vec<MleIndex<F>>>> {
        let mut input_mle_num_vars = input_mles
            .iter()
            .map(|input_mle| input_mle.num_iterated_vars())
            .collect_vec();

        // --- Add input-output MLE length if needed ---
        input_mle_num_vars.extend(extra_mle_num_vars.iter().flatten().cloned());
        let mle_combine_indices = argsort(&input_mle_num_vars, true);

        // --- Get the total needed capacity by rounding the raw capacity up to the nearest power of 2 ---
        let raw_needed_capacity = input_mle_num_vars
            .into_iter()
            .fold(0, |prev, input_mle_num_vars| {
                prev + 2_usize.pow(input_mle_num_vars as u32)
            });
        let padded_needed_capacity = (1 << log2(raw_needed_capacity)) as usize;
        let total_num_vars = log2(padded_needed_capacity) as usize;

        let mut extra_mle_indices = Vec::with_capacity(
            extra_mle_num_vars
                .as_ref()
                .map(|num_vars| num_vars.len())
                .unwrap_or(0),
        );

        // --- Go through individual MLEs and add prefix bits ---
        let mut current_padded_usage: u32 = 0;
        mle_combine_indices.into_iter().for_each(|input_mle_idx| {
            // --- Only add prefix bits to the non-input-output MLEs ---
            if input_mle_idx < input_mles.len() {
                let input_mle = &mut input_mles[input_mle_idx];

                // --- Grab the prefix bits and add them to the individual MLEs ---
                let prefix_bits: Vec<MleIndex<F>> = get_prefix_bits_from_capacity(
                    current_padded_usage,
                    total_num_vars,
                    input_mle.num_iterated_vars(),
                );
                input_mle.add_prefix_bits(prefix_bits);
                current_padded_usage += 2_u32.pow(input_mle.num_iterated_vars() as u32);
            } else {
                let extra_mle_index = input_mles.len() - input_mle_idx;

                // --- Grab the prefix bits for the dummy padded MLE (this should ONLY happen if we have a dummy padded MLE) ---
                let prefix_bits: Vec<MleIndex<F>> = get_prefix_bits_from_capacity(
                    current_padded_usage,
                    total_num_vars,
                    extra_mle_num_vars.as_ref().unwrap()[extra_mle_index],
                );
                extra_mle_indices.push(prefix_bits);
                current_padded_usage +=
                    2_u32.pow(extra_mle_num_vars.as_ref().unwrap()[extra_mle_index] as u32);
            }
        });

        if !extra_mle_indices.is_empty() {
            Some(extra_mle_indices)
        } else {
            None
        }
    }

    /// Add a concrete value for the extra_mle declared at the start.
    pub fn add_extra_mle(
        &mut self,
        extra_mle: &mut (dyn Mle<F> + 'static),
    ) -> Result<(), &'static str> {
        let new_bits = self.extra_mle_indices.as_mut().ok_or("Called add_extra_mle too many times compared to the extra_mles that were declared when creating the builder")?;
        let new_bits = new_bits.remove(0);
        extra_mle.add_prefix_bits(new_bits);
        let extra_mle = dyn_clone::clone_box(extra_mle);
        self.mles.push(extra_mle);
        Ok(())
    }

    /// Combines the list of input MLEs in the input layer into one giant MLE by interleaving them
    /// assuming that the indices of the bookkeeping table are stored in little endian.
    fn combine_input_mles(&self) -> DenseMle<F> {
        let input_mles = &self.mles;
        let mle_combine_indices = argsort(
            &input_mles
                .iter()
                .map(|mle| mle.num_iterated_vars())
                .collect_vec(),
            true,
        );

        let final_bookkeeping_table = mle_combine_indices.into_iter().fold(
            vec![],
            |current_bookkeeping_table, input_mle_idx| {
                // --- Grab from the list of input MLEs OR the input-output MLE if the index calls for it ---
                let input_mle = &input_mles[input_mle_idx];

                // --- Basically, everything is stored in big-endian (including bookkeeping tables ---
                // --- and indices), BUT the indexing functions all happen as if we're interpreting ---
                // --- the indices as little-endian. Therefore we need to merge the input MLEs via ---
                // --- interleaving, or alternatively by converting everything to "big-endian", ---
                // --- merging the usual big-endian way, and re-converting the merged version back to ---
                // --- "little-endian" ---
                let inverted_input_mle =
                    invert_mle_bookkeeping_table(input_mle.get_padded_evaluations());

                // --- Fold the new (padded) bookkeeping table with the old ones ---
                // let padded_bookkeeping_table = input_mle.get_padded_evaluations();
                current_bookkeeping_table
                    .into_iter()
                    .chain(inverted_input_mle)
                    .collect()
            },
        );

        // --- Convert the final bookkeeping table back to "little-endian" ---
        let re_inverted_final_bookkeeping_table =
            invert_mle_bookkeeping_table(final_bookkeeping_table);
        DenseMle::new_from_raw(re_inverted_final_bookkeeping_table, self.layer_id)
    }

    /// Turn this builder into an input layer.
    pub fn to_input_layer<I: MleInputLayer<F>>(self) -> I {
        let final_mle: DenseMle<F> = self.combine_input_mles();
        I::new(final_mle, self.layer_id)
    }

    /// Turn the builder into an input layer WITH a pre-commitment.
    pub fn to_input_layer_with_precommit(
        self,
        ligero_comm: LcCommit<PoseidonSpongeHasher<F>, LigeroAuxInfo<F>, F>,
        ligero_aux: LigeroAuxInfo<F>,
        ligero_root: LcRoot<LigeroAuxInfo<F>, F>,
        verifier_is_precommit: bool,
    ) -> LigeroInputLayer<F> {
        let final_mle: DenseMle<F> = self.combine_input_mles();
        LigeroInputLayer::<F>::new_with_ligero_commitment(
            final_mle,
            self.layer_id,
            ligero_comm,
            ligero_aux,
            ligero_root,
            verifier_is_precommit,
        )
    }

    /// Turn the builder into input layer with rho inv specified.
    pub fn to_ligero_input_layer_with_rho_inv(
        self,
        rho_inv: u8,
        ratio: f64,
    ) -> LigeroInputLayer<F> {
        let final_mle: DenseMle<F> = self.combine_input_mles();
        LigeroInputLayer::<F>::new_with_rho_inv_ratio(final_mle, self.layer_id, rho_inv, ratio)
    }
}
