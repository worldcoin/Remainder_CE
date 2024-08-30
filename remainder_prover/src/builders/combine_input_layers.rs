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

use crate::input_layer::ligero_input_layer::LigeroInputLayer;

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
