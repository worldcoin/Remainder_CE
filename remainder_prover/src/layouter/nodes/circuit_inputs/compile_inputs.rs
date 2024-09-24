use ark_std::log2;
use itertools::Itertools;
use remainder_ligero::ligero_structs::LigeroAuxInfo;
use remainder_shared_types::Field;

use crate::{
    input_layer::{
        enum_input_layer::CircuitInputLayerEnum, hyrax_input_layer::CircuitHyraxInputLayer,
        ligero_input_layer::CircuitLigeroInputLayer, public_input_layer::CircuitPublicInputLayer,
    },
    layer::LayerId,
    layouter::{
        layouting::{CircuitDescriptionMap, CircuitLocation, DAGError},
        nodes::CircuitNode,
    },
    mle::evals::{Evaluations, MultilinearExtension},
    utils::mle::{argsort, pad_to_nearest_power_of_two},
};

use super::{InputLayerNode, InputLayerType};

/// Function which returns a vector of `MleIndex::Fixed` for prefix bits according to which
/// position we are in the range from 0 to `total_num_bits` - `num_iterated_bits`.
fn get_prefix_bits_from_capacity(
    capacity: u32,
    total_num_bits: usize,
    num_iterated_bits: usize,
) -> Vec<bool> {
    (0..total_num_bits - num_iterated_bits)
        .map(|bit_position| {
            // Divide capacity by 2**(total_num_bits - bit_position - 1) and see whether the last bit is 1
            let bit_val = (capacity >> (total_num_bits - bit_position - 1)) & 1;
            bit_val == 1
        })
        .collect()
}

fn index_input_mles(input_mle_num_vars: &[usize]) -> (Vec<Vec<bool>>, Vec<usize>, usize) {
    // Add input-output MLE length if needed
    let mle_combine_indices = argsort(input_mle_num_vars, true);

    // Get the total needed capacity by rounding the raw capacity up to the nearest power of 2
    let raw_needed_capacity = input_mle_num_vars
        .iter()
        .fold(0, |prev, input_mle_num_vars| {
            prev + 2_usize.pow(*input_mle_num_vars as u32)
        });
    let padded_needed_capacity = (1 << log2(raw_needed_capacity)) as usize;
    let total_num_vars = log2(padded_needed_capacity) as usize;

    // Go through individual MLEs and collect the prefix bits that need to be added to each one
    let mut current_padded_usage: u32 = 0;
    let res = mle_combine_indices
        .iter()
        .map(|input_mle_idx| {
            let input_mle_bits = input_mle_num_vars[*input_mle_idx];

            // Collect the prefix bits for each MLE
            let prefix_bits: Vec<_> =
                get_prefix_bits_from_capacity(current_padded_usage, total_num_vars, input_mle_bits);
            current_padded_usage += 2_u32.pow(input_mle_bits as u32);
            prefix_bits
        })
        .collect();
    (res, mle_combine_indices, total_num_vars)
}

/// Combines the list of input MLEs in the input layer into one giant MLE by interleaving them
/// assuming that the indices of the bookkeeping table are stored in little endian.
pub fn combine_input_mles<F: Field>(
    input_mles: &[&MultilinearExtension<F>],
) -> MultilinearExtension<F> {
    let mle_combine_indices = argsort(
        &input_mles.iter().map(|mle| mle.num_vars()).collect_vec(),
        true,
    );

    let final_bookkeeping_table =
        mle_combine_indices
            .into_iter()
            .fold(vec![], |current_bookkeeping_table, input_mle_idx| {
                // --- Grab from the list of input MLEs OR the input-output MLE if the index calls for it ---
                let input_mle = &input_mles[input_mle_idx];

                // --- Basically, everything is stored in big-endian (including bookkeeping tables ---
                // --- and indices), BUT the indexing functions all happen as if we're interpreting ---
                // --- the indices as little-endian. Therefore we need to merge the input MLEs via ---
                // --- interleaving, or alternatively by converting everything to "big-endian", ---
                // --- merging the usual big-endian way, and re-converting the merged version back to ---
                // --- "little-endian" ---
                let inverted_input_mle =
                    invert_mle_bookkeeping_table(input_mle.get_evals_vector().to_vec());

                // --- Fold the new (padded) bookkeeping table with the old ones ---
                // let padded_bookkeeping_table = input_mle.get_padded_evaluations();
                current_bookkeeping_table
                    .into_iter()
                    .chain(inverted_input_mle)
                    .collect()
            });

    let num_vars = log2(final_bookkeeping_table.len()) as usize;

    // --- Convert the final bookkeeping table back to "little-endian" ---
    MultilinearExtension::new_from_evals(Evaluations::new(
        num_vars,
        invert_mle_bookkeeping_table(final_bookkeeping_table),
    ))
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
fn invert_mle_bookkeeping_table<F: Field>(bookkeeping_table: Vec<F>) -> Vec<F> {
    // --- This should only happen the first time!!! ---
    let padded_bookkeeping_table = pad_to_nearest_power_of_two(&bookkeeping_table);

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

impl InputLayerNode {
    /// From the circuit description map and a starting layer id,
    /// create the circuit description of an input layer.
    pub fn generate_input_circuit_description<F: Field>(
        &self,
        layer_id: &mut LayerId,
        circuit_description_map: &mut CircuitDescriptionMap,
    ) -> Result<CircuitInputLayerEnum<F>, DAGError> {
        let input_layer_id = layer_id.get_and_inc();
        let Self {
            id: _,
            children,
            input_layer_type,
        } = &self;

        let input_mle_num_vars = children
            .iter()
            .map(|node| node.get_num_vars())
            .collect_vec();

        let (prefix_bits, input_shred_indices, num_vars_combined_mle) =
            index_input_mles(&input_mle_num_vars);
        debug_assert_eq!(input_shred_indices.len(), children.len());

        let out = match input_layer_type {
            InputLayerType::LigeroInputLayer((rho_inv, ratio)) => {
                let aux = LigeroAuxInfo::new(
                    (1 << num_vars_combined_mle) as usize,
                    *rho_inv,
                    *ratio,
                    None,
                );
                let ligero_input_layer_description: CircuitLigeroInputLayer<F> =
                    CircuitLigeroInputLayer::new(
                        input_layer_id.to_owned(),
                        num_vars_combined_mle,
                        aux,
                    );
                CircuitInputLayerEnum::LigeroInputLayer(ligero_input_layer_description)
            }
            InputLayerType::PublicInputLayer => {
                let public_input_layer_description =
                    CircuitPublicInputLayer::new(input_layer_id.to_owned(), num_vars_combined_mle);
                CircuitInputLayerEnum::PublicInputLayer(public_input_layer_description)
            }
            InputLayerType::HyraxInputLayer => {
                let hyrax_input_layer_description =
                    CircuitHyraxInputLayer::new(input_layer_id.to_owned(), num_vars_combined_mle);
                CircuitInputLayerEnum::HyraxInputLayer(hyrax_input_layer_description)
            }
        };

        input_shred_indices
            .iter()
            .zip(prefix_bits)
            .for_each(|(input_shred_index, prefix_bits)| {
                let input_shred = &children[*input_shred_index];
                circuit_description_map.add_node_id_and_location_num_vars(
                    input_shred.id,
                    (
                        CircuitLocation::new(input_layer_id, prefix_bits),
                        input_shred.get_num_vars(),
                    ),
                );
            });
        Ok(out)
    }
}
