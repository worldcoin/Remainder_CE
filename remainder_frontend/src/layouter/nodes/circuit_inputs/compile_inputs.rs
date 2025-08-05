use ark_std::log2;
use itertools::Itertools;
use remainder_shared_types::Field;

use crate::{
    input_layer::InputLayerDescription,
    layouter::{builder::CircuitMap, layouting::CircuitLocation, nodes::CircuitNode},
    utils::mle::argsort,
};

use super::InputLayerNode;

use anyhow::Result;

/// Function which returns a vector of the values for prefix bits according to which
/// position we are in the range from 0 to `total_num_bits` - `num_free_bits`.
fn get_prefix_bits_from_capacity(
    capacity: u32,
    total_num_bits: usize,
    num_free_bits: usize,
) -> Vec<bool> {
    (0..total_num_bits - num_free_bits)
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

impl InputLayerNode {
    /// From the circuit description map and a starting layer id, create the circuit description of
    /// an input layer, adding the input shreds to the circuit map.
    pub fn generate_input_layer_description<F: Field>(
        &self,
        circuit_map: &mut CircuitMap,
    ) -> Result<InputLayerDescription> {
        let Self {
            id: _,
            input_layer_id,
            input_shreds,
        } = &self;
        let input_mle_num_vars = input_shreds
            .iter()
            .map(|node| node.get_num_vars())
            .collect_vec();

        let (prefix_bits, input_shred_indices, num_vars_combined_mle) =
            index_input_mles(&input_mle_num_vars);
        debug_assert_eq!(input_shred_indices.len(), input_shreds.len());

        let input_layer_description = InputLayerDescription {
            layer_id: *input_layer_id,
            num_vars: num_vars_combined_mle,
        };

        input_shred_indices
            .iter()
            .zip(prefix_bits)
            .for_each(|(input_shred_index, prefix_bits)| {
                let input_shred = &input_shreds[*input_shred_index];
                circuit_map.add_node_id_and_location_num_vars(
                    input_shred.id,
                    (
                        CircuitLocation::new(*input_layer_id, prefix_bits),
                        input_shred.get_num_vars(),
                    ),
                );
            });

        Ok(input_layer_description)
    }
}
