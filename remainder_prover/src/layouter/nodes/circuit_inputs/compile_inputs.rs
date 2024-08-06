use ark_std::log2;
use itertools::Itertools;
use remainder_shared_types::FieldExt;

use crate::{
    input_layer::{
        hyrax_placeholder_input_layer::HyraxPlaceholderInputLayer,
        hyrax_precommit_placeholder_input_layer::HyraxPrecommitPlaceholderInputLayer,
        ligero_input_layer::LigeroInputLayer, public_input_layer::PublicInputLayer, InputLayer,
        MleInputLayer,
    },
    layouter::{
        compiling::WitnessBuilder,
        layouting::{CircuitLocation, CircuitMap, DAGError},
        nodes::CompilableNode,
    },
    mle::evals::{Evaluations, MultilinearExtension},
    prover::proof_system::ProofSystem,
    utils::{argsort, pad_to_nearest_power_of_two},
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

fn index_input_mles<F: FieldExt>(
    input_mles: &[&MultilinearExtension<F>],
) -> (Vec<Vec<bool>>, Vec<usize>) {
    let input_mle_num_vars = input_mles
        .iter()
        .map(|input_mle| input_mle.num_vars())
        .collect_vec();

    // Add input-output MLE length if needed
    let mle_combine_indices = argsort(&input_mle_num_vars, true);

    // Get the total needed capacity by rounding the raw capacity up to the nearest power of 2
    let raw_needed_capacity = input_mle_num_vars
        .into_iter()
        .fold(0, |prev, input_mle_num_vars| {
            prev + 2_usize.pow(input_mle_num_vars as u32)
        });
    let padded_needed_capacity = (1 << log2(raw_needed_capacity)) as usize;
    let total_num_vars = log2(padded_needed_capacity) as usize;

    // Go through individual MLEs and collect the prefix bits that need to be added to each one
    let mut current_padded_usage: u32 = 0;
    let res = mle_combine_indices
        .iter()
        .map(|input_mle_idx| {
            let input_mle = &input_mles[*input_mle_idx];

            // Collect the prefix bits for each MLE
            let prefix_bits: Vec<_> = get_prefix_bits_from_capacity(
                current_padded_usage,
                total_num_vars,
                input_mle.num_vars(),
            );
            current_padded_usage += 2_u32.pow(input_mle.num_vars() as u32);
            prefix_bits
        })
        .collect();
    (res, mle_combine_indices)
}

/// Combines the list of input MLEs in the input layer into one giant MLE by interleaving them
/// assuming that the indices of the bookkeeping table are stored in little endian.
fn combine_input_mles<F: FieldExt>(
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

impl<F: FieldExt, Pf: ProofSystem<F, InputLayer = IL>, IL> CompilableNode<F, Pf>
    for InputLayerNode<F>
where
    IL: InputLayer<F>
        + From<PublicInputLayer<F>>
        + From<LigeroInputLayer<F>>
        + From<HyraxPlaceholderInputLayer<F>>
        + From<HyraxPrecommitPlaceholderInputLayer<F>>,
{
    fn compile<'a>(
        &'a self,
        witness: &mut WitnessBuilder<F, Pf>,
        circuit_map: &mut CircuitMap<'a, F>,
    ) -> Result<(), DAGError> {
        let layer_id = witness.next_input_layer();
        let Self {
            id: _,
            children,
            input_layer_type,
        } = &self;

        let input_mles = children
            .iter()
            .map(|input_shred| &input_shred.data)
            .collect_vec();
        let mle = combine_input_mles(&input_mles);
        let (prefix_bits, input_shred_indices) = index_input_mles(&input_mles);
        debug_assert_eq!(input_shred_indices.len(), children.len());

        let out: IL = match input_layer_type {
            InputLayerType::LigeroInputLayer => LigeroInputLayer::new(mle, layer_id).into(),
            InputLayerType::PublicInputLayer => PublicInputLayer::new(mle, layer_id).into(),
            InputLayerType::Default => PublicInputLayer::new(mle, layer_id).into(),
            InputLayerType::HyraxPlaceholderInputLayer => {
                HyraxPlaceholderInputLayer::new(mle, layer_id).into()
            }
            InputLayerType::HyraxPrecommitPlaceholderInputLayer => {
                HyraxPrecommitPlaceholderInputLayer::new(mle, layer_id).into()
            }
        };

        witness.add_input_layer(out);
        input_shred_indices
            .iter()
            .zip(prefix_bits)
            .for_each(|(input_shred_index, prefix_bits)| {
                let input_shred = &children[*input_shred_index];
                circuit_map.add_node(
                    input_shred.id,
                    (
                        CircuitLocation::new(layer_id, prefix_bits),
                        &input_shred.data,
                    ),
                );
            });
        Ok(())
    }
}
