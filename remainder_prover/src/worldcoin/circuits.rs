use crate::layouter::compiling::LayouterCircuit;
use crate::layouter::component::{Component, ComponentSet};
use crate::layouter::nodes::circuit_inputs::{InputLayerNode, InputLayerType};
use crate::layouter::nodes::circuit_outputs::OutputNode;
use crate::layouter::nodes::identity_gate::IdentityGateNode;
use crate::layouter::nodes::matmult::MatMultNode;
use crate::layouter::nodes::node_enum::NodeEnum;
use crate::layouter::nodes::{ClaimableNode, Context};
use crate::mle::circuit_mle::CircuitMle;
use crate::utils::get_input_shred_from_vec;
use crate::utils::pad_to_nearest_power_of_two;
use crate::worldcoin::components::DigitRecompComponent;
use crate::worldcoin::components::SignCheckerComponent;
use crate::worldcoin::data::{WorldcoinCircuitData, WorldcoinData};
use crate::worldcoin::digit_decomposition::BASE;
use itertools::Itertools;
use remainder_shared_types::Fr;

use super::components::BitsAreBinary;

/// Builds the worldcoin circuit.
pub fn build_circuit(data: WorldcoinCircuitData<Fr>)
    -> LayouterCircuit<Fr, ComponentSet<NodeEnum<Fr>>, impl FnMut(&Context) -> ComponentSet<NodeEnum<Fr>>> {
    LayouterCircuit::new(move |ctx| {
        let WorldcoinCircuitData {
            image_matrix_mle,
            reroutings: wirings,
            num_placements,
            kernel_matrix_mle,
            kernel_matrix_dims,
            digits,
            iris_code,
            digit_multiplicities,
        } = &data;
        let mut output_nodes = vec![];

        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let image = get_input_shred_from_vec(image_matrix_mle.clone(), ctx, &input_layer);
        let matrix_a = IdentityGateNode::new(ctx, &image, wirings.clone());

        let (filter_num_rows, _) = kernel_matrix_dims;
        let matrix_a_num_rows_cols = (*num_placements, *filter_num_rows);
        let matrix_b = get_input_shred_from_vec(kernel_matrix_mle.clone(), ctx, &input_layer);

        let matmult = MatMultNode::new(
            ctx,
            &matrix_a,
            matrix_a_num_rows_cols,
            &matrix_b,
            *kernel_matrix_dims,
        );

        let digits_input_shreds = digits.make_input_shreds(ctx, &input_layer);
        let digits_input_refs = digits_input_shreds
            .iter()
            .map(|shred| shred as &dyn ClaimableNode<F = Fr>)
            .collect_vec();
        let recomp_of_abs_value =
            DigitRecompComponent::new(ctx, &digits_input_refs, BASE as u64);

        let iris_code_input_shred = get_input_shred_from_vec(
            pad_to_nearest_power_of_two(iris_code.clone()),
            ctx,
            &input_layer,
        );
        let recomp_check_builder = SignCheckerComponent::new(
            ctx,
            &matmult,
            &iris_code_input_shred,
            &recomp_of_abs_value.sector,
        );
        output_nodes.push(OutputNode::new_zero(ctx, &recomp_check_builder.sector));

        let bits_are_binary = BitsAreBinary::new(ctx, &iris_code_input_shred);
        output_nodes.push(OutputNode::new_zero(ctx, &bits_are_binary.sector));

        // Collect all the nodes, starting with the input nodes
        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            image.into(),
            matrix_a.into(),
            matrix_b.into(),
            iris_code_input_shred.into(),
        ];
        all_nodes.extend(digits_input_shreds.into_iter().map(|node| node.into()));

        // Add matmult node
        all_nodes.push(matmult.into());

        // Add nodes from components
        all_nodes.extend(bits_are_binary.yield_nodes().into_iter());
        all_nodes.extend(recomp_of_abs_value.yield_nodes().into_iter());
        all_nodes.extend(recomp_check_builder.yield_nodes().into_iter());

        // Add output nodes
        all_nodes.extend(output_nodes.into_iter().map(|node| node.into()));

        ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes)
    })
}