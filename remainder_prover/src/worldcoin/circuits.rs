use crate::layouter::compiling::LayouterCircuit;
use crate::layouter::component::ComponentSet;
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
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let input_shred_matrix_a =
            get_input_shred_from_vec(image_matrix_mle.clone(), ctx, &input_layer);

        let matrix_a = IdentityGateNode::new(ctx, &input_shred_matrix_a, wirings.clone());
        let (filter_num_rows, filter_num_cols) = kernel_matrix_dims;
        let matrix_a_num_rows_cols = (*num_placements, *filter_num_rows);
        let matrix_b = get_input_shred_from_vec(kernel_matrix_mle.clone(), ctx, &input_layer);
        let matrix_b_num_rows_cols = (*filter_num_rows, *filter_num_cols);

        let result_of_matmult = MatMultNode::new(
            ctx,
            &matrix_a,
            matrix_a_num_rows_cols,
            &matrix_b,
            matrix_b_num_rows_cols,
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
            &result_of_matmult,
            &iris_code_input_shred,
            &recomp_of_abs_value.sector,
        );

        let output = OutputNode::new_zero(ctx, &recomp_check_builder.sector);

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            input_shred_matrix_a.into(),
            matrix_a.into(),
            matrix_b.into(),
            result_of_matmult.into(),
            recomp_of_abs_value.sector.into(),
            iris_code_input_shred.into(),
            recomp_check_builder.sector.into(),
            output.into(),
        ];

        all_nodes.extend(digits_input_shreds.into_iter().map(|shred| shred.into()));

        ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes)
    })
}