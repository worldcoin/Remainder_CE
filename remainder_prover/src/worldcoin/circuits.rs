use crate::layouter::compiling::LayouterCircuit;
use crate::layouter::component::{Component, ComponentSet};
use crate::layouter::nodes::circuit_inputs::{InputLayerData, InputLayerNode, InputLayerType};
use crate::layouter::nodes::circuit_outputs::OutputNode;
use crate::layouter::nodes::identity_gate::IdentityGateNode;
use crate::layouter::nodes::lookup::{LookupConstraint, LookupTable};
use crate::layouter::nodes::matmult::MatMultNode;
use crate::layouter::nodes::node_enum::NodeEnum;
use crate::layouter::nodes::verifier_challenge::VerifierChallengeNode;
use crate::layouter::nodes::{CircuitNode, Context};
use crate::mle::circuit_mle::BundledInputMle;
use crate::utils::pad_to_nearest_power_of_two;
use crate::utils::{get_input_shred_and_data_from_vec, get_input_shred_from_num_vars};
use crate::worldcoin::components::SignCheckerComponent;
use crate::worldcoin::components::{DigitalRecompositionComponent, SubtractionComponent};
use crate::worldcoin::data::WorldcoinCircuitData;
use crate::worldcoin::digit_decomposition::BASE;
use itertools::Itertools;
use remainder_shared_types::FieldExt;

use super::components::{BitsAreBinary, DigitsConcatenator};

/// Builds the worldcoin circuit with all public input layers.
pub fn build_circuit_public_il<F: FieldExt>(
    data: WorldcoinCircuitData<F>,
) -> LayouterCircuit<
    F,
    ComponentSet<NodeEnum<F>>,
    impl FnMut(&Context) -> (ComponentSet<NodeEnum<F>>, Vec<InputLayerData<F>>),
> {
    LayouterCircuit::new(move |ctx| {
        let WorldcoinCircuitData {
            image_matrix_mle,
            reroutings: wirings,
            num_placements,
            kernel_matrix_mle: kernel_matrix,
            kernel_matrix_dims,
            digits,
            code: iris_code,
            digit_multiplicities,
            thresholds_matrix,
        } = &data;
        let mut output_nodes = vec![];

        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        println!("Input layer = {:?}", input_layer.id());
        let (image, image_data) =
            get_input_shred_and_data_from_vec(image_matrix_mle.clone(), ctx, &input_layer);
        println!("Image input = {:?}", image.id());
        let (thresholds, thresholds_data) = get_input_shred_and_data_from_vec(
            pad_to_nearest_power_of_two(thresholds_matrix.clone()),
            ctx,
            &input_layer,
        );
        println!("Thresholds input = {:?}", thresholds.id());
        let rerouted_image = IdentityGateNode::new(ctx, &image, wirings.clone());
        println!("Identity gate = {:?}", rerouted_image.id());

        let (filter_num_values, _) = kernel_matrix_dims;
        let matrix_a_num_rows_cols = (num_placements.next_power_of_two(), *filter_num_values);
        let (kernel_matrix, kernel_matrix_data) =
            get_input_shred_and_data_from_vec(kernel_matrix.clone(), ctx, &input_layer);
        println!("Kernel values input = {:?}", kernel_matrix.id());

        let matmult = MatMultNode::new(
            ctx,
            &rerouted_image,
            matrix_a_num_rows_cols,
            &kernel_matrix,
            *kernel_matrix_dims,
        );
        println!("Matmult = {:?}", matmult.id());

        let subtract_thresholds = SubtractionComponent::new(ctx, &matmult, &thresholds);

        let digits_input_shreds = digits.make_input_shreds(ctx, &input_layer);
        for (i, shred) in digits_input_shreds.iter().enumerate() {
            println!("{}th digit input = {:?}", i, shred.id());
        }
        let digits_refs = digits_input_shreds
            .iter()
            .map(|shred| shred as &dyn CircuitNode)
            .collect_vec();
        // Concatenate the digits (which are stored for each digital place separately) into a single
        // MLE
        let digits_concatenator = DigitsConcatenator::new(ctx, &digits_refs);

        // Use a lookup to range check the digits to the range 0..BASE
        let (lookup_table_values, lookup_table_values_data) = get_input_shred_and_data_from_vec(
            (0..BASE as u64).map(F::from).collect(),
            ctx,
            &input_layer,
        );
        println!("Digit range check input = {:?}", lookup_table_values.id());

        let verifier_challenge_node = VerifierChallengeNode::new(ctx, 1);
        let lookup_table =
            LookupTable::new::<F>(ctx, &lookup_table_values, false, verifier_challenge_node);
        println!("Lookup table = {:?}", lookup_table.id());
        let (digit_multiplicities, digit_multiplicities_data) =
            get_input_shred_and_data_from_vec(digit_multiplicities.clone(), ctx, &input_layer);
        println!("Digit multiplicities = {:?}", digit_multiplicities.id());
        let lookup_constraint = LookupConstraint::new::<F>(
            ctx,
            &lookup_table,
            &&digits_concatenator.sector,
            &digit_multiplicities,
        );
        println!("Lookup constraint = {:?}", lookup_constraint.id());

        let recomp_of_abs_value =
            DigitalRecompositionComponent::new(ctx, &digits_refs, BASE as u64);

        let (iris_code, iris_code_data) = get_input_shred_and_data_from_vec(
            pad_to_nearest_power_of_two(iris_code.clone()),
            ctx,
            &input_layer,
        );
        println!("Iris code input = {:?}", iris_code.id());
        let sign_checker = SignCheckerComponent::new(
            ctx,
            &&subtract_thresholds.sector,
            &iris_code,
            &&recomp_of_abs_value.sector,
        );
        output_nodes.push(OutputNode::new_zero(ctx, &sign_checker.sector));

        let bits_are_binary = BitsAreBinary::new(ctx, &iris_code);
        output_nodes.push(OutputNode::new_zero(ctx, &bits_are_binary.sector));

        let input_layer_data = InputLayerData::new(
            input_layer.id(),
            vec![
                image_data,
                thresholds_data,
                kernel_matrix_data,
                lookup_table_values_data,
                digit_multiplicities_data,
                iris_code_data,
            ],
            None,
        );

        // Collect all the nodes, starting with the input nodes
        let mut all_nodes: Vec<NodeEnum<F>> = vec![
            input_layer.into(),
            image.into(),
            kernel_matrix.into(),
            iris_code.into(),
            thresholds.into(),
            digit_multiplicities.into(),
            lookup_table_values.into(),
        ];
        all_nodes.extend(digits_input_shreds.into_iter().map(|node| node.into()));

        // Add the identity gate node
        all_nodes.push(rerouted_image.into());

        // Add matmult node
        all_nodes.push(matmult.into());

        // Add the lookup nodes
        all_nodes.push(lookup_table.into());
        all_nodes.push(lookup_constraint.into());

        // Add nodes from components
        all_nodes.extend(subtract_thresholds.yield_nodes().into_iter());
        all_nodes.extend(digits_concatenator.yield_nodes().into_iter());
        all_nodes.extend(bits_are_binary.yield_nodes().into_iter());
        all_nodes.extend(recomp_of_abs_value.yield_nodes().into_iter());
        all_nodes.extend(sign_checker.yield_nodes().into_iter());

        // Add output nodes
        all_nodes.extend(output_nodes.into_iter().map(|node| node.into()));

        (
            ComponentSet::<NodeEnum<F>>::new_raw(all_nodes),
            vec![input_layer_data],
        )
    })
}
