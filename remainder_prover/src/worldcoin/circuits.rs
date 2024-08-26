use crate::layouter::compiling::LayouterCircuit;
use crate::layouter::component::{Component, ComponentSet};
use crate::layouter::nodes::circuit_inputs::{InputLayerNode, InputLayerType};
use crate::layouter::nodes::circuit_outputs::OutputNode;
use crate::layouter::nodes::identity_gate::IdentityGateNode;
use crate::layouter::nodes::lookup::{LookupConstraint, LookupTable};
use crate::layouter::nodes::matmult::MatMultNode;
use crate::layouter::nodes::node_enum::NodeEnum;
use crate::layouter::nodes::{CircuitNode, ClaimableNode, Context};
use crate::mle::circuit_mle::CircuitMle;
use crate::utils::get_input_shred_from_vec;
use crate::utils::pad_to_nearest_power_of_two;
use crate::digits::components::{ComplementaryRecompChecker, UnsignedRecomposition, BitsAreBinary, DigitsConcatenator};
use crate::worldcoin::components::Thresholder;
use crate::worldcoin::data::WorldcoinCircuitData;
use crate::worldcoin::{BASE, NUM_DIGITS};
use itertools::Itertools;
use remainder_shared_types::FieldExt;


/// Builds the worldcoin circuit.
pub fn build_circuit<F: FieldExt>(
    data: WorldcoinCircuitData<F>,
) -> LayouterCircuit<F, ComponentSet<NodeEnum<F>>, impl FnMut(&Context) -> ComponentSet<NodeEnum<F>>>
{
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
            equality_allowed,
        } = &data;
        let mut output_nodes = vec![];

        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        println!("Input layer = {:?}", input_layer.id());
        let image = get_input_shred_from_vec(image_matrix_mle.clone(), ctx, &input_layer);
        println!("Image input = {:?}", image.id());
        let thresholds = get_input_shred_from_vec(
            pad_to_nearest_power_of_two(thresholds_matrix.clone()),
            ctx,
            &input_layer,
        );
        println!("Thresholds input = {:?}", thresholds.id());
        let equality_allowed = get_input_shred_from_vec(vec![equality_allowed.clone()], ctx, &input_layer);
        println!("Equality allowed input = {:?}", equality_allowed.id());

        let rerouted_image = IdentityGateNode::new(ctx, &image, wirings.clone());
        println!("Identity gate = {:?}", rerouted_image.id());

        let (filter_num_values, _) = kernel_matrix_dims;
        let matrix_a_num_rows_cols = (num_placements.next_power_of_two(), *filter_num_values);
        let kernel_matrix = get_input_shred_from_vec(kernel_matrix.clone(), ctx, &input_layer);
        println!("Kernel values input = {:?}", kernel_matrix.id());

        let matmult = MatMultNode::new(
            ctx,
            &rerouted_image,
            matrix_a_num_rows_cols,
            &kernel_matrix,
            *kernel_matrix_dims,
        );
        println!("Matmult = {:?}", matmult.id());

        let thresholder = Thresholder::new(ctx, &matmult, &thresholds, &equality_allowed);

        let digits_input_shreds = digits.make_input_shreds(ctx, &input_layer);
        for (i, shred) in digits_input_shreds.iter().enumerate() {
            println!("{}th digit input = {:?}", i, shred.id());
        }
        let digits_refs = digits_input_shreds
            .iter()
            .map(|shred| shred as &dyn ClaimableNode<F = F>)
            .collect_vec();
        // Concatenate the digits (which are stored for each digital place separately) into a single
        // MLE
        let digits_concatenator = DigitsConcatenator::new(ctx, &digits_refs);

        // Use a lookup to range check the digits to the range 0..BASE
        let lookup_table_values =
            get_input_shred_from_vec((0..BASE as u64).map(F::from).collect(), ctx, &input_layer);
        println!("Digit range check input = {:?}", lookup_table_values.id());
        let lookup_table = LookupTable::new(ctx, &lookup_table_values, false);
        println!("Lookup table = {:?}", lookup_table.id());
        let digit_multiplicities =
            get_input_shred_from_vec(digit_multiplicities.clone(), ctx, &input_layer);
        println!("Digit multiplicities = {:?}", digit_multiplicities.id());
        let lookup_constraint = LookupConstraint::new(
            ctx,
            &lookup_table,
            &digits_concatenator.sector,
            &digit_multiplicities,
        );
        println!("Lookup constraint = {:?}", lookup_constraint.id());

        let unsigned_recomp = UnsignedRecomposition::new(ctx, &digits_refs, BASE as u64);

        let iris_code = get_input_shred_from_vec(
            pad_to_nearest_power_of_two(iris_code.clone()),
            ctx,
            &input_layer,
        );
        println!("Iris code input = {:?}", iris_code.id());
        let complementary_checker = ComplementaryRecompChecker::new(
            ctx,
            &thresholder.sector,
            &iris_code,
            &unsigned_recomp.sector,
            BASE as u64,
            NUM_DIGITS,
        );
        output_nodes.push(OutputNode::new_zero(ctx, &complementary_checker.sector));

        let bits_are_binary = BitsAreBinary::new(ctx, &iris_code);
        output_nodes.push(OutputNode::new_zero(ctx, &bits_are_binary.sector));

        // Collect all the nodes, starting with the input nodes
        let mut all_nodes: Vec<NodeEnum<F>> = vec![
            input_layer.into(),
            image.into(),
            kernel_matrix.into(),
            iris_code.into(),
            thresholds.into(),
            digit_multiplicities.into(),
            lookup_table_values.into(),
            equality_allowed.into(),
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
        all_nodes.extend(thresholder.yield_nodes().into_iter());
        all_nodes.extend(digits_concatenator.yield_nodes().into_iter());
        all_nodes.extend(bits_are_binary.yield_nodes().into_iter());
        all_nodes.extend(unsigned_recomp.yield_nodes().into_iter());
        all_nodes.extend(complementary_checker.yield_nodes().into_iter());

        // Add output nodes
        all_nodes.extend(output_nodes.into_iter().map(|node| node.into()));

        ComponentSet::<NodeEnum<F>>::new_raw(all_nodes)
    })
}
