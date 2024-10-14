use std::collections::HashMap;

use crate::digits::components::{
    BitsAreBinary, ComplementaryRecompChecker, DigitsConcatenator, UnsignedRecomposition,
};
use crate::input_layer::InputLayerDescription;
use crate::layer::LayerId;
use crate::layouter::compiling::LayouterCircuit;
use crate::layouter::component::{Component, ComponentSet};
use crate::layouter::nodes::circuit_inputs::{
    InputLayerNode, InputLayerNodeData, InputShred,
};
use crate::layouter::nodes::circuit_outputs::OutputNode;
use crate::layouter::nodes::fiat_shamir_challenge::FiatShamirChallengeNode;
use crate::layouter::nodes::identity_gate::IdentityGateNode;
use crate::layouter::nodes::lookup::{LookupConstraint, LookupTable};
use crate::layouter::nodes::matmult::MatMultNode;
use crate::layouter::nodes::node_enum::NodeEnum;
use crate::layouter::nodes::{CircuitNode, Context, NodeId};
use crate::mle::evals::MultilinearExtension;
use crate::prover::{generate_circuit_description, GKRCircuitDescription};
use crate::utils::{build_input_shred_and_data, get_input_shred_and_data};
use crate::worldcoin::components::Subtractor;

use itertools::Itertools;
use remainder_shared_types::Field;

use super::data::IriscodeCircuitData;

/// Builds the iriscode circuit, return the circuit description, the input node map and the input
/// layer ids of the input layers for the public and private inputs.
pub fn build_iriscode_circuit_description<
    F: Field,
    const TO_REROUTE_NUM_VARS: usize,
    const MATMULT_ROWS_NUM_VARS: usize,
    const MATMULT_COLS_NUM_VARS: usize,
    const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
    const BASE: u64,
    const NUM_DIGITS: usize,
>(
    reroutings: Vec<(usize, usize)>,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(IriscodeCircuitData<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
    InputLayerDescription
) {
    assert!(BASE.is_power_of_two());
    let log_base = BASE.ilog2() as usize;
    let mut output_nodes = vec![];
    let ctx = Context::new();

    // Private inputs
    let private_input_layer = InputLayerNode::new(&ctx, None);
    println!(
        "{:?} = Input layer for private values",
        private_input_layer.id()
    );
    let to_reroute = InputShred::new(&ctx, TO_REROUTE_NUM_VARS, &private_input_layer);
    println!("{:?} = Image to_reroute input", to_reroute.id());

    let digits_input_shreds: Vec<_> = (0..NUM_DIGITS)
        .map(|i| {
            let shred = InputShred::new(
                &ctx,
                MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS,
                &private_input_layer,
            );
            println!("{:?} = {}th digit input", shred.id(), i);
            shred
        })
        .collect();

    let digit_multiplicities = InputShred::new(&ctx, log_base, &private_input_layer);
    println!("{:?} = Digit multiplicities", digit_multiplicities.id());

    let sign_bits = InputShred::new(
        &ctx,
        MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS,
        &private_input_layer,
    );
    println!("{:?} = Sign bits (iris code) input", sign_bits.id());

    // Public inputs
    let public_input_layer = InputLayerNode::new(&ctx, None);
    println!(
        "{:?} = Input layer for public values",
        public_input_layer.id()
    );

    let to_sub_from_matmult = InputShred::new(
        &ctx,
        MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS,
        &public_input_layer,
    );
    println!("{:?} = input to sub from matmult", to_sub_from_matmult.id());

    let rh_matmult_multiplicand = InputShred::new(
        &ctx,
        MATMULT_INTERNAL_DIM_NUM_VARS + MATMULT_COLS_NUM_VARS,
        &public_input_layer,
    );
    println!(
        "{:?} = RH multiplicand of matmult (input)",
        rh_matmult_multiplicand.id()
    );

    let lookup_table_values = InputShred::new(&ctx, log_base, &public_input_layer);
    println!(
        "{:?} = Lookup table values for digit range check (input)",
        lookup_table_values.id()
    );

    // Intermediate layers
    let rerouted_image = IdentityGateNode::new(&ctx, &to_reroute, reroutings);
    println!("{:?} = Identity gate", rerouted_image.id());

    let matmult = MatMultNode::new(
        &ctx,
        &rerouted_image,
        (MATMULT_ROWS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS),
        &rh_matmult_multiplicand,
        (MATMULT_INTERNAL_DIM_NUM_VARS, MATMULT_COLS_NUM_VARS),
    );
    println!("{:?} = Matmult", matmult.id());

    let subtractor = Subtractor::new(&ctx, &matmult, &to_sub_from_matmult);

    // Concatenate the digits (which are stored for each digital place separately) into a single
    // MLE
    let digits_refs = digits_input_shreds
        .iter()
        .map(|shred| shred as &dyn CircuitNode)
        .collect_vec();
    let digits_concatenator = DigitsConcatenator::new(&ctx, &digits_refs);

    let fiat_shamir_challenge_node = FiatShamirChallengeNode::new(&ctx, 1);
    let lookup_table =
        LookupTable::new::<F>(&ctx, &lookup_table_values, &fiat_shamir_challenge_node);
    println!("{:?} = Lookup table", lookup_table.id());

    let lookup_constraint = LookupConstraint::new::<F>(
        &ctx,
        &lookup_table,
        &&digits_concatenator.sector,
        &digit_multiplicities,
    );
    println!("{:?} = Lookup constraint", lookup_constraint.id());

    let unsigned_recomp = UnsignedRecomposition::new(&ctx, &digits_refs, BASE);

    let complementary_checker = ComplementaryRecompChecker::new(
        &ctx,
        &&subtractor.sector,
        &sign_bits,
        &&unsigned_recomp.sector,
        BASE,
        NUM_DIGITS,
    );
    output_nodes.push(OutputNode::new_zero(&ctx, &complementary_checker.sector));

    let bits_are_binary = BitsAreBinary::new(&ctx, &sign_bits);
    output_nodes.push(OutputNode::new_zero(&ctx, &bits_are_binary.sector));

    // Collect all the nodes, starting with the input nodes
    let mut all_nodes: Vec<NodeEnum<F>> = vec![
        private_input_layer.clone().into(),
        public_input_layer.clone().into(),
        fiat_shamir_challenge_node.clone().into(),
        to_reroute.clone().into(),
        rh_matmult_multiplicand.clone().into(),
        sign_bits.clone().into(),
        to_sub_from_matmult.clone().into(),
        digit_multiplicities.clone().into(),
        lookup_table_values.clone().into(),
    ];
    all_nodes.extend(digits_input_shreds.iter().map(|node| node.clone().into()));

    // Add the identity gate node
    all_nodes.push(rerouted_image.into());

    // Add matmult node
    all_nodes.push(matmult.into());

    // Add the lookup nodes
    all_nodes.push(lookup_table.into());
    all_nodes.push(lookup_constraint.into());

    // Add nodes from components
    all_nodes.extend(subtractor.yield_nodes());
    all_nodes.extend(digits_concatenator.yield_nodes());
    all_nodes.extend(bits_are_binary.yield_nodes());
    all_nodes.extend(unsigned_recomp.yield_nodes());
    all_nodes.extend(complementary_checker.yield_nodes());

    // Add output nodes
    all_nodes.extend(output_nodes.into_iter().map(|node| node.into()));

    let (circ_desc, input_builder_from_shred_map, input_node_id_to_layer_id) =
        generate_circuit_description(all_nodes).unwrap();

    let input_builder = move |data: IriscodeCircuitData<F>| {
        let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<F>> = HashMap::new();
        input_shred_id_to_data.insert(to_reroute.id(), data.to_reroute);
        input_shred_id_to_data.insert(rh_matmult_multiplicand.id(), data.rh_matmult_multiplicand);
        data.digits
            .into_iter()
            .zip(digits_input_shreds.iter())
            .for_each(|(mle, shred)| {
                input_shred_id_to_data.insert(shred.id(), mle);
            });
        input_shred_id_to_data.insert(sign_bits.id(), data.sign_bits);
        input_shred_id_to_data.insert(to_sub_from_matmult.id(), data.to_sub_from_matmult);
        input_shred_id_to_data.insert(digit_multiplicities.id(), data.digit_multiplicities);
        input_shred_id_to_data.insert(
            lookup_table_values.id(),
            MultilinearExtension::new((0..BASE).map(F::from).collect()),
        );
        input_builder_from_shred_map(input_shred_id_to_data).unwrap()
    };

    let private_input_layer_id = input_node_id_to_layer_id.get(&private_input_layer.id()).unwrap();
    let private_input_layer = circ_desc.input_layers.iter().find(|il| il.layer_id == *private_input_layer_id).unwrap().clone();
    (
        circ_desc,
        input_builder,
        private_input_layer,
    )
}
