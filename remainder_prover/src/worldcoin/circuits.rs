use std::collections::HashMap;

use crate::digits::components::{
    BitsAreBinary, ComplementaryRecompChecker, DigitsConcatenator, UnsignedRecomposition,
};
use crate::layer::LayerId;
use crate::layouter::compiling::LayouterCircuit;
use crate::layouter::component::{Component, ComponentSet};
use crate::layouter::nodes::circuit_inputs::{
    InputLayerNode, InputLayerNodeData, InputLayerType, InputShred,
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

/// Builds the iriscode circuit.
///
/// This circuit is satisfied iff the (entry-wise) complementary decomposition of the result of the
/// matrix multiplication, minus `to_sub_from_matmult`, has digits `digits` and sign bits
/// `sign_bits`.  In more detail:
/// + The matrix multiplication has two multiplicands.  The left-hand multiplicand is obtained by
///   selecting the entries of `to_reroute` in the manner specified by `wirings`; the right-hand
///   multiplicand is given by `rh_matmult_multiplicand`.  All matrix enumerations are row-major.
/// + The complementary decomposition is the analog of 2's complement, base `BASE` and with
///   `NUM_DIGITS` digits.  See [crate::digits::complementary_decomposition] and
///   [Notion](https://www.notion.so/Constraining-for-the-response-zero-case-using-the-complementary-representation-d77ddfe258a74a9ab949385cc6f7eda4).
/// + The sign bits are range checked using a degree two polynomial.  The digits are range-checked
///   using a lookup and the multiplicities `digit_multiplicities`.
///
/// The structure of the circuit is determined by:
/// + the `reroutings`, which determine how to build the MLE of left-hand multiplicand of the matrix
///   multiplication from the MLE of the input `to_reroute`;
/// + the generics `BASE`, `NUM_DIGITS`, `MATMULT_ROWS_NUM_VARS`, `MATMULT_COLS_NUM_VARS`,
///   `MATMULT_INTERNAL_DIM_NUM_VARS`;
/// + the length of the MLE `to_reroute`.
///
/// See [CircuitData] for a detailed description of each generic and argument with all public input layers.
pub fn build_circuit<
    F: Field,
    const MATMULT_ROWS_NUM_VARS: usize,
    const MATMULT_COLS_NUM_VARS: usize,
    const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
    const BASE: u64,
    const NUM_DIGITS: usize,
>(
    data: IriscodeCircuitData<F>,
    reroutings: Vec<(usize, usize)>,
) -> LayouterCircuit<
    F,
    ComponentSet<NodeEnum<F>>,
    impl FnMut(&Context) -> (ComponentSet<NodeEnum<F>>, Vec<InputLayerNodeData<F>>),
> {
    LayouterCircuit::new(move |ctx| {
        let IriscodeCircuitData {
            to_reroute,
            rh_matmult_multiplicand,
            digits,
            sign_bits,
            digit_multiplicities,
            to_sub_from_matmult,
        } = &data;
        let mut output_nodes = vec![];

        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        println!("{:?} = Input layer", input_layer.id());
        // TODO shouldn't have to clone here, but need to change library functions
        let (to_reroute, to_reroute_data) =
            build_input_shred_and_data(to_reroute.clone(), ctx, &input_layer);
        println!("{:?} = Image to_reroute input", to_reroute.id());
        let (to_sub_from_matmult, to_sub_from_matmult_data) =
            build_input_shred_and_data(to_sub_from_matmult.clone(), ctx, &input_layer);
        println!("{:?} = input to sub from matmult", to_sub_from_matmult.id());
        let rerouted_image = IdentityGateNode::new(ctx, &to_reroute, reroutings.clone());
        println!("{:?} = Identity gate", rerouted_image.id());

        let (rh_matmult_multiplicand, rh_matmult_multiplicand_data) =
            build_input_shred_and_data(rh_matmult_multiplicand.clone(), ctx, &input_layer);
        println!(
            "{:?} = Kernel values (RH multiplicand of matmult) input",
            rh_matmult_multiplicand.id()
        );

        let matmult = MatMultNode::new(
            ctx,
            &rerouted_image,
            (MATMULT_ROWS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS),
            &rh_matmult_multiplicand,
            (MATMULT_INTERNAL_DIM_NUM_VARS, MATMULT_COLS_NUM_VARS),
        );
        println!("{:?} = Matmult", matmult.id());

        let subtractor = Subtractor::new(ctx, &matmult, &to_sub_from_matmult);

        let (digits_input_shreds, digits_input_shreds_data): (Vec<_>, Vec<_>) = digits
            .iter()
            .map(|digit_values| build_input_shred_and_data(digit_values.clone(), ctx, &input_layer))
            .unzip();
        for (i, shred) in digits_input_shreds.iter().enumerate() {
            println!("{:?} = {}th digit input", shred.id(), i);
        }
        let digits_refs = digits_input_shreds
            .iter()
            .map(|shred| shred as &dyn CircuitNode)
            .collect_vec();
        // Concatenate the digits (which are stored for each digital place separately) into a single
        // MLE
        let digits_concatenator = DigitsConcatenator::new(ctx, &digits_refs);

        // Use a lookup to range check the digits to the range 0..BASE
        let (lookup_table_values, lookup_table_values_data) =
            get_input_shred_and_data((0..BASE).map(F::from).collect(), ctx, &input_layer);
        println!("{:?} = Digit range check input", lookup_table_values.id());

        let fiat_shamir_challenge_node = FiatShamirChallengeNode::new(ctx, 1);
        let lookup_table =
            LookupTable::new::<F>(ctx, &lookup_table_values, &fiat_shamir_challenge_node);
        println!("{:?} = Lookup table", lookup_table.id());
        let (digit_multiplicities, digit_multiplicities_data) =
            build_input_shred_and_data(digit_multiplicities.clone(), ctx, &input_layer);
        println!("{:?} = Digit multiplicities", digit_multiplicities.id());
        let lookup_constraint = LookupConstraint::new::<F>(
            ctx,
            &lookup_table,
            &&digits_concatenator.sector,
            &digit_multiplicities,
        );
        println!("{:?} = Lookup constraint", lookup_constraint.id());

        let unsigned_recomp = UnsignedRecomposition::new(ctx, &digits_refs, BASE);

        let (sign_bits, sign_bits_data) =
            build_input_shred_and_data(sign_bits.clone(), ctx, &input_layer);
        println!("{:?} = Sign bits (iris code) input", sign_bits.id());
        let complementary_checker = ComplementaryRecompChecker::new(
            ctx,
            &&subtractor.sector,
            &sign_bits,
            &&unsigned_recomp.sector,
            BASE,
            NUM_DIGITS,
        );
        output_nodes.push(OutputNode::new_zero(ctx, &complementary_checker.sector));

        let bits_are_binary = BitsAreBinary::new(ctx, &sign_bits);
        output_nodes.push(OutputNode::new_zero(ctx, &bits_are_binary.sector));

        let mut input_data_shreds = vec![
            to_reroute_data,
            rh_matmult_multiplicand_data,
            sign_bits_data,
            to_sub_from_matmult_data,
            digit_multiplicities_data,
            lookup_table_values_data,
        ];
        input_data_shreds.extend(digits_input_shreds_data);
        let input_layer_data = InputLayerNodeData::new(input_layer.id(), input_data_shreds, None);

        // Collect all the nodes, starting with the input nodes
        let mut all_nodes: Vec<NodeEnum<F>> = vec![
            input_layer.into(),
            // FIXME restore old order
            lookup_table_values.into(),
            fiat_shamir_challenge_node.into(),
            to_reroute.into(),
            rh_matmult_multiplicand.into(),
            sign_bits.into(),
            to_sub_from_matmult.into(),
            digit_multiplicities.into(),
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
        all_nodes.extend(subtractor.yield_nodes());
        all_nodes.extend(digits_concatenator.yield_nodes());
        all_nodes.extend(bits_are_binary.yield_nodes());
        all_nodes.extend(unsigned_recomp.yield_nodes());
        all_nodes.extend(complementary_checker.yield_nodes());

        // Add output nodes
        all_nodes.extend(output_nodes.into_iter().map(|node| node.into()));

        (
            ComponentSet::<NodeEnum<F>>::new_raw(all_nodes),
            vec![input_layer_data],
        )
    })
}

/// Builds the iriscode circuit, return the circuit description, the input node map and the node ids
/// of the public and the private input layers.
pub fn build_circuit_description<
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
) {
    assert!(BASE.is_power_of_two());
    let log_base = BASE.ilog2() as usize;
    let mut output_nodes = vec![];
    let ctx = Context::new();

    // Private inputs
    // FIXME(Ben) this will be fine once we get rid of InputLayerType, but it does look funny for now
    let private_input_layer = InputLayerNode::new(&ctx, None, InputLayerType::PublicInputLayer);
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
    let public_input_layer = InputLayerNode::new(&ctx, None, InputLayerType::PublicInputLayer);
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

    let (circ_desc, _, input_builder_from_shred_map) =
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

    (circ_desc, input_builder)
}
