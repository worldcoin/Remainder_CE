use crate::digits::components::{
    BitsAreBinary, ComplementaryRecompChecker, DigitsConcatenator, UnsignedRecomposition,
};
use crate::layouter::compiling::LayouterCircuit;
use crate::layouter::component::{Component, ComponentSet};
use crate::layouter::nodes::circuit_inputs::{InputLayerData, InputLayerNode, InputLayerType};
use crate::layouter::nodes::circuit_outputs::OutputNode;
use crate::layouter::nodes::identity_gate::IdentityGateNode;
use crate::layouter::nodes::lookup::{LookupConstraint, LookupTable};
use crate::layouter::nodes::matmult::MatMultNode;
use crate::layouter::nodes::node_enum::NodeEnum;
use crate::layouter::nodes::fiat_shamir::FiatShamirChallengeNode;
use crate::layouter::nodes::{CircuitNode, Context};
use crate::mle::bundled_input_mle::BundledInputMle;
use crate::utils::get_input_shred_and_data;
use crate::worldcoin::components::Subtractor;

use itertools::Itertools;
use remainder_shared_types::Field;

use super::data::CircuitData;

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
    data: CircuitData<
        F,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >,
) -> LayouterCircuit<
    F,
    ComponentSet<NodeEnum<F>>,
    impl FnMut(&Context) -> (ComponentSet<NodeEnum<F>>, Vec<InputLayerData<F>>),
> {
    LayouterCircuit::new(move |ctx| {
        let CircuitData {
            to_reroute,
            reroutings,
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
            get_input_shred_and_data(to_reroute.clone(), ctx, &input_layer);
        println!("{:?} = Image to_reroute input", to_reroute.id());
        let (to_sub_from_matmult, to_sub_from_matmult_data) =
            get_input_shred_and_data(to_sub_from_matmult.clone(), ctx, &input_layer);
        println!("{:?} = input to sub from matmult", to_sub_from_matmult.id());
        let rerouted_image = IdentityGateNode::new(ctx, &to_reroute, reroutings.clone());
        println!("{:?} = Identity gate", rerouted_image.id());

        let (rh_matmult_multiplicand, rh_matmult_multiplicand_data) =
            get_input_shred_and_data(rh_matmult_multiplicand.clone(), ctx, &input_layer);
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

        let (digits_input_shreds, digits_input_shreds_data) =
            digits.make_input_shred_and_data(ctx, &input_layer);
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
            LookupTable::new::<F>(ctx, &lookup_table_values, false, &fiat_shamir_challenge_node);
        println!("{:?} = Lookup table", lookup_table.id());
        let (digit_multiplicities, digit_multiplicities_data) =
            get_input_shred_and_data(digit_multiplicities.clone(), ctx, &input_layer);
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
            get_input_shred_and_data(sign_bits.clone(), ctx, &input_layer);
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
        let input_layer_data = InputLayerData::new(input_layer.id(), input_data_shreds, None);

        // Collect all the nodes, starting with the input nodes
        let mut all_nodes: Vec<NodeEnum<F>> = vec![
            input_layer.into(),
            fiat_shamir_challenge_node.into(),
            to_reroute.into(),
            rh_matmult_multiplicand.into(),
            sign_bits.into(),
            to_sub_from_matmult.into(),
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
