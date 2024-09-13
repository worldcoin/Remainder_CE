use crate::digits::components::{
    BitsAreBinary, ComplementaryRecompChecker, DigitsConcatenator, UnsignedRecomposition,
};
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
use crate::utils::mle::get_input_shred_from_vec;
use crate::worldcoin::components::Subtractor;
use crate::worldcoin::data::CircuitData;
use itertools::Itertools;
use remainder_shared_types::Field;

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
/// + the generics `BASE`, `NUM_DIGITS`, `MATMULT_NUM_ROWS`, `MATMULT_NUM_COLS`,
///   `MATMULT_INTERNAL_DIM`;
/// + the length of the MLE `to_reroute`.
///
/// See [CircuitData] for a detailed description of each generic and argument.
pub fn build_circuit<
    F: FieldExt,
    const MATMULT_NUM_ROWS_VARS: usize,
    const MATMULT_NUM_COLS_VARS: usize,
    const MATMULT_INTERNAL_DIM_VARS: usize,
    const BASE: u64,
    const NUM_DIGITS: usize,
>(
    data: CircuitData<
        F,
        MATMULT_NUM_ROWS_VARS,
        MATMULT_NUM_COLS_VARS,
        MATMULT_INTERNAL_DIM_VARS,
        BASE,
        NUM_DIGITS,
    >,
) -> LayouterCircuit<F, ComponentSet<NodeEnum<F>>, impl FnMut(&Context) -> ComponentSet<NodeEnum<F>>>
{
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
        let to_reroute = get_input_shred_from_vec(to_reroute.clone(), ctx, &input_layer);
        println!("{:?} = Image to_reroute input", to_reroute.id());
        let to_sub_from_matmult =
            get_input_shred_from_vec(to_sub_from_matmult.clone(), ctx, &input_layer);
        println!("{:?} = input to sub from matmult", to_sub_from_matmult.id());
        let rerouted_image = IdentityGateNode::new(ctx, &to_reroute, reroutings.clone());
        println!("{:?} = Identity gate", rerouted_image.id());

        let rh_matmult_multiplicand =
            get_input_shred_from_vec(rh_matmult_multiplicand.clone(), ctx, &input_layer);
        println!(
            "{:?} = Kernel values (RH multiplicand of matmult) input",
            rh_matmult_multiplicand.id()
        );

        let matmult = MatMultNode::new(
            ctx,
            &rerouted_image,
            (MATMULT_NUM_ROWS_VARS, MATMULT_INTERNAL_DIM_VARS),
            &rh_matmult_multiplicand,
            (MATMULT_INTERNAL_DIM_VARS, MATMULT_NUM_COLS_VARS),
        );
        println!("{:?} = Matmult", matmult.id());

        let subtractor = Subtractor::new(ctx, &matmult, &to_sub_from_matmult);

        let digits_input_shreds = digits.make_input_shreds(ctx, &input_layer);
        for (i, shred) in digits_input_shreds.iter().enumerate() {
            println!("{:?} = {}th digit input", shred.id(), i);
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
        println!("{:?} = Digit range check input", lookup_table_values.id());
        let lookup_table = LookupTable::new(ctx, &lookup_table_values, false);
        println!("{:?} = Lookup table", lookup_table.id());
        let digit_multiplicities =
            get_input_shred_from_vec(digit_multiplicities.clone(), ctx, &input_layer);
        println!("{:?} = Digit multiplicities", digit_multiplicities.id());
        let lookup_constraint = LookupConstraint::new(
            ctx,
            &lookup_table,
            &digits_concatenator.sector,
            &digit_multiplicities,
        );
        println!("{:?} = Lookup constraint", lookup_constraint.id());

        let unsigned_recomp = UnsignedRecomposition::new(ctx, &digits_refs, BASE as u64);

        let sign_bits = get_input_shred_from_vec(sign_bits.clone(), ctx, &input_layer);
        println!("{:?} = Sign bits (iris code) input", sign_bits.id());
        let complementary_checker = ComplementaryRecompChecker::new(
            ctx,
            &subtractor.sector,
            &sign_bits,
            &unsigned_recomp.sector,
            BASE as u64,
            NUM_DIGITS,
        );
        output_nodes.push(OutputNode::new_zero(ctx, &complementary_checker.sector));

        let bits_are_binary = BitsAreBinary::new(ctx, &sign_bits);
        output_nodes.push(OutputNode::new_zero(ctx, &bits_are_binary.sector));

        // Collect all the nodes, starting with the input nodes
        let mut all_nodes: Vec<NodeEnum<F>> = vec![
            input_layer.into(),
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
        all_nodes.extend(subtractor.yield_nodes().into_iter());
        all_nodes.extend(digits_concatenator.yield_nodes().into_iter());
        all_nodes.extend(bits_are_binary.yield_nodes().into_iter());
        all_nodes.extend(unsigned_recomp.yield_nodes().into_iter());
        all_nodes.extend(complementary_checker.yield_nodes().into_iter());

        // Add output nodes
        all_nodes.extend(output_nodes.into_iter().map(|node| node.into()));

        ComponentSet::<NodeEnum<F>>::new_raw(all_nodes)
    })
}
