use ark_std::test_rng;

use frontend::{
    layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef},
    sel_expr,
};
use remainder::prover::helpers::test_circuit_with_runtime_optimized_config;
use shared_types::Field;
use utils::get_dummy_random_mle;

use crate::utils::TestUtilComponents;
pub mod utils;

/// A builder which returns the following expression:
/// `sel(left_sel_mle, right_sel_mle) + right_prod_mle_1 * right_prod_mle_2`
///
/// The idea is that this builder has one selector bit which is nonlinear.
///
/// ## Arguments
/// * `left_sel_mle` - An MLE with arbitrary bookkeeping table values.
/// * `right_sel_mle` - An MLE with arbitrary bookkeeping table values, same size as `left_sel_mle`.
/// * `right_prod_mle_1`, `right_prod_mle_2` - MLEs with arbitrary bookkeeping table values, same size,
/// one more variable than `right_sel_mle`.
pub fn nonlinear_selector<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    left_sel_mle: &NodeRef<F>,
    right_sel_mle: &NodeRef<F>,
    right_prod_mle_1: &NodeRef<F>,
    right_prod_mle_2: &NodeRef<F>,
) -> NodeRef<F> {
    builder_ref
        .add_sector(sel_expr!(left_sel_mle, right_sel_mle) + right_prod_mle_1 * right_prod_mle_2)
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving.
fn build_nonlinear_selector_test_circuit<F: Field>(
    num_vars_product_side: usize,
    num_vars_sel_side: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // All inputs are public
    let public_input_layer_node =
        builder.add_input_layer("Public Input Layer", LayerVisibility::Public);

    // Inputs to the circuit are the MLEs at the leaves of the selector + product
    let left_sel_mle_shred = builder.add_input_shred(
        "Left Selector MLE",
        num_vars_sel_side,
        &public_input_layer_node,
    );
    let right_sel_mle_shred = builder.add_input_shred(
        "Right Selector MLE",
        num_vars_sel_side,
        &public_input_layer_node,
    );
    let right_prod_mle_1_shred = builder.add_input_shred(
        "Right Product MLE 1",
        num_vars_product_side,
        &public_input_layer_node,
    );
    let right_prod_mle_2_shred = builder.add_input_shred(
        "Right Product MLE 2",
        num_vars_product_side,
        &public_input_layer_node,
    );

    // Create the circuit components
    let component_1 = nonlinear_selector(
        &mut builder,
        &left_sel_mle_shred,
        &right_sel_mle_shred,
        &right_prod_mle_1_shred,
        &right_prod_mle_2_shred,
    );
    let _component_2 = TestUtilComponents::difference(&mut builder, &component_1);

    builder.build_with_layer_combination().unwrap()
}

/// A circuit which does the following:
/// * Layer 0: [NonlinearSelectorBuilderComponent] with all inputs.
/// * Layer 1: [DifferenceBuilderComponent] with output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// See [NonlinearSelectorBuilderComponent].
#[test]
fn test_nonlinear_sel_circuit_newmainder() {
    const NUM_VARS_PRODUCT_SIDE: usize = 3;
    const NUM_VARS_SEL_SIDE: usize = NUM_VARS_PRODUCT_SIDE - 1;
    let mut rng = &mut test_rng();

    let left_sel_mle = get_dummy_random_mle(NUM_VARS_SEL_SIDE, &mut rng).mle;
    let right_sel_mle = get_dummy_random_mle(NUM_VARS_SEL_SIDE, &mut rng).mle;
    let right_prod_mle_1 = get_dummy_random_mle(NUM_VARS_PRODUCT_SIDE, &mut rng).mle;
    let right_prod_mle_2 = get_dummy_random_mle(NUM_VARS_PRODUCT_SIDE, &mut rng).mle;

    // Create circuit description + input helper function
    let mut circuit =
        build_nonlinear_selector_test_circuit(NUM_VARS_PRODUCT_SIDE, NUM_VARS_SEL_SIDE);

    circuit.set_input("Left Selector MLE", left_sel_mle);
    circuit.set_input("Right Selector MLE", right_sel_mle);
    circuit.set_input("Right Product MLE 1", right_prod_mle_1);
    circuit.set_input("Right Product MLE 2", right_prod_mle_2);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
