use ark_std::test_rng;

use remainder::prover::helpers::test_circuit_with_runtime_optimized_config;
use remainder_frontend::{
    layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef},
    sel_expr,
};
use remainder_shared_types::Field;
use utils::get_dummy_random_mle;

use crate::utils::TestUtilComponents;
pub mod utils;

/// A builder which returns the following expression:
/// ```text
/// sel(sel(left_inner_sel_mle, right_inner_sel_mle), right_outer_sel_mle)
///     + right_prod_mle_1 * right_prod_mle_2
/// ```
///
/// The idea is that this builder has two selector bits which are nonlinear.
///
/// ## Arguments
/// * `left_inner_sel_mle` - An MLE with arbitrary bookkeeping table values.
/// * `right_inner_sel_mle` - An MLE with arbitrary bookkeeping table values, same size as `left_inner_sel_mle`.
/// * `right_outer_sel_mle` - An MLE with arbitrary bookkeeping table values, one more variable
/// than `right_inner_sel_mle`.
/// * `right_prod_mle_1`, `right_prod_mle_2` - MLEs with arbitrary bookkeeping table values, same size,
/// one more variable than `right_outer_sel_mle`.
pub fn nonlinear_nested_selector<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    left_inner_sel_mle: &NodeRef<F>,
    right_inner_sel_mle: &NodeRef<F>,
    right_outer_sel_mle: &NodeRef<F>,
    right_prod_mle_1: &NodeRef<F>,
    right_prod_mle_2: &NodeRef<F>,
) -> NodeRef<F> {
    builder_ref.add_sector(
        sel_expr!(left_inner_sel_mle, right_inner_sel_mle, right_outer_sel_mle)
            + right_prod_mle_1 * right_prod_mle_2,
    )
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
fn build_nonlinear_nested_sel_test_circuit<F: Field>(
    num_vars_product_side: usize,
    num_vars_outer_sel_side: usize,
    num_vars_inner_sel_side: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // All inputs are public
    let public_input_layer_node = builder.add_input_layer(LayerVisibility::Public);

    // Inputs to the circuit are the MLEs at the leaves of the nested selector + product
    let left_inner_sel_mle_shred = builder.add_input_shred(
        "Left Inner Selector MLE",
        num_vars_inner_sel_side,
        &public_input_layer_node,
    );
    let right_inner_sel_mle_shred = builder.add_input_shred(
        "Right Inner Selector MLE",
        num_vars_inner_sel_side,
        &public_input_layer_node,
    );
    let right_outer_sel_mle_shred = builder.add_input_shred(
        "Right Outer Selector MLE",
        num_vars_outer_sel_side,
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
    let component_1 = nonlinear_nested_selector(
        &mut builder,
        &left_inner_sel_mle_shred,
        &right_inner_sel_mle_shred,
        &right_outer_sel_mle_shred,
        &right_prod_mle_1_shred,
        &right_prod_mle_2_shred,
    );
    let _component_2 = TestUtilComponents::difference(&mut builder, &component_1);

    builder.build().unwrap()
}

/// A circuit which does the following:
/// * Layer 0: [NonlinearNestedSelectorBuilderComponent] with all inputs.
/// * Layer 1: [DifferenceBuilderComponent] with output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// See [NonlinearNestedSelectorBuilderComponent].
#[test]
fn test_nonlinear_nested_sel_circuit_newmainder() {
    const NUM_VARS_PRODUCT_SIDE: usize = 5;
    const NUM_VARS_OUTER_SEL_SIDE: usize = NUM_VARS_PRODUCT_SIDE - 1;
    const NUM_VARS_INNER_SEL_SIDE: usize = NUM_VARS_OUTER_SEL_SIDE - 1;
    let mut rng = test_rng();

    let left_inner_sel_mle = get_dummy_random_mle(NUM_VARS_INNER_SEL_SIDE, &mut rng).mle;
    let right_inner_sel_mle = get_dummy_random_mle(NUM_VARS_INNER_SEL_SIDE, &mut rng).mle;
    let right_outer_sel_mle = get_dummy_random_mle(NUM_VARS_OUTER_SEL_SIDE, &mut rng).mle;
    let right_prod_mle_1 = get_dummy_random_mle(NUM_VARS_PRODUCT_SIDE, &mut rng).mle;
    let right_prod_mle_2 = get_dummy_random_mle(NUM_VARS_PRODUCT_SIDE, &mut rng).mle;

    // Create circuit description + input helper function
    let mut circuit = build_nonlinear_nested_sel_test_circuit(
        NUM_VARS_PRODUCT_SIDE,
        NUM_VARS_OUTER_SEL_SIDE,
        NUM_VARS_INNER_SEL_SIDE,
    );

    circuit.set_input("Left Inner Selector MLE", left_inner_sel_mle);
    circuit.set_input("Right Inner Selector MLE", right_inner_sel_mle);
    circuit.set_input("Right Outer Selector MLE", right_outer_sel_mle);
    circuit.set_input("Right Product MLE 1", right_prod_mle_1);
    circuit.set_input("Right Product MLE 2", right_prod_mle_2);

    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
