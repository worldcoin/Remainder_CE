use ark_std::test_rng;

use remainder::{
    layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef},
    prover::helpers::test_circuit_with_runtime_optimized_config,
    sel_expr,
};
use remainder_shared_types::Field;
use utils::{get_dummy_random_mle, TestUtilComponents};

pub mod utils;

/// A builder which returns the following expression:
/// `sel(mle_1, mle_1) + mle_2 * mle_2`
///
/// The idea is that the last bit in this expression is linear.
///
/// ## Arguments
/// * `sel_mle` - An MLE with arbitrary bookkeeping table values.
/// * `prod_mle` - An MLE with arbitrary bookkeeping table values; same size as `sel_mle`.
pub fn last_var_linear<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    sel_node: &NodeRef<F>,
    prod_node: &NodeRef<F>,
) -> NodeRef<F> {
    builder_ref.add_sector(sel_expr!(sel_node, sel_node) + prod_node * prod_node)
}

/// A builder which returns the following expression:
/// `sel(mle_1 * mle_1, mle_1)`
///
/// The idea is that the first bit (selector bit) in this expression is linear.
///
/// ## Arguments
/// * `sel_mle` - An MLE with arbitrary bookkeeping table values.
pub fn first_var_linear<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    sel_node: &NodeRef<F>,
) -> NodeRef<F> {
    builder_ref.add_sector(sel_expr!(sel_node * sel_node, sel_node))
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
fn build_linear_and_nonlinear_vars_test_circuit<F: Field>(
    selector_mle_num_vars: usize,
    product_mle_num_vars: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // All inputs are public
    let public_input_layer_node = builder.add_input_layer(LayerVisibility::Public);

    // "Semantic" circuit inputs
    let selector_mle_shred = builder.add_input_shred(
        "Selector MLE",
        selector_mle_num_vars,
        &public_input_layer_node,
    );
    let product_mle_shred = builder.add_input_shred(
        "Product MLE",
        product_mle_num_vars,
        &public_input_layer_node,
    );

    // Create the circuit components
    let component_1 = last_var_linear(&mut builder, &selector_mle_shred, &product_mle_shred);
    let component_2 = first_var_linear(&mut builder, &component_1);
    let _output_component = TestUtilComponents::difference(&mut builder, &component_2);

    builder.build().unwrap()
}

/// A circuit which does the following:
/// * Layer 0: [LastVarLinearBuilderComponent] with `sel_mle`, `prod_mle`
/// * Layer 1: [FirstVarLinearBuilderComponent] with `sel_mle`
/// * Layer 2: [DifferenceBuilderComponent] with the output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `sel_mle`, `prod_mle` both MLEs with arbitrary bookkeeping table values, same size.

#[test]
fn test_linear_and_nonlinear_vars_circuit_newmainder() {
    const SELECTOR_MLE_NUM_VARS: usize = 1;
    const PRODUCT_MLE_NUM_VARS: usize = SELECTOR_MLE_NUM_VARS;
    let mut rng = test_rng();

    // Generate circuit inputs
    let selector_mle = get_dummy_random_mle(SELECTOR_MLE_NUM_VARS, &mut rng).mle;
    let product_mle = get_dummy_random_mle(PRODUCT_MLE_NUM_VARS, &mut rng).mle;

    // Create circuit description + input helper function
    let mut circuit =
        build_linear_and_nonlinear_vars_test_circuit(SELECTOR_MLE_NUM_VARS, PRODUCT_MLE_NUM_VARS);

    circuit.set_input("Selector MLE", selector_mle);
    circuit.set_input("Product MLE", product_mle);

    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
