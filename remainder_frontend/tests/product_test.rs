use remainder::mle::evals::MultilinearExtension;
use remainder::prover::helpers::test_circuit_with_runtime_optimized_config;
use remainder_frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef};
use remainder_shared_types::{Field, Fr};

// Checks that factor1 * factor2 - expected_product == 0.
fn product_check<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    factor1: &NodeRef<F>,
    factor2: &NodeRef<F>,
    expected_product: &NodeRef<F>,
) -> NodeRef<F> {
    builder_ref.add_sector(factor1 * factor2 - expected_product)
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
///
/// Note that this additionally returns the [LayerId] of the Ligero input layer!
fn build_product_checker_test_circuit<F: Field>(num_free_vars: usize) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // The multiplicands are public...
    let public_input_layer_node = builder.add_input_layer(LayerVisibility::Public);
    let mle_1_shred = builder.add_input_shred("MLE 1", num_free_vars, &public_input_layer_node);
    let mle_2_shred = builder.add_input_shred("MLE 2", num_free_vars, &public_input_layer_node);

    // ...while the expected output is private
    let ligero_input_layer_node = builder.add_input_layer(LayerVisibility::Private);
    let mle_expected_shred =
        builder.add_input_shred("Expected MLE", num_free_vars, &ligero_input_layer_node);

    // Create the circuit components
    let checker = product_check(
        &mut builder,
        &mle_1_shred,
        &mle_2_shred,
        &mle_expected_shred,
    );
    let _output = builder.set_output(&checker);

    builder.build().unwrap()
}

#[test]
fn test_product_checker() {
    const NUM_FREE_VARS: usize = 2;

    let mle_1 = MultilinearExtension::new(vec![
        Fr::from(3u64),
        Fr::from(2u64),
        Fr::from(3u64),
        Fr::from(2u64),
    ]);
    let mle_2 = MultilinearExtension::new(vec![
        Fr::from(5u64),
        Fr::from(6u64),
        Fr::from(5u64),
        Fr::from(6u64),
    ]);
    let mle_expected = MultilinearExtension::new(vec![
        Fr::from(15u64),
        Fr::from(12u64),
        Fr::from(15u64),
        Fr::from(12u64),
    ]);

    // Create circuit description + input helper function
    let mut circuit = build_product_checker_test_circuit(NUM_FREE_VARS);

    circuit.set_input("MLE 1", mle_1);
    circuit.set_input("MLE 2", mle_2);
    circuit.set_input("Expected MLE", mle_expected);

    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config::<Fr, Fr>(&provable_circuit);
}
