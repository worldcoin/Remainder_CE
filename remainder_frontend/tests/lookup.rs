use remainder::{
    mle::evals::MultilinearExtension, prover::helpers::test_circuit_with_runtime_optimized_config,
};
use remainder_frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility};
use remainder_shared_types::{Field, Fr};

pub mod utils;

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
///
/// Note that this function also returns the layer ID of the Ligero input layer!
fn build_single_shred_lookup_test_circuit<F: Field>(
    table_mle_num_vars: usize,
    witness_mle_num_vars: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // Lookup table is public
    let public_input_layer_node = builder.add_input_layer(LayerVisibility::Public);
    let table_mle_shred =
        builder.add_input_shred("Table MLE", table_mle_num_vars, &public_input_layer_node);

    // Witness values are private, as are multiplicities
    let ligero_input_layer_node = builder.add_input_layer(LayerVisibility::Private);
    let witness_mle_shred = builder.add_input_shred(
        "Witness MLE",
        witness_mle_num_vars,
        &ligero_input_layer_node,
    );
    let multiplicities_mle_shred = builder.add_input_shred(
        "Multiplicities MLE",
        table_mle_num_vars,
        &ligero_input_layer_node,
    );

    // Create the circuit components
    let fiat_shamir_challenge_node = builder.add_fiat_shamir_challenge_node(1);
    let lookup_table = builder.add_lookup_table(&table_mle_shred, &fiat_shamir_challenge_node);
    let _lookup_constraint =
        builder.add_lookup_constraint(&lookup_table, &witness_mle_shred, &multiplicities_mle_shred);

    builder.build().unwrap()
}

/// Test the case where there is only one LookupConstraint for the LookupTable i.e. just one constrained
/// MLE.
#[test]
pub fn single_shred_test() {
    const TABLE_MLE_NUM_VARS: usize = 1;
    const WITNESS_MLE_NUM_VARS: usize = 2;

    // Input generation
    let table_mle = MultilinearExtension::new(vec![Fr::from(0u64), Fr::from(1u64)]);
    let witness_mle = MultilinearExtension::new(vec![
        Fr::from(0u64),
        Fr::from(1u64),
        Fr::from(1u64),
        Fr::from(1u64),
    ]);
    let multiplicities_mle = MultilinearExtension::new(vec![Fr::from(1u64), Fr::from(3u64)]);

    // Create circuit description + input helper function
    let mut circuit =
        build_single_shred_lookup_test_circuit(TABLE_MLE_NUM_VARS, WITNESS_MLE_NUM_VARS);

    circuit.set_input("Table MLE", table_mle);
    circuit.set_input("Witness MLE", witness_mle);
    circuit.set_input("Multiplicities MLE", multiplicities_mle);

    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

/// Test the case where there is only one LookupConstraint for the LookupTable i.e. just one constrained
/// MLE.
#[test]
pub fn single_shred_test_non_power_of_2() {
    const TABLE_MLE_NUM_VARS: usize = 1;
    const WITNESS_MLE_NUM_VARS: usize = 2;

    // Input generation
    let table_mle = MultilinearExtension::new(vec![Fr::from(0u64), Fr::from(1u64)]);
    let witness_mle =
        MultilinearExtension::new(vec![Fr::from(0u64), Fr::from(1u64), Fr::from(1u64)]);
    let multiplicities_mle = MultilinearExtension::new(vec![Fr::from(2u64), Fr::from(2u64)]);

    // Create circuit description + input helper function
    let mut circuit =
        build_single_shred_lookup_test_circuit(TABLE_MLE_NUM_VARS, WITNESS_MLE_NUM_VARS);

    circuit.set_input("Table MLE", table_mle);
    circuit.set_input("Witness MLE", witness_mle);
    circuit.set_input("Multiplicities MLE", multiplicities_mle);
    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the multi-input lookup test circuit.
///
/// Note that this function also returns the layer ID of the Ligero input layer!
fn build_multi_shred_lookup_test_circuit<F: Field>(
    table_mle_num_vars: usize,
    witness_mle_1_num_vars: usize,
    witness_mle_2_num_vars: usize,
    witness_mle_3_num_vars: usize,
    witness_mle_4_num_vars: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // Lookup table is public
    let public_input_layer_node = builder.add_input_layer(LayerVisibility::Public);
    let table_mle_shred =
        builder.add_input_shred("Table MLE", table_mle_num_vars, &public_input_layer_node);

    // Witness values are private, as are multiplicities
    let ligero_input_layer_node = builder.add_input_layer(LayerVisibility::Private);

    let witness_mle_1_shred = builder.add_input_shred(
        "Witness MLE 1",
        witness_mle_1_num_vars,
        &ligero_input_layer_node,
    );
    let multiplicities_mle_1_shred = builder.add_input_shred(
        "Multiplicities MLE 1",
        table_mle_num_vars,
        &ligero_input_layer_node,
    );

    let witness_mle_2_shred = builder.add_input_shred(
        "Witness MLE 2",
        witness_mle_2_num_vars,
        &ligero_input_layer_node,
    );
    let multiplicities_mle_2_shred = builder.add_input_shred(
        "Multiplicities MLE 2",
        table_mle_num_vars,
        &ligero_input_layer_node,
    );

    let witness_mle_3_shred = builder.add_input_shred(
        "Witness MLE 3",
        witness_mle_3_num_vars,
        &ligero_input_layer_node,
    );
    let multiplicities_mle_3_shred = builder.add_input_shred(
        "Multiplicities MLE 3",
        table_mle_num_vars,
        &ligero_input_layer_node,
    );

    let witness_mle_4_shred = builder.add_input_shred(
        "Witness MLE 4",
        witness_mle_4_num_vars,
        &ligero_input_layer_node,
    );
    let multiplicities_mle_4_shred = builder.add_input_shred(
        "Multiplicities MLE 4",
        table_mle_num_vars,
        &ligero_input_layer_node,
    );

    // Create the circuit components
    let fiat_shamir_challenge_node = builder.add_fiat_shamir_challenge_node(1);
    let lookup_table = builder.add_lookup_table(&table_mle_shred, &fiat_shamir_challenge_node);
    let _lookup_constraint_1 = builder.add_lookup_constraint(
        &lookup_table,
        &witness_mle_1_shred,
        &multiplicities_mle_1_shred,
    );
    let _lookup_constraint_2 = builder.add_lookup_constraint(
        &lookup_table,
        &witness_mle_2_shred,
        &multiplicities_mle_2_shred,
    );
    let _lookup_constraint_3 = builder.add_lookup_constraint(
        &lookup_table,
        &witness_mle_3_shred,
        &multiplicities_mle_3_shred,
    );
    let _lookup_constraint_4 = builder.add_lookup_constraint(
        &lookup_table,
        &witness_mle_4_shred,
        &multiplicities_mle_4_shred,
    );

    builder.build().unwrap()
}

/// Test the lookup functionality when there are multiple LookupConstraints for the same LookupTable.
#[test]
pub fn multi_shred_test() {
    const TABLE_MLE_NUM_VARS: usize = 1;
    const WITNESS_MLE_1_NUM_VARS: usize = 2;
    const WITNESS_MLE_2_NUM_VARS: usize = 2;
    const WITNESS_MLE_3_NUM_VARS: usize = 2;
    const WITNESS_MLE_4_NUM_VARS: usize = 2;

    // Input generation
    let table_mle = MultilinearExtension::new(vec![Fr::from(3u64), Fr::from(4u64)]);
    let witness_mle_1 = MultilinearExtension::new(vec![
        Fr::from(3u64),
        Fr::from(3u64),
        Fr::from(3u64),
        Fr::from(4u64),
    ]);
    let multiplicities_mle_1 = MultilinearExtension::new(vec![Fr::from(3u64), Fr::from(1u64)]);
    let witness_mle_2 = MultilinearExtension::new(vec![
        Fr::from(4u64),
        Fr::from(4u64),
        Fr::from(4u64),
        Fr::from(4u64),
    ]);
    let multiplicities_mle_2 = MultilinearExtension::new(vec![Fr::from(0u64), Fr::from(4u64)]);
    let witness_mle_3 = MultilinearExtension::new(vec![
        Fr::from(4u64),
        Fr::from(3u64),
        Fr::from(3u64),
        Fr::from(4u64),
    ]);
    let multiplicities_mle_3 = MultilinearExtension::new(vec![Fr::from(2u64), Fr::from(2u64)]);
    let witness_mle_4 = MultilinearExtension::new(vec![
        Fr::from(4u64),
        Fr::from(4u64),
        Fr::from(4u64),
        Fr::from(3u64),
    ]);
    let multiplicities_mle_4 = MultilinearExtension::new(vec![Fr::from(1u64), Fr::from(3u64)]);

    // Create circuit description + input helper function
    let mut circuit = build_multi_shred_lookup_test_circuit(
        TABLE_MLE_NUM_VARS,
        WITNESS_MLE_1_NUM_VARS,
        WITNESS_MLE_2_NUM_VARS,
        WITNESS_MLE_3_NUM_VARS,
        WITNESS_MLE_4_NUM_VARS,
    );

    circuit.set_input("Table MLE", table_mle);
    circuit.set_input("Witness MLE 1", witness_mle_1);
    circuit.set_input("Multiplicities MLE 1", multiplicities_mle_1);
    circuit.set_input("Witness MLE 2", witness_mle_2);
    circuit.set_input("Multiplicities MLE 2", multiplicities_mle_2);
    circuit.set_input("Witness MLE 3", witness_mle_3);
    circuit.set_input("Multiplicities MLE 3", multiplicities_mle_3);
    circuit.set_input("Witness MLE 4", witness_mle_4);
    circuit.set_input("Multiplicities MLE 4", multiplicities_mle_4);

    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

/// Test that a panic occurs when the constrained MLE contains values not in the lookup table.
#[test]
#[should_panic]
pub fn test_not_satisfied() {
    const TABLE_MLE_NUM_VARS: usize = 1;
    const WITNESS_MLE_NUM_VARS: usize = 2;

    // Input generation
    let table_mle = MultilinearExtension::new(vec![Fr::from(0u64), Fr::from(1u64)]);
    let witness_mle = MultilinearExtension::new(vec![
        Fr::from(3u64),
        Fr::from(1u64),
        Fr::from(1u64),
        Fr::from(1u64),
    ]);
    let multiplicities_mle = MultilinearExtension::new(vec![Fr::from(1u64), Fr::from(3u64)]);

    // Create circuit description + input helper function
    let mut circuit =
        build_single_shred_lookup_test_circuit(TABLE_MLE_NUM_VARS, WITNESS_MLE_NUM_VARS);

    circuit.set_input("Table MLE", table_mle);
    circuit.set_input("Witness MLE", witness_mle);
    circuit.set_input("Multiplicities MLE", multiplicities_mle);
    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
