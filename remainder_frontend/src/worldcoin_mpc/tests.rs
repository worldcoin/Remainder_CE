use remainder_shared_types::Fr;

use crate::layouter::builder::LayerVisibility;
use crate::worldcoin_mpc::circuits::build_circuit;
use crate::worldcoin_mpc::circuits::mpc_attach_data;
use crate::worldcoin_mpc::data::fetch_inversed_test_data;
use crate::worldcoin_mpc::data::generate_trivial_test_data;
use remainder::prover::helpers::test_circuit_with_memory_optimized_config;

#[test]
fn test_mpc_circuit_with_mock_data() {
    const NUM_IRIS_4_CHUNKS: usize = 1;
    const PARTY_IDX: usize = 0;

    // Create circuit description + input helper function
    let mut circuit = build_circuit::<Fr, NUM_IRIS_4_CHUNKS>(LayerVisibility::Committed);

    let (const_data, input_data) = generate_trivial_test_data::<Fr, NUM_IRIS_4_CHUNKS, PARTY_IDX>();

    // Convert input data into circuit inputs which are assignable by prover
    mpc_attach_data(&mut circuit, const_data, input_data);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    test_circuit_with_memory_optimized_config(&provable_circuit);
}

#[ignore] // takes a long time to run!
#[test]
fn test_mpc_circuit_with_inverse_data() {
    const NUM_IRIS_4_CHUNKS: usize = 1;
    const PARTY_IDX: usize = 0;

    // Create circuit description + input helper function
    let circuit = build_circuit::<Fr, NUM_IRIS_4_CHUNKS>(LayerVisibility::Committed);

    for test_idx in 10..12 {
        let mut circuit = circuit.clone();

        let (const_data, input_data) =
            fetch_inversed_test_data::<Fr, NUM_IRIS_4_CHUNKS, PARTY_IDX>(test_idx);

        mpc_attach_data(&mut circuit, const_data, input_data);

        let provable_circuit = circuit.gen_provable_circuit().unwrap();

        test_circuit_with_memory_optimized_config(&provable_circuit);
    }
}

#[ignore] // takes a long time to run!
#[test]
fn test_mpc_circuit_batched_with_mock_data() {
    const NUM_IRIS_4_CHUNKS: usize = 4;
    const PARTY_IDX: usize = 0;

    // Create circuit description + input helper function
    let mut circuit = build_circuit::<Fr, NUM_IRIS_4_CHUNKS>(LayerVisibility::Committed);

    let (const_data, input_data) = generate_trivial_test_data::<Fr, NUM_IRIS_4_CHUNKS, PARTY_IDX>();

    mpc_attach_data(&mut circuit, const_data, input_data);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    test_circuit_with_memory_optimized_config(&provable_circuit);
}

#[ignore] // takes a long time to run!
#[test]
fn test_mpc_circuit_batched_with_inverse_data() {
    const NUM_IRIS_4_CHUNKS: usize = 16;
    const PARTY_IDX: usize = 0;
    const TEST_IDX_START: usize = 2;

    // Create circuit description + input helper function
    let mut circuit = build_circuit::<Fr, NUM_IRIS_4_CHUNKS>(LayerVisibility::Committed);

    let (const_data, input_data) =
        fetch_inversed_test_data::<Fr, NUM_IRIS_4_CHUNKS, PARTY_IDX>(TEST_IDX_START);

    mpc_attach_data(&mut circuit, const_data, input_data);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    test_circuit_with_memory_optimized_config(&provable_circuit);
}

#[ignore] // takes a long time to run!
#[test]
fn test_mpc_circuit_batched_non_power_of_2_with_mock_data() {
    const NUM_IRIS_4_CHUNKS: usize = 3200;
    const PARTY_IDX: usize = 0;

    // Create circuit description + input helper function
    let mut circuit = build_circuit::<Fr, NUM_IRIS_4_CHUNKS>(LayerVisibility::Committed);

    let (const_data, input_data) = generate_trivial_test_data::<Fr, NUM_IRIS_4_CHUNKS, PARTY_IDX>();

    mpc_attach_data(&mut circuit, const_data, input_data);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    test_circuit_with_memory_optimized_config(&provable_circuit);
}

#[ignore] // takes a long time to run!
#[test]
fn test_mpc_circuit_batched_non_power_of_2_with_inverse_data() {
    const NUM_IRIS_4_CHUNKS: usize = 15;
    const PARTY_IDX: usize = 0;
    const TEST_IDX_START: usize = 2;

    // Create circuit description + input helper function
    let mut circuit = build_circuit::<Fr, NUM_IRIS_4_CHUNKS>(LayerVisibility::Committed);

    let (const_data, input_data) =
        fetch_inversed_test_data::<Fr, NUM_IRIS_4_CHUNKS, PARTY_IDX>(TEST_IDX_START);

    mpc_attach_data(&mut circuit, const_data, input_data);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    test_circuit_with_memory_optimized_config(&provable_circuit);
}
