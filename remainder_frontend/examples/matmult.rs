use remainder::{
    mle::evals::MultilinearExtension,
    prover::helpers::{
        prove_circuit_with_runtime_optimized_config, verify_circuit_with_proof_config,
    },
};
use remainder_frontend::layouter::builder::{CircuitBuilder, LayerVisibility};
use remainder_shared_types::{transcript::poseidon_sponge::PoseidonSponge, Fr};

fn main() {
    const PADDED_MATRIX_A_LOG_NUM_ROWS: usize = 2;
    const PADDED_MATRIX_A_LOG_NUM_COLS: usize = 2;
    const PADDED_MATRIX_B_LOG_NUM_ROWS: usize = 2;
    const PADDED_MATRIX_B_LOG_NUM_COLS: usize = 1;

    const MATRIX_A_NUM_VARS: usize = 4;
    const MATRIX_B_NUM_VARS: usize = 3;
    const MATRIX_C_NUM_VARS: usize = 3;

    let matrix_a_data: MultilinearExtension<Fr> = vec![0, 1, 2, 1, 2, 3, 2, 3, 4].into();
    let matrix_b_data: MultilinearExtension<Fr> = vec![3, 4, 4, 5, 5, 6].into();
    let matrix_c_data: MultilinearExtension<Fr> = vec![14, 17, 26, 32, 38, 47].into();

    let matrix_a_padding_wiring = vec![
        (0, 0),
        (1, 1),
        (2, 2),
        (4, 3),
        (5, 4),
        (6, 5),
        (8, 6),
        (9, 7),
        (10, 8),
    ];

    let mut builder = CircuitBuilder::<Fr>::new();

    let inputs = builder.add_input_layer("Matrices", LayerVisibility::Public);

    let matrix_a = builder.add_input_shred("Matrix A", MATRIX_A_NUM_VARS, &inputs);
    let matrix_b = builder.add_input_shred("Matrix B", MATRIX_B_NUM_VARS, &inputs);
    let expected_matrix_c =
        builder.add_input_shred("Expected Matrix C", MATRIX_C_NUM_VARS, &inputs);

    let padded_matrix_a =
        builder.add_identity_gate_node(&matrix_a, matrix_a_padding_wiring, MATRIX_A_NUM_VARS, None);

    let matrix_c = builder.add_matmult_node(
        &padded_matrix_a,
        (PADDED_MATRIX_A_LOG_NUM_ROWS, PADDED_MATRIX_A_LOG_NUM_COLS),
        &matrix_b,
        (PADDED_MATRIX_B_LOG_NUM_ROWS, PADDED_MATRIX_B_LOG_NUM_COLS),
    );

    let output = builder.add_sector(matrix_c - expected_matrix_c);
    builder.set_output(&output);

    let circuit = builder.build().unwrap();

    // Create circuit description.
    let mut prover_circuit = circuit.clone();
    let mut verifier_circuit = circuit.clone();

    prover_circuit.set_input("Matrix A", matrix_a_data.clone());
    prover_circuit.set_input("Matrix B", matrix_b_data.clone());
    prover_circuit.set_input("Expected Matrix C", matrix_c_data.clone());

    let provable_circuit = prover_circuit.gen_provable_circuit().unwrap();

    // Prove the circuit.
    let (proof_config, proof_as_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable_circuit);

    // Create verifier circuit description and attach inputs.
    verifier_circuit.set_input("Matrix A", matrix_a_data);
    verifier_circuit.set_input("Matrix B", matrix_b_data);
    verifier_circuit.set_input("Expected Matrix C", matrix_c_data);

    let verifiable_circuit = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable_circuit, &proof_config, proof_as_transcript);
}
