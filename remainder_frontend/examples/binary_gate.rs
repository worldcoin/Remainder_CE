use remainder::{
    layer::gate::BinaryOperation,
    mle::evals::MultilinearExtension,
    prover::helpers::{
        prove_circuit_with_runtime_optimized_config, verify_circuit_with_proof_config,
    },
};
use remainder_frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility};
use remainder_shared_types::{transcript::poseidon_sponge::PoseidonSponge, Field, Fr};

fn build_example_binary_gate_circuit<F: Field>(
    input_num_vars_lhs: usize,
    input_num_vars_rhs: usize,
    wiring: Vec<(u32, u32, u32)>,
    binary_operation: BinaryOperation,
    output_num_vars: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    // The left-hand side candidates for the input to the binary gate
    let lhs_input = builder.add_input_shred(
        "LHS candidates for binary gate",
        input_num_vars_lhs,
        &public,
    );
    // The right-hand side candidates for the input to the binary gate
    let rhs_input = builder.add_input_shred(
        "RHS candidates for binary gate",
        input_num_vars_rhs,
        &public,
    );
    // The expected output of the gate operation
    let expected_output = builder.add_input_shred("Expected output", output_num_vars, &public);

    let gate_result = builder.add_gate_node(&lhs_input, &rhs_input, wiring, binary_operation, None);

    let output = builder.add_sector(gate_result - expected_output);
    builder.set_output(&output);

    builder.build().unwrap()
}

fn main() {
    const LHS_NUM_VARS: usize = 3;
    const RHS_NUM_VARS: usize = 2;
    const OUTPUT_NUM_VARS: usize = 2;

    // Example inputs to the gate function
    let lhs_mle: MultilinearExtension<Fr> = vec![5, 7, 2, 9, 13, 1, 11, 2].into();
    let rhs_mle: MultilinearExtension<Fr> = vec![11, 13, 15, 3].into();
    // Example wiring
    let wiring = vec![
        (0, 0, 1),
        (0, 1, 3),
        (1, 5, 3),
        (2, 6, 2),
        (2, 7, 1),
        (3, 2, 0),
    ];
    let expected_output_mle: MultilinearExtension<Fr> = vec![28, 4, 41, 13].into();

    // Create circuit description
    let mut prover_circuit = build_example_binary_gate_circuit::<Fr>(
        LHS_NUM_VARS,
        RHS_NUM_VARS,
        wiring,
        BinaryOperation::Add,
        OUTPUT_NUM_VARS,
    );
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("LHS candidates for binary gate", lhs_mle.clone());
    prover_circuit.set_input("RHS candidates for binary gate", rhs_mle.clone());
    prover_circuit.set_input("Expected output", expected_output_mle.clone());

    let provable_circuit = prover_circuit.finalize().unwrap();

    // Prove the circuit
    let (proof_config, proof_as_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable_circuit);

    // Create verifier circuit description and attach inputs.
    verifier_circuit.set_input("LHS candidates for binary gate", lhs_mle);
    verifier_circuit.set_input("RHS candidates for binary gate", rhs_mle);
    verifier_circuit.set_input("Expected output", expected_output_mle);

    let (verifiable_circuit, predetermined_public_inputs) =
        verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(
        &verifiable_circuit,
        predetermined_public_inputs,
        &proof_config,
        proof_as_transcript,
    );
}
