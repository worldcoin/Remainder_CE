use remainder::{
    mle::evals::MultilinearExtension,
    prover::helpers::{
        prove_circuit_with_runtime_optimized_config, verify_circuit_with_proof_config,
    },
};
use remainder_frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility};
use remainder_shared_types::{transcript::poseidon_sponge::PoseidonSponge, Field, Fr};

fn build_example_identity_gate_circuit_dataparallel<F: Field>(
    num_dataparallel_vars: usize,
    source_num_vars: usize,
    wiring: Vec<(u32, u32)>,
    output_num_vars: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    // The MLE that we are routing via the wiring
    let source = builder.add_input_shred("Source for identity gate", source_num_vars, &public);
    // Expected routing result from the wiring
    let expected_output = builder.add_input_shred("Expected output", output_num_vars, &public);

    let gate_result = builder.add_identity_gate_node(
        &source,
        wiring,
        output_num_vars,
        Some(num_dataparallel_vars),
    );

    let output = builder.add_sector(gate_result - expected_output);
    builder.set_output(&output);

    builder.build().unwrap()
}

fn main() {
    const NUM_DATAPARALLEL_VARS: usize = 1;
    const SOURCE_NUM_VARS: usize = 3;
    const OUTPUT_NUM_VARS: usize = 2;

    // Example input
    let source_mle: MultilinearExtension<Fr> = vec![5, 7, 2, 9, 13, 1, 11, 2].into();
    // Example wiring. This is repeated across (1 << [NUM_DATAPARALLEL_VARS]) copies of the circuit.
    let wiring = vec![(0, 1), (0, 3), (1, 2)];
    let expected_output_mle: MultilinearExtension<Fr> = vec![16, 2, 3, 11].into();

    // Create circuit description
    let mut prover_circuit = build_example_identity_gate_circuit_dataparallel::<Fr>(
        NUM_DATAPARALLEL_VARS,
        SOURCE_NUM_VARS,
        wiring,
        OUTPUT_NUM_VARS,
    );
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("Source for identity gate", source_mle.clone());
    prover_circuit.set_input("Expected output", expected_output_mle.clone());

    let provable_circuit = prover_circuit.finalize().unwrap();

    // Prove the circuit
    let (proof_config, proof_as_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable_circuit);

    // Create verifier circuit description and attach inputs.
    verifier_circuit.set_input("Source for identity gate", source_mle);
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
