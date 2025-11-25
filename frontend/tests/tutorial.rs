use frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility};
use remainder::prover::helpers::{
    prove_circuit_with_runtime_optimized_config, verify_circuit_with_proof_config,
};
use shared_types::transcript::poseidon_sponge::PoseidonSponge;
use shared_types::Fr;

use tracing::Level;
use tracing_subscriber::fmt;
use tracing_subscriber::{self};

fn build_circuit() -> Circuit<Fr> {
    let mut builder = CircuitBuilder::<Fr>::new();

    let lhs_rhs_input_layer =
        builder.add_input_layer("LHS RHS input layer", LayerVisibility::Committed);
    let expected_output_input_layer =
        builder.add_input_layer("Expected output", LayerVisibility::Public);

    let lhs = builder.add_input_shred("LHS", 2, &lhs_rhs_input_layer);
    let rhs = builder.add_input_shred("RHS", 2, &lhs_rhs_input_layer);
    let expected_output =
        builder.add_input_shred("Expected output", 2, &expected_output_input_layer);

    // let multiplication_sector = lhs * rhs;
    let multiplication_sector = builder.add_sector(lhs * rhs);

    let subtraction_sector = builder.add_sector(multiplication_sector - expected_output);

    builder.set_output(&subtraction_sector);

    builder.build_with_layer_combination().unwrap()
}

#[test]
fn tutorial_test() {
    // For tracing.
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    // Create the base layered circuit description.
    let base_circuit = build_circuit();
    let mut prover_circuit = base_circuit.clone();
    let verifier_circuit = base_circuit.clone();

    // Generate circuit inputs.
    let lhs_data = vec![1, 2, 3, 4].into();
    let rhs_data = vec![5, 6, 7, 8].into();
    let expected_output_data = vec![5, 12, 21, 32].into();

    // Append circuit inputs to their respective input "shreds" in the prover's
    // view of the circuit.
    prover_circuit.set_input("LHS", lhs_data); // This is committed!
    prover_circuit.set_input("RHS", rhs_data); // This is committed!
    prover_circuit.set_input("Expected output", expected_output_data); // This is public!

    // Create a version of the circuit description which the prover can use.
    let provable_circuit = prover_circuit.gen_provable_circuit().unwrap();

    let (proof_config, proof_as_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable_circuit);

    // ------------ VERIFIER ------------

    // Here we don't have any pre-determined public inputs from the verifier,
    // so we can directly call the `gen_verifiable_circuit()` function.
    let verifiable_circuit = verifier_circuit.gen_verifiable_circuit().unwrap();

    verify_circuit_with_proof_config::<Fr, PoseidonSponge<Fr>>(
        &verifiable_circuit,
        &proof_config,
        proof_as_transcript,
    );
}
