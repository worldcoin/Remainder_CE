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
        builder.add_input_layer("LHS RHS input layer", LayerVisibility::Public);

    let lhs = builder.add_input_shred("LHS", 1, &lhs_rhs_input_layer);
    let matrix = builder.add_input_shred("Matrix", 2, &lhs_rhs_input_layer);
    let sq_sector = builder.add_matmult_node(&matrix, (1, 1), &lhs, (1, 0));
    let sq_sector = builder.add_sector(sq_sector + Fr::zero());
    let first = builder.add_split_node(&sq_sector, 1)[0].clone();

    // let multiplication_sector = lhs * rhs;
    let sub_sector = builder.add_sector(first.clone() - first);

    builder.set_output(&sub_sector);

    builder.build().expect("Failed to build circuit")
}

fn main() {
    // For tracing.
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    // Create the base layered circuit description.
    let base_circuit = build_circuit();
    let mut prover_circuit = base_circuit.clone();
    let verifier_circuit = base_circuit.clone();

    // Generate circuit inputs.
    let lhs_data = vec![1, 2].into();
    let matrix_data = vec![1; 4].into();

    // Append circuit inputs to their respective input "shreds" in the prover's
    // view of the circuit.
    prover_circuit.set_input("LHS", lhs_data);
    prover_circuit.set_input("Matrix", matrix_data);

    // Create a version of the circuit description which the prover can use.
    let provable_circuit = prover_circuit
        .gen_provable_circuit()
        .expect("Failed to generate provable circuit");

    let (proof_config, proof_as_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable_circuit);

    // ------------ VERIFIER ------------

    // Here we don't have any pre-determined public inputs from the verifier,
    // so we can directly call the `gen_verifiable_circuit()` function.
    let verifiable_circuit = verifier_circuit
        .gen_verifiable_circuit()
        .expect("Failed to generate verifiable circuit");

    verify_circuit_with_proof_config::<Fr, PoseidonSponge<Fr>>(
        &verifiable_circuit,
        &proof_config,
        proof_as_transcript,
    );

    println!("All done! GKR proof generated + verified.");
}
