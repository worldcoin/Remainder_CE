use std::collections::HashMap;

use remainder::{
    circuit_layout::{ProvableCircuit, VerifiableCircuit},
    layer::LayerId,
    mle::evals::MultilinearExtension,
    prover::{prove, verify},
};
use remainder_frontend::{
    abstract_expr::AbstractExpression,
    const_expr,
    layouter::builder::{Circuit, CircuitBuilder, LayerVisibility},
};
use remainder_shared_types::{
    config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig, ProofConfig},
    perform_function_under_prover_config, perform_function_under_verifier_config,
    transcript::{poseidon_sponge::PoseidonSponge, Transcript, TranscriptReader, TranscriptWriter},
    Fr,
};

use tracing::Level;
use tracing_subscriber::fmt;
use tracing_subscriber::{self};

fn build_circuit() -> Circuit<Fr> {
    let mut builder = CircuitBuilder::<Fr>::new();

    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);

    let lhs = builder.add_input_shred("LHS MLE", 0, &input_layer);
    let rhs = builder.add_input_shred("RHS MLE", 0, &input_layer);
    let expected_output = builder.add_input_shred("Expected Output MLE", 0, &input_layer);

    let main_sector_1 = builder.add_sector(
        // 2 * l * 2 * r * 2
        const_expr!(Fr::from(2)) * &lhs * Fr::from(2) * &rhs * Fr::from(2) - &expected_output,
    );
    builder.set_output(&main_sector_1);

    let main_sector_2 = builder.add_sector(
        // l * (3 + 5) * r
        &lhs * (Fr::from(3) + Fr::from(5)) * &rhs - &expected_output,
    );
    builder.set_output(&main_sector_2);

    let main_sector_3 = builder.add_sector(
        // (l + 3 * r) * (l + l + l) + l * -8
        (&lhs + &rhs * Fr::from(3)) * (&lhs + &lhs + &lhs) + &lhs * -const_expr!(Fr::from(8))
            - &expected_output,
    );
    builder.set_output(&main_sector_3);

    let main_sector_4 = builder.add_sector(
        // -(l + 3 * r) * -(l * l + l) + r + l * 3
        -(&lhs + &rhs * Fr::from(3)) * -(&lhs * &lhs + &lhs) + &rhs + &lhs * Fr::from(3)
            - &expected_output,
    );
    builder.set_output(&main_sector_4);

    let main_sector_5 = builder.add_sector(
        // (1 + l)^6 - r * 5 + 1
        AbstractExpression::pow(6, &lhs + Fr::from(1)) - &rhs * Fr::from(5) + Fr::from(1)
            - &expected_output,
    );
    builder.set_output(&main_sector_5);

    builder.build().unwrap()
}

fn prove_circuit(provable_circuit: &ProvableCircuit<Fr>) -> (Transcript<Fr>, ProofConfig) {
    let mut transcript_writer = TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("dummy label");

    let proof_config = prove(
        provable_circuit,
        remainder_shared_types::circuit_hash::CircuitHashType::Poseidon,
        &mut transcript_writer,
    )
    .expect("Proving failed!");

    let proof = transcript_writer.get_transcript();
    
    (proof, proof_config)
}

fn verify_circuit(
    verifiable_circuit: &VerifiableCircuit<Fr>,
    predetermined_public_inputs: HashMap<LayerId, MultilinearExtension<Fr>>,
    proof: Transcript<Fr>,
    proof_config: &ProofConfig,
) {
    let mut transcript_reader = TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(proof);

    verify(
        verifiable_circuit,
        predetermined_public_inputs,
        remainder_shared_types::circuit_hash::CircuitHashType::Poseidon,
        &mut transcript_reader,
        proof_config,
    )
    .expect("Verification Failed!");
}

#[test]
fn mult_overload_checks() {
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let prover_config = GKRCircuitProverConfig::memory_optimized_default();

    let mut circuit = perform_function_under_prover_config!(build_circuit, &prover_config,);

    let lhs_data = MultilinearExtension::<Fr>::new([1].into_iter().map(Fr::from).collect());
    let rhs_data = MultilinearExtension::<Fr>::new([5].into_iter().map(Fr::from).collect());
    let expected_output_data =
        MultilinearExtension::<Fr>::new([40].into_iter().map(Fr::from).collect());

    circuit.set_input("LHS MLE", lhs_data);
    circuit.set_input("RHS MLE", rhs_data);
    circuit.set_input("Expected Output MLE", expected_output_data);

    let provable_circuit = circuit.finalize().unwrap();

    let (proof, proof_config) =
        perform_function_under_prover_config!(prove_circuit, &prover_config, &provable_circuit);
    let verifier_config = GKRCircuitVerifierConfig::new_from_proof_config(&proof_config, true);

    let verifiable_circuit = provable_circuit._gen_verifiable_circuit();
    let predetermined_public_inputs = HashMap::new();

    perform_function_under_verifier_config!(
        verify_circuit,
        &verifier_config,
        &verifiable_circuit,
        predetermined_public_inputs,
        proof,
        &proof_config
    );
}
