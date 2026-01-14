use frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility};
use remainder::{
    mle::evals::MultilinearExtension,
    prover::helpers::{
        prove_circuit_with_runtime_optimized_config, verify_circuit_with_proof_config,
    },
};
use shared_types::{transcript::poseidon_sponge::PoseidonSponge, Field, Fr};

fn build_example_lookup_circuit<F: Field>(
    table_num_vars: usize,
    witness_num_vars: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // Lookup table is typically public
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let table = builder.add_input_shred("Table", table_num_vars, &public);

    // Witness values are typically committed, as are multiplicities
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);
    let witness = builder.add_input_shred("Witness", witness_num_vars, &committed);
    let multiplicities = builder.add_input_shred("Multiplicities", table_num_vars, &committed);

    // Create the circuit components
    let fiat_shamir_challenge_node = builder.add_fiat_shamir_challenge_node(1);
    let lookup_table = builder.add_lookup_table(&table, &fiat_shamir_challenge_node);
    let _lookup_constraint =
        builder.add_lookup_constraint(&lookup_table, &witness, &multiplicities);

    builder.build().unwrap()
}

/// Example demonstrating a range check using a lookup table.
fn main() {
    const TABLE_NUM_VARS: usize = 8;
    const WITNESS_NUM_VARS: usize = 2;
    const RANGE_LIMIT: u64 = 1 << TABLE_NUM_VARS; // 256

    // The lookup table contains the values 0 thru 255
    let table_mle = MultilinearExtension::new((0u64..RANGE_LIMIT).map(|x| Fr::from(x)).collect());
    // Some example witness values to be range checked
    let witness_values = vec![233u64, 233u64, 0u64, 1u64];
    // Count the number of times each value occurs to build the multiplicities MLE.
    let mut multiplicities: Vec<u32> = vec![0; RANGE_LIMIT as usize];
    witness_values.iter().for_each(|value| {
        multiplicities[*value as usize] += 1;
    });
    let witness_mle: MultilinearExtension<Fr> = witness_values.into();
    let multiplicities_mle: MultilinearExtension<Fr> = multiplicities.into();

    // Create circuit description
    let mut prover_circuit = build_example_lookup_circuit::<Fr>(TABLE_NUM_VARS, WITNESS_NUM_VARS);
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("Table", table_mle.clone());
    prover_circuit.set_input("Witness", witness_mle);
    prover_circuit.set_input("Multiplicities", multiplicities_mle);

    let provable_circuit = prover_circuit.gen_provable_circuit().unwrap();

    // Prove the circuit
    let (proof_config, proof_as_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable_circuit);

    // Create verifier circuit description and attach lookup table as public
    // input to it.
    verifier_circuit.set_input("Table", table_mle);
    let verifiable_circuit = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable_circuit, &proof_config, proof_as_transcript);
}
