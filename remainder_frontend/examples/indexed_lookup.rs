use remainder::{
    mle::evals::MultilinearExtension,
    prover::helpers::{
        prove_circuit_with_runtime_optimized_config, verify_circuit_with_proof_config,
    },
};
use remainder_frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility};
use remainder_shared_types::{transcript::poseidon_sponge::PoseidonSponge, Field, Fr};

fn build_example_indexed_lookup_circuit<F: Field>(
    table_num_vars: usize,
    witness_num_vars: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // Lookup table is typically public
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let table_input = builder.add_input_shred("Table input", table_num_vars, &public);
    let table_output = builder.add_input_shred("Table output", table_num_vars, &public);

    // Witness values are typically private, as are multiplicities
    let private = builder.add_input_layer("Private", LayerVisibility::Private);
    let witness_input = builder.add_input_shred("Witness input", witness_num_vars, &private);
    let witness_output = builder.add_input_shred("Witness output", witness_num_vars, &private);
    let multiplicities = builder.add_input_shred("Multiplicities", table_num_vars, &private);

    // A Fiat-Shamir challenge node is needed to combine input and output values
    let rlc_fiat_shamir_challenge_node = builder.add_fiat_shamir_challenge_node(1);

    // Combine input and output values for the indexed lookup
    let table_values =
        builder.add_sector(&table_input + &rlc_fiat_shamir_challenge_node * &table_output);
    let witness_values =
        builder.add_sector(&witness_input + &rlc_fiat_shamir_challenge_node * &witness_output);

    // Add the usual lookup components
    let logup_fiat_shamir_challenge_node = builder.add_fiat_shamir_challenge_node(1);
    let lookup_table = builder.add_lookup_table(&table_values, &logup_fiat_shamir_challenge_node);
    let _lookup_constraint =
        builder.add_lookup_constraint(&lookup_table, &witness_values, &multiplicities);

    builder.build().unwrap()
}

fn main() {
    // Uses an indexed lookup to check the application of a function defined by a lookup table.
    // The sigmoid function is used.
    // Inputs and outputs are both scaled and discretized: for all integers `2^9 <= i < 2^9`, the corresponding field element $i \in \mathbb{F}$ represents the real value $i / 2^8$.
    const TABLE_NUM_VARS: usize = 10;
    const WITNESS_NUM_VARS: usize = 2;
    let range_limit: i64 = 1 << (TABLE_NUM_VARS - 1);

    let sigmoid = |x: i64| -> i64 {
        // Sigmoid function scaled by 2^5
        let x_real = (x as f64) / 32.0;
        let sigmoid_real = 1.0 / (1.0 + (-x_real).exp());
        (sigmoid_real * 32.0).round() as i64
    };

    // The lookup table will contain the input and output values for the sigmoid for input values
    let input_values_mle: MultilinearExtension<Fr> =
        (-range_limit..range_limit).collect::<Vec<_>>().into();
    let output_values_mle: MultilinearExtension<Fr> = (-range_limit..range_limit)
        .map(|x| sigmoid(x))
        .collect::<Vec<_>>()
        .into();

    // Some example witness input values to be evaluated through the lookup table
    let witness_input_values = vec![-20i64, 0i64, 12i64, 12i64];
    let witness_output_values: Vec<i64> =
        witness_input_values.iter().map(|&x| sigmoid(x)).collect();
    let witness_input_mle: MultilinearExtension<Fr> = witness_input_values.clone().into();
    let witness_output_mle: MultilinearExtension<Fr> = witness_output_values.into();

    // Count the number of times each (input, output) pair occurs to build the multiplicities MLE.
    let mut multiplicities: Vec<u32> = vec![0; 1 << TABLE_NUM_VARS];
    witness_input_values.iter().for_each(|&input_value| {
        // Compute the index in the table for the (input, output) pair
        let index = input_value + range_limit;
        multiplicities[index as usize] += 1;
    });
    let multiplicities_mle: MultilinearExtension<Fr> = multiplicities.into();

    // Create circuit description
    let mut prover_circuit =
        build_example_indexed_lookup_circuit::<Fr>(TABLE_NUM_VARS, WITNESS_NUM_VARS);
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("Table input", input_values_mle.clone());
    prover_circuit.set_input("Table output", output_values_mle.clone());
    prover_circuit.set_input("Witness input", witness_input_mle.clone());
    prover_circuit.set_input("Witness output", witness_output_mle.clone());
    prover_circuit.set_input("Multiplicities", multiplicities_mle);

    let provable_circuit = prover_circuit.finalize().unwrap();

    // Prove the circuit
    let (proof_config, proof_as_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable_circuit);

    // Create verifier circuit description and attach lookup table as public
    // input to it.
    verifier_circuit.set_input("Table input", input_values_mle);
    verifier_circuit.set_input("Table output", output_values_mle);
    let (verifiable_circuit, predetermined_public_inputs) =
        verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(
        &verifiable_circuit,
        predetermined_public_inputs,
        &proof_config,
        proof_as_transcript,
    );
}
