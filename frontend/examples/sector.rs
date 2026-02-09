use frontend::{
    layouter::builder::{Circuit, CircuitBuilder, LayerVisibility},
    sel_expr,
};
use remainder::{
    mle::evals::MultilinearExtension,
    prover::helpers::{
        prove_circuit_with_runtime_optimized_config, verify_circuit_with_proof_config,
    },
};
use shared_types::{transcript::poseidon_sponge::PoseidonSponge, Field, Fr};

// ========== Example 1a: Naive Sector Circuit ==========

fn build_naive_sector_circuit<F: Field>() -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);

    // Input MLEs V1 and V2.
    let v1 = builder.add_input_shred("MLE 1", 2, &input_layer);
    let v2 = builder.add_input_shred("MLE 2", 2, &input_layer);

    // Expected Output MLE.
    let expected_output = builder.add_input_shred("Expected Output", 2, &input_layer);

    let v3 = builder.add_sector(v1 * v2);
    let v4 = builder.add_sector(&v3 * &v3);

    // OR
    // let v4 = builder.add_sector((&v1 * &v2) * (v1 * v2));

    // Subtract the expected output from the sector output and set as circuit output.
    let output = builder.add_sector(v4 - expected_output);
    builder.set_output(&output);

    builder.build().unwrap()
}

fn run_example_1a() {
    let v1: MultilinearExtension<Fr> = vec![1, 2, 3, 4].into();
    let v2: MultilinearExtension<Fr> = vec![5, 6, 7, 8].into();
    let expected_v4: MultilinearExtension<Fr> = vec![25, 144, 441, 1024].into();

    // Create circuit description.
    let mut prover_circuit = build_naive_sector_circuit::<Fr>();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("MLE 1", v1.clone());
    prover_circuit.set_input("MLE 2", v2.clone());
    prover_circuit.set_input("Expected Output", expected_v4.clone());

    let provable_circuit = prover_circuit.gen_provable_circuit().unwrap();

    // Prove the circuit.
    let (proof_config, proof_as_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable_circuit);

    // Create verifier circuit description and attach inputs.
    verifier_circuit.set_input("MLE 1", v1);
    verifier_circuit.set_input("MLE 2", v2);
    verifier_circuit.set_input("Expected Output", expected_v4);

    let verifiable_circuit = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable_circuit, &proof_config, proof_as_transcript);
}

// ========== Example 1b: Multiplying MLEs of different sizes ==========

fn build_heterogenous_sector_circuit<F: Field>() -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);

    // Input MLEs V1 and V2.
    let v1 = builder.add_input_shred("MLE 1", 2, &input_layer);
    let v2 = builder.add_input_shred("MLE 2", 1, &input_layer);

    // Expected Output MLE.
    let expected_output = builder.add_input_shred("Expected Output", 2, &input_layer);

    let v3 = builder.add_sector(v1 * v2);

    // Subtract the expected output from the sector output and set as circuit output.
    let output = builder.add_sector(v3 - expected_output);
    builder.set_output(&output);

    builder.build().unwrap()
}

fn run_example_1b() {
    let v1: MultilinearExtension<Fr> = vec![1, 2, 3, 4].into();
    let v2: MultilinearExtension<Fr> = vec![5, 6].into();
    let expected_v3: MultilinearExtension<Fr> = vec![1 * 5, 2 * 5, 3 * 6, 4 * 6].into();

    // Create circuit description.
    let mut prover_circuit = build_heterogenous_sector_circuit::<Fr>();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("MLE 1", v1.clone());
    prover_circuit.set_input("MLE 2", v2.clone());
    prover_circuit.set_input("Expected Output", expected_v3.clone());

    let provable_circuit = prover_circuit.gen_provable_circuit().unwrap();

    // Prove the circuit.
    let (proof_config, proof_as_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable_circuit);

    // Create verifier circuit description and attach inputs.
    verifier_circuit.set_input("MLE 1", v1);
    verifier_circuit.set_input("MLE 2", v2);
    verifier_circuit.set_input("Expected Output", expected_v3);

    let verifiable_circuit = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable_circuit, &proof_config, proof_as_transcript);
}

// ========== Example 2: Using a Split Node ==========

fn build_sector_with_split_node<F: Field>() -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);

    // The Input MLE [a, b, c, d, e, f, g, h].
    let mle = builder.add_input_shred("Input MLE", 3, &input_layer);

    // Expected Output MLE.
    let expected_output = builder.add_input_shred("Expected Output", 2, &input_layer);

    // Split the MLE into two halves: left = [a, b, c, d] and right = [e, f, g, h].
    let [left, right]: [_; 2] = builder.add_split_node(&mle, 1).try_into().unwrap();

    // Multiply the two halves together using a sector node and the expression `left * right`,
    // producing the output MLE [a*e, b*f, c*g, d*h].
    let sector = builder.add_sector(left * right);

    // Subtract the expected output from the sector output and set as circuit output.
    let output = builder.add_sector(sector - expected_output);
    builder.set_output(&output);

    builder.build().unwrap()
}

fn run_example_2() {
    let input_mle: MultilinearExtension<Fr> = vec![1, 2, 3, 4, 5, 6, 7, 8].into();
    let expected_output_mle: MultilinearExtension<Fr> = vec![1 * 5, 2 * 6, 3 * 7, 4 * 8].into();

    // Create circuit description.
    let mut prover_circuit = build_sector_with_split_node::<Fr>();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("Input MLE", input_mle.clone());
    prover_circuit.set_input("Expected Output", expected_output_mle.clone());

    let provable_circuit = prover_circuit.gen_provable_circuit().unwrap();

    // Prove the circuit.
    let (proof_config, proof_as_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable_circuit);

    // Create verifier circuit description and attach inputs.
    verifier_circuit.set_input("Input MLE", input_mle);
    verifier_circuit.set_input("Expected Output", expected_output_mle);

    let verifiable_circuit = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable_circuit, &proof_config, proof_as_transcript);
}

// ========== Example 3: Using Constants in Sector Expressions ==========

fn build_sector_with_constant_circuit<F: Field>() -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);

    // Input MLEs V1 and V2.
    let v1 = builder.add_input_shred("MLE 1", 2, &input_layer);
    let v2 = builder.add_input_shred("MLE 2", 2, &input_layer);

    // Expected Output MLE.
    let expected_output = builder.add_input_shred("Expected Output", 2, &input_layer);

    let v3 = builder.add_sector(v1 + v2 * F::from(42));

    // Subtract the expected output from the sector output and set as circuit output.
    let output = builder.add_sector(v3 - expected_output);
    builder.set_output(&output);

    builder.build().unwrap()
}

fn run_example_3() {
    let v1: MultilinearExtension<Fr> = vec![1, 2, 3, 4].into();
    let v2: MultilinearExtension<Fr> = vec![5, 6, 7, 8].into();
    let expected_output_mle: MultilinearExtension<Fr> =
        vec![1 + 42 * 5, 2 + 42 * 6, 3 + 42 * 7, 4 + 42 * 8].into();

    // Create circuit description.
    let mut prover_circuit = build_sector_with_constant_circuit::<Fr>();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("MLE 1", v1.clone());
    prover_circuit.set_input("MLE 2", v2.clone());
    prover_circuit.set_input("Expected Output", expected_output_mle.clone());

    let provable_circuit = prover_circuit.gen_provable_circuit().unwrap();

    // Prove the circuit.
    let (proof_config, proof_as_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable_circuit);

    // Create verifier circuit description and attach inputs.
    verifier_circuit.set_input("MLE 1", v1);
    verifier_circuit.set_input("MLE 2", v2);
    verifier_circuit.set_input("Expected Output", expected_output_mle);

    let verifiable_circuit = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable_circuit, &proof_config, proof_as_transcript);
}

// ========== Example 4: Using Selector Expressions in Sector Nodes ==========

fn build_selector_expression_circuit<F: Field>() -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);

    // Input MLE V1.
    let v1 = builder.add_input_shred("MLE", 2, &input_layer);

    // Expected Output MLE.
    let expected_output = builder.add_input_shred("Expected Output", 2, &input_layer);

    // Split V1(z_1, z_2) into V1_l(z_1) = V1(0, z_1) and V1_r(z_1) = V1(1, z_1).
    let [v1_l, v1_r]: [_; 2] = builder.add_split_node(&v1, 1).try_into().unwrap();

    // Selector layer.
    let v2 = builder.add_sector(sel_expr!(&v1_l * &v1_l, &v1_r * F::from(2)));

    // Subtract the expected output from the sector output and set as circuit output.
    let output = builder.add_sector(v2 - expected_output);
    builder.set_output(&output);

    builder.build().unwrap()
}

fn run_example_4() {
    let v1: MultilinearExtension<Fr> = vec![10, 11, 12, 13].into();
    let expected_output_mle: MultilinearExtension<Fr> =
        vec![10 * 10, 11 * 11, 2 * 12, 2 * 13].into();

    // Create circuit description.
    let mut prover_circuit = build_selector_expression_circuit::<Fr>();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("MLE", v1.clone());
    prover_circuit.set_input("Expected Output", expected_output_mle.clone());

    let provable_circuit = prover_circuit.gen_provable_circuit().unwrap();

    // Prove the circuit.
    let (proof_config, proof_as_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable_circuit);

    // Create verifier circuit description and attach inputs.
    verifier_circuit.set_input("MLE", v1);
    verifier_circuit.set_input("Expected Output", expected_output_mle);

    let verifiable_circuit = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable_circuit, &proof_config, proof_as_transcript);
}

fn main() {
    // `Sector` nodes with `AbstractExpressions` built from MLEs.
    run_example_1a();
    run_example_1b();

    // Using a `Split` node.
    run_example_2();

    // Using constants in the `AbstractExpression` of a `Sector` node.
    run_example_3();

    // Using a `Selector` expressions in a `Sector` node.
    run_example_4();
}
