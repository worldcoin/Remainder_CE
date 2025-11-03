use ark_std::test_rng;
use rand::Rng;
use remainder::mle::evals::MultilinearExtension;
use remainder::prover::helpers::test_circuit_with_runtime_optimized_config;
use remainder_frontend::abstract_expr::AbstractExpression;
use remainder_frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef};
use remainder_shared_types::{Field, Fr};

/// What's the idea behind ReLU again? So we should pass in an MLE which is
/// a bunch of signed `u32`s.
///
/// We are also gonna use twos complement here. But that shouldn't matter
/// since we aren't trying to take an absolute value?
///
/// This function returns the MLE of ReLU outputs, i.e. ReLU(x).
fn add_relu_check<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    signed_u32_relu_input: &NodeRef<F>,
    signed_decomp: &NodeRef<F>,
) -> NodeRef<F> {
    // First, compute the signed recomposition of the `signed_decomp`
    // and ensure that it is equivalent to the `signed_u32_relu_input`.
    let all_bits_nodes = builder_ref.add_split_node(&signed_decomp, 5);
    assert_eq!(all_bits_nodes.len(), 32);

    // The first bit is the sign bit. Everything else are bits in big-endian.
    let recomposition_expr = (0..31)
        .rev()
        .fold(all_bits_nodes[31].expr(), |acc, bit_idx| {
            // Get the correct power of two to scale by.
            let scale_factor = F::from(2_u64.pow(32 - bit_idx - 1));
            acc + AbstractExpression::scaled(all_bits_nodes[bit_idx as usize].expr(), scale_factor)
        });

    // Add the sign bit as -2^{32}
    let recomposition_expr = recomposition_expr
        - AbstractExpression::scaled(all_bits_nodes[0].expr(), F::from(2_u64.pow(32)));

    // Finally, assert that the recomposition is identical to the ReLU output.
    let check_expr = signed_u32_relu_input - recomposition_expr;
    let recomp_check = builder_ref.add_sector(check_expr);
    builder_ref.set_output(&recomp_check);

    // Next, we use the sign bit to compute the actual ReLU value.
    builder_ref.add_sector(
        signed_u32_relu_input.expr() - signed_u32_relu_input.expr() * all_bits_nodes[0].expr(),
    )
}

/// Neural network circuit begins with just two layers.
///
/// We commit to all of the model weights and decompositions needed for ReLU
/// for now. These will be in two separate input layers.
fn build_simple_two_layer_nn_circuit<F: Field>(
    input_num_free_vars: usize,
    hidden_layer_1_num_free_vars: usize,
    output_layer_num_free_vars: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // Create an input layer for the model input.
    let model_input_input_layer_node =
        builder.add_input_layer("Model input input layer", LayerVisibility::Public);
    let model_input_mle = builder.add_input_shred(
        "Model input MLE",
        input_num_free_vars,
        &model_input_input_layer_node,
    );

    // Create an input layer for the expected model output.
    let expected_output_input_layer_node =
        builder.add_input_layer("Expected output input layer", LayerVisibility::Public);
    let expected_output_mle = builder.add_input_shred(
        "Expected output MLE",
        output_layer_num_free_vars,
        &expected_output_input_layer_node,
    );

    // Create an input layer for the weights.
    let model_weights_input_layer_node =
        builder.add_input_layer("Model weights input layer", LayerVisibility::Private);
    let w1_mle = builder.add_input_shred(
        "W1 MLE",
        input_num_free_vars + hidden_layer_1_num_free_vars,
        &model_weights_input_layer_node,
    );
    let b1_mle = builder.add_input_shred(
        "b1 MLE",
        hidden_layer_1_num_free_vars,
        &model_weights_input_layer_node,
    );
    let w2_mle = builder.add_input_shred(
        "W2 MLE",
        hidden_layer_1_num_free_vars + output_layer_num_free_vars,
        &model_weights_input_layer_node,
    );
    let b2_mle = builder.add_input_shred(
        "b2 MLE",
        output_layer_num_free_vars,
        &model_weights_input_layer_node,
    );

    // Create an input layer for the ReLU decompositions.
    // Let's have our post-mul things be `u32`s, so that pre-mul we can have
    // at most `u8`s.
    let relu_decomps_input_layer_node =
        builder.add_input_layer("ReLU decomps input layer", LayerVisibility::Private);
    let layer_1_relu_decomp = builder.add_input_shred(
        "Layer 1 ReLU decomp MLE",
        hidden_layer_1_num_free_vars + 5,
        &relu_decomps_input_layer_node,
    );

    // Let's do our first check: bits are binary on the ReLU decomps.
    let bits_are_binary_output_node = builder.add_sector(
        layer_1_relu_decomp.expr() * layer_1_relu_decomp.expr() - layer_1_relu_decomp.expr(),
    );
    builder.set_output(&bits_are_binary_output_node);

    // Let's compute the first hidden layer.
    let w1x = builder.add_matmult_node(
        &w1_mle,
        (hidden_layer_1_num_free_vars, input_num_free_vars),
        &model_input_mle,
        (input_num_free_vars, 0),
    );
    let w1x_plus_b1 = builder.add_sector(w1x.expr() + b1_mle.expr());

    // Let's compute the ReLU of the first hidden layer.
    let h1 = add_relu_check(&mut builder, &w1x_plus_b1, &layer_1_relu_decomp);

    // W_2 h_1
    let w2h1 = builder.add_matmult_node(
        &w2_mle,
        (output_layer_num_free_vars, hidden_layer_1_num_free_vars),
        &h1,
        (hidden_layer_1_num_free_vars, 0),
    );

    // W_2 h_1 + b_2
    let w2h1_plus_b2 = builder.add_sector(w2h1.expr() + b2_mle.expr());

    // Check that the circuit-computed output equals the expected output
    let model_output_check = builder.add_sector(w2h1_plus_b2.expr() - expected_output_mle.expr());
    builder.set_output(&model_output_check);

    builder.build().unwrap()
}

/// Takes in a slice of `u32` values, e.g. [a, b, c, d], and returns a `Vec<F>`
/// containing the `u32` decomposition of all of the values, concatenated in
/// first- to last-bit order, with all `i`th bits in chunks.
///
/// In other words, the returned vector should appear as follows:
/// [a_0, b_0, c_0, d_0; a_1, b_1, c_1, d_1; ...; a_{31}, b_{31}, c_{31}, d_{31}]
///
/// where `a_i` is the `i`th bit in the 32-bit decomposition of `a`.
fn compute_binary_decomposition_u32<F: Field>(vals: &[F]) -> Vec<F> {
    (0..32)
        .flat_map(|bit_idx| {
            let result: Vec<F> = vals
                .iter()
                .map(|val| {
                    let val_as_u64 = val.to_u64s_le()[0];
                    let ith_bit = (val_as_u64 >> (31 - bit_idx)) & 1;
                    F::from(ith_bit)
                })
                .collect();
            result
        })
        .collect()
}

fn matmult_over_flattened_matrix_reprs<F: Field>(
    a: &[F],
    a_num_rows: usize,
    a_num_cols: usize,
    b: &[F],
    b_num_cols: usize,
) -> Vec<F> {
    assert_eq!(a.len(), a_num_rows * a_num_cols);
    assert_eq!(b.len(), a_num_cols * b_num_cols);

    // Row-major order in the final destination matrix
    (0..a_num_rows)
        .flat_map(|i| {
            (0..b_num_cols)
                .map(|k| {
                    (0..a_num_cols).fold(F::ZERO, |acc, j| {
                        // \sum_j a[i, j] * b[j, k]
                        acc + a[i * a_num_cols + j] * b[j * b_num_cols + k]
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn eltwise_add<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(a_i, b_i)| *a_i + b_i)
        .collect()
}

/// Generates MLEs with evaluations of the form [0, ..., 2^{num_vars} - 1].
/// For debugging purposes, mostly.
#[allow(dead_code)]
fn generate_range_mle_evaluations<F: Field>(num_vars: usize) -> Vec<F> {
    (0..1 << num_vars).map(|x| F::from(x)).collect()
}

/// Generates MLEs with 2^{num_vars} random (u8) evaluations.
fn generate_random_mle_u8_evaluations<F: Field>(num_vars: usize) -> Vec<F> {
    let mut test_rng = test_rng();
    (0..1 << num_vars)
        .map(|_| F::from(test_rng.gen::<u8>() as u64).neg())
        .collect()
}

#[test]
fn test_build_simple_two_layer_nn_circuit() {
    const NUM_INPUT_VARS: usize = 5;
    const HIDDEN_LAYER_1_NUM_VARS: usize = 5;
    const OUTPUT_LAYER_NUM_VARS: usize = 5;

    // Create circuit description
    let mut circuit = build_simple_two_layer_nn_circuit(
        NUM_INPUT_VARS,
        HIDDEN_LAYER_1_NUM_VARS,
        OUTPUT_LAYER_NUM_VARS,
    );

    // Generate model inputs, weights, biases
    let input_vals_vec: Vec<Fr> = generate_random_mle_u8_evaluations(NUM_INPUT_VARS);
    let w1_vec: Vec<Fr> =
        generate_random_mle_u8_evaluations(HIDDEN_LAYER_1_NUM_VARS + NUM_INPUT_VARS);
    let b1_vec: Vec<Fr> = generate_random_mle_u8_evaluations(HIDDEN_LAYER_1_NUM_VARS);
    let w2_vec: Vec<Fr> =
        generate_random_mle_u8_evaluations(OUTPUT_LAYER_NUM_VARS + HIDDEN_LAYER_1_NUM_VARS);
    let b2_vec: Vec<Fr> = generate_random_mle_u8_evaluations(OUTPUT_LAYER_NUM_VARS);

    // Compute h1 + binary decomp, and generate weights + bias as well
    let w1x_vec = matmult_over_flattened_matrix_reprs(
        &w1_vec,
        1 << HIDDEN_LAYER_1_NUM_VARS,
        1 << NUM_INPUT_VARS,
        &input_vals_vec,
        1,
    );
    let h1_vec = eltwise_add(&w1x_vec, &b1_vec);
    let h1_bin_decomp = compute_binary_decomposition_u32(&h1_vec);
    let h1_bin_decomp_mle = MultilinearExtension::new(h1_bin_decomp);
    circuit.set_input("Layer 1 ReLU decomp MLE", h1_bin_decomp_mle);

    // Compute expected outputs from circuit since Remainder doesn't support
    // non-zero output layers for now...
    let w2h1_vec = matmult_over_flattened_matrix_reprs(
        &w2_vec,
        1 << OUTPUT_LAYER_NUM_VARS,
        1 << HIDDEN_LAYER_1_NUM_VARS,
        &h1_vec,
        1,
    );
    let expected_output_vec = eltwise_add(&w2h1_vec, &b2_vec);
    let expected_output_mle = MultilinearExtension::new(expected_output_vec);
    circuit.set_input("Expected output MLE", expected_output_mle);

    // Set everything else (in this order for ownership reasons)
    let model_input_mle = MultilinearExtension::new(input_vals_vec);
    circuit.set_input("Model input MLE", model_input_mle);

    let w1_mle = MultilinearExtension::new(w1_vec);
    let b1_mle = MultilinearExtension::new(b1_vec);
    circuit.set_input("W1 MLE", w1_mle);
    circuit.set_input("b1 MLE", b1_mle);

    let w2_mle = MultilinearExtension::new(w2_vec);
    let b2_mle = MultilinearExtension::new(b2_vec);
    circuit.set_input("W2 MLE", w2_mle);
    circuit.set_input("b2 MLE", b2_mle);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
