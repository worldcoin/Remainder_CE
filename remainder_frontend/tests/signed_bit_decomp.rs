use itertools::Itertools;
use remainder::{
    mle::evals::MultilinearExtension, prover::helpers::test_circuit_with_runtime_optimized_config,
};
use remainder_frontend::{
    const_expr,
    layouter::builder::{Circuit, CircuitBuilder, LayerVisibility},
};
use remainder_shared_types::Fr;

fn build_signed_bit_decomp_circuit() -> Circuit<Fr> {
    let mut builder = CircuitBuilder::<Fr>::new();

    // 0. Metadata
    let k = 2; // Log of numbers of x in parallel
    let n = 4; // Total number of bits, which decomposes x into (b_s, b_0, b_1, ..., b_{n-2})

    // 1. Build circuit
    let int_layer = builder.add_input_layer(LayerVisibility::Public); // for x
    let x = builder.add_input_shred("x", k, &int_layer);

    let bit_layer = builder.add_input_layer(LayerVisibility::Private); // for bits
    let bs = builder.add_input_shred("bs", k, &bit_layer);
    let bi = (0..n - 1)
        .map(|i| builder.add_input_shred(&format!("b{i}"), k, &bit_layer))
        .collect_vec();

    (0..n).for_each(|i| {
        let next_bit = if i == 0 {
            bs.clone()
        } else {
            bi[i - 1].clone()
        };
        let bit_check_sector =
            builder.add_sector(&next_bit * (const_expr!(Fr::from(1)) - &next_bit));
        builder.set_output(&bit_check_sector);
    });

    // Assemble checks
    let mut assemb_expr = bs * Fr::from(1 << (n - 1)).neg();
    for i in 0..n - 1 {
        assemb_expr += &bi[i] * Fr::from(1 << (n - i - 2));
    }
    let assemb_sector = builder.add_sector(assemb_expr - x);
    builder.set_output(&assemb_sector);

    builder.build().unwrap()
}

#[test]
// A simple circuit that verifies that (bs, bi) is a two's-complemented decomposition of x
fn signed_bit_decomp_test() {
    let mut circuit = build_signed_bit_decomp_circuit();

    // 2. Attach input data.
    let x_data = MultilinearExtension::<Fr>::new(
        [4, 5, 6, 7]
            .into_iter()
            .map(|i| Fr::from(i).neg())
            .collect(),
    );
    let bs_data = MultilinearExtension::<Fr>::new([1, 1, 1, 1].into_iter().map(Fr::from).collect());
    let bi_data = vec![
        MultilinearExtension::<Fr>::new([1, 0, 0, 0].into_iter().map(Fr::from).collect()),
        MultilinearExtension::<Fr>::new([0, 1, 1, 0].into_iter().map(Fr::from).collect()),
        MultilinearExtension::<Fr>::new([0, 1, 0, 1].into_iter().map(Fr::from).collect()),
    ];

    circuit.set_input("x", x_data);
    circuit.set_input("bs", bs_data);
    bi_data.into_iter().enumerate().for_each(|(i, mle)| {
        circuit.set_input(&format!("b{i}"), mle);
    });

    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
