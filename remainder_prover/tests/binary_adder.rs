use itertools::Itertools;
use remainder::provable_circuit::ProvableCircuit;
use remainder::{
    binary_operations::binary_adder::BinaryAdder,
    expression::abstract_expr::ExprBuilder,
    layouter::builder::{Circuit, CircuitBuilder, LayerKind, ProvableCircuit},
    mle::evals::MultilinearExtension,
    prover::helpers::test_circuit_with_memory_optimized_config,
};
use remainder_shared_types::Fr;

use tracing::Level;
use tracing_subscriber::fmt;
use tracing_subscriber::{self};

fn build_circuit() -> Circuit<Fr> {
    let mut builder = CircuitBuilder::new();

    let input_layer = builder.add_input_layer(LayerKind::Public);

    // Create a circuit for adding two `2^log_bit_width = 8`-bit binary unsigned integers.
    let log_bit_width = 3;

    // We're packing all inputs into a single `InputShred``. This is convenient for applying
    // a binary-checker on all of them at once, before spliting them into individual pieces later
    // using a `SplitNode`.
    // Here's the intended structure of this input shred:
    // LHS: [a0, a1, ..., a7]
    // RHS: [b0, b1, ..., b7]
    // Carries: [c0, d1, ..., c7]
    // Results: [r0, r1, ..., r7]
    // Packing four MLEs like that requires two extra selector bits, hence the `2 + log_bit_width`
    // variables on this input shred.
    let all_inputs = builder.add_input_shred("All Inputs", 2 + log_bit_width, &input_layer);

    let b = &all_inputs;
    let b_sq = ExprBuilder::products(vec![b.id(), b.id()]);
    let b = b.expr();

    // Check that all input bits are binary.
    let binary_sector = builder.add_sector(
        // b * (1 - b) = b - b^2
        b - b_sq,
    );
    builder.set_output(&binary_sector);

    let splits = builder.add_split_node(&all_inputs, 2);

    // lhs = [a0, ..., a7],
    // rhs = [b0, ..., b7],
    // carries = [c0, ..., c7],
    // results = [r0, ..., r7],
    let [lhs, rhs, carries, results] = splits.try_into().unwrap();

    let adder = BinaryAdder::new(&mut builder, &lhs, &rhs, &carries);
    let compare_sector = builder.add_sector(results.expr() - adder.get_output().expr());
    builder.set_output(&compare_sector);

    builder.build().unwrap()
}

fn attach_data(mut circuit: Circuit<Fr>) -> ProvableCircuit<Fr> {
    // 2. Attach input data.
    let lhs = [0, 0, 1, 0, 1, 1, 0, 1];
    let rhs = [1, 0, 0, 1, 1, 1, 1, 1];
    let carries = [0, 0, 1, 1, 1, 1, 1, 1];
    let expected_results = [1, 1, 0, 0, 1, 1, 0, 0];

    let input_mle = MultilinearExtension::new(
        lhs.into_iter()
            .chain(rhs.into_iter())
            .chain(carries.into_iter())
            .chain(expected_results.into_iter())
            .map(Fr::from)
            .collect_vec(),
    );
    // dbg!(&input_mle);

    circuit.set_input("All Inputs", input_mle);

    circuit.gen_provable_circuit().unwrap()
}

#[test]
fn adder_test() {
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_circuit();
    let provable_circuit = attach_data(circuit);

    test_circuit_with_memory_optimized_config(&provable_circuit);
}
