use itertools::Itertools;
use remainder::{
    binary_operations::rotate_bits::RotateNode,
    expression::abstract_expr::ExprBuilder,
    layouter::builder::{Circuit, CircuitBuilder, LayerKind, ProvableCircuit},
    mle::evals::MultilinearExtension,
    prover::helpers::test_circuit_with_memory_optimized_config,
};
use remainder_shared_types::Fr;
// use tracing::Level;
// use tracing_subscriber::fmt;
// use tracing_subscriber::{self};

fn build_rotate_circuit(arity: usize, rotation_bits: i32) -> Circuit<Fr> {
    let mut builder = CircuitBuilder::new();

    let input_layer = builder.add_input_layer(LayerKind::Public);

    // Keep input and output in the same shred.
    let all_inputs = builder.add_input_shred("All Inputs", 1 + arity, &input_layer);

    let b = &all_inputs;
    let b_sq = ExprBuilder::products(vec![b.id(), b.id()]);
    let b = b.expr();

    // Check that all input bits are binary, including the expected output.
    let binary_sector = builder.add_sector(
        // b * (1 - b) = b - b^2
        b - b_sq,
    );

    builder.set_output(&binary_sector);

    let splits = builder.add_split_node(&all_inputs, 1);

    let [lhs, rhs] = splits.try_into().unwrap();

    let rotator = RotateNode::new(&mut builder, arity, rotation_bits, &lhs);
    let compare_sector = builder.add_sector(rhs.expr() - rotator.get_output().expr());

    builder.set_output(&compare_sector);

    builder.build().unwrap()
}

fn sample_left_rot_data(mut circuit: Circuit<Fr>) -> ProvableCircuit<Fr> {
    // 2. Attach input data.
    let lhs = [0, 0, 1, 0, 1, 1, 0, 1];
    // rotate left by 2
    let rhs = [1, 0, 1, 1, 0, 1, 0, 0];

    let input_mle = MultilinearExtension::new(
        lhs.into_iter()
            .chain(rhs.into_iter())
            .map(Fr::from)
            .collect_vec(),
    );
    // dbg!(&input_mle);

    circuit.set_input("All Inputs", input_mle);

    circuit.finalize().unwrap()
}

fn sample_right_rot_data(mut circuit: Circuit<Fr>) -> ProvableCircuit<Fr> {
    // 2. Attach input data.
    let lhs = [0, 0, 1, 0, 1, 1, 0, 1];
    // rotate right by 5
    let rhs = [0, 1, 1, 0, 1, 0, 0, 1];

    let input_mle = MultilinearExtension::new(
        lhs.into_iter()
            .chain(rhs.into_iter())
            .map(Fr::from)
            .collect_vec(),
    );
    // dbg!(&input_mle);

    circuit.set_input("All Inputs", input_mle);

    circuit.finalize().unwrap()
}

#[test]
fn left_rotate_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_rotate_circuit(3, -2);
    let provable_circuit = sample_left_rot_data(circuit);
    test_circuit_with_memory_optimized_config(&provable_circuit);
}

#[test]
fn right_rotate_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_rotate_circuit(3, 5);
    let provable_circuit = sample_right_rot_data(circuit);
    test_circuit_with_memory_optimized_config(&provable_circuit);
}
