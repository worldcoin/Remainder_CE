use itertools::Itertools;
use remainder::{
    binary_operations::{binary_adder::BinaryAdder, logical_shift::ShiftNode},
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

    let all_inputs = builder.add_input_shred("All Inputs", 3, &input_layer);

    let shifter = ShiftNode::new(&mut builder, 3, -1, &all_inputs);

    let shifter_splits = builder.add_split_node(&shifter.get_output(), 1);

    // Dummy output node involving first split
    let output_part_1 = builder.add_sector(shifter_splits[0].expr() - shifter_splits[0].expr());
    // Dummy output node involving second split
    let output_part_2 = builder.add_sector(shifter_splits[1].expr() - shifter_splits[1].expr());

    builder.set_output(&output_part_1);
    builder.set_output(&output_part_2);

    builder.build().unwrap()
}

fn attach_data(mut circuit: Circuit<Fr>) -> ProvableCircuit<Fr> {
    let data = [0, 0, 1, 0, 1, 1, 0, 1];

    let input_mle = MultilinearExtension::new(data.into_iter().map(Fr::from).collect_vec());
    // dbg!(&input_mle);

    circuit.set_input("All Inputs", input_mle);

    circuit.finalize().unwrap()
}

#[test]
fn adder_test() {
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_circuit();
    let provable_circuit = attach_data(circuit);

    test_circuit_with_memory_optimized_config(&provable_circuit);
}
