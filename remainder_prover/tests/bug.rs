use itertools::Itertools;
use remainder::{
    binary_operations::logical_shift::ShiftNode,
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
    let primary_input_mle = builder.add_input_shred("Primary Input MLE", 3, &input_layer);
    let left_half_shifted_expected_mle =
        builder.add_input_shred("Left Half Shifted Expected MLE", 2, &input_layer);
    let right_half_shifted_expected_mle =
        builder.add_input_shred("Right Half Shifted Expected MLE", 2, &input_layer);

    let shifter = ShiftNode::new(&mut builder, 3, 1, &primary_input_mle);

    let shifter_splits = builder.add_split_node(&shifter.get_output(), 1);

    let output_part_1 =
        builder.add_sector(shifter_splits[0].expr() - left_half_shifted_expected_mle.expr());
    let output_part_2 =
        builder.add_sector(shifter_splits[1].expr() - right_half_shifted_expected_mle.expr());

    builder.set_output(&output_part_1);
    builder.set_output(&output_part_2);

    builder.build().unwrap()
}

fn attach_data(mut circuit: Circuit<Fr>) -> ProvableCircuit<Fr> {
    let data = [1, 2, 3, 4, 5, 6, 7, 8];
    // We are shifting evaluations right by 1.
    let left_half_shifted_expected_data = [0, 1, 2, 3];
    let right_half_shifted_expected_data = [4, 5, 6, 7];

    let input_mle = MultilinearExtension::new(data.into_iter().map(Fr::from).collect_vec());
    let left_half_shifted_expected_mle = MultilinearExtension::new(
        left_half_shifted_expected_data
            .into_iter()
            .map(Fr::from)
            .collect_vec(),
    );
    let right_half_shifted_expected_mle = MultilinearExtension::new(
        right_half_shifted_expected_data
            .into_iter()
            .map(Fr::from)
            .collect_vec(),
    );

    circuit.set_input("Primary Input MLE", input_mle);
    circuit.set_input(
        "Left Half Shifted Expected MLE",
        left_half_shifted_expected_mle,
    );
    circuit.set_input(
        "Right Half Shifted Expected MLE",
        right_half_shifted_expected_mle,
    );

    circuit.finalize().unwrap()
}

#[test]
fn adder_test() {
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_circuit();
    let provable_circuit = attach_data(circuit);

    test_circuit_with_memory_optimized_config(&provable_circuit);
}
