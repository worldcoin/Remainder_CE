use remainder::prover::helpers::test_circuit_with_runtime_optimized_config;
use remainder_frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility};
use remainder_shared_types::Fr;

use tracing::Level;
use tracing_subscriber::fmt;
use tracing_subscriber::{self};

fn build_circuit() -> Circuit<Fr> {
    let mut builder = CircuitBuilder::<Fr>::new();

    let input_layer = builder.add_input_layer(LayerVisibility::Private);

    let lhs = builder.add_input_shred("LHS", 2, &input_layer);
    let rhs = builder.add_input_shred("RHS", 2, &input_layer);
    let expected_output = builder.add_input_shred("Output", 2, &input_layer);

    // let multiplication_sector = lhs * rhs;
    let multiplication_sector = builder.add_sector(lhs * rhs);

    let subtraction_sector = builder.add_sector(multiplication_sector - expected_output);

    builder.set_output(&subtraction_sector);

    builder.build().unwrap()
}

#[allow(dead_code)]
fn build_circuit_concise() -> Circuit<Fr> {
    let mut builder = CircuitBuilder::<Fr>::new();

    let input_layer = builder.add_input_layer(LayerVisibility::Public);

    let lhs = builder.add_input_shred("LHS", 2, &input_layer);
    let rhs = builder.add_input_shred("RHS", 2, &input_layer);
    let expected_output = builder.add_input_shred("Output", 2, &input_layer);

    // let multiplication_sector = lhs * rhs;
    let main_sector = builder.add_sector(lhs * rhs - expected_output);

    builder.set_output(&main_sector);

    builder.build().unwrap()
}

#[test]
fn tutorial_test() {
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let mut circuit = build_circuit();

    let lhs_data = vec![1, 2, 3, 4].into();
    let rhs_data = vec![5, 6, 7, 8].into();
    let expected_output_data = vec![5, 12, 21, 32].into();

    // TODO: Check input lengths.
    circuit.set_input("LHS", lhs_data);
    circuit.set_input("RHS", rhs_data);
    circuit.set_input("Output", expected_output_data);

    let provable_circuit = circuit.finalize().unwrap();

    dbg!(&provable_circuit);

    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
