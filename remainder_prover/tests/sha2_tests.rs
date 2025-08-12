use itertools::Itertools;
use rand::{thread_rng, RngCore};
use remainder::{
    binary_operations::binary_adder::BinaryAdder,
    components::sha2::nonlinear_gates as sha2,
    expression::abstract_expr::ExprBuilder,
    layouter::builder::{Circuit, CircuitBuilder, LayerKind, ProvableCircuit},
    mle::evals::MultilinearExtension,
    prover::helpers::test_circuit_with_memory_optimized_config,
};
use remainder_shared_types::Fr;

use tracing::Level;
use tracing_subscriber::fmt;
use tracing_subscriber::{self};

fn bit_decompose(input: u32) -> [u64; 32] {
    let mut result = [0; 32];

    for i in 0..u32::BITS.try_into().unwrap() {
        result[i] = ((input >> i) & 0x1) as u64;
    }
    result
}

fn build_ch_gate_circuit() -> Circuit<Fr> {
    let mut builder = CircuitBuilder::new();

    let input_layer = builder.add_input_layer(LayerKind::Public);
    let log_bit_width = 5; // SHA-256 num vars

    // All inputs includes 32*3 input wires + 32 expected output
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

    let [x_vars, y_vars, z_vars, expected_result] = splits.try_into().unwrap();

    let ch_gate = sha2::ChGate::new(&mut builder, &x_vars, &y_vars, &z_vars);

    let compare_sector = builder.add_sector(expected_result.expr() - ch_gate.get_output().expr());
    builder.set_output(&compare_sector);
    builder.build().unwrap()
}

fn attach_ch_gate_data<const POSITIVE: bool>(mut circuit: Circuit<Fr>) -> ProvableCircuit<Fr> {
    // 2. Attach random data
    let mut trng = thread_rng();
    let x = trng.next_u32();
    let y = trng.next_u32();
    let z = trng.next_u32();

    let expected_value = if POSITIVE {
        (x & y) ^ ((!x) & z)
    } else {
        // Randomly generate an output unrelated to input. This should fail with probability 1-1/(2^32)
        trng.next_u32()
    };

    let x_bits = bit_decompose(x);
    let y_bits = bit_decompose(y);
    let z_bits = bit_decompose(z);
    let expected_bits = bit_decompose(expected_value);

    let input_mle = MultilinearExtension::new(
        x_bits
            .into_iter()
            .chain(y_bits.into_iter())
            .chain(z_bits.into_iter())
            .chain(expected_bits.into_iter())
            .map(Fr::from)
            .collect_vec(),
    );

    circuit.set_input("All Inputs", input_mle);

    circuit.finalize().unwrap()
}

fn build_maj_gate_circuit() -> Circuit<Fr> {
    let mut builder = CircuitBuilder::new();

    let input_layer = builder.add_input_layer(LayerKind::Public);
    let log_bit_width = 5; // SHA-256 num vars

    // All inputs includes 32*3 input wires + 32 expected output
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

    let [x_vars, y_vars, z_vars, expected_result] = splits.try_into().unwrap();

    let ch_gate = sha2::MajGate::new(&mut builder, &x_vars, &y_vars, &z_vars);

    let compare_sector = builder.add_sector(expected_result.expr() - ch_gate.get_output().expr());
    builder.set_output(&compare_sector);
    builder.build().unwrap()
}

fn attach_maj_gate_data<const POSITIVE: bool>(mut circuit: Circuit<Fr>) -> ProvableCircuit<Fr> {
    // 2. Attach random data
    let mut trng = thread_rng();
    let x = trng.next_u32();
    let y = trng.next_u32();
    let z = trng.next_u32();

    let expected_value = if POSITIVE {
        (x & y) ^ (y & z) ^ (x & z)
    } else {
        // Randomly generate an output unrelated to input. This should fail with probability 1-1/(2^32)
        trng.next_u32()
    };

    let x_bits = bit_decompose(x);
    let y_bits = bit_decompose(y);
    let z_bits = bit_decompose(z);
    let expected_bits = bit_decompose(expected_value);

    let input_mle = MultilinearExtension::new(
        x_bits
            .into_iter()
            .chain(y_bits.into_iter())
            .chain(z_bits.into_iter())
            .chain(expected_bits.into_iter())
            .map(Fr::from)
            .collect_vec(),
    );

    circuit.set_input("All Inputs", input_mle);

    circuit.finalize().unwrap()
}

#[test]
fn sha2_ch_gate_positive_test() {
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_ch_gate_circuit();

    for _ in 0..10 {
        let provable_circuit = attach_ch_gate_data::<true>(circuit.clone());
        test_circuit_with_memory_optimized_config(&provable_circuit);
    }
}

#[test]
fn sha2_ch_gate_negative_test() {
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_ch_gate_circuit();

    for _ in 0..10 {
        let provable_circuit = attach_ch_gate_data::<false>(circuit.clone());
        let result = std::panic::catch_unwind(|| {
            test_circuit_with_memory_optimized_config(&provable_circuit)
        });
        assert!(result.is_err(), "A -ve test should panic");
    }
}

#[test]
fn sha2_maj_gate_positive_test() {
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_maj_gate_circuit();

    for _ in 0..10 {
        let provable_circuit = attach_maj_gate_data::<true>(circuit.clone());
        test_circuit_with_memory_optimized_config(&provable_circuit);
    }
}

#[test]
fn sha2_maj_gate_negative_test() {
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_maj_gate_circuit();

    for _ in 0..10 {
        let provable_circuit = attach_maj_gate_data::<false>(circuit.clone());
        let result = std::panic::catch_unwind(|| {
            test_circuit_with_memory_optimized_config(&provable_circuit)
        });
        assert!(result.is_err(), "A -ve test should panic");
    }
}
