use itertools::Itertools;
use rand::{thread_rng, RngCore};
use remainder::{
    components::sha2::nonlinear_gates as sha2,
    components::sha2::nonlinear_gates::IsBitDecomposable,
    components::sha2::ripple_carry_adder as rca,
    components::sha2::sha256_bit_decomp as sha256,
    expression::abstract_expr::ExprBuilder,
    layouter::builder::{Circuit, CircuitBuilder, LayerKind, ProvableCircuit},
    mle::evals::MultilinearExtension,
    prover::helpers::test_circuit_with_memory_optimized_config,
};
use remainder_shared_types::Fr;
use std::ops::{BitAnd, BitOr, BitXor, Shl, Shr};

// use tracing::Level;
// use tracing_subscriber::fmt;
// use tracing_subscriber::{self};

fn rotate_right<T>(val: T, bits: usize) -> T
where
    T: Shr<usize, Output = T> + Shl<usize, Output = T> + BitOr<T, Output = T> + Copy,
{
    (val >> bits) | (val << (8 * std::mem::size_of::<T>() - bits))
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

    let x_bits = sha2::bit_decompose_msb_first(x);
    let y_bits = sha2::bit_decompose_msb_first(y);
    let z_bits = sha2::bit_decompose_msb_first(z);
    let expected_bits = sha2::bit_decompose_msb_first(expected_value);

    let input_mle = MultilinearExtension::new(
        x_bits
            .into_iter()
            .chain(y_bits.into_iter())
            .chain(z_bits.into_iter())
            .chain(expected_bits.into_iter())
            .map(u64::from)
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

    let x_bits = sha2::bit_decompose_msb_first(x);
    let y_bits = sha2::bit_decompose_msb_first(y);
    let z_bits = sha2::bit_decompose_msb_first(z);
    let expected_bits = sha2::bit_decompose_msb_first(expected_value);

    let input_mle = MultilinearExtension::new(
        x_bits
            .into_iter()
            .chain(y_bits.into_iter())
            .chain(z_bits.into_iter())
            .chain(expected_bits.into_iter())
            .map(u64::from)
            .map(Fr::from)
            .collect_vec(),
    );

    circuit.set_input("All Inputs", input_mle);

    circuit.finalize().unwrap()
}

fn build_sigma_gate_circuit<
    const WORD_SIZE: usize,
    const ROTR1: i32,
    const ROTR2: i32,
    const ROTR3: i32,
>(
    is_big_sigma_gate: bool,
) -> Circuit<Fr> {
    let mut builder = CircuitBuilder::new();

    let input_layer = builder.add_input_layer(LayerKind::Public);
    let log_bit_width = if WORD_SIZE == 32 { 5 } else { 6 }; // SHA-256 num vars

    // All inputs includes 32*3 input wires + 32 expected output
    let all_inputs = builder.add_input_shred("All Inputs", 1 + log_bit_width, &input_layer);

    let b = &all_inputs;
    let b_sq = ExprBuilder::products(vec![b.id(), b.id()]);
    let b = b.expr();

    // Check that all input bits are binary.
    let binary_sector = builder.add_sector(
        // b * (1 - b) = b - b^2
        b - b_sq,
    );
    builder.set_output(&binary_sector);

    let splits = builder.add_split_node(&all_inputs, 1);

    let [x_vars, expected_result] = splits.try_into().unwrap();

    let compare_sector = if is_big_sigma_gate {
        let sigma_gate = sha2::Sigma::<WORD_SIZE, ROTR1, ROTR2, ROTR3>::new(&mut builder, &x_vars);
        builder.add_sector(expected_result.expr() - sigma_gate.get_output().expr())
    } else {
        let sigma_gate =
            sha2::SmallSigma::<WORD_SIZE, ROTR1, ROTR2, ROTR3>::new(&mut builder, &x_vars);
        builder.add_sector(expected_result.expr() - sigma_gate.get_output().expr())
    };

    builder.set_output(&compare_sector);
    builder.build().unwrap()
}

fn attach_sigma_gate_data<
    const ROTR1: i32,
    const ROTR2: i32,
    const ROTR3: i32,
    const POSITIVE: bool,
    T,
>(
    mut circuit: Circuit<Fr>,
    random_data: T,
    is_big_sigma_gate: bool,
) -> ProvableCircuit<Fr>
where
    T: IsBitDecomposable
        + BitOr<T, Output = T>
        + BitXor<T, Output = T>
        + BitAnd<u32, Output = T>
        + Copy,
    u64: From<T>,
{
    // 2. Attach random data
    let expected_value = if POSITIVE {
        let r1 = rotate_right(random_data, ROTR1 as usize);
        let r2 = rotate_right(random_data, ROTR2 as usize);
        let r3 = if is_big_sigma_gate {
            rotate_right(random_data, ROTR3 as usize)
        } else {
            random_data >> ROTR3.try_into().unwrap()
        };
        r1 ^ r2 ^ r3
    } else {
        // Randomly generate an output unrelated to input. This should fail with probability 1-1/(2^32)
        random_data
    };

    let x_bits = sha2::bit_decompose_msb_first(random_data);
    let expected_bits = sha2::bit_decompose_msb_first(expected_value);

    let input_mle = MultilinearExtension::new(
        x_bits
            .into_iter()
            .chain(expected_bits.into_iter())
            .map(u64::from)
            .map(Fr::from)
            .collect_vec(),
    );

    circuit.set_input("All Inputs", input_mle);

    circuit.finalize().unwrap()
}

// Adder Gate Tests
fn build_full_adder_gate_circuit() -> Circuit<Fr> {
    let mut builder = CircuitBuilder::new();

    let input_layer = builder.add_input_layer(LayerKind::Public);

    // Inputs are x, y, c, 0 and outputs are expected to be s,c,0,0
    let all_inputs = builder.add_input_shred("All Inputs", 3, &input_layer);

    let b = &all_inputs;
    let b_sq = ExprBuilder::products(vec![b.id(), b.id()]);
    let b = b.expr();

    // Check that all input bits are binary.
    let binary_sector = builder.add_sector(
        // b * (1 - b) = b - b^2
        b - b_sq,
    );

    builder.set_output(&binary_sector);

    let splits = builder.add_split_node(&all_inputs, 1);

    let [data, expected] = splits.try_into().unwrap();
    let args_split = builder.add_split_node(&data, 2);
    let res_split = builder.add_split_node(&expected, 2);
    let [x, y, c, _] = args_split.try_into().unwrap();
    let [expected_sum, expected_carry, _, _] = res_split.try_into().unwrap();

    let fa_gate = rca::FullAdder::new(&mut builder, &x, &y, &c);
    let (s, c) = fa_gate.get_output();
    let sum_is_valid = builder.add_sector(s.expr() - expected_sum.expr());
    let carry_is_valid = builder.add_sector(c.expr() - expected_carry.expr());

    builder.set_output(&sum_is_valid);
    builder.set_output(&carry_is_valid);
    builder.build().unwrap()
}

fn attach_full_adder_gate_data(mut circuit: Circuit<Fr>) -> ProvableCircuit<Fr> {
    // 2. Attach random data
    let mut trng = thread_rng();
    let x = trng.next_u64() & 0x1;
    let y = trng.next_u64() & 0x1;
    let c = trng.next_u64() & 0x1;

    let expected_sum = (x + y + c) % 2;
    let expected_carry = if (x + y + c) >= 2 { 1 } else { 0 };

    let input_vec = [x, y, c, 0, expected_sum, expected_carry, 0, 0];

    let input_mle = MultilinearExtension::new(input_vec.into_iter().map(Fr::from).collect_vec());

    circuit.set_input("All Inputs", input_mle);

    circuit.finalize().unwrap()
}

fn constant_gate_circuit<T>(gate_data: T) -> Circuit<Fr>
where
    T: IsBitDecomposable,
    u64: From<T>,
{
    let mut builder = CircuitBuilder::new();

    let input_layer = builder.add_input_layer(LayerKind::Public);

    let num_vars = 3 + std::mem::size_of::<T>().ilog2() as usize;

    let const_msb_gate = sha2::ConstInputGate::new(&mut builder, "random_data", gate_data);

    // Inputs for expected values
    let expected_output = builder.add_input_shred("Expected output", num_vars, &input_layer);

    // Check the outputs are correct
    let const_is_valid =
        builder.add_sector(const_msb_gate.get_output().expr() - expected_output.expr());

    builder.set_output(&const_is_valid);
    let mut circuit = builder.build().unwrap();
    const_msb_gate.add_to_circuit(&mut circuit);
    circuit
}

fn attach_constant_gate_data<T>(mut circuit: Circuit<Fr>, x: T) -> ProvableCircuit<Fr>
where
    T: IsBitDecomposable,
    u64: From<T>,
{
    let x_bits = sha2::bit_decompose_msb_first(x);
    let expected_mle = MultilinearExtension::new(
        x_bits
            .into_iter()
            .map(u64::from)
            .map(Fr::from)
            .collect_vec(),
    );
    circuit.set_input("Expected output", expected_mle);
    circuit.finalize().unwrap()
}

#[test]
fn attach_constant_gate_data_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();
    let mut trng = rand::thread_rng();

    for _ in 0..10 {
        let data = trng.next_u32();
        let circuit = constant_gate_circuit(data);
        let provable_circuit = attach_constant_gate_data(circuit, data);
        test_circuit_with_memory_optimized_config(&provable_circuit);
    }

    for _ in 0..10 {
        let data = trng.next_u64();
        let circuit = constant_gate_circuit(data);
        let provable_circuit = attach_constant_gate_data(circuit, data);
        test_circuit_with_memory_optimized_config(&provable_circuit);
    }
}

#[test]
fn full_adder_gate_positive_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_full_adder_gate_circuit();

    for _ in 0..10 {
        let provable_circuit = attach_full_adder_gate_data(circuit.clone());
        test_circuit_with_memory_optimized_config(&provable_circuit);
    }
}

#[test]
fn sha2_ch_gate_positive_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_ch_gate_circuit();

    for _ in 0..10 {
        let provable_circuit = attach_ch_gate_data::<true>(circuit.clone());
        test_circuit_with_memory_optimized_config(&provable_circuit);
    }
}

#[test]
fn sha2_ch_gate_negative_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

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
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_maj_gate_circuit();

    for _ in 0..10 {
        let provable_circuit = attach_maj_gate_data::<true>(circuit.clone());
        test_circuit_with_memory_optimized_config(&provable_circuit);
    }
}

#[test]
fn sha2_maj_gate_negative_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = build_maj_gate_circuit();

    for _ in 0..10 {
        let provable_circuit = attach_maj_gate_data::<false>(circuit.clone());
        let result = std::panic::catch_unwind(|| {
            test_circuit_with_memory_optimized_config(&provable_circuit)
        });
        assert!(result.is_err(), "A -ve test should panic");
    }
}

#[test]
fn sha256_big_sigma_gate_positive_tests() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();
    let mut trng = rand::thread_rng();

    // The shift values are taken from NIST SP-180-4:
    // https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
    // printed page number 10 for SHA-256.

    let sigma_0 = build_sigma_gate_circuit::<{ sha256::WORD_SIZE }, 2, 13, 22>(true);
    let sigma_1 = build_sigma_gate_circuit::<{ sha256::WORD_SIZE }, 6, 11, 25>(true);

    let small_sigma_0 = build_sigma_gate_circuit::<{ sha256::WORD_SIZE }, 7, 18, 3>(false);
    let small_sigma_1 = build_sigma_gate_circuit::<{ sha256::WORD_SIZE }, 17, 19, 10>(false);

    for _ in 0..10 {
        let random_data = trng.next_u32();
        let sha256_sigma_0 =
            attach_sigma_gate_data::<2, 13, 22, true, _>(sigma_0.clone(), random_data, true);

        let sha256_sigma_1 =
            attach_sigma_gate_data::<6, 11, 25, true, _>(sigma_1.clone(), random_data, true);

        let sha256_small_sigma_0 =
            attach_sigma_gate_data::<7, 18, 3, true, _>(small_sigma_0.clone(), random_data, false);

        let sha256_small_sigma_1 = attach_sigma_gate_data::<17, 19, 10, true, _>(
            small_sigma_1.clone(),
            random_data,
            false,
        );

        test_circuit_with_memory_optimized_config(&sha256_sigma_0.clone());
        test_circuit_with_memory_optimized_config(&sha256_sigma_1.clone());
        test_circuit_with_memory_optimized_config(&sha256_small_sigma_0.clone());
        test_circuit_with_memory_optimized_config(&sha256_small_sigma_1.clone());
    }
}
