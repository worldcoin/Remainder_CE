use itertools::Itertools;
use rand::{thread_rng, RngCore};
use remainder::{
    components::sha2::nonlinear_gates as sha2,
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

fn bit_decompose_msb_first<T>(input: T) -> Vec<u64>
where
    T: Shr<usize, Output = T> + BitAnd<T, Output = T> + Copy + From<u32> + TryInto<u64>,
{
    let bit_count = 8 * std::mem::size_of::<T>();
    let mut result = Vec::<u64>::with_capacity(bit_count);

    for i in 0..bit_count {
        let v = ((input >> (bit_count - 1 - i)) & From::from(0x1))
            .try_into()
            .unwrap_or_default();
        result.push(v);
    }

    result
}

#[allow(unused)]
fn bit_decompose_lsb_first<T>(input: T) -> Vec<u64>
where
    T: Shr<usize, Output = T> + BitAnd<T, Output = T> + Copy + From<u32> + TryInto<u64>,
{
    let bit_count = 8 * std::mem::size_of::<T>();
    let mut result = Vec::<u64>::with_capacity(bit_count);

    for i in 0..bit_count {
        let v = ((input >> i) & From::from(0x1))
            .try_into()
            .unwrap_or_default();
        result.push(v);
    }

    result
}

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

    let x_bits = bit_decompose_msb_first(x);
    let y_bits = bit_decompose_msb_first(y);
    let z_bits = bit_decompose_msb_first(z);
    let expected_bits = bit_decompose_msb_first(expected_value);

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

    let x_bits = bit_decompose_msb_first(x);
    let y_bits = bit_decompose_msb_first(y);
    let z_bits = bit_decompose_msb_first(z);
    let expected_bits = bit_decompose_msb_first(expected_value);

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
    T: Shr<usize, Output = T>
        + Shl<usize, Output = T>
        + BitOr<T, Output = T>
        + BitXor<T, Output = T>
        + BitAnd<T, Output = T>
        + From<u32>
        + Copy,
    u64: TryFrom<T>,
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

    let x_bits = bit_decompose_msb_first(random_data);
    let expected_bits = bit_decompose_msb_first(expected_value);

    let input_mle = MultilinearExtension::new(
        x_bits
            .into_iter()
            .chain(expected_bits.into_iter())
            .map(Fr::from)
            .collect_vec(),
    );

    circuit.set_input("All Inputs", input_mle);

    circuit.finalize().unwrap()
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
