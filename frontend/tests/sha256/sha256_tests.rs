use frontend::{
    abstract_expr::AbstractExpression,
    components::sha2_gkr::{self, AdderGateTrait},
    layouter::builder::{Circuit, CircuitBuilder, LayerVisibility},
};
use itertools::Itertools;
use rand::{thread_rng, RngCore};
use remainder::{
    mle::evals::MultilinearExtension,
    provable_circuit::ProvableCircuit,
    prover::helpers::{
        test_circuit_with_memory_optimized_config, test_circuit_with_runtime_optimized_config,
    },
};
use shared_types::{Field, Fr, Halo2FFTFriendlyField};

fn create_adder_test_circuit<F: Field, Adder: AdderGateTrait<F>>(
    carry_layer_kind: Option<LayerVisibility>,
) -> (Circuit<F>, Adder) {
    let mut builder = CircuitBuilder::new();

    let input_layer = builder.add_input_layer("adder_test_circuit_input", LayerVisibility::Public);
    let carry_layer = carry_layer_kind
        .map(|layer_kind| builder.add_input_layer("adder_test_circuit_carry", layer_kind));

    let num_vars = 5; // 32-bit adder

    // Inputs are 32-bit x, 32-bit y, 32-bit z, rest all zeros.
    let all_inputs = builder.add_input_shred("All Inputs", 2 + num_vars, &input_layer);

    let b = &all_inputs;
    let b_sq = AbstractExpression::products(vec![b.id(), b.id()]);
    let b = b.expr();

    // Check that all input bits are binary.
    let binary_sector = builder.add_sector(
        // b * (1 - b) = b - b^2
        b - b_sq,
    );

    // Make sure all inputs are either `0` or `1`
    builder.set_output(&binary_sector);

    let splits = builder.add_split_node(&all_inputs, 2);

    let [x, y, expected_sum, _] = splits.try_into().unwrap();

    let adder = Adder::layout_adder_circuit(&mut builder, &x, &y, carry_layer);
    let sum_is_valid = builder.add_sector(adder.get_output().expr() - expected_sum.expr());

    builder.set_output(&sum_is_valid);
    (builder.build().unwrap(), adder)
}

fn attach_data_to_adder_gate<const POSITIVE: bool, F: Halo2FFTFriendlyField, Adder>(
    mut circuit: Circuit<F>,
    adder: Adder,
) -> ProvableCircuit<F>
where
    F: Field + From<u64>,
    Adder: AdderGateTrait<F, IntegralType = u32>,
{
    let mut trng = thread_rng();
    // 1. Attach random data
    let x = trng.next_u32();
    let y = trng.next_u32();
    let expected_sum = if POSITIVE {
        x.wrapping_add(y)
    } else {
        // Will be exact sum with probability 1/2^32
        trng.next_u32()
    };

    let x_bits = sha2_gkr::nonlinear_gates::bit_decompose_msb_first(x);
    let y_bits = sha2_gkr::nonlinear_gates::bit_decompose_msb_first(y);
    let expected_bits = sha2_gkr::nonlinear_gates::bit_decompose_msb_first(expected_sum);

    let input_mle = MultilinearExtension::new(
        x_bits
            .into_iter()
            .chain(y_bits.into_iter())
            .chain(expected_bits.into_iter())
            .chain(std::iter::repeat(0).take(32))
            .map(u64::from)
            .map(F::from)
            .collect_vec(),
    );

    let performed_sum = adder.perform_addition(&mut circuit, x, y);
    assert!(!POSITIVE || performed_sum == expected_sum);
    circuit.set_input("All Inputs", input_mle);
    circuit.gen_provable_circuit().unwrap()
}

fn sha256_message_schedule_circuit<F, Adder>(
    with_layer_kind: Option<LayerVisibility>,
) -> (
    Circuit<F>,
    sha2_gkr::sha256_bit_decomp::MessageSchedule<F, Adder>,
)
where
    F: Field + From<u64>,
    Adder: AdderGateTrait<F, IntegralType = u32> + Clone,
{
    let mut builder = CircuitBuilder::new();

    let input_layer = builder.add_input_layer(
        "sha256_message_schedule_input_layer",
        LayerVisibility::Public,
    );
    let carry_layer = with_layer_kind.map(|layer_kind| {
        builder.add_input_layer("sha256_message_schedule_carry_layer", layer_kind)
    });
    // Each word is 32-bit input
    let word_size = 32_usize;

    // There are 16 input values word size
    let input_word_count = 16;

    // Output has 64 words
    let output_word_count = 64;

    let input_num_vars = (word_size * input_word_count).ilog2() as usize;
    let output_num_vars = (word_size * output_word_count).ilog2() as usize;

    let message_inputs = builder.add_input_shred("message_input", input_num_vars, &input_layer);

    let expected_output = builder.add_input_shred("expected_output", output_num_vars, &input_layer);

    let b = &message_inputs;
    let b_sq = AbstractExpression::products(vec![b.id(), b.id()]);
    let b = b.expr();

    // Check that all input bits are binary.
    let binary_sector = builder.add_sector(
        // b * (1 - b) = b - b^2
        b - b_sq,
    );

    // Make sure all inputs are either `0` or `1`
    builder.set_output(&binary_sector);

    let word_splits = builder.add_split_node(&message_inputs, input_word_count.ilog2() as usize);

    let msg_schedule = sha2_gkr::sha256_bit_decomp::MessageSchedule::<F, Adder>::new(
        &mut builder,
        carry_layer.as_ref(),
        &word_splits,
    );
    let schedule_is_valid =
        builder.add_sector(msg_schedule.get_output_expr() - expected_output.expr());

    builder.set_output(&schedule_is_valid);
    (builder.build().unwrap(), msg_schedule)
}

fn attach_data_sha256_message_schedule<const POSITIVE: bool, F: Halo2FFTFriendlyField, Adder>(
    mut circuit: Circuit<F>,
    msg_sched: &mut sha2_gkr::sha256_bit_decomp::MessageSchedule<F, Adder>,
) -> ProvableCircuit<F>
where
    F: Field + From<u64>,
    Adder: AdderGateTrait<F, IntegralType = u32> + Clone,
{
    let mut rng = rand::thread_rng();
    let mut input_message = [0u8; 16 * 4];
    rng.try_fill_bytes(&mut input_message).unwrap();

    let mut input_message_u32: [u32; 16] = [0; 16];

    for (w, d) in input_message_u32
        .iter_mut()
        .zip(input_message.iter().step_by(4))
        .take(16)
    {
        *w = u32::from_be_bytes(unsafe { *(d as *const u8 as *const [u8; 4]) });
    }

    let computed_sched = msg_sched.populate_message_schedule(&mut circuit, &input_message_u32);

    let expected_output = if POSITIVE {
        sha256_simple::Sha256::update_message_schedule(&input_message)
    } else {
        let mut random_output = [0u32; 64];
        for v in random_output.iter_mut() {
            *v = rng.next_u32();
        }
        random_output
    };

    (0..64)
        .into_iter()
        .for_each(|i| assert!(computed_sched[i] == expected_output[i]));

    let message_mle = MultilinearExtension::new(
        input_message
            .clone()
            .into_iter()
            .map(sha2_gkr::nonlinear_gates::bit_decompose_msb_first)
            .flatten()
            .map(u64::from)
            .map(F::from)
            .collect(),
    );

    let expected_mle = MultilinearExtension::new(
        expected_output
            .into_iter()
            .map(sha2_gkr::nonlinear_gates::bit_decompose_msb_first)
            .flatten()
            .map(u64::from)
            .map(F::from)
            .collect(),
    );

    circuit.set_input("message_input", message_mle);
    circuit.set_input("expected_output", expected_mle);
    circuit.gen_provable_circuit().unwrap()
}

fn sha256_test_circuit<F, Adder>(
    input_data: Vec<u8>,
    with_carry_layer: Option<LayerVisibility>,
) -> Circuit<F>
where
    F: Field + From<u64>,
    Adder: AdderGateTrait<F, IntegralType = u32> + Clone,
{
    let mut builder = CircuitBuilder::<F>::new();
    let mut hasher = sha256_simple::Sha256::default();
    hasher.update(&input_data);
    let expected_hash = hasher.finish();

    let hash_value_mle = MultilinearExtension::<F>::new(
        expected_hash
            .iter()
            .map(|v| sha2_gkr::nonlinear_gates::bit_decompose_msb_first(*v))
            .flatten()
            .map(u64::from)
            .map(F::from)
            .collect(),
    );

    let input_layer = builder.add_input_layer("sha256_test_circuit_input", LayerVisibility::Public);
    let committed_carry_layer = with_carry_layer
        .map(|layer_kind| builder.add_input_layer("sha256_test_circuit_carry", layer_kind));

    let hash_size_bits = 256u32.ilog2() as usize;

    let expected_output = builder.add_input_shred("expected_output", hash_size_bits, &input_layer);

    let output_words = builder.add_split_node(&expected_output, 3);

    let sha256_ckt = sha2_gkr::sha256_bit_decomp::Sha256::<F, Adder>::new(
        &mut builder,
        &input_layer,
        committed_carry_layer.as_ref(),
        input_data,
    );

    let output_nodes = sha256_ckt.get_output_node();

    let m = output_words
        .into_iter()
        .zip(output_nodes.into_iter())
        .map(|(expected, computed)| builder.add_sector(expected.expr() - computed.expr()))
        .collect_vec();

    m.into_iter().for_each(|v| builder.set_output(&v));

    let mut ckt = builder.build().unwrap();

    let computed_hash = sha256_ckt
        .populate_circuit(&mut ckt)
        .into_iter()
        .map(u32::to_be_bytes)
        .flatten()
        .collect_vec();

    assert_eq!(computed_hash, expected_hash);

    ckt.set_input("expected_output", hash_value_mle);
    ckt
}

fn sha256_test_with_data<F, Adder>(data: Vec<u8>, carry_layer_kind: Option<LayerVisibility>)
where
    F: Halo2FFTFriendlyField,
    Adder: AdderGateTrait<F, IntegralType = u32> + Clone,
{
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();
    let sha_circuit = sha256_test_circuit::<F, Adder>(data, carry_layer_kind);

    let provable_circuit = sha_circuit.gen_provable_circuit().unwrap();
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

#[ignore = "Test takes a long time"]
#[test]
fn sha256_single_round_committed_carry() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let data: Vec<u8> = vec!['a', 'b', 'c']
        .into_iter()
        .map(|x| x.try_into().unwrap())
        .collect_vec();
    sha256_test_with_data::<Fr, sha2_gkr::sha256_bit_decomp::CommittedCarryAdder<_>>(
        data,
        Some(LayerVisibility::Public),
    );
}

#[ignore = "Test takes a long time"]
#[test]
fn sha256_single_round_pp_adder() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let data: Vec<u8> = vec!['a', 'b', 'c']
        .into_iter()
        .map(|x| x.try_into().unwrap())
        .collect_vec();
    sha256_test_with_data::<Fr, sha2_gkr::brent_kung_adder::BKAdder<32, _>>(data, None);
}

#[ignore = "Test takes a long time"]
#[test]
fn sha256_single_round_rc_adder() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let data: Vec<u8> = vec!['a', 'b', 'c']
        .into_iter()
        .map(|x| x.try_into().unwrap())
        .collect_vec();
    sha256_test_with_data::<Fr, sha2_gkr::ripple_carry_adder::RippleCarryAdderMod2w<32, _>>(
        data, None,
    );
}

#[test]
fn sha256_message_schedule_positive_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let (circuit, mut sched) = sha256_message_schedule_circuit(Some(LayerVisibility::Public));
    let provable_circuit = attach_data_sha256_message_schedule::<
        true,
        Fr,
        sha2_gkr::sha256_bit_decomp::CommittedCarryAdder<Fr>,
    >(circuit.clone(), &mut sched);
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

#[test]
fn sha256_pp_adder_positive_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let (circuit, adder) =
        create_adder_test_circuit::<Fr, sha2_gkr::brent_kung_adder::BKAdder<32, Fr>>(None);

    for _ in 0..10 {
        let provable_circuit =
            attach_data_to_adder_gate::<true, _, _>(circuit.clone(), adder.clone());
        test_circuit_with_memory_optimized_config(&provable_circuit);
    }
}

#[test]
fn sha256_pp_adder_gate_negative_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let (circuit, adder) =
        create_adder_test_circuit::<Fr, sha2_gkr::brent_kung_adder::BKAdder<32, Fr>>(None);

    for _ in 0..10 {
        let provable_circuit =
            attach_data_to_adder_gate::<false, _, _>(circuit.clone(), adder.clone());
        let result = std::panic::catch_unwind(|| {
            test_circuit_with_memory_optimized_config(&provable_circuit)
        });
        assert!(result.is_err(), "A -ve test should panic");
    }
}

#[test]
fn sha256_ripple_adder_gate_positive_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let (circuit, adder) = create_adder_test_circuit::<
        Fr,
        sha2_gkr::ripple_carry_adder::RippleCarryAdderMod2w<32, Fr>,
    >(None);

    for _ in 0..10 {
        let provable_circuit =
            attach_data_to_adder_gate::<true, _, _>(circuit.clone(), adder.clone());
        test_circuit_with_memory_optimized_config(&provable_circuit);
    }
}

#[test]
fn sha256_ripple_adder_gate_negative_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let (circuit, adder) = create_adder_test_circuit::<
        Fr,
        sha2_gkr::ripple_carry_adder::RippleCarryAdderMod2w<32, Fr>,
    >(None);

    for _ in 0..10 {
        let provable_circuit =
            attach_data_to_adder_gate::<false, _, _>(circuit.clone(), adder.clone());
        let result = std::panic::catch_unwind(|| {
            test_circuit_with_memory_optimized_config(&provable_circuit)
        });
        assert!(result.is_err(), "A -ve test should panic");
    }
}

#[test]
fn sha256_committed_carry_adder_priv_positive_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let (circuit, adder) = create_adder_test_circuit::<
        Fr,
        sha2_gkr::sha256_bit_decomp::CommittedCarryAdder<Fr>,
    >(Some(LayerVisibility::Public));

    for _ in 0..10 {
        let provable_circuit =
            attach_data_to_adder_gate::<true, _, _>(circuit.clone(), adder.clone());
        test_circuit_with_memory_optimized_config(&provable_circuit);
    }
}

#[test]
fn sha256_committed_carry_adder_pub_positive_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let (circuit, adder) = create_adder_test_circuit::<
        Fr,
        sha2_gkr::sha256_bit_decomp::CommittedCarryAdder<_>,
    >(Some(LayerVisibility::Public));

    for _ in 0..10 {
        let provable_circuit =
            attach_data_to_adder_gate::<true, _, _>(circuit.clone(), adder.clone());
        test_circuit_with_memory_optimized_config(&provable_circuit);
    }
}

#[test]
fn sha256_committed_carry_adder_gate_negative_test() {
    let (circuit, adder) = create_adder_test_circuit::<
        Fr,
        sha2_gkr::sha256_bit_decomp::CommittedCarryAdder<_>,
    >(Some(LayerVisibility::Public));

    for _ in 0..10 {
        let provable_circuit =
            attach_data_to_adder_gate::<false, _, _>(circuit.clone(), adder.clone());
        let result = std::panic::catch_unwind(|| {
            test_circuit_with_memory_optimized_config(&provable_circuit)
        });
        assert!(result.is_err(), "A -ve test should panic");
    }
}

#[test]
fn sha256_test_padding() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let mut builder = CircuitBuilder::<Fr>::new();
    let input_layer =
        builder.add_input_layer("sha256_test_padding_input_layer", LayerVisibility::Public);
    let carry_layer =
        builder.add_input_layer("sha256_test_padding_carry_layer", LayerVisibility::Public);
    let input_data = ['a', 'b', 'c']
        .into_iter()
        .map(|x| x.try_into().unwrap())
        .collect_vec();
    let padded_data = [
        0x61626380, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000018,
    ];

    let sha256 = sha2_gkr::sha256_bit_decomp::Sha256::<
        Fr,
        sha2_gkr::sha256_bit_decomp::CommittedCarryAdder<Fr>,
    >::new(&mut builder, &input_layer, Some(&carry_layer), input_data);
    assert_eq!(sha256.padded_data_chunks(), padded_data.to_vec());
}

fn test_sha_value_evaluation<F, Adder>()
where
    F: Field,
    Adder: AdderGateTrait<F, IntegralType = u32> + Clone,
{
    let mut builder = CircuitBuilder::<F>::new();
    let expected_hash = vec![
        0xba7816bf_u32,
        0x8f01cfea,
        0x414140de,
        0x5dae2223,
        0xb00361a3,
        0x96177a9c,
        0xb410ff61,
        0xf20015ad,
    ];

    let input_layer =
        builder.add_input_layer("sha_value_evaluation_input_layer", LayerVisibility::Public);
    let carry_layer =
        builder.add_input_layer("sha_value_evaluation_carry_layer", LayerVisibility::Public);
    let input_data = ['a', 'b', 'c']
        .into_iter()
        .map(|x| x.try_into().unwrap())
        .collect_vec();
    let sha = sha2_gkr::sha256_bit_decomp::Sha256::<F, Adder>::new(
        &mut builder,
        &input_layer,
        Some(&carry_layer),
        input_data,
    );
    let mut ckt = builder.build().unwrap();
    let hash_values = sha.populate_circuit(&mut ckt);
    assert_eq!(hash_values, expected_hash);
}

#[test]
fn sha256_ripple_carry_test_evaluation() {
    test_sha_value_evaluation::<Fr, sha2_gkr::ripple_carry_adder::RippleCarryAdderMod2w<32, _>>();
}

#[test]
fn sha256_pp_adder_full_round_test() {
    test_sha_value_evaluation::<Fr, sha2_gkr::brent_kung_adder::BKAdder<32, _>>();
}

#[test]
fn sha256_committed_carry_adder_full_round_test() {
    test_sha_value_evaluation::<Fr, sha2_gkr::sha256_bit_decomp::CommittedCarryAdder<Fr>>();
}

///
/// This is an independent pure Rust implementation of SHA256 taken from
/// https://github.com/nanpuyue/sha256/ Licensed under Apache
///
#[allow(dead_code)]
pub(crate) mod sha256_simple {
    #![allow(clippy::unreadable_literal)]

    use core::default::Default;

    const H: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    pub struct Sha256 {
        state: [u32; 8],
        completed_data_blocks: u64,
        pending: [u8; 64],
        num_pending: usize,
    }

    impl Default for Sha256 {
        fn default() -> Self {
            Self {
                state: H,
                completed_data_blocks: 0,
                pending: [0u8; 64],
                num_pending: 0,
            }
        }
    }

    impl Sha256 {
        pub fn with_state(state: [u32; 8]) -> Self {
            Self {
                state,
                completed_data_blocks: 0,
                pending: [0u8; 64],
                num_pending: 0,
            }
        }

        pub fn update_message_schedule(data: &[u8; 64]) -> [u32; 64] {
            let mut w = [0; 64];
            for (w, d) in w.iter_mut().zip(data.iter().step_by(4)).take(16) {
                *w = u32::from_be_bytes(unsafe { *(d as *const u8 as *const [u8; 4]) });
            }

            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            w
        }

        pub fn update_state(state: &mut [u32; 8], data: &[u8; 64]) {
            let w = Self::update_message_schedule(data);

            // println!(
            //     "Input message schedule: \n{}",
            //     w.iter()
            //         .map(|v| format!("  0x{:08x}", v))
            //         .collect::<Vec<_>>()
            //         .join("\n")
            // );

            let mut h = *state;

            // println!("=============");

            // println!(
            //     "Input state: \n{}",
            //     h.iter()
            //         .map(|v| format!("  0x{:08x}", v))
            //         .collect::<Vec<_>>()
            //         .join("\n")
            // );

            for i in 0..64 {
                let ch = (h[4] & h[5]) ^ (!h[4] & h[6]);
                let ma = (h[0] & h[1]) ^ (h[0] & h[2]) ^ (h[1] & h[2]);
                let s0 = h[0].rotate_right(2) ^ h[0].rotate_right(13) ^ h[0].rotate_right(22);
                let s1 = h[4].rotate_right(6) ^ h[4].rotate_right(11) ^ h[4].rotate_right(25);
                let t0 = h[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let t1 = s0.wrapping_add(ma);

                h[7] = h[6];
                h[6] = h[5];
                h[5] = h[4];
                h[4] = h[3].wrapping_add(t0);
                h[3] = h[2];
                h[2] = h[1];
                h[1] = h[0];
                h[0] = t0.wrapping_add(t1);

                //     println!("------> {i} <-------");

                //     println!(
                //         "\t{}",
                //         h.iter()
                //             .map(|v| format!("0x{:08x}", v))
                //             .collect::<Vec<_>>()
                //             .join("\n\t")
                //     );
            }

            for (i, v) in state.iter_mut().enumerate() {
                *v = v.wrapping_add(h[i]);
            }

            // println!("=============");

            // println!(
            //     "\nFinal state: {}",
            //     state
            //         .iter()
            //         .map(|v| format!("0x{:08x}", v))
            //         .collect::<Vec<_>>()
            //         .join("\n\t")
            // );
        }

        pub fn update(&mut self, data: &[u8]) {
            let mut len = data.len();
            let mut offset = 0;

            if self.num_pending > 0 && self.num_pending + len >= 64 {
                self.pending[self.num_pending..].copy_from_slice(&data[..64 - self.num_pending]);
                Self::update_state(&mut self.state, &self.pending);
                self.completed_data_blocks += 1;
                offset = 64 - self.num_pending;
                len -= offset;
                self.num_pending = 0;
            }

            let data_blocks = len / 64;
            let remain = len % 64;
            for _ in 0..data_blocks {
                Self::update_state(&mut self.state, unsafe {
                    &*(data.as_ptr().add(offset) as *const [u8; 64])
                });
                offset += 64;
            }
            self.completed_data_blocks += data_blocks as u64;

            if remain > 0 {
                self.pending[self.num_pending..self.num_pending + remain]
                    .copy_from_slice(&data[offset..]);
                self.num_pending += remain;
            }
        }

        pub fn finish(mut self) -> [u8; 32] {
            let data_bits = self.completed_data_blocks * 512 + self.num_pending as u64 * 8;
            let mut pending = [0u8; 72];
            pending[0] = 128;

            let offset = if self.num_pending < 56 {
                56 - self.num_pending
            } else {
                120 - self.num_pending
            };

            pending[offset..offset + 8].copy_from_slice(&data_bits.to_be_bytes());
            self.update(&pending[..offset + 8]);

            for h in self.state.iter_mut() {
                *h = h.to_be();
            }
            unsafe { *(self.state.as_ptr() as *const [u8; 32]) }
        }

        pub fn digest(data: &[u8]) -> [u8; 32] {
            let mut sha256 = Self::default();
            sha256.update(data);
            sha256.finish()
        }

        pub fn state(&self) -> [u32; 8] {
            self.state
        }
    }

    // #[cfg(test)]
    // mod test {
    //     use super::Sha256;
    //     use itertools::Itertools;
    //     #[test]
    //     fn sample_sha256_evaluation() {
    //         let msg: Vec<u8> = ['a', 'b', 'c']
    //             .into_iter()
    //             .map(|x| x.try_into().unwrap())
    //             .collect_vec();
    //         let mut sha = Sha256::default();
    //         sha.update(&msg);
    //         let result = sha.finish();
    //         println!("{:?}", result);
    //     }
    // }
}
