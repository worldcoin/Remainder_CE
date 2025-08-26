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
use remainder_shared_types::{Field, Fr};
use std::ops::{BitAnd, BitOr, BitXor, Shl, Shr};

fn sha256_adder_gate_circuit() -> Circuit<Fr> {
    let mut builder = CircuitBuilder::new();

    let input_layer = builder.add_input_layer(LayerKind::Public);

    let num_vars = 5; // 32-bit adder

    // Inputs are 32-bit x, 32-bit y, 32-bit z, rest all zeros.
    let all_inputs = builder.add_input_shred("All Inputs", 2 + num_vars, &input_layer);

    let b = &all_inputs;
    let b_sq = ExprBuilder::products(vec![b.id(), b.id()]);
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

    let mod32_rpa = sha256::Sha256Adder::new(&mut builder, &x, &y);
    let sum_is_valid = builder.add_sector(mod32_rpa.get_output().expr() - expected_sum.expr());

    builder.set_output(&sum_is_valid);
    builder.build().unwrap()
}

fn attach_sha256_adder_gate_data<const POSITIVE: bool>(
    mut circuit: Circuit<Fr>,
) -> ProvableCircuit<Fr> {
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

    let x_bits = sha2::bit_decompose_msb_first(x);
    let y_bits = sha2::bit_decompose_msb_first(y);
    let expected_bits = sha2::bit_decompose_msb_first(expected_sum);

    let input_mle = MultilinearExtension::new(
        x_bits
            .into_iter()
            .chain(y_bits.into_iter())
            .chain(expected_bits.into_iter())
            .chain(std::iter::repeat(0).take(32))
            .map(u64::from)
            .map(Fr::from)
            .collect_vec(),
    );

    circuit.set_input("All Inputs", input_mle);
    circuit.finalize().unwrap()
}

fn sha256_message_schedule_circuit() -> Circuit<Fr> {
    let mut builder = CircuitBuilder::new();

    let input_layer = builder.add_input_layer(LayerKind::Public);

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
    let b_sq = ExprBuilder::products(vec![b.id(), b.id()]);
    let b = b.expr();

    // Check that all input bits are binary.
    let binary_sector = builder.add_sector(
        // b * (1 - b) = b - b^2
        b - b_sq,
    );

    // Make sure all inputs are either `0` or `1`
    builder.set_output(&binary_sector);

    let word_splits = builder.add_split_node(&message_inputs, input_word_count.ilog2() as usize);

    let msg_schedile = sha256::MessageSchedule::new(&mut builder, &word_splits);
    let schedule_is_valid =
        builder.add_sector(msg_schedile.get_output_expr::<Fr>() - expected_output.expr());

    builder.set_output(&schedule_is_valid);
    builder.build().unwrap()
}

fn attach_data_sha256_message_schedule<const POSITIVE: bool>(
    mut circuit: Circuit<Fr>,
) -> ProvableCircuit<Fr> {
    let mut rng = rand::thread_rng();
    let mut input_message = [0u8; 16 * 4];
    rng.try_fill_bytes(&mut input_message).unwrap();

    let expected_output = if POSITIVE {
        sha256_simple::Sha256::update_message_schedule(&input_message)
    } else {
        let mut random_output = [0u32; 64];
        for v in random_output.iter_mut() {
            *v = rng.next_u32();
        }
        random_output
    };

    let message_mle = MultilinearExtension::new(
        input_message
            .clone()
            .into_iter()
            .map(sha2::bit_decompose_msb_first)
            .flatten()
            .map(u64::from)
            .map(Fr::from)
            .collect(),
    );

    let expected_mle = MultilinearExtension::new(
        expected_output
            .into_iter()
            .map(sha2::bit_decompose_msb_first)
            .flatten()
            .map(u64::from)
            .map(Fr::from)
            .collect(),
    );

    circuit.set_input("message_input", message_mle);
    circuit.set_input("expected_output", expected_mle);
    circuit.finalize().unwrap()
}

#[test]
fn sha256_message_schedule_positive_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = sha256_message_schedule_circuit();

    let provable_circuit = attach_data_sha256_message_schedule::<true>(circuit.clone());
    test_circuit_with_memory_optimized_config(&provable_circuit);
}

#[test]
fn sha256_adder_gate_positive_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = sha256_adder_gate_circuit();

    for _ in 0..10 {
        let provable_circuit = attach_sha256_adder_gate_data::<true>(circuit.clone());
        test_circuit_with_memory_optimized_config(&provable_circuit);
    }
}

#[test]
fn sha256_adder_gate_negative_test() {
    // let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let circuit = sha256_adder_gate_circuit();

    for _ in 0..10 {
        let provable_circuit = attach_sha256_adder_gate_data::<false>(circuit.clone());
        let result = std::panic::catch_unwind(|| {
            test_circuit_with_memory_optimized_config(&provable_circuit)
        });
        assert!(result.is_err(), "A -ve test should panic");
    }
}

///
/// This is an independent pure Rust implementation of SHA256 taken from
/// https://github.com/nanpuyue/sha256/ Licensed under Apache
///
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
            let mut w = Self::update_message_schedule(data);

            let mut h = *state;
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
            }

            for (i, v) in state.iter_mut().enumerate() {
                *v = v.wrapping_add(h[i]);
            }
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
}
