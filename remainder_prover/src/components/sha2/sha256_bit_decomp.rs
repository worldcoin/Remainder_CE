//!
//! Implementation of SHA-256 circuit using bitwise decomposition
//!

use super::nonlinear_gates::*;
use super::ripple_carry_adder::RippleCarryAdderMod2w;
use crate::binary_operations::binary_adder::BinaryAdder;
use crate::expression::abstract_expr::ExprBuilder;
use crate::layouter::builder::{Circuit, CircuitBuilder, LayerKind, NodeRef};
use crate::mle::evals::MultilinearExtension;
use itertools::Itertools;
use rand::RngCore;
use remainder_shared_types::Field;
use std::ops::Index;

pub const WORD_SIZE: usize = 32;

// See https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf for details about constants
pub type Sigma0 = Sigma<WORD_SIZE, 2, 13, 22>;
pub type Sigma1 = Sigma<WORD_SIZE, 6, 11, 25>;
pub type SmallSigma0 = SmallSigma<WORD_SIZE, 7, 18, 3>;
pub type SmallSigma1 = SmallSigma<WORD_SIZE, 17, 19, 10>;
pub type Sha256Adder = CommittedCarryAdder;

fn random256bit_string() -> String {
    let mut rng = rand::thread_rng();
    [
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
    ]
    .into_iter()
    .map(|v| format!("{:08x}", v))
    .collect::<Vec<_>>()
    .join("-")
}

const fn ch_gate_eval(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (!x & z)
}

const fn maj_eval(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (y & z) ^ (x & x)
}

fn add_get_carry_bits_lsb(x: u32, y: u32, mut c_in: u32) -> (u32, Vec<u32>) {
    debug_assert!(c_in == 0 || c_in == 1);
    let x_vec = bit_decompose_lsb_first(x);
    let y_vec = bit_decompose_lsb_first(y);
    let mut carry = Vec::<u32>::with_capacity(33);
    let mut sum = 0x0u32;
    for (i, (x, y)) in x_vec.into_iter().zip(y_vec.into_iter()).enumerate() {
        let this_sum = x + y + c_in;
        c_in = this_sum / 2;
        carry.push(c_in);
        sum = sum.wrapping_add((this_sum % 2) << i);
    }
    (sum, carry)
}

fn add_get_carry_bits_msb(x: u32, y: u32, c_in: u32) -> (u32, Vec<u32>) {
    let (s, mut c) = add_get_carry_bits_lsb(x, y, c_in);
    c.reverse();
    (s, c)
}

/// An adder that just checks the carry bits instead of explicitly
/// computing it through Ripple Carry Adder.
#[derive(Debug, Clone)]
pub struct CommittedCarryAdder {
    // Automatically generated carry shred name
    carry_shred_name: String,
    /// Node representing the sum
    sum_node: NodeRef,
}

impl CommittedCarryAdder {
    /// Adder that internally generates an input node for carry bits and
    /// only checks that the sums are correct instead of computing it.
    pub fn new<F: Field>(
        ckt_builder: &mut CircuitBuilder<F>,
        x_node: &NodeRef,
        y_node: &NodeRef,
    ) -> Self {
        debug_assert_eq!(x_node.get_num_vars(), 5);
        debug_assert_eq!(y_node.get_num_vars(), 5);
        // TODO: Needs to be Private!
        let carry_bits_node = ckt_builder.add_input_layer(LayerKind::Public);
        // Probability of collision is 2^{-128}
        let carry_shred_name = random256bit_string();
        let carry_shred =
            ckt_builder.add_input_shred(&carry_shred_name, x_node.get_num_vars(), &carry_bits_node);

        // Check that all input bits are binary.
        let b_sq = ExprBuilder::products(vec![carry_shred.id(), carry_shred.id()]);
        let b = carry_shred.expr();

        // Check that all input bits are binary.
        let binary_sector = ckt_builder.add_sector(
            // b * (1 - b) = b - b^2
            b - b_sq,
        );

        ckt_builder.set_output(&binary_sector);

        let binary_adder = BinaryAdder::new(ckt_builder, x_node, y_node, &carry_shred);

        Self {
            carry_shred_name,
            sum_node: binary_adder.get_output(),
        }
    }

    pub fn get_output(&self) -> NodeRef {
        self.sum_node.clone()
    }

    /// Given an instantiated circuit, adds gate to the circuit as input
    /// with correct input label name.
    pub fn populate_carry<F: Field>(
        &self,
        circuit: &mut Circuit<F>,
        x_val: u32,
        y_val: u32,
    ) -> u32 {
        let (s, carries) = add_get_carry_bits_msb(x_val, y_val, 0);
        debug_assert_eq!(s, x_val.wrapping_add(y_val));

        let carry_mle = MultilinearExtension::new(
            carries
                .into_iter()
                .map(u64::from)
                .map(F::from)
                .collect_vec(),
        );
        circuit.set_input(&self.carry_shred_name, carry_mle);
        s
    }
}

#[derive(Debug, Clone)]
struct MessageScheduleAdderTree {
    sum_a_leaf: CommittedCarryAdder, // A = Sum of SmallSigma1(state[t-2]) + state[t - 7]
    sum_b_leaf: CommittedCarryAdder, // B = Sum of SmallSigma0(state[t-15]) + state[t - 16]
    sum_a_b: CommittedCarryAdder,    // C = A + B
}

#[derive(Debug, Clone)]
struct MessageScheduleState {
    state_node: NodeRef,
    state_adders: Option<MessageScheduleAdderTree>,
}

/// Represents the 64 rounds of message schedule. Each Round consists of
/// 32 Wires where the first 16 rounds are identity gates, and the rest
/// are computed as per the spec
pub struct MessageSchedule {
    msg_schedule: Vec<MessageScheduleState>,
}

impl MessageSchedule {
    /// Given the 16, 32-bit inputs in MBS format, computes the 64
    /// rounds of message schedule corresponding to the input. The
    /// `msg_vars` must be the 16 32-bit words decomposed in MBS format.
    pub fn new<F: Field>(ckt_builder: &mut CircuitBuilder<F>, msg_vars: &[NodeRef]) -> Self {
        debug_assert_eq!(msg_vars.len(), 16);

        let mut state: Vec<MessageScheduleState> = Vec::with_capacity(64);

        (0..16).for_each(|i| {
            debug_assert!(msg_vars[i].get_num_vars() == 5);
            state.push(MessageScheduleState {
                state_node: msg_vars[i].clone(),
                state_adders: None,
            })
        });

        for t in 16..64 {
            assert!(state.len() >= t - 16);
            let small_sigma_1_val = SmallSigma1::new(ckt_builder, &state[t - 2].state_node);
            let w_first = state[t - 7].state_node.clone();

            let small_sigma_0_val = SmallSigma0::new(ckt_builder, &state[t - 15].state_node);
            let w_second = state[t - 16].state_node.clone();

            let sum_a_leaf =
                Sha256Adder::new(ckt_builder, &small_sigma_1_val.get_output(), &w_first);

            let sum_b_leaf =
                Sha256Adder::new(ckt_builder, &small_sigma_0_val.get_output(), &w_second);

            let sum_a_b = Sha256Adder::new(ckt_builder, &sum_a_leaf.sum_node, &sum_b_leaf.sum_node);
            let current_state = MessageScheduleState {
                state_node: sum_a_b.get_output(),
                state_adders: Some(MessageScheduleAdderTree {
                    sum_a_leaf,
                    sum_b_leaf,
                    sum_a_b,
                }),
            };

            state.push(current_state);
        }

        Self {
            msg_schedule: state,
        }
    }

    /// Returns the list of 64 nodes corresponding to SHA256 message schedule
    pub fn get_output_nodes(&self) -> Vec<NodeRef> {
        self.msg_schedule
            .iter()
            .map(|st| st.state_node.clone())
            .collect()
    }

    /// Attaches a message schedule to the SHA Circuit. Note that this
    /// needs to match with the way adder tree is implemented
    pub fn populate_message_schedule<F: Field>(
        &mut self,
        circuit: &mut Circuit<F>,
        input_data: &[u32],
    ) -> Vec<u32> {
        debug_assert_eq!(input_data.len(), 16);
        let mut state: Vec<u32> = Vec::with_capacity(64);

        (0..16).for_each(|i| state.push(input_data[i]));

        for t in 16..64 {
            assert!(state.len() >= t - 16);
            let small_sigma_1_val = SmallSigma1::evaluate(state[t - 2]);
            let w_first = state[t - 7];

            let small_sigma_0_val = SmallSigma0::evaluate(state[t - 15]);
            let w_second = state[t - 16];

            let add_tree = self.msg_schedule[t].state_adders.clone().unwrap();

            let sum_a_value =
                add_tree
                    .sum_a_leaf
                    .populate_carry(circuit, small_sigma_1_val, w_first);

            let sum_b_value =
                add_tree
                    .sum_b_leaf
                    .populate_carry(circuit, small_sigma_0_val, w_second);

            let sum_a_b_value = add_tree
                .sum_a_b
                .populate_carry(circuit, sum_a_value, sum_b_value);

            debug_assert_eq!(
                sum_a_b_value,
                small_sigma_1_val
                    .wrapping_add(w_first)
                    .wrapping_add(small_sigma_0_val)
                    .wrapping_add(w_second)
            );
            state.push(sum_a_b_value);
        }

        state
    }

    /// Creates the output expression that can be tested with other
    /// output expressions
    pub fn get_output_expr<F: Field>(&self) -> ExprBuilder<F> {
        ExprBuilder::<F>::selectors(
            self.msg_schedule
                .iter()
                .map(|st| st.state_node.expr())
                .collect(),
        )
    }
}

pub struct KeySchedule<F: Field> {
    keys: Vec<ConstInputGate<F>>,
}

pub struct HConstants<F: Field> {
    ivs: Vec<ConstInputGate<F>>,
}

impl<F: Field> HConstants<F> {
    const H: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    fn iv_name(index: usize) -> String {
        format!("sha256-iv-{index:x}")
    }

    pub fn new(ckt_builder: &mut CircuitBuilder<F>) -> Self {
        Self {
            ivs: Self::H
                .iter()
                .enumerate()
                .map(|(i, val)| ConstInputGate::new(ckt_builder, &Self::iv_name(i), *val))
                .collect(),
        }
    }

    pub fn add_iv_to_circuit(&self, circuit: &mut Circuit<F>) {
        for (ndx, const_iv) in self.ivs.iter().enumerate() {
            circuit.set_input(&Self::iv_name(ndx), const_iv.input_mle().clone());
        }
    }
}

impl<F: Field> KeySchedule<F> {
    const ROUND_KEYS: [u32; 64] = [
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

    fn key_name(index: usize) -> String {
        format!("sha256-key-schedule-{index}")
    }

    pub fn new(ckt_builder: &mut CircuitBuilder<F>) -> Self {
        Self {
            keys: Self::ROUND_KEYS
                .iter()
                .enumerate()
                .map(|(i, val)| ConstInputGate::new(ckt_builder, &Self::key_name(i), *val))
                .collect(),
        }
    }

    pub fn add_key_schedule_to_circuit(&self, circuit: &mut Circuit<F>) {
        for (i, key_const) in self.keys.iter().enumerate() {
            circuit.set_input(&Self::key_name(i), key_const.input_mle().clone());
        }
    }
}

impl<F: Field> Index<usize> for KeySchedule<F> {
    type Output = NodeRef;

    fn index(&self, index: usize) -> &Self::Output {
        self.keys[index].get_output_ref()
    }
}

impl<F: Field> Index<usize> for HConstants<F> {
    type Output = NodeRef;

    fn index(&self, index: usize) -> &Self::Output {
        self.ivs[index].get_output_ref()
    }
}

/// Computes the single round of compression function
pub struct CompressionFn {
    output: Vec<NodeRef>,
}

impl CompressionFn {
    /// A Single Round of SHA-256 compression function. The
    /// `msg_schedule` is the 64 rounds of expanded message schedule.
    /// The `input_schedule` is the 256-bits of Input values and
    /// `round_keys`
    pub fn new<F: Field>(
        ckt_builder: &mut CircuitBuilder<F>,
        msg_schedule: &MessageSchedule,
        input_schedule: &[NodeRef],
        round_keys: KeySchedule<F>,
    ) -> Self {
        let msg_schedule = msg_schedule.get_output_nodes();
        debug_assert_eq!(msg_schedule.len(), 64);
        debug_assert_eq!(input_schedule.len(), 64);
        (0..64).for_each(|i| debug_assert_eq!(msg_schedule[i].get_num_vars(), 5));
        (0..8).for_each(|i| debug_assert_eq!(input_schedule[i].get_num_vars(), 5));

        let mut a = input_schedule[0].clone();
        let mut b = input_schedule[1].clone();
        let mut c = input_schedule[2].clone();
        let mut d = input_schedule[3].clone();
        let mut e = input_schedule[4].clone();
        let mut f = input_schedule[5].clone();
        let mut g = input_schedule[6].clone();
        let mut h = input_schedule[7].clone();

        for t in 0..64 {
            let w_t = msg_schedule[t].clone();
            let k_t = round_keys[t].clone();
            let t1 = Self::compute_t1(ckt_builder, &e, &f, &g, &h, &w_t, &k_t);
            let t2 = Self::compute_t2(ckt_builder, &a, &b, &c);
            h = ckt_builder.add_sector(g.expr());
            g = ckt_builder.add_sector(f.expr());
            f = ckt_builder.add_sector(e.expr());
            e = Sha256Adder::new(ckt_builder, &d, &t1).get_output();
            d = ckt_builder.add_sector(c.expr());
            c = ckt_builder.add_sector(b.expr());
            b = ckt_builder.add_sector(a.expr());
            a = Sha256Adder::new(ckt_builder, &t1, &t2).get_output();
        }

        let intermediates = [a, b, c, d, e, f, g, h];

        let output = input_schedule
            .iter()
            .zip(intermediates.iter())
            .map(|(h, x)| Sha256Adder::new(ckt_builder, h, x).get_output())
            .collect();

        Self { output }
    }

    fn compute_t1<F: Field>(
        ckt_builder: &mut CircuitBuilder<F>,
        e: &NodeRef,
        f: &NodeRef,
        g: &NodeRef,
        h: &NodeRef,
        w_t: &NodeRef,
        k_t: &NodeRef,
    ) -> NodeRef {
        debug_assert!(e.get_num_vars() == 5);
        debug_assert!(f.get_num_vars() == 5);
        debug_assert!(g.get_num_vars() == 5);
        debug_assert!(h.get_num_vars() == 5);
        debug_assert!(w_t.get_num_vars() == 5);
        debug_assert!(k_t.get_num_vars() == 5);

        let t1_sigma_1 = Sigma1::new(ckt_builder, e);
        let t1_ch = ChGate::new(ckt_builder, e, f, g);

        // h1 + Sigma1(e)
        let sum1 = Sha256Adder::new(ckt_builder, h, &t1_sigma_1.get_output());

        // ch(e,f,g) + K_t
        let sum2 = Sha256Adder::new(ckt_builder, &t1_ch.get_output(), k_t);

        // h1 + Sigma1(e) + ch(e,f,g) + K_t
        let sum3 = Sha256Adder::new(ckt_builder, &sum1.get_output(), &sum2.get_output());

        // h1 + Sigma1(e) + ch(e,f,g) + K_t + W_t
        let sum4 = Sha256Adder::new(ckt_builder, &sum3.get_output(), w_t);

        sum4.get_output()
    }

    fn compute_t2<F: Field>(
        ckt_builder: &mut CircuitBuilder<F>,
        a: &NodeRef,
        b: &NodeRef,
        c: &NodeRef,
    ) -> NodeRef {
        let s1 = Sigma0::new(ckt_builder, a);
        let m1 = MajGate::new(ckt_builder, a, b, c);
        Sha256Adder::new(ckt_builder, &s1.get_output(), &m1.get_output()).get_output()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bit_decomp_carry() {
        let x = 0xffffffffu32;
        let y = 0x1u32;
        let z = 0x80000000u32;

        #[rustfmt::skip]
        let carry_x_plus_y = vec![
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
        ];

        let (s, c) = super::add_get_carry_bits_lsb(x, y, 0);
        assert_eq!(s, x.wrapping_add(y));
        assert_eq!(c, carry_x_plus_y);

        let (s, c) = super::add_get_carry_bits_lsb(x, y, 1);
        assert_eq!(s, x.wrapping_add(y).wrapping_add(1));
        assert_eq!(c, carry_x_plus_y);

        #[rustfmt::skip]
        let carry_x_plus_z = vec![
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1
        ];

        let (s, c) = super::add_get_carry_bits_lsb(x, z, 0);
        assert_eq!(s, x.wrapping_add(z));
        assert_eq!(c, carry_x_plus_z);

        #[rustfmt::skip]
        let carry_x_plus_z = vec![
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
        ];

        let (s, c) = super::add_get_carry_bits_lsb(x, z, 1);
        assert_eq!(s, x.wrapping_add(z).wrapping_add(1));
        assert_eq!(c, carry_x_plus_z);
    }
}
