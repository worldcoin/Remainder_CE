//!
//! Implementation of SHA-256 circuit using bitwise decomposition
//!

use super::nonlinear_gates::*;
use super::ripple_carry_adder::AdderNoCarry;
use crate::expression::abstract_expr::ExprBuilder;
use crate::layouter::builder::{CircuitBuilder, NodeRef};
use remainder_shared_types::Field;

pub const WORD_SIZE: usize = 32;

// See https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf for details about constants
pub type Sigma0 = Sigma<WORD_SIZE, 2, 13, 22>;
pub type Sigma1 = Sigma<WORD_SIZE, 6, 11, 25>;
pub type SmallSigma0 = SmallSigma<WORD_SIZE, 7, 18, 3>;
pub type SmallSigma1 = SmallSigma<WORD_SIZE, 17, 19, 10>;
pub type Sha256Adder = AdderNoCarry<32>;

const ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Represents the 64 rounds of message schedule. Each Round consists of
/// 32 Wires where the first 16 rounds are identity gates, and the rest
/// are computed as per the spec
pub struct MessageSchedule {
    msg_schedule: Vec<NodeRef>,
}

impl MessageSchedule {
    /// Given the 16, 32-bit inputs in MBS format, computes the 64
    /// rounds of message schedule corresponding to the input. The
    /// `msg_vars` must be the 16 32-bit words decomposed in MBS format.
    pub fn new<F: Field>(ckt_builder: &mut CircuitBuilder<F>, msg_vars: &[NodeRef]) -> Self {
        let mut state: Vec<NodeRef> = Vec::with_capacity(64);

        debug_assert!(msg_vars.len() == 16);
        (0..16).for_each(|i| debug_assert!(msg_vars[i].get_num_vars() == 5));

        state.extend_from_slice(msg_vars);

        for t in 16..64 {
            assert!(state.len() >= t - 16);
            let small_sigma_1_val = SmallSigma1::new(ckt_builder, &state[t - 2]);
            let w_first = state[t - 7].clone();

            let small_sigma_0_val = SmallSigma0::new(ckt_builder, &state[t - 15]);
            let w_second = state[t - 16].clone();

            let add1 = Sha256Adder::new(ckt_builder, &small_sigma_1_val.get_output(), &w_first);

            let add2 = Sha256Adder::new(ckt_builder, &small_sigma_0_val.get_output(), &w_second);

            state.push(
                Sha256Adder::new(ckt_builder, &add1.get_output(), &add2.get_output()).get_output(),
            );
        }

        Self {
            msg_schedule: state,
        }
    }

    pub fn get_output_nodes(&self) -> Vec<NodeRef> {
        self.msg_schedule.clone()
    }

    pub fn get_output_expr<F: Field>(&self) -> ExprBuilder<F> {
        ExprBuilder::<F>::selectors(self.msg_schedule.iter().map(|n| n.expr()).collect())
    }
}

/// Computes the single round of compression function
pub struct CompressionFn {
    state: NodeRef,
}

impl CompressionFn {
    pub fn new<F: Field>(
        ckt_builder: &mut CircuitBuilder<F>,
        msg_schedule: &MessageSchedule,
        key_schedule: &[NodeRef],
    ) {
        let msg_schedule = msg_schedule.get_output_nodes();
        debug_assert_eq!(msg_schedule.len(), 64);
        debug_assert_eq!(key_schedule.len(), 8);
        (0..64).for_each(|i| debug_assert_eq!(msg_schedule[i].get_num_vars(), 5));
        (0..8).for_each(|i| debug_assert_eq!(key_schedule[i].get_num_vars(), 5));
    }

    fn compute_t1<F: Field>(
        ckt_builder: &mut CircuitBuilder<F>,
        e: NodeRef,
        f: NodeRef,
        g: NodeRef,
        h: NodeRef,
        w: NodeRef,
        msg_schedule: &NodeRef,
        key_schedule: &NodeRef,
    ) -> NodeRef {
        debug_assert!(e.get_num_vars() == 5);
        debug_assert!(f.get_num_vars() == 5);
        debug_assert!(g.get_num_vars() == 5);
        debug_assert!(h.get_num_vars() == 5);
        debug_assert!(w.get_num_vars() == 5);
        debug_assert!(msg_schedule.get_num_vars() == 5);
        debug_assert!(key_schedule.get_num_vars() == 5);

        let t1_sigma_1 = Sigma1::new(ckt_builder, &e);
        let t1_ch = ChGate::new(ckt_builder, &e, &f, &g);

        // h1 + Sigma1(e)
        let sum1 = Sha256Adder::new(ckt_builder, &h, &t1_sigma_1.get_output());

        // ch(e,f,g) + K_t
        let sum2 = Sha256Adder::new(ckt_builder, &t1_ch.get_output(), key_schedule);

        // h1 + Sigma1(e) + ch(e,f,g) + K_t
        let sum3 = Sha256Adder::new(ckt_builder, &sum1.get_output(), &sum2.get_output());

        // h1 + Sigma1(e) + ch(e,f,g) + K_t + W_t
        let sum4 = Sha256Adder::new(ckt_builder, &sum3.get_output(), msg_schedule);

        sum4.get_output()
    }
}
