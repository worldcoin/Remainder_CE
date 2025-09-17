//! Brent-Kung adder is a parallel prefix adder described here https://maths-people.anu.edu.au/~brent/pd/rpb060_IEEETC.pdf
//!
//! For a comparison of different adders, see also https://www.lirmm.fr/arith18/papers/patil-RobustEnergyEffcientAdder.pdf

use crate::layouter::builder::{CircuitBuilder, NodeRef};
use remainder_shared_types::Field;

pub struct BKAdder<const BitWidth: usize> {
    sum_value: NodeRef,
}
