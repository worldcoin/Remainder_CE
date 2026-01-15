//! Implements bit shifting gates.

use std::cmp::{max, min};

use itertools::Itertools;
use shared_types::Field;

use crate::layouter::builder::{CircuitBuilder, NodeRef};

/// A component that performs logical bit shift operations using [IdentityGateNode].
///
/// Both left and right shifts are supported by using negative/positive values in the `shift_amount`
/// parameter.
///
/// This is a _logical_ shift, meaning that bits which are shifted out are discarded, and zeros are
/// filled in on the other side.
///
/// Requires that the input node has already been verified to contain binary digits.
#[derive(Clone, Debug)]
pub struct ShiftNode<F: Field> {
    output: NodeRef<F>,
}

impl<F: Field> ShiftNode<F> {
    /// Create a new [ShiftNode] that performs a shift by `shift_amount` (to the right if
    /// `shift_amount > 0` or to the left if `shift_amount < 0`) on `input` node which contains
    /// `2^num_vars` binary digits.
    ///
    /// # Requires
    /// `input` is assumed to only contain binary digits (i.e. only values from the set
    /// `{F::ZERO, F::ONE}` for a field `F`).
    pub fn new(
        builder_ref: &mut CircuitBuilder<F>,
        num_vars: usize,
        shift_amount: i32,
        input: &NodeRef<F>,
    ) -> Self {
        // Compute the bit reroutings that effectively shift the
        // input MLE by the appropriate amount.
        let shift_wirings = generate_shift_wirings(num_vars, shift_amount);
        let output = builder_ref.add_identity_gate_node(input, shift_wirings, num_vars, None);

        Self { output }
    }

    /// Returns a reference to the node containing the shifted value.
    pub fn get_output(&self) -> NodeRef<F> {
        self.output.clone()
    }
}

fn generate_shift_wirings(num_vars: usize, shift_amount: i32) -> Vec<(u32, u32)> {
    // Ensure `shift_amount` can represent all possible shift amounts for a given value of
    // `num_vars`.
    // In general, if `shift_amount` is a signed `n`-bit integer, it can represent shift amounts
    // in the integer range `[-2^(n-1), +2^(n-1) - 1]`. For a `2^num_vars`-bit shifter,
    // ideally we'd like to support all bit shift values in the integer range `[-2^num_vars,
    // +2^num_vars]`, therefore we have to work under the assumption that `num_vars < n-1`.
    // Here `shift_amount` is of type `i32` (`n == 32`), so we need `num_vars <= 30`.
    assert!(num_vars <= 30);

    // Cap the shift amount to be in the range `[-2^num_vars, 2^num_vars]`.
    let shift_amount = max(min(shift_amount, 1 << num_vars), -(1 << num_vars));

    if shift_amount >= 0 {
        let shift_amount = shift_amount as u32;
        (0..(1 << num_vars) - shift_amount)
            .map(|i| (i + shift_amount, i))
            .collect_vec()
    } else {
        let shift_amount = shift_amount.unsigned_abs();
        (0..(1 << num_vars) - shift_amount as u32)
            .map(|i| (i, i + shift_amount))
            .collect_vec()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_8bit_right_shift_by_1() {
        let shift_wirings = generate_shift_wirings(3, 1);

        assert_eq!(
            shift_wirings,
            vec![(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6)]
        );
    }

    #[test]
    fn test_256bit_right_shift_by_1() {
        let shift_wirings = generate_shift_wirings(8, 1);

        assert_eq!(
            shift_wirings,
            (0..255).into_iter().map(|i| (i + 1, i)).collect_vec()
        );
    }

    #[test]
    fn test_8bit_left_shift_by_1() {
        let shift_wirings = generate_shift_wirings(3, -1);

        assert_eq!(
            shift_wirings,
            vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        );
    }

    #[test]
    fn test_256bit_left_shift_by_1() {
        let shift_wirings = generate_shift_wirings(8, -1);

        assert_eq!(
            shift_wirings,
            (0..255).into_iter().map(|i| (i, i + 1)).collect_vec()
        );
    }

    #[test]
    fn test_zero_shift() {
        let shift_wirings = generate_shift_wirings(3, 0);
        // let zero_wirings = generate_zero_wirings(3, 0);

        assert_eq!(
            shift_wirings,
            vec![
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7)
            ]
        );
    }

    #[test]
    fn test_8bit_right_shift_by_bit_length() {
        let shift_wirings = generate_shift_wirings(3, 1 << 3);

        assert_eq!(shift_wirings, vec![]);
    }

    #[test]
    fn test_8bit_right_shift_by_more_than_bit_length() {
        let shift_wirings = generate_shift_wirings(3, 100);

        assert_eq!(shift_wirings, vec![]);
    }
}
