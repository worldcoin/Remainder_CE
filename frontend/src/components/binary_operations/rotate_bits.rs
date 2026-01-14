//! Implements bit shifting gates.

use shared_types::Field;

use crate::layouter::builder::{CircuitBuilder, NodeRef};

/// A component that performs bit (wire rotation) using
/// [IdentityGateNode] (i.e., rewire).
///
/// TODO: Generalize to a wire shuffle (permute) gate.
///
/// Both left and right rotations are supported by using
/// negative/positive values in the `rotate_amount` parameter. +ve value
/// means rotate left
///
/// This is a rotate instruction, meaning that bits which are shifted
/// beyond num_vars of the word, are appended to the other side in the
/// same order.
///
/// Requires that the input node has already been verified to contain
/// binary digits.
#[derive(Clone, Debug)]
pub struct RotateNode<F: Field> {
    output: NodeRef<F>,
}

impl<F: Field> RotateNode<F> {
    /// Create a new [ShiftNode] that performs a rotation by `rotate_amount` (to the right if
    /// `rotate_amount > 0` or to the left if `rotate_amount < 0`) on `input` node which contains
    /// `2^num_vars` binary digits.
    ///
    /// # Requires
    /// `input` is assumed to only contain binary digits (i.e. only values from the set
    /// `{F::ZERO, F::ONE}` for a field `F`).
    pub fn new(
        builder_ref: &mut CircuitBuilder<F>,
        num_vars: usize,
        rotate_amount: i32,
        input: &NodeRef<F>,
    ) -> Self {
        // Compute the bit reroutings that effectively shift the
        // input MLE by the appropriate amount.
        let rot_wirings = generate_rot_wirings(num_vars, rotate_amount);
        let output = builder_ref.add_identity_gate_node(input, rot_wirings, num_vars, None);

        Self { output }
    }

    /// Returns a reference to the node containing the shifted value.
    pub fn get_output(&self) -> NodeRef<F> {
        self.output.clone()
    }
}

fn generate_rot_wirings(arity: usize, rot_amount: i32) -> Vec<(u32, u32)> {
    // Ensure `rot_amount` can represent all possible rotations for a
    // given value of `arity`. Here `rot_amount` is of type `i32` and a
    // -ve value means left rotation and a +ve value means right
    // rotation. The amount of rotation

    assert!(arity <= 30);

    let arr_sz = 1 << arity;

    let mod_n = |x: i32| {
        let v = x % arr_sz;
        if v < 0 {
            v + arr_sz
        } else {
            v
        }
    };

    (0..arr_sz)
        .map(|i| (mod_n(i + rot_amount) as _, i as _))
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn test_8bit_right_rotate_by_one() {
        let shift_wirings = generate_rot_wirings(3, 1);

        assert_eq!(
            shift_wirings,
            vec![
                (1, 0),
                (2, 1),
                (3, 2),
                (4, 3),
                (5, 4),
                (6, 5),
                (7, 6),
                (0, 7)
            ]
        );
    }

    #[test]
    fn test_8bit_left_rotate_by_one() {
        let shift_wirings = generate_rot_wirings(3, -1);

        assert_eq!(
            shift_wirings,
            vec![
                (7, 0),
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7)
            ]
        );
    }

    #[test]
    fn test_256bit_right_rotate_by_rand() {
        let shift_wirings = generate_rot_wirings(8, 1);
        let expected = (1..256)
            .map(|i| (i, i - 1))
            .chain((255..256).map(|_| (0, 255)))
            .collect_vec();
        assert_eq!(shift_wirings, expected);
    }

    #[test]
    fn test_256bit_left_rotate_by_rand() {
        let rot_wirings = generate_rot_wirings(8, -1);
        let mut expected_wires = vec![(255, 0)];
        (1..256).for_each(|i| expected_wires.push((i - 1 as u32, i as u32)));
        assert_eq!(rot_wirings, expected_wires);
    }
}
