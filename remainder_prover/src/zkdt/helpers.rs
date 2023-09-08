//! Miscellaneous helper functions for `dt2zkdt`.
use crate::zkdt::structs::{BinDecomp16Bit, DecisionNode, LeafNode};
use crate::zkdt::data_pipeline::trees::*;
use remainder_shared_types::FieldExt;
use std::iter::repeat;

/// Helper function for conversion to field elements, handling negative values.
pub fn i32_to_field<F: FieldExt>(value: i32) -> F {
    if value >= 0 {
        F::from(value as u64)
    } else {
        F::from(value.abs() as u64).neg()
    }
}

/// Helper function for conversion to field elements, handling negative values.
pub fn i64_to_field<F: FieldExt>(value: i64) -> F {
    if value >= 0 {
        F::from(value as u64)
    } else {
        F::from(value.abs() as u64).neg()
    }
}

/// Return the first power of two that is greater than or equal to the argument, or None if this
/// would exceed the range of a u32.
pub fn next_power_of_two(n: usize) -> Option<usize> {
    if n == 0 {
        return Some(1);
    }

    for pow in 1..32 {
        let value = 1_usize << (pow - 1);
        if value >= n {
            return Some(value);
        }
    }
    None
}

const BIT_DECOMPOSITION_LENGTH: usize = 16;
pub const SIGNED_DECOMPOSITION_MAX_ARG_ABS: u32 =
    2_u32.pow(BIT_DECOMPOSITION_LENGTH as u32 - 1) - 1;
const UNSIGNED_DECOMPOSITION_MAX_ARG: u32 = 2_u32.pow(BIT_DECOMPOSITION_LENGTH as u32) - 1;

/// Build a 16 bit signed decomposition of the specified i32, or None if the argument is too large
/// in absolute value (exceeding SIGNED_DECOMPOSITION_MAX_ARG_ABS).
/// Result is little endian (so LSB has index 0).
/// Sign bit has maximal index.
pub fn build_signed_bit_decomposition<F: FieldExt>(value: i32) -> Option<BinDecomp16Bit<F>> {
    let abs_val = value.abs() as u32;
    if abs_val > SIGNED_DECOMPOSITION_MAX_ARG_ABS {
        return None;
    }
    let mut decomposition = build_unsigned_bit_decomposition(abs_val).unwrap();
    // set the sign bit
    decomposition.bits[BIT_DECOMPOSITION_LENGTH - 1] =
        if value >= 0 { F::zero() } else { F::one() };
    Some(decomposition)
}

/// Build a 16 bit decomposition of the specified u32, or None if the argument is too large
/// (exceeding UNSIGNED_DECOMPOSITION_MAX_ARG_ABS).
/// Result is little endian i.e. LSB has index 0.
pub fn build_unsigned_bit_decomposition<F: FieldExt>(mut value: u32) -> Option<BinDecomp16Bit<F>> {
    if value > UNSIGNED_DECOMPOSITION_MAX_ARG {
        return None;
    }
    let mut bits = [F::zero(); BIT_DECOMPOSITION_LENGTH];
    for i in 0..BIT_DECOMPOSITION_LENGTH {
        if value & 1 == 1 {
            bits[i] = F::one();
        }
        value >>= 1;
    }
    Some(BinDecomp16Bit { bits: bits })
}

/// Repeat the items of the provided slice `repetitions` times, before padding with the minimal
/// number of zeros such that the length is a power of two.
pub fn repeat_and_pad<T: Clone>(values: &[T], repetitions: usize, padding: T) -> Vec<T> {
    // repeat
    let repeated_length = repetitions * values.len();
    let repeated_iter = values.into_iter().cycle().take(repeated_length);
    // pad to nearest power of two
    let padding_length = next_power_of_two(repeated_length).unwrap() - repeated_length;
    let padding_iter = repeat(&padding).take(padding_length);
    // chain together and convert to a vector
    repeated_iter.chain(padding_iter).cloned().collect()
}

/// Return a Vec containing a DecisionNode for each Node::Internal appearing in this tree, in arbitrary order.
/// Pre: if `node` is any descendent of this Node then `node.get_id()` is not None.
pub fn extract_decision_nodes<T: Copy, F: FieldExt>(tree: &Node<T>) -> Vec<DecisionNode<F>> {
    let mut decision_nodes = Vec::new();
    append_decision_nodes(tree, &mut decision_nodes);
    decision_nodes
}

/// Helper function to extract_decision_nodes.
fn append_decision_nodes<T: Copy, F: FieldExt>(
    tree: &Node<T>,
    decision_nodes: &mut Vec<DecisionNode<F>>,
) {
    if let Node::Internal {
        id,
        left,
        right,
        feature_index,
        threshold,
    } = tree
    {
        decision_nodes.push(DecisionNode {
            node_id: F::from(id.unwrap() as u64),
            attr_id: F::from(*feature_index as u64),
            threshold: F::from(*threshold as u64),
        });
        append_decision_nodes(left, decision_nodes);
        append_decision_nodes(right, decision_nodes);
    }
}

/// Return a Vec containing a LeafNode for each Node::Leaf appearing in this tree, in order of
/// id, where the ids are allocated as in extract_decision_nodes().
pub fn extract_leaf_nodes<F: FieldExt>(tree: &Node<i64>) -> Vec<LeafNode<F>> {
    let mut leaf_nodes = Vec::new();
    append_leaf_nodes(tree, &mut leaf_nodes);
    leaf_nodes
}

/// Helper function for extract_leaf_nodes.
fn append_leaf_nodes<F: FieldExt>(tree: &Node<i64>, leaf_nodes: &mut Vec<LeafNode<F>>) {
    match tree {
        Node::Leaf { id, value } => {
            leaf_nodes.push(LeafNode {
                node_id: F::from(id.unwrap() as u64),
                node_val: i64_to_field(*value),
            });
        }
        Node::Internal { left, right, .. } => {
            append_leaf_nodes(left, leaf_nodes);
            append_leaf_nodes(right, leaf_nodes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;

    /// Returns a small tree for testing:
    ///      .
    ///     / \
    ///    .  1.2
    ///   / \
    /// 0.1 0.2
    fn build_small_tree() -> Node<f64> {
        let left = Node::new_leaf(None, 0.1);
        let middle = Node::new_leaf(None, 0.2);
        let right = Node::new_leaf(None, 1.2);
        let internal = Node::new_internal(None, 0, 2, left, middle);
        Node::new_internal(None, 1, 1, internal, right)
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), Some(1));
        assert_eq!(next_power_of_two(3), Some(4));
        assert_eq!(next_power_of_two(4), Some(4));
        assert_eq!(next_power_of_two(5), Some(8));
        assert_eq!(next_power_of_two(usize::MAX), None); // NOTE this will only fail on 64bit
                                                         // platforms
    }

    #[test]
    fn test_extract_decision_nodes() {
        // test trivial boundary case
        let leaf = Node::new_leaf(Some(1), 3);
        let decision_nodes = extract_decision_nodes::<u32, Fr>(&leaf);
        assert_eq!(decision_nodes.len(), 0);
        // test in non-trivial case
        let mut tree = build_small_tree();
        let root_id = 6;
        tree.assign_id(root_id);
        let mut decision_nodes = extract_decision_nodes::<f64, Fr>(&tree);
        assert_eq!(decision_nodes.len(), 2);
        decision_nodes.sort_by_key(|node| node.node_id);
        assert_eq!(decision_nodes[0].node_id, Fr::from(root_id as u64));
        assert_eq!(decision_nodes[0].attr_id, Fr::from(1));
        assert_eq!(decision_nodes[1].node_id, Fr::from(2 * (root_id as u64) + 1));
    }

    #[test]
    fn test_extract_leaf_nodes() {
        // trivial case
        let leaf = Node::new_leaf(Some(5), -3);
        let leaf_nodes = extract_leaf_nodes::<Fr>(&leaf);
        assert_eq!(leaf_nodes.len(), 1);
        assert_eq!(leaf_nodes[0].node_id, Fr::from(5));
        assert_eq!(leaf_nodes[0].node_val, -Fr::from(3));
        // non-trivial
        let mut tree = build_small_tree().map(&|x| x as i64);
        tree.assign_id(0);
        let mut leaf_nodes = extract_leaf_nodes::<Fr>(&tree);
        assert_eq!(leaf_nodes.len(), 3);
        leaf_nodes.sort_by_key(|node| node.node_id);
        assert_eq!(leaf_nodes[0].node_id, Fr::from(2));
        assert_eq!(leaf_nodes[0].node_val, Fr::from(1));
        assert_eq!(leaf_nodes[2].node_id, Fr::from(4));
        assert_eq!(leaf_nodes[2].node_val, Fr::from(0));
    }

    #[test]
    fn test_repeat_and_pad() {
        let result = repeat_and_pad(&vec![3_u16, 1_u16], 3, 0_u16);
        assert_eq!(result.len(), 8);
        assert_eq!(result[0], 3_u16);
        assert_eq!(result[1], 1_u16);
        assert_eq!(result[2], 3_u16);
        assert_eq!(result[4], 3_u16);
        assert_eq!(result[6], 0_u16);
        assert_eq!(result[7], 0_u16);
    }

    #[test]
    fn test_repeat_and_pad_boundary() {
        let result = repeat_and_pad(&vec![3_u32, 1_u32], 2, 0_u32);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_build_signed_bit_decomposition() {
        // test vanilla case
        let result = build_signed_bit_decomposition::<Fr>(-3);
        if let Some(bit_decomp) = result {
            assert_eq!(bit_decomp.bits[0], Fr::from(1));
            assert_eq!(bit_decomp.bits[1], Fr::from(1));
            assert_eq!(bit_decomp.bits[2], Fr::from(0));
            assert_eq!(bit_decomp.bits[3], Fr::from(0));
            assert_eq!(bit_decomp.bits[15], Fr::from(1));
        } else {
            assert!(false);
        }
        // test overflow handling
        assert_eq!(build_signed_bit_decomposition::<Fr>(2_i32.pow(30)), None);
        assert_eq!(
            build_signed_bit_decomposition::<Fr>(-1 * 2_i32.pow(30)),
            None
        );
    }

    #[test]
    fn test_build_unsigned_bit_decomposition() {
        // test vanilla case
        let result = build_unsigned_bit_decomposition::<Fr>(6);
        if let Some(bit_decomp) = result {
            assert_eq!(bit_decomp.bits[0], Fr::from(0));
            assert_eq!(bit_decomp.bits[1], Fr::from(1));
            assert_eq!(bit_decomp.bits[2], Fr::from(1));
            assert_eq!(bit_decomp.bits[3], Fr::from(0));
            assert_eq!(bit_decomp.bits[15], Fr::from(0));
        } else {
            assert!(false);
        }
        // test overflow handling
        assert_eq!(build_unsigned_bit_decomposition::<Fr>(2_u32.pow(30)), None);
    }
}
