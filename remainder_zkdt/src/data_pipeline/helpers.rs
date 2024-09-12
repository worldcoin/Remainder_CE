//! Miscellaneous helper functions for `dt2zkdt`.
use crate::data_pipeline::trees::*;
use crate::structs::{DecisionNode, LeafNode};
use remainder_ligero::utils::get_least_significant_bits_to_usize_little_endian;
use remainder_shared_types::Field;

/// Helper function for conversion to field elements, handling negative values.
pub fn i32_to_field<F: Field>(value: i32) -> F {
    if value >= 0 {
        F::from(value as u64)
    } else {
        F::from(value.unsigned_abs() as u64).neg()
    }
}

/// Helper function for conversion to field elements, handling negative values.
pub fn i64_to_field<F: Field>(value: i64) -> F {
    if value >= 0 {
        F::from(value as u64)
    } else {
        F::from(value.unsigned_abs()).neg()
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

/// Return the `bit_length` bit signed decomposition of the specified i32, or None if the argument is too large.
/// Result is little endian (so LSB has index 0).
/// Sign bit has maximal index.
/// Pre: bit_length > 1.
pub fn build_signed_bit_decomposition(value: i32, bit_length: usize) -> Option<Vec<bool>> {
    let unsigned = build_unsigned_bit_decomposition(value.unsigned_abs(), bit_length - 1);
    if let Some(mut bits) = unsigned {
        bits.push(value < 0);
        return Some(bits);
    }
    None
}

/// Return the `bit_length` bit decomposition of the specified u32, or None if the argument is too large.
/// Result is little endian i.e. LSB has index 0.
pub fn build_unsigned_bit_decomposition(mut value: u32, bit_length: usize) -> Option<Vec<bool>> {
    let mut bits = vec![];
    for _ in 0..bit_length {
        bits.push((value & 1) != 0);
        value >>= 1;
    }
    if value == 0 {
        Some(bits)
    } else {
        None
    }
}

/// Return a Vec containing a [`DecisionNode`] for each Node::Internal appearing in this tree, in arbitrary order.
/// Pre: if `node` is any descendent of this Node then `node.get_id()` is not None.
pub fn extract_decision_nodes<T: Copy, F: Field>(tree: &Node<T>) -> Vec<DecisionNode<F>> {
    let mut decision_nodes = Vec::new();
    append_decision_nodes(tree, &mut decision_nodes);
    decision_nodes
}

/// Helper function to [`extract_decision_nodes`].
fn append_decision_nodes<T: Copy, F: Field>(
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

/// Return a Vec containing a [`LeafNode`] for each Node::Leaf appearing in this tree, in order of
/// id, where the ids are allocated as in [`extract_decision_nodes`].
pub fn extract_leaf_nodes<F: Field>(tree: &Node<i64>) -> Vec<LeafNode<F>> {
    let mut leaf_nodes = Vec::new();
    append_leaf_nodes(tree, &mut leaf_nodes);
    leaf_nodes
}

/// Helper function for extract_leaf_nodes.
fn append_leaf_nodes<F: Field>(tree: &Node<i64>, leaf_nodes: &mut Vec<LeafNode<F>>) {
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

pub fn get_field_val_as_usize_vec<F: Field>(value: F) -> Vec<usize> {
    let value_le_bytes = value.to_repr().as_ref().to_vec();
    let first_result =
        get_least_significant_bits_to_usize_little_endian(value_le_bytes.clone(), 64);
    let second_result =
        get_least_significant_bits_to_usize_little_endian(value_le_bytes[8..].to_vec(), 64);
    let third_result =
        get_least_significant_bits_to_usize_little_endian(value_le_bytes[16..].to_vec(), 64);
    let fourth_result =
        get_least_significant_bits_to_usize_little_endian(value_le_bytes[24..].to_vec(), 64);
    vec![first_result, second_result, third_result, fourth_result]
}

#[cfg(test)]
mod tests {
    use super::*;
    use remainder_shared_types::Fr;

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
    fn test_least_significant_bits_limbs() {
        let hi = Fr::from(1).neg();
        let fr_limbs = get_field_val_as_usize_vec(hi);
        dbg!(fr_limbs);
        // return vec![first_result, second_result, third_result, fourth_result];
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
        assert_eq!(
            decision_nodes[1].node_id,
            Fr::from(2 * (root_id as u64) + 1)
        );
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
    fn test_build_signed_bit_decomposition() {
        // test vanilla case
        let result = build_signed_bit_decomposition(-3, 16);
        if let Some(bit_decomp) = result {
            assert!(bit_decomp[0]);
            assert!(bit_decomp[1]);
            assert!(!bit_decomp[2]);
            assert!(!bit_decomp[3]);
            assert!(bit_decomp[15]);
        } else {
            assert!(false);
        }
        // test overflow handling
        assert_eq!(build_signed_bit_decomposition(2_i32.pow(30), 16), None);
        assert_eq!(build_signed_bit_decomposition(-2_i32.pow(30), 16), None);
    }

    #[test]
    fn test_build_unsigned_bit_decomposition() {
        // test vanilla case
        let result = build_unsigned_bit_decomposition(6, 16);
        if let Some(bit_decomp) = result {
            assert!(!bit_decomp[0]);
            assert!(bit_decomp[1]);
            assert!(bit_decomp[2]);
            assert!(!bit_decomp[3]);
            assert!(!bit_decomp[15]);
        } else {
            assert!(false);
        }
        // test overflow handling
        assert_eq!(build_unsigned_bit_decomposition(2_u32.pow(30), 16), None);
    }
}
