extern crate num;
extern crate serde;
extern crate serde_json;

use super::structs::*;
use crate::FieldExt;
use ndarray::Array2;
use ndarray_npy::{read_npy, ReadNpyError};
use num::pow;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct TreesInfo {
    trees: Vec<Node<f64>>,
    bias: f64,
    scale: f64,
} 

type FlatTree<F> = (Vec<DecisionNode<F>>, Vec<LeafNode<F>>);
/// Given a TreesInfo object representing a decision tree model operating on u32 samples, prepare
/// the model for circuitization:
/// 1. scale and bias are folded into the leaf values;
/// 2. leaf values are symmetrically quantized to i32;
/// 3. all trees are padded such that they are all perfect and of uniform depth (without modifying
///    the predictions of any tree);
/// The tree of the decision tree model is then "flattened" to a 2-tuple of type:
///   (Vec<DecisionNode>, Vec<LeafNode<i32>>).
/// Both these vectors are ordered by the id attributes of their members.
/// Ids for DecisionNodes are 0 .. 2^(depth - 1) - 2 (inclusive).
/// Ids for LeafNodes are 2^(depth - 1) - 1 .. 2^depth - 1 (inclusive).
/// The vector of all such pairs is then returned, along with
/// the (uniform) depth of the trees, and the scaling factor to approximately undo the quantization
/// (via division) after aggregating scores.
fn prepare_for_circuitization<F: FieldExt>(trees_info: &TreesInfo) -> (Vec<FlatTree<F>>, u32, f64) {
    let mut trees_f64 = trees_info.trees.clone();
    // fold scale into all trees
    for tree in &mut trees_f64 {
        tree.transform_values(&|value| trees_info.scale * value);
    }
    // fold bias into first tree
    trees_f64[0].transform_values(&|value| value + trees_info.bias);

    // quantize the leaf values
    let (qtrees, rescaling) = quantize_trees(&trees_f64);

    // pad the trees for perfection
    let max_depth = qtrees
        .iter()
        .map(|tree: &Node<i32>| tree.depth(std::cmp::max))
        .max()
        .unwrap();
    // we'll insert DecisionNodes with feature_index=0 and threshold=0 where needed
    let leaf_expander = |depth: u32, value: i32| Node::new_constant_tree(depth, 0, 0, value);
    let qtrees: Vec<Node<i32>> = qtrees
        .iter()
        .map(|tree: &Node<i32>| tree.perfect_to_depth(max_depth, &leaf_expander))
        .collect();

    let mut flattened_trees: Vec<(Vec<DecisionNode<F>>, Vec<LeafNode<F>>)> = vec![];
    for qtree in qtrees {
        flattened_trees.push((qtree.extract_decision_nodes::<F>(0), qtree.extract_leaf_nodes::<F>(0)));
    }

    (flattened_trees, max_depth, rescaling)
}

type Sample<F> = Vec<InputAttribute<F>>;
/// Read in the 2d array of u16s serialized in `npy` format from the filename specified, and return
/// its conversion to Vec<Vec<InputAttribute>> where the attr_id of the InputAttribute is given by
/// the column index of the value.
fn read_sample_array<F: FieldExt>(filename: &String) -> Result<Vec<Sample<F>>, ReadNpyError> {
    let input_arr: Array2<u16> = read_npy(filename)?;
    let mut samples: Vec<Vec<InputAttribute<F>>> = vec![];
    for row in input_arr.outer_iter() {
        let sample = row
            .iter()
            .enumerate()
            .map(|(index, value)| InputAttribute {
                attr_id: F::from(index as u32),
                attr_val: F::from(*value),
            })
            .collect();
        samples.push(sample);
    }
    Ok(samples)
}


/// Struct for representing a tree in recursive form
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Node<T: Copy> {
    Internal {
        feature_index: u32,
        threshold: u32,
        left: Box<Node<T>>,
        right: Box<Node<T>>,
    },
    Leaf {
        value: T,
    },
}

impl<T: Copy> Node<T> {
    fn new_leaf(value: T) -> Self {
        Node::Leaf { value }
    }

    fn new_internal(feature_index: u32, threshold: u32, left: Node<T>, right: Node<T>) -> Self {
        Node::Internal {
            feature_index,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Helper function: return a new perfect tree where all internal nodes have the feature index
    /// and threshold specified, and all leaf nodes have the value specified.
    fn new_constant_tree(depth: u32, feature_index: u32, threshold: u32, value: T) -> Node<T> {
        if depth > 1 {
            let left = Self::new_constant_tree(depth - 1, feature_index, threshold, value);
            let right = Self::new_constant_tree(depth - 1, feature_index, threshold, value);
            Self::new_internal(feature_index, threshold, left, right)
        } else {
            Self::new_leaf(value)
        }
    }

    /// Return the depth of the tree (a tree with one node has depth 1).
    /// Example:
    /// ```ignore
    /// tree.depth(std::cmp::max);
    /// ```
    fn depth(&self, aggregator: fn(u32, u32) -> u32) -> u32 {
        match self {
            Node::Internal { left, right, .. } => {
                1 + aggregator(left.depth(aggregator), right.depth(aggregator))
            }
            Node::Leaf { .. } => 1,
        }
    }

    /// Aggregate the leaf values using the binary function provided (e.g. `f64::max`).
    /// Example:
    /// ```ignore
    /// let max = tree.aggregate_values(f64::max);
    /// ```
    fn aggregate_values(&self, operation: fn(T, T) -> T) -> T {
        match self {
            Node::Leaf { value } => *value,
            Node::Internal { left, right, .. } => operation(
                left.aggregate_values(operation),
                right.aggregate_values(operation),
            ),
        }
    }

    /// Transform the leaf values of the tree using the function provided.
    /// Example:
    /// ```ignore
    /// tree.transform_values(&|x| -1.0 * x);
    /// ```
    fn transform_values<F>(&mut self, transform: &F)
    where
        F: Fn(T) -> T,
    {
        match self {
            Node::Leaf { value } => {
                *value = transform(*value);
            }
            Node::Internal { left, right, .. } => {
                left.transform_values(transform);
                right.transform_values(transform);
            }
        }
    }

    /// As per transform_values, but handling the case of a function whose return type is different
    /// from the input type, at the cost of reconstructing the tree.
    fn map<U, F>(&self, f: &F) -> Node<U>
    where
        F: Fn(T) -> U,
        U: Copy,
    {
        match self {
            Node::Leaf { value } => Node::new_leaf(f(*value)),
            Node::Internal {
                feature_index,
                threshold,
                left,
                right,
            } => Node::new_internal(*feature_index, *threshold, left.map(f), right.map(f)),
        }
    }

    /// Return if the tree is perfect, i.e. all children are of maximal depth.
    fn is_perfect(&self) -> bool {
        self.depth(std::cmp::max) == self.depth(std::cmp::min)
    }

    /// Return a new Node<T> instance which is perfect with the specified depth, given a function
    /// `leaf_expander(depth, value)` that returns a subtree of height `depth` to replace the
    /// premature Leaf with value `value`.
    /// Pre: depth >= self.depth()
    /// Post: self.is_perfect()
    fn perfect_to_depth<F>(&self, depth: u32, leaf_expander: &F) -> Node<T>
    where
        F: Fn(u32, T) -> Node<T>,
    {
        assert!(depth >= 1);
        match self {
            Node::Internal {
                left,
                right,
                feature_index,
                threshold,
            } => {
                let _left = left.perfect_to_depth(depth - 1, leaf_expander);
                let _right = right.perfect_to_depth(depth - 1, leaf_expander);
                Node::new_internal(*feature_index, *threshold, _left, _right)
            }
            Node::Leaf { value } => leaf_expander(depth, *value),
        }
    }
}

const LEAF_QUANTILE_BITWIDTH: u32 = 29; // FIXME why doesn't this work with 32?
/// Given a vector of trees (Node instances) with f64 leaf values, quantize the leaf values
/// symmetrically, returning the quantized trees and the rescaling factor.  The scale is chosen
/// such that all possible _aggregate_ scores will fit within the LEAF_QUANTILE_BITWIDTH.
/// Post: (quantized leaf values) / rescaling ~= (original leaf values)
fn quantize_trees(trees: &[Node<f64>]) -> (Vec<Node<i32>>, f64) {
    // determine the spread of the scores
    let max_score: f64 = trees
        .iter()
        .map(|tree| tree.aggregate_values(f64::max))
        .sum();
    let min_score: f64 = trees
        .iter()
        .map(|tree| tree.aggregate_values(f64::min))
        .sum();
    let spread = f64::max(max_score.abs(), min_score.abs());

    // quantize the leaf values
    let quant_max = ((1_u32) << (LEAF_QUANTILE_BITWIDTH - 1)) - 2;
    let rescaling = (quant_max as f64) / spread;
    let qtrees: Vec<Node<i32>> = trees
        .iter()
        .map(|tree| tree.map(&|value| (value * rescaling) as i32))
        .collect();
    (qtrees, rescaling)
}

impl<T: Copy> Node<T> {
    /// Return a Vec containing a DecisionNode for each Node::Internal appearing in this tree, in
    /// arbitrary order, with ids allocated according to:
    /// + Root node has id root_id;
    /// + Left child id is 2 * parent_id + 1;
    /// + Right child is 2 * parent_id + 2.
    pub fn extract_decision_nodes<F: FieldExt>(&self, root_id: u32) -> Vec<DecisionNode<F>> {
        let mut decision_nodes = Vec::new();
        self.append_decision_nodes(root_id, &mut decision_nodes);
        decision_nodes
    }

    /// Helper function to extract_decision_nodes.
    fn append_decision_nodes<F: FieldExt>(&self, root_id: u32, decision_nodes: &mut Vec<DecisionNode<F>>) {
        if let Node::Internal {
            left,
            right,
            feature_index,
            threshold,
        } = self
        {
            decision_nodes.push(DecisionNode {
                node_id: F::from(root_id),
                attr_id: F::from(*feature_index),
                threshold: F::from(*threshold),
            });
            left.append_decision_nodes(2 * root_id + 1, decision_nodes);
            right.append_decision_nodes(2 * root_id + 2, decision_nodes);
        }
    }
}

impl Node<i32> {
    /// Return a Vec containing a LeafNode for each Node::Leaf appearing in this tree, in order of
    /// id, where the ids are allocated as in extract_decision_nodes().
    fn extract_leaf_nodes<F: FieldExt>(&self, root_id: u32) -> Vec<LeafNode<F>> {
        let mut leaf_nodes = Vec::new();
        self.append_leaf_nodes(root_id, &mut leaf_nodes);
        leaf_nodes
    }

    fn append_leaf_nodes<F: FieldExt>(&self, root_id: u32, leaf_nodes: &mut Vec<LeafNode<F>>) {
        match self {
            Node::Leaf { value } => {
                leaf_nodes.push(LeafNode {
                    node_id: F::from(root_id),
                    node_val: if *value >= 0 { F::from(*value as u32) } else { F::from(value.abs() as u32).neg() },
                });
            }
            Node::Internal { left, right, .. } => {
                left.append_leaf_nodes(2 * root_id + 1, leaf_nodes);
                right.append_leaf_nodes(2 * root_id + 2, leaf_nodes);
            }
        }
    }
}

/// Return the first power of two that is greater than or equal to the argument, or None if this
/// would exceed the range of u32.
fn next_power_of_two(n: u32) -> Option<u32> {
    if n == 0 {
        return Some(1);
    }

    for pow in 1..32 {
        let value = 1_u32 << (pow - 1);
        if value >= n {
            return Some(value);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_test_curves::fp128::Fq;
    use std::fs::File;

    #[test]
    fn test_read_sample_array() {
        let filename = String::from("src/zkdt/test_qsamples.npy");
        let result = read_sample_array::<Fq>(&filename);
        match result {
            Ok(result) => {}
            Err(why) => {
                panic!("{}", why);
            }
        }
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), Some(1));
        assert_eq!(next_power_of_two(3), Some(4));
        assert_eq!(next_power_of_two(4), Some(4));
        assert_eq!(next_power_of_two(5), Some(8));
        assert_eq!(next_power_of_two(u32::MAX), None);
    }

    #[test]
    fn test_depth() {
        let tree = Node::new_leaf(0);
        assert_eq!(tree.depth(std::cmp::max), 1);
        let tree = build_small_tree();
        assert_eq!(tree.depth(std::cmp::max), 3);
    }

    #[test]
    fn test_deserialization() {
        let json = r#"
        {
          "feature_index": 2,
          "threshold": 58,
          "left": {
            "feature_index": 4,
            "threshold": 570,
            "left": {
              "value": -64.13449274301529
            },
            "right": {
              "value": 10.048001098632813
            }
          },
          "right": {
            "feature_index": 6,
            "threshold": 823,
            "left": {
              "value": 56.9426991389348
            },
            "right": {
              "value": 110.34857613699776
            }
          }
        }
        "#;
        let tree: Node<f64> = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_debug_printing() {
        let tree = Node::new_internal(0, 1, Node::new_leaf(0), Node::new_leaf(1));
        println!("{:?}", tree);
    }

    /// Returns a small tree for testing. Shape:
    ///      .
    ///     / \
    ///    .   .
    ///   / \
    ///  .   .
    fn build_small_tree() -> Node<f64> {
        let left = Node::new_leaf(0.1);
        let middle = Node::new_leaf(0.2);
        let right = Node::new_leaf(1.2);
        let internal = Node::new_internal(1, 2, left, middle);
        Node::new_internal(0, 1, internal, right)
    }

    #[test]
    fn test_aggregate_values() {
        let tree = build_small_tree();
        assert_eq!(tree.aggregate_values(f64::max), 1.2);
        assert_eq!(tree.aggregate_values(f64::min), 0.1);
    }

    #[test]
    fn test_transform_values() {
        let mut tree = build_small_tree();
        tree.transform_values(&|x| -1.0 * x);
        if let Node::Internal { right, .. } = tree {
            if let Node::Leaf { value } = *right {
                assert_eq!(value, -1.2);
                return;
            }
        }
        panic!();
    }

    #[test]
    fn test_map() {
        let tree_f64 = build_small_tree();
        let tree_i32 = tree_f64.map(&|x| x as i32);
        if let Node::Internal { right, .. } = tree_i32 {
            if let Node::Leaf { value } = *right {
                assert_eq!(value, 1);
                return;
            }
        }
        panic!();
    }

    #[test]
    fn test_new_constant_tree() {
        let _feature_index = 2;
        let _threshold = 6;
        let _value = 3;
        let tree = Node::new_constant_tree(3, _feature_index, _threshold, _value);
        assert_eq!(tree.depth(std::cmp::max), 3);
        if let Node::Internal {
            left,
            right,
            feature_index,
            threshold,
        } = tree
        {
            assert_eq!(feature_index, _feature_index);
            assert_eq!(threshold, _threshold);
            if let Node::Internal { left, .. } = *left {
                if let Node::Leaf { value } = *left {
                    assert_eq!(value, _value);
                    return;
                }
            }
        }
        panic!();
    }

    #[test]
    fn test_perfect_to_depth() {
        let tree = build_small_tree();
        let leaf_expander = |depth: u32, value: f64| Node::new_constant_tree(depth, 0, 0, value);
        let perfect_tree = tree.perfect_to_depth(3, &leaf_expander);
        assert_eq!(perfect_tree.depth(std::cmp::max), 3);
        assert!(perfect_tree.is_perfect());
        let perfect_tree = tree.perfect_to_depth(4, &leaf_expander);
        assert_eq!(perfect_tree.depth(std::cmp::max), 4);
        assert!(perfect_tree.is_perfect());
    }

    #[test]
    fn test_extract_decision_nodes() {
        // test trivial boundary case
        let leaf = Node::new_leaf(3);
        let decision_nodes = leaf.extract_decision_nodes::<Fq>(0);
        assert_eq!(decision_nodes.len(), 0);
        // test in non-trivial case
        let tree = build_small_tree();
        let root_id = 6;
        let mut decision_nodes = tree.extract_decision_nodes::<Fq>(root_id);
        assert_eq!(decision_nodes.len(), 2);
        decision_nodes.sort_by_key(|node| node.node_id);
        assert_eq!(decision_nodes[0].node_id, Fq::from(root_id));
        assert_eq!(decision_nodes[0].attr_id, Fq::from(0));
        assert_eq!(decision_nodes[1].node_id, Fq::from(2 * root_id + 1));
    }

    #[test]
    fn test_extract_leaf_nodes() {
        // trivial case
        let leaf = Node::new_leaf(-3);
        let root_id = 5;
        let leaf_nodes = leaf.extract_leaf_nodes::<Fq>(root_id);
        assert_eq!(leaf_nodes.len(), 1);
        assert_eq!(leaf_nodes[0].node_id, Fq::from(root_id));
        assert_eq!(leaf_nodes[0].node_val, -Fq::from(3));
        // non-trivial
        let tree = build_small_tree().map(&|x| x as i32);
        let mut leaf_nodes = tree.extract_leaf_nodes::<Fq>(0);
        assert_eq!(leaf_nodes.len(), 3);
        leaf_nodes.sort_by_key(|node| node.node_id);
        assert_eq!(leaf_nodes[0].node_id, Fq::from(2));
        assert_eq!(leaf_nodes[0].node_val, Fq::from(1));
        assert_eq!(leaf_nodes[2].node_id, Fq::from(4));
        assert_eq!(leaf_nodes[2].node_val, Fq::from(0));
    }

    #[test]
    fn test_quantize_trees() {
        let value0 = -123.1;
        let value1 = 145.1;
        let trees = vec![Node::new_leaf(value0), Node::new_leaf(value1)];
        let (qtrees, rescaling) = quantize_trees(&trees);
        assert_eq!(qtrees.len(), trees.len());
        if let Node::Leaf { value: qvalue0 } = qtrees[0] {
            assert!((value0 - (qvalue0 as f64) / rescaling).abs() < 1e-5);
            if let Node::Leaf { value: qvalue1 } = qtrees[1] {
                assert!((value1 - (qvalue1 as f64) / rescaling).abs() < 1e-5);
                return;
            }
        }
        panic!("The quantized trees should each have been leaf nodes");
    }

    const TEST_TREES_INFO: &str = "src/zkdt/test_qtrees.json";
    #[test]
    fn test_trees_info_loading() {
        let file = File::open(TEST_TREES_INFO)
            .expect(&format!("'{}' should be available.", TEST_TREES_INFO));
        let trees_info: TreesInfo = serde_json::from_reader(file).expect(&format!(
            "'{}' should be valid TreesInfo JSON.",
            TEST_TREES_INFO
        ));
    }

    #[test]
    fn test_prepare_for_circuitization() {
        let trees_info = TreesInfo {
            trees: vec![build_small_tree(), Node::new_leaf(3.0)],
            bias: 1.1,
            scale: 6.6,
        };
        let (flattened_trees, depth, rescaling) = prepare_for_circuitization::<Fq>(&trees_info);
        assert_eq!(flattened_trees.len(), 2);
        assert_eq!(depth, 3);
        let mut acc_score: Fq = Fq::from(0);
        for (decision_nodes, leaf_nodes) in &flattened_trees {
            assert_eq!(decision_nodes.len(), 3);
            assert_eq!(leaf_nodes.len(), 4);
            // check the ids of the decision nodes
            assert!(decision_nodes
                .iter()
                .map(|node| node.node_id)
                .zip(0..3)
                .map(|(a, b)| a == Fq::from(b))
                .all(|x| x));
            // check the ids of the leaf nodes
            assert!(leaf_nodes
                .iter()
                .map(|node| node.node_id)
                .zip(3..7)
                .map(|(a, b)| a == Fq::from(b))
                .all(|x| x));
            // accumulate score by taking the value of the first leaf node
            acc_score += leaf_nodes[0].node_val;
        }
        // check that the quantized scores accumulated as expected
        let expected_score = trees_info.scale * (0.1 + 3.0) + trees_info.bias;
        let quant_score = (expected_score * rescaling) as i32;
        let f_quant_score = if quant_score >= 0 { Fq::from(quant_score) } else { -Fq::from(quant_score.abs() as u32) };
        // just check that's it's close
        assert_eq!(f_quant_score, acc_score + Fq::from(1));
    }
}
