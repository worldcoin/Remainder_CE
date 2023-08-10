extern crate num;
extern crate serde;
extern crate serde_json;

use crate::zkdt::structs::{DecisionNode,LeafNode,InputAttribute,BinDecomp16Bit};
use crate::FieldExt;
use ndarray::Array2;
use ndarray_npy::{read_npy, ReadNpyError};
use num::pow;
use serde::{Deserialize, Serialize};
use itertools::{repeat_n, Itertools};

struct CircuitizedPath<F: FieldExt> {
    sample: Sample<F>,
    permuted_sample: Sample<F>,
    decision_path: Vec<DecisionNode<F>>,
    path_end: LeafNode<F>,
    differences: Vec<BinDecomp16Bit<F>>,
    //? multiplicities: Vec<BinDecomp16Bit<F>>,
}
// FIXME define a struct for the return value of this method?
// But what about the multiplicities in each tree?  how does this interact with batching?
fn circuitize_samples<F: FieldExt>(sample: &Vec<u32>, pqtrees: &PaddedQuantizedTrees) -> Vec<InputAttribute<F>> {
    // repeat the attributes of the sample
    let sample = sample.iter()
        .cycle()
        .take((pqtrees.depth - 1) as usize * sample.len())
        .cloned()
        .collect();
    // get the paths
    let paths = pqtrees.trees.iter().map(|x| x.get_path(&sample)).collect_vec();

    // pad sample to the next power of two
    vec![]
}


#[derive(Debug, Serialize, Deserialize)]
struct TreesInfo {
    trees: Vec<Node<f64>>,
    bias: f64,
    scale: f64,
    n_features: usize,
}

/// Ids are allocated according to:
/// + Root node has id 0;
/// + Left child id is 2 * parent_id + 1;
/// + Right child is 2 * parent_id + 2.
/// Decision nodes and leaf nodes are both ordered by id.
/// CircuitizedTrees are always perfect.
/// Decision node ids are 0..K and leaf node ids are K..L (excluding endpoints) where:
/// K = 2^(depth - 1) - 1 and
/// L = 2^depth - 1
struct CircuitizedTree<F: FieldExt> {
    decision_nodes: Vec<DecisionNode<F>>,
    leaf_nodes: Vec<LeafNode<F>>,
}

impl<F: FieldExt> From<&Node<i32>> for CircuitizedTree<F> {
    /// Pre: tree.is_perfect()
    fn from(tree: &Node<i32>) -> Self {
        CircuitizedTree {
            decision_nodes: tree.extract_decision_nodes::<F>(),
            leaf_nodes: tree.extract_leaf_nodes::<F>(),
        }
    }
}

struct CircuitizedTrees<F: FieldExt> {
    trees: Vec<CircuitizedTree<F>>,
    depth: u32,
    scaling: f64,
}

struct PaddedQuantizedTrees {
    trees: Vec<Node<i32>>,
    depth: u32,
    scaling: f64,
}


impl From<&TreesInfo> for PaddedQuantizedTrees {
    /// Given a TreesInfo object representing a decision tree model operating on u32 samples, prepare the model for circuitization:
    /// 1. scale and bias are folded into the leaf values;
    /// 2. leaf values are symmetrically quantized to i32;
    /// 3. all trees are padded such that they are all perfect and of uniform depth (without modifying
    ///    the predictions of any tree) where the uniform depth is chosen to be 2^l + 1 for minimal l >= 0;
    /// 4. ids are assigned to all nodes (as per assign_id());
    /// 5. the feature indexes are transformed according to
    ///      idx -> (depth_of_node - 1) * trees_info.n_features + idx
    ///    thereby ensuring that each feature index occurs only once on each descent path.
    /// The resulting PaddedQuantizedTrees incorporates all the CircuitizedTree instances, the (uniform) depth
    /// of the trees, and the scaling factor to approximately undo the quantization (via division)
    /// after aggregating the scores.
    fn from(trees_info: &TreesInfo) -> Self {
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
        let target_depth = next_power_of_two(max_depth - 1).unwrap() + 1;
        // we'll insert DecisionNodes with feature_index=0 and threshold=0 where needed
        let leaf_expander = |depth: u32, value: i32| Node::new_constant_tree(depth, 0, 0, value);
        let mut qtrees: Vec<Node<i32>> = qtrees
            .iter()
            .map(|tree: &Node<i32>| tree.perfect_to_depth(target_depth, &leaf_expander))
            .collect();
        // assign ids to all nodes
        for tree in &mut qtrees {
            tree.assign_id(0);
        }
        // transform feature indices such that they never repeat along a path
        for tree in &mut qtrees {
            tree.offset_feature_indices(trees_info.n_features as u32);
        }

        PaddedQuantizedTrees {
            trees: qtrees,
            depth: target_depth,
            scaling: rescaling,
        }
    }
}

// FIXME this is now redundant
struct CircuitizedPaths<'a, F: FieldExt> {
    trees: &'a CircuitizedTrees<F>,
    samples: Vec<Sample<F>>,
    permuted_samples: Vec<Sample<F>>,
    decision_paths: Vec<Vec<DecisionNode<F>>>,
    path_ends: Vec<LeafNode<F>>,
    // differences: Vec<Vec<BinDecomp16Bit<F>>>,
    // multiplicities: Vec<BinDecomp16Bit<F>>,
}

impl<'a, F: FieldExt> From<(&'a CircuitizedTrees<F>, Vec<Sample<F>>)> for CircuitizedPaths<'a, F> {
    // NOTE can't live longer than the CircuitizedTrees given as input
    // NOTE takes ownership of the vector of samples
    fn from(inputs: (&'a CircuitizedTrees<F>, Vec<Sample<F>>)) -> Self {
        let (trees, samples) = inputs;
        
        // Repeat the attributes depth - 1 times
        let repeated_samples = samples.iter()
            .map(|row| row.iter()
                 .cycle()
                 .take(row.len() * (trees.depth as usize - 1))
                 .cloned().collect())
            .collect();
        CircuitizedPaths {
            trees: trees,
            samples: repeated_samples,
            permuted_samples: vec![],
            decision_paths: vec![vec![]],
            path_ends: vec![],
        }
    }
}

// TODO consider doing the conversion to samples in the from function
// do we want to use InputAttribute as late as possible?  Redundant encoding of the attr_id
type Sample<F> = Vec<InputAttribute<F>>;
/// Read in the 2d array of u16s serialized in `npy` format from the filename specified, and return
/// its conversion to Vec<Sample> where the attr_id of the InputAttribute is given by
/// the column index of the value.
/// Samples have a uniform number of InputAttributes, equal to the next_power_of_two of the number
/// of columns in the numpy array.
fn read_sample_array<F: FieldExt>(filename: &String) -> Result<Vec<Sample<F>>, ReadNpyError> {
    let input_arr: Array2<u16> = read_npy(filename)?;
    let target_length = next_power_of_two(input_arr.shape()[1] as u32).unwrap();
    let mut samples: Vec<Vec<InputAttribute<F>>> = vec![];
    for row in input_arr.outer_iter() {
        let mut sample: Sample<F> = row
            .iter()
            .enumerate()
            .map(|(index, value)| InputAttribute {
                attr_id: F::from(index as u32),
                attr_val: F::from(*value),
            })
            .collect();
        
        // pad the Sample with as many dummy attributes as required
        let mut next_index = sample.len() as u32;
        while next_index < target_length {
            sample.push(InputAttribute {
                attr_id: F::from(next_index),
                attr_val: F::from(0_u32),
            });
            next_index += 1;
        }
        samples.push(sample);
    }
    Ok(samples)
}

/// Struct for representing a tree in recursive form
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Node<T: Copy> {
    Internal {
        id: Option<u32>,
        feature_index: u32,  // FIXME should be usize?
        threshold: u32,
        left: Box<Node<T>>,
        right: Box<Node<T>>,
    },
    Leaf {
        id: Option<u32>,
        value: T,
    },
}

impl<T: Copy> Node<T> {
    fn new_leaf(id: Option<u32>, value: T) -> Self {
        Node::Leaf { id, value }
    }

    fn new_internal(id: Option<u32>, feature_index: u32, threshold: u32, left: Node<T>, right: Node<T>) -> Self {
        Node::Internal {
            id,
            feature_index,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Helper function: return a new perfect tree where all internal nodes have the feature index
    /// and threshold specified, and all leaf nodes have the value specified.
    /// Ids are not assigned.
    fn new_constant_tree(depth: u32, feature_index: u32, threshold: u32, value: T) -> Node<T> {
        if depth > 1 {
            let left = Self::new_constant_tree(depth - 1, feature_index, threshold, value);
            let right = Self::new_constant_tree(depth - 1, feature_index, threshold, value);
            Self::new_internal(None, feature_index, threshold, left, right)
        } else {
            Self::new_leaf(None, value)
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
            Node::Leaf { value, .. } => *value,
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
            Node::Leaf { value, .. } => {
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
            Node::Leaf { id, value } => Node::new_leaf(*id, f(*value)),
            Node::Internal {
                id, 
                feature_index,
                threshold,
                left,
                right,
            } => Node::new_internal(*id, *feature_index, *threshold, left.map(f), right.map(f)),
        }
    }

    /// Return if the tree is perfect, i.e. all children are of maximal depth.
    fn is_perfect(&self) -> bool {
        self.depth(std::cmp::max) == self.depth(std::cmp::min)
    }

    /// Return a new Node<T> instance which is perfect with the specified depth, given a function
    /// `leaf_expander(depth, value)` that returns a subtree of depth `depth` to replace the
    /// premature Leaf with value `value`.
    /// New Nodes will not have ids assigned.
    /// Pre: depth >= self.depth()
    /// Post: self.is_perfect()
    fn perfect_to_depth<F>(&self, depth: u32, leaf_expander: &F) -> Node<T>
    where
        F: Fn(u32, T) -> Node<T>,
    {
        assert!(depth >= 1);
        match self {
            Node::Internal {
                id, 
                left,
                right,
                feature_index,
                threshold,
            } => {
                let _left = left.perfect_to_depth(depth - 1, leaf_expander);
                let _right = right.perfect_to_depth(depth - 1, leaf_expander);
                Node::new_internal(None, *feature_index, *threshold, _left, _right)
            }
            Node::Leaf { value, .. } => leaf_expander(depth, *value),
        }
    }

    /// Assign the specified id to this Node, and assign ids to any child nodes according to the
    /// rule:
    /// left_child_id = 2 * id + 1
    /// right_child_id = 2 * id + 2
    fn assign_id(&mut self, new_id: u32) {
        match self {
            Node::Internal { id, left, right, .. } => {
                *id = Some(new_id);
                left.assign_id(2 * new_id + 1);
                right.assign_id(2 * new_id + 2);
            }
            Node::Leaf { id, .. } => {
                *id = Some(new_id);
            }
        }
    }

    fn get_id(&self) -> Option<u32> {
        match self {
            Node::Internal { id, .. } => *id,
            Node::Leaf { id, .. } => *id,
        }
    }

    /// Add (depth_of_node - 1) * multiplier to the feature index of all internal nodes in this
    /// tree.
    fn offset_feature_indices(&mut self, multiplier: u32) {
        self.offset_feature_indices_for_depth(multiplier, 1);
    }

    /// Helper to offset_feature_indices.
    fn offset_feature_indices_for_depth(&mut self, multiplier: u32, depth: u32) {
        if let Node::Internal { feature_index, left, right, .. } = self {
            *feature_index = (depth - 1) * multiplier + *feature_index;
            left.offset_feature_indices_for_depth(multiplier, depth + 1);
            right.offset_feature_indices_for_depth(multiplier, depth + 1);
        }
    }

    /// Return the path traced by the specified sample down this tree.
    /// Pre: sample.len() > node.feature_index for this node and all descendents.
    fn get_path<'a>(&'a self, sample: &Vec<u32>) -> Vec<&'a Node<T>> {
        let mut path = Vec::new();
        self.append_path(sample, &mut path);
        path
    }

    /// Helper function to get_path.
    /// Appends self to path, then, if internal, calls on the appropriate child node.
    fn append_path<'a>(&'a self, sample: &Vec<u32>, path_to_here: &mut Vec<&'a Node<T>>) {
        path_to_here.push(self);
        if let Node::Internal {
            left,
            right,
            feature_index,
            threshold,
            ..
        } = self {
            let next = if sample[*feature_index as usize] >= *threshold {
                right
            } else {
                left
            };
            next.append_path(sample, path_to_here);
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
    /// Return a Vec containing a DecisionNode for each Node::Internal appearing in this tree, in arbitrary order.
    /// Pre: if `node` is any descendent of this Node then `node.get_id()` is not None.
    pub(crate) fn extract_decision_nodes<F: FieldExt>(&self) -> Vec<DecisionNode<F>> {
        let mut decision_nodes = Vec::new();
        self.append_decision_nodes(&mut decision_nodes);
        decision_nodes
    }

    /// Helper function to extract_decision_nodes.
    fn append_decision_nodes<F: FieldExt>(
        &self,
        decision_nodes: &mut Vec<DecisionNode<F>>,
    ) {
        if let Node::Internal {
            id,
            left,
            right,
            feature_index,
            threshold,
        } = self
        {
            decision_nodes.push(DecisionNode {
                node_id: F::from(id.unwrap()),
                attr_id: F::from(*feature_index),
                threshold: F::from(*threshold),
            });
            left.append_decision_nodes(decision_nodes);
            right.append_decision_nodes(decision_nodes);
        }
    }
}

impl Node<i32> {
    /// Return a Vec containing a LeafNode for each Node::Leaf appearing in this tree, in order of
    /// id, where the ids are allocated as in extract_decision_nodes().
    fn extract_leaf_nodes<F: FieldExt>(&self) -> Vec<LeafNode<F>> {
        let mut leaf_nodes = Vec::new();
        self.append_leaf_nodes(&mut leaf_nodes);
        leaf_nodes
    }

    fn append_leaf_nodes<F: FieldExt>(&self, leaf_nodes: &mut Vec<LeafNode<F>>) {
        match self {
            Node::Leaf { id, value } => {
                leaf_nodes.push(LeafNode {
                    node_id: F::from(id.unwrap()),
                    node_val: if *value >= 0 {
                        F::from(*value as u32)
                    } else {
                        F::from(value.abs() as u32).neg()
                    },
                });
            }
            Node::Internal { left, right, .. } => {
                left.append_leaf_nodes(leaf_nodes);
                right.append_leaf_nodes(leaf_nodes);
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
    use ark_bn254::Fr;
    use std::fs::File;

    #[test]
    fn test_read_sample_array() {
        let filename = String::from("src/zkdt/test_samples_10x6.npy");
        let samples = read_sample_array::<Fr>(&filename);
        match samples {
            Ok(samples) => {
                assert_eq!(samples.len(), 10);
                // check that number of InputAttributes is the next power of two
                assert_eq!(samples[0].len(), 8);
            }
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
        let tree = Node::new_leaf(None, 0);
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
        let tree = Node::new_internal(None, 0, 1, Node::new_leaf(None, 0), Node::new_leaf(None, 1));
        println!("{:?}", tree);
    }

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
        let internal = Node::new_internal(None, 1, 2, left, middle);
        Node::new_internal(None, 0, 1, internal, right)
    }

    /// Returns a skinny looking tree for testing.
    /// Pre: target_depth >= 1.
    ///        .
    ///       / \
    ///      .   -2.
    ///     / \
    ///    .   -2.
    ///   / \
    ///  1.  -2.
    fn build_skinny_tree(target_depth: u32) -> Node<f64> {
        let mut tree = Node::new_leaf(None, 1.0);
        let mut depth = 1;
        while depth < target_depth {
            let premature_leaf = Node::new_leaf(None, -2.0);
            tree = Node::new_internal(None, 1, 2, tree, premature_leaf);
            depth += 1;
        }
        tree
    }

    /// Helper function for testing.
    fn quantize_and_perfect(tree_f64: Node<f64>, depth: u32) -> Node<i32> {
        let leaf_expander = |depth: u32, value: i32| Node::new_constant_tree(depth, 0, 0, value);
        tree_f64.map(&|x| x as i32)
            .perfect_to_depth(3, &leaf_expander)
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
            if let Node::Leaf { value, .. } = *right {
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
            if let Node::Leaf { value, .. } = *right {
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
            id,
            left,
            right,
            feature_index,
            threshold,
        } = tree
        {
            assert_eq!(id, None);
            assert_eq!(feature_index, _feature_index);
            assert_eq!(threshold, _threshold);
            if let Node::Internal { left, .. } = *left {
                if let Node::Leaf { value, .. } = *left {
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
    fn test_assign_id() {
        let mut tree = build_small_tree();
        tree.assign_id(0);
        assert_eq!(tree.get_id(), Some(0));
        if let Node::Internal { left, .. } = tree {
            assert_eq!(left.get_id(), Some(1));
            if let Node::Internal { left, right, .. } = *left {
                assert_eq!(left.get_id(), Some(3));
                assert_eq!(right.get_id(), Some(4));
            }
        }
    }

    #[test]
    fn test_extract_decision_nodes() {
        // test trivial boundary case
        let leaf = Node::new_leaf(Some(1), 3);
        let decision_nodes = leaf.extract_decision_nodes::<Fr>();
        assert_eq!(decision_nodes.len(), 0);
        // test in non-trivial case
        let mut tree = build_small_tree();
        let root_id = 6;
        tree.assign_id(root_id);
        let mut decision_nodes = tree.extract_decision_nodes::<Fr>();
        assert_eq!(decision_nodes.len(), 2);
        decision_nodes.sort_by_key(|node| node.node_id);
        assert_eq!(decision_nodes[0].node_id, Fr::from(root_id));
        assert_eq!(decision_nodes[0].attr_id, Fr::from(0));
        assert_eq!(decision_nodes[1].node_id, Fr::from(2 * root_id + 1));
    }

    #[test]
    fn test_extract_leaf_nodes() {
        // trivial case
        let leaf = Node::new_leaf(Some(5), -3);
        let leaf_nodes = leaf.extract_leaf_nodes::<Fr>();
        assert_eq!(leaf_nodes.len(), 1);
        assert_eq!(leaf_nodes[0].node_id, Fr::from(5));
        assert_eq!(leaf_nodes[0].node_val, -Fr::from(3));
        // non-trivial
        let mut tree = build_small_tree().map(&|x| x as i32);
        tree.assign_id(0);
        let mut leaf_nodes = tree.extract_leaf_nodes::<Fr>();
        assert_eq!(leaf_nodes.len(), 3);
        leaf_nodes.sort_by_key(|node| node.node_id);
        assert_eq!(leaf_nodes[0].node_id, Fr::from(2));
        assert_eq!(leaf_nodes[0].node_val, Fr::from(1));
        assert_eq!(leaf_nodes[2].node_id, Fr::from(4));
        assert_eq!(leaf_nodes[2].node_val, Fr::from(0));
    }

    #[test]
    fn test_quantize_trees() {
        let value0 = -123.1;
        let value1 = 145.1;
        let trees = vec![Node::new_leaf(None, value0), Node::new_leaf(None, value1)];
        let (qtrees, rescaling) = quantize_trees(&trees);
        assert_eq!(qtrees.len(), trees.len());
        if let Node::Leaf { value: qvalue0, .. } = qtrees[0] {
            assert!((value0 - (qvalue0 as f64) / rescaling).abs() < 1e-5);
            if let Node::Leaf { value: qvalue1, .. } = qtrees[1] {
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
    fn test_offset_feature_indices() {
        let mut tree = build_small_tree();
        tree.offset_feature_indices(10);
        if let Node::Internal { left, feature_index, .. } = tree {
            assert_eq!(feature_index, 0);
            if let Node::Internal { feature_index, .. } = *left {
                assert_eq!(feature_index, 11);
                return;
            }
        }
        panic!("Should be inaccessible");
    }

    //#[test]
    //fn test_circuitizedtrees_from() {
    //    let mut tree = build_small_tree();
    //    tree.assign_id(0);
    //    let trees_info = TreesInfo {
    //        trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
    //        bias: 1.1,
    //        scale: 6.6,
    //    };
    //    let circ_trees: CircuitizedTrees<Fr> = (&trees_info).into();
    //    assert_eq!(circ_trees.trees.len(), 2);
    //    assert_eq!(circ_trees.depth, 3);
    //    let mut acc_score: Fr = Fr::from(0);
    //    for flat_tree in &circ_trees.trees {
    //        assert_eq!(flat_tree.decision_nodes.len(), 3);
    //        assert_eq!(flat_tree.leaf_nodes.len(), 4);
    //        // accumulate score by taking the value of the first leaf node
    //        acc_score += flat_tree.leaf_nodes[0].node_val;
    //    }
    //    // check that the quantized scores accumulated as expected
    //    let expected_score = trees_info.scale * (0.1 + 3.0) + trees_info.bias;
    //    let quant_score = (expected_score * circ_trees.scaling) as i32;
    //    let f_quant_score = if quant_score >= 0 {
    //        Fr::from(quant_score)
    //    } else {
    //        -Fr::from(quant_score.abs() as u32)
    //    };
    //    // just check that's it's close
    //    assert_eq!(f_quant_score, acc_score + Fr::from(1));
    //}

    //#[test]
    //fn test_prepare_for_circuitization_depth() {
    //    // create a tree of depth 6
    //    let mut tree = build_skinny_tree(6);
    //    tree.assign_id(0);
    //    let trees_info = TreesInfo {
    //        trees: vec![tree],
    //        bias: 1.1,
    //        scale: 6.6,
    //    };
    //    let circ_trees: CircuitizedTrees<Fr> = (&trees_info).into();
    //    // check that the depth is now 2^l + 1 for minimal l, i.e. equal to 9.
    //    assert_eq!(circ_trees.depth, 9);
    //}

    #[test]
    fn test_flattree_from() {
        let mut tree = quantize_and_perfect(build_small_tree(), 3);
        tree.assign_id(0);
        let flat_tree: CircuitizedTree<Fr> = (&tree).into();
        assert_eq!(flat_tree.decision_nodes.len(), 3);
        assert_eq!(flat_tree.leaf_nodes.len(), 4);
        assert!(flat_tree.decision_nodes
                .iter()
                .map(|node| node.node_id)
                .zip(0..3)
                .map(|(a, b)| a == Fr::from(b))
                .all(|x| x));
        assert!(flat_tree.leaf_nodes
                .iter()
                .map(|node| node.node_id)
                .zip(3..7)
                .map(|(a, b)| a == Fr::from(b))
                .all(|x| x));
    }

    #[test]
    fn test_get_path() {
        let mut tree = quantize_and_perfect(build_small_tree(), 3);
        tree.assign_id(0);
        let path = tree.get_path(&vec![0_u32, 2_u32]);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].get_id(), Some(0));
        assert_eq!(path[1].get_id(), Some(1));
        assert_eq!(path[2].get_id(), Some(4));
    }
}
