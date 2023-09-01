//! Recursive data structure for decision trees
//!
//! Trees contains all the functionality for a recursive representation of a decision tree that is
//! amenable to manipulation.  Thresholds and sample values are `u16`.  Leaf values are generic.
//!
//! ## Usage
//!
//! ```
//! let n_features = 4;
//! let depth = 6;
//! // start with a random (probably not perfect) tree with f64 leaf values
//! let tree_f64 = generate_tree(depth, n_features, 0.5);
//! // quantize the leaf values (see also quantize_leaf_values())
//! let tree = tree_f64.map(&|x| x as i64);
//! // perfect the tree
//! let mut perfect_tree = tree.perfect_to_depth(depth);
//! assert!(perfect_tree.is_perfect());
//! // recursively assign ids to nodes
//! perfect_tree.assign_id(0);
//! // get the path of a sample through the tree
//! let sample: Vec<u16> = vec![11, 0, 1, 2];
//! let path: Vec<&Node<i64>> = perfect_tree.get_path(&sample);
//! ```
extern crate serde;
extern crate serde_json;
use crate::zkdt::helpers::SIGNED_DECOMPOSITION_MAX_ARG_ABS;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Enum for representing a tree in a recursive form amenable to manipulation and path
/// determination (given a sample).
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Node<T: Copy> {
    Internal {
        id: Option<u32>,
        feature_index: usize,
        threshold: u16,
        left: Box<Node<T>>,
        right: Box<Node<T>>,
    },
    Leaf {
        id: Option<u32>,
        value: T,
    },
}

impl<T: Copy> Node<T> {
    pub fn new_leaf(id: Option<u32>, value: T) -> Self {
        Node::Leaf { id, value }
    }

    pub fn new_internal(
        id: Option<u32>,
        feature_index: usize,
        threshold: u16,
        left: Node<T>,
        right: Node<T>,
    ) -> Self {
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
    pub fn new_constant_tree(
        depth: usize,
        feature_index: usize,
        threshold: u16,
        value: T,
    ) -> Node<T> {
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
    pub fn depth(&self, aggregator: fn(usize, usize) -> usize) -> usize {
        match self {
            Node::Internal { left, right, .. } => {
                1 + aggregator(left.depth(aggregator), right.depth(aggregator))
            }
            Node::Leaf { .. } => 1,
        }
    }

    /// Transform the leaf values of the tree using the function provided.
    /// Example:
    /// ```ignore
    /// tree.transform_values(&|x| -1.0 * x);
    /// ```
    pub fn transform_values<F>(&mut self, transform: &F)
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
    pub fn map<U, F>(&self, f: &F) -> Node<U>
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
    pub fn is_perfect(&self) -> bool {
        self.depth(std::cmp::max) == self.depth(std::cmp::min)
    }

    const DUMMY_THRESHOLD: u16 = 0;
    const DUMMY_FEATURE_INDEX: usize = 0;
    /// Return a new `Node<T>` instance which is perfect with the specified depth.
    /// New Nodes will not have ids assigned.
    /// All new Leaf nodes will have the value of the Leaf they replaced.
    /// The feature index and threshold of all new Internal nodes are `DUMMY_THRESHOLD` and
    /// `DUMMY_FEATURE_INDEX`, respectively.
    /// Pre: `depth >= self.depth()`
    /// Post: `self.is_perfect()`
    pub fn perfect_to_depth(&self, depth: usize) -> Node<T> {
        assert!(depth >= 1);
        match self {
            Node::Internal {
                left,
                right,
                feature_index,
                threshold,
                ..
            } => {
                let _left = left.perfect_to_depth(depth - 1);
                let _right = right.perfect_to_depth(depth - 1);
                Node::new_internal(None, *feature_index, *threshold, _left, _right)
            }
            Node::Leaf { value, .. } => Node::new_constant_tree(
                depth,
                Self::DUMMY_FEATURE_INDEX,
                Self::DUMMY_THRESHOLD,
                *value,
            ),
        }
    }

    /// Assign the specified id to this Node, and assign ids to any child nodes according to the
    /// rule:
    /// left_child_id = 2 * id + 1
    /// right_child_id = 2 * id + 2
    pub fn assign_id(&mut self, new_id: u32) {
        match self {
            Node::Internal {
                id, left, right, ..
            } => {
                *id = Some(new_id);
                left.assign_id(2 * new_id + 1);
                right.assign_id(2 * new_id + 2);
            }
            Node::Leaf { id, .. } => {
                *id = Some(new_id);
            }
        }
    }

    /// Example:
    /// ```
    /// tree.assign_id(3);
    /// assert_eq!(tree.get_id().unwrap(), 3);
    /// ```
    pub fn get_id(&self) -> Option<u32> {
        match self {
            Node::Internal { id, .. } => *id,
            Node::Leaf { id, .. } => *id,
        }
    }

    /// Add (depth_of_node - 1) * multiplier to the feature index of all internal nodes in this
    /// tree.
    pub fn offset_feature_indices(&mut self, multiplier: usize) {
        self.offset_feature_indices_for_depth(multiplier, 1);
    }

    /// Helper to offset_feature_indices.
    fn offset_feature_indices_for_depth(&mut self, multiplier: usize, depth: usize) {
        if let Node::Internal {
            feature_index,
            left,
            right,
            ..
        } = self
        {
            *feature_index = (depth - 1) * multiplier + *feature_index;
            left.offset_feature_indices_for_depth(multiplier, depth + 1);
            right.offset_feature_indices_for_depth(multiplier, depth + 1);
        }
    }

    /// Return the path traced by the specified sample down this tree.
    /// Pre: sample.len() > node.feature_index for this node and all descendents.
    pub fn get_path<'a>(&'a self, sample: &[u16]) -> Vec<&'a Node<T>> {
        let mut path = Vec::new();
        self.append_path(sample, &mut path);
        path
    }

    /// Helper function to get_path.
    /// Appends self to path, then, if internal, calls this function on the appropriate child node.
    fn append_path<'a>(&'a self, sample: &[u16], path_to_here: &mut Vec<&'a Node<T>>) {
        path_to_here.push(self);
        if let Node::Internal {
            left,
            right,
            feature_index,
            threshold,
            ..
        } = self
        {
            let next = if sample[*feature_index] >= *threshold {
                right
            } else {
                left
            };
            next.append_path(sample, path_to_here);
        }
    }
}

const LEAF_QUANTILE_BITWIDTH: u64 = 52; // 52 is mantissa length of a f64
/// Given a vector of trees (Node instances) with f64 leaf values, quantize the leaf values
/// symmetrically, returning the quantized trees and the rescaling factor.  The scale is chosen
/// such that no leaf value or sum trees.len() distinct leaf values will exceed
/// LEAF_QUANTILE_BITWIDTH.
/// Pre: trees.len() > 0; no leaf values are NaN.
/// Post: (quantized leaf values) / rescaling ~= (original leaf values)
pub fn quantize_leaf_values(trees: &[Node<f64>]) -> (Vec<Node<i64>>, f64) {
    let max_radius = trees
        .iter()
        .map(|tree| tree.leaf_value_radius())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let quant_max = ((1_u64) << (LEAF_QUANTILE_BITWIDTH - 1)) - 1;
    let quant_max_per_tree = quant_max / trees.len() as u64;

    // quantize the leaf values
    let rescaling = (quant_max_per_tree as f64) / max_radius;
    let qtrees: Vec<Node<i64>> = trees
        .iter()
        .map(|tree| tree.map(&|value| (value * rescaling) as i64))
        .collect();
    (qtrees, rescaling)
}

/// Randomly generate a tree with f64 leaf values.
/// Each potential decision node can degenerate to a leaf node with probability `premature_leaf_proba`.
/// Tree is guaranteed to be perfect of depth `target_depth` if `premature_leaf_proba` is 0,
/// otherwise `target_depth` is an upper bound.
/// Node ids are not assigned.
/// Thresholds are bounded SIGNED_DECOMPOSITION_MAX_ARG_ABS.
/// Pre: target_depth >= 1; n_features >= 1.
pub fn generate_tree(
    target_depth: usize,
    n_features: usize,
    premature_leaf_proba: f64,
) -> Node<f64> {
    let mut rng = rand::thread_rng();
    let premature_leaf: bool = rng.gen::<f64>() < premature_leaf_proba;
    if (target_depth == 1) | premature_leaf {
        Node::new_leaf(None, rng.gen())
    } else {
        Node::new_internal(
            None,
            rng.gen_range(0..n_features) as usize,
            rng.gen_range(0..SIGNED_DECOMPOSITION_MAX_ARG_ABS) as u16,
            generate_tree(target_depth - 1, n_features, premature_leaf_proba),
            generate_tree(target_depth - 1, n_features, premature_leaf_proba),
        )
    }
}

impl Node<f64> {
    /// Return the minimal non-negative value x such that all leaf values fit within the interval
    /// [-x, x].
    /// Pre: No leaf values are NaN.
    pub fn leaf_value_radius(&self) -> f64 {
        match self {
            Node::Leaf { value, .. } => value.abs(),
            Node::Internal { left, right, .. } => {
                f64::max(left.leaf_value_radius(), right.leaf_value_radius())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let _tree: Node<f64> = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_debug_printing() {
        let tree = Node::new_internal(None, 0, 1, Node::new_leaf(None, 0), Node::new_leaf(None, 1));
        println!("{:?}", tree);
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
        let tree_i64 = tree_f64.map(&|x| x as i64);
        if let Node::Internal { right, .. } = tree_i64 {
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
            feature_index,
            threshold,
            ..
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
        let perfect_tree = tree.perfect_to_depth(3);
        assert_eq!(perfect_tree.depth(std::cmp::max), 3);
        assert!(perfect_tree.is_perfect());
        let perfect_tree = tree.perfect_to_depth(4);
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
    fn test_offset_feature_indices() {
        let mut tree = build_small_tree();
        tree.offset_feature_indices(10);
        if let Node::Internal {
            left,
            feature_index,
            ..
        } = tree
        {
            assert_eq!(feature_index, 1);
            if let Node::Internal { feature_index, .. } = *left {
                assert_eq!(feature_index, 10);
                return;
            }
        }
        panic!("Should be inaccessible");
    }

    #[test]
    fn test_get_path() {
        let mut tree = build_small_tree().map(&|x| x as i64).perfect_to_depth(3);
        tree.assign_id(0);
        let path = tree.get_path(&vec![2_u16, 0_u16]);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].get_id(), Some(0));
        assert_eq!(path[1].get_id(), Some(1));
        assert_eq!(path[2].get_id(), Some(4));
    }

    // Helper function
    fn _test_quantize_leaf_values(leaf_values: &Vec<f64>) {
        let mut trees: Vec<Node<f64>> = vec![];
        for value in leaf_values.iter() {
            trees.push(Node::new_leaf(None, *value));
        }
        let (qtrees, rescaling) = quantize_leaf_values(&trees);
        assert_eq!(qtrees.len(), trees.len());
        for (value, qtree) in leaf_values.iter().zip(qtrees.iter()) {
            if let Node::Leaf { value: qvalue, .. } = qtree {
                assert!((value - (*qvalue as f64) / rescaling).abs() < 1e-5);
            } else {
                panic!("The quantized trees should have been leaf node");
            }
        }
    }

    #[test]
    fn test_quantize_leaf_values() {
        _test_quantize_leaf_values(&vec![123.11111111111, 145.1]);
        _test_quantize_leaf_values(&vec![-123.11111111111, 145.1]);
        _test_quantize_leaf_values(&vec![-0.12311111111111, 145.1]);
        _test_quantize_leaf_values(&vec![145.1]);
        _test_quantize_leaf_values(&vec![145.1; 100]);
    }

    #[test]
    fn test_generate_tree() {
        let target_depth = 3;
        let n_features = 6;
        // check that trees are perfect when premature_leaf_proba==0.
        for _ in (0..10) {
            let tree = generate_tree(target_depth, n_features, 0.);
            assert_eq!(tree.depth(std::cmp::max), target_depth);
            assert!(tree.is_perfect());
        }
        // check that target_depth is always upper bound and that premature_leaf_proba > 0 results
        // in some imperfect trees.
        let mut n_perfect: usize = 0;
        let n_iter = 50;
        for _ in (0..n_iter) {
            let tree = generate_tree(target_depth, n_features, 0.5);
            assert!(tree.depth(std::cmp::max) <= target_depth);
            if tree.is_perfect() {
                n_perfect += 1;
            }
        }
        assert!(n_perfect < n_iter);
    }

    #[test]
    fn test_leaf_value_radius() {
        let left = Node::new_leaf(None, -0.1);
        let middle = Node::new_leaf(None, 0.2);
        let right = Node::new_leaf(None, 1.2);
        let internal = Node::new_internal(None, 0, 2, left.clone(), middle);
        let tree = Node::new_internal(None, 1, 1, internal.clone(), right);
        // test for a single leaf
        assert_eq!(left.leaf_value_radius(), 0.1);
        // .. for a cherry
        assert_eq!(internal.leaf_value_radius(), 0.2);
        // .. for depth 3
        assert_eq!(tree.leaf_value_radius(), 1.2);
    }

    #[test]
    fn test_documentation() {
        // TODO remove once the doctests are being run
        let n_features = 4;
        let depth = 6;
        // start with a random (probably not perfect) tree with f64 leaf values
        let tree_f64 = generate_tree(depth, n_features, 0.5);
        // quantize the leaf values (see also quantize_leaf_values())
        let tree = tree_f64.map(&|x| x as i64);
        // perfect the tree
        let mut perfect_tree = tree.perfect_to_depth(depth);
        assert!(perfect_tree.is_perfect());
        // recursively assign ids to nodes
        perfect_tree.assign_id(0);
        // get the path of a sample through the tree
        let sample: Vec<u16> = vec![11, 0, 1, 2];
        let path: Vec<&Node<i64>> = perfect_tree.get_path(&sample);
    }
}
