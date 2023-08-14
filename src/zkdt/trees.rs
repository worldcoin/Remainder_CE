extern crate serde;
extern crate serde_json;
use serde::{Deserialize, Serialize};

/// Struct for representing a tree in recursive form
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
    pub fn new_constant_tree(depth: usize, feature_index: usize, threshold: u16, value: T) -> Node<T> {
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

    /// Aggregate the leaf values using the binary function provided (e.g. `f64::max`).
    /// Example:
    /// ```ignore
    /// let max = tree.aggregate_values(f64::max);
    /// ```
    pub fn aggregate_values(&self, operation: fn(T, T) -> T) -> T {
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

    /// Return a new Node<T> instance which is perfect with the specified depth, given a function
    /// `leaf_expander(depth, value)` that returns a subtree of depth `depth` to replace the
    /// premature Leaf with value `value`.
    /// New Nodes will not have ids assigned.
    /// Pre: depth >= self.depth()
    /// Post: self.is_perfect()
    pub fn perfect_to_depth<F>(&self, depth: usize, leaf_expander: &F) -> Node<T>
    where
        F: Fn(usize, T) -> Node<T>,
    {
        assert!(depth >= 1);
        match self {
            Node::Internal {
                left,
                right,
                feature_index,
                threshold,
                ..
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

const LEAF_QUANTILE_BITWIDTH: u32 = 29; // FIXME why doesn't this work with 32?
/// Given a vector of trees (Node instances) with f64 leaf values, quantize the leaf values
/// symmetrically, returning the quantized trees and the rescaling factor.  The scale is chosen
/// such that all possible _aggregate_ scores will fit within the LEAF_QUANTILE_BITWIDTH.
/// Post: (quantized leaf values) / rescaling ~= (original leaf values)
pub fn quantize_trees(trees: &[Node<f64>]) -> (Vec<Node<i32>>, f64) {
    // determine the spread of the scores
    let max_score: f64 = trees
        .iter()
        .map(|tree| tree.aggregate_values(f64::max))
        .sum();
    let min_score: f64 = trees
        .iter()
        .map(|tree| tree.aggregate_values(f64::min))
        .sum();
    let radius = f64::max(max_score.abs(), min_score.abs());

    // quantize the leaf values
    let quant_max = ((1_u32) << (LEAF_QUANTILE_BITWIDTH - 1)) - 2;
    let rescaling = (quant_max as f64) / radius;
    let qtrees: Vec<Node<i32>> = trees
        .iter()
        .map(|tree| tree.map(&|value| (value * rescaling) as i32))
        .collect();
    (qtrees, rescaling)
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
        let leaf_expander = |depth: usize, value: f64| Node::new_constant_tree(depth, 0, 0, value);
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


}

