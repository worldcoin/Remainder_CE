extern crate num;
extern crate serde;
extern crate serde_json;

use crate::zkdt::structs::{DecisionNode,LeafNode,InputAttribute,BinDecomp16Bit};
use crate::FieldExt;
use ndarray::Array2;
use ndarray_npy::{read_npy, ReadNpyError};
use num::pow;
use serde::{Deserialize, Serialize};
use std::iter::repeat;
use itertools::Itertools;

const BIT_DECOMPOSITION_LENGTH: usize = 16;
const SIGNED_DECOMPOSITION_MAX_ARG_ABS: u32 = 2_u32.pow(BIT_DECOMPOSITION_LENGTH as u32 - 1) - 1;
const UNSIGNED_DECOMPOSITION_MAX_ARG: u32 = 2_u32.pow(BIT_DECOMPOSITION_LENGTH as u32) - 1;

/// Build a 16 bit signed decomposition of the specified i32, or None if the argument is too large
/// in absolute value (exceeding SIGNED_DECOMPOSITION_MAX_ARG_ABS).
/// Result is little endian (so LSB has index 0).
/// Sign bit has maximal index.
fn build_signed_bit_decomposition<F: FieldExt>(value: i32) -> Option<BinDecomp16Bit<F>> {
    let abs_val = value.abs() as u32;
    if abs_val > SIGNED_DECOMPOSITION_MAX_ARG_ABS {
        return None;
    }
    let mut decomposition = build_unsigned_bit_decomposition(abs_val).unwrap();
    // set the sign bit
    decomposition.bits[BIT_DECOMPOSITION_LENGTH - 1] = if value >= 0 {
        F::zero()
    } else {
        F::one()
    };
    Some(decomposition)
}

/// Build a 16 bit decomposition of the specified u32, or None if the argument is too large
/// (exceeding UNSIGNED_DECOMPOSITION_MAX_ARG_ABS).
/// Result is little endian i.e. LSB has index 0.
fn build_unsigned_bit_decomposition<F: FieldExt>(mut value: u32) -> Option<BinDecomp16Bit<F>> {
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

type Sample<F> = Vec<InputAttribute<F>>;
/// Convert a vector of u16s to a Sample (consisting of field elements).
fn build_sample<F: FieldExt>(values: &Vec<u16>) -> Sample<F> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| InputAttribute {
            attr_id: F::from(index as u16),
            attr_val: F::from(*value),
        })
        .collect()
}

/// Repeat the items of the provided vector `repetitions` times, before padding with the minimal
/// number of zeros such that the length is a power of two.
fn repeat_and_pad<T: Clone>(values: &Vec<T>, repetitions: usize, padding: T) -> Vec<T> {
    // repeat
    let repeated_length = repetitions * values.len();
    let repeated_iter = values.into_iter().cycle().take(repeated_length);
    // pad to nearest power of two
    let padding_length = (next_power_of_two(repeated_length as u32).unwrap() as usize - repeated_length);
    let padding_iter = repeat(&padding).take(padding_length);
    // chain together and convert to Sample
    repeated_iter
        .chain(padding_iter)
        .cloned()
        .collect()
}

// TODO:
// Confirm index orders below
// Confirm that multiplicities is just over node indices, i.e. not counted separately per tree
// Ask: do the number of trees need to be a power of two?
// Double-check: input attributes are repeated _then_ padded to a power of two.
struct CircuitizedSamples<F: FieldExt> {
    samples: Vec<Sample<F>>, // indexed by samples
    permuted_samples: Vec<Vec<Sample<F>>>, // indexed by samples, trees
    decision_paths: Vec<Vec<Vec<DecisionNode<F>>>>, // indexed by samples, trees, steps in path
    differences: Vec<Vec<Vec<BinDecomp16Bit<F>>>>, // indexed by samples, trees, steps in path
    path_ends: Vec<Vec<LeafNode<F>>>, // indexed by samples, trees
    multiplicities: Vec<BinDecomp16Bit<F>>, // indexed by tree node indices
}

/// TODO describe return values
/// multiplicities is 2 ** depth in size.
/// Pre: values_array is not empty
fn circuitize_samples<F: FieldExt>(values_array: &Vec<Vec<u16>>, pqtrees: &PaddedQuantizedTrees) -> CircuitizedSamples<F> {
    // repeat and pad the attributes of the sample
    let values_array: Vec<Vec<u16>> = values_array
        .iter()
        .map(|x| repeat_and_pad(x, (pqtrees.depth - 1) as usize, 0_u16))
        .collect();
    let sample_length = values_array[0].len();
    
    // TODO can we think of a better naming convention than appending 's'?
    let mut samples: Vec<Sample<F>> = vec![];
    let mut permuted_sampless: Vec<Vec<Sample<F>>> = vec![];
    let mut decision_pathss: Vec<Vec<Vec<DecisionNode<F>>>> = vec![];
    let mut path_endss: Vec<Vec<LeafNode<F>>> = vec![];
    let mut differencesss: Vec<Vec<Vec<BinDecomp16Bit<F>>>> = vec![];

    // initialize the node visit counts "multiplicities"
    let mut multiplicities: Vec<u32> = vec![0_u32; 2_usize.pow(pqtrees.depth)];

    for values in values_array {
        let sample = build_sample(&values);
        samples.push(sample.clone());
        let mut permuted_samples: Vec<Sample<F>> = vec![];
        let mut decision_paths: Vec<Vec<DecisionNode<F>>> = vec![];
        let mut path_ends: Vec<LeafNode<F>> = vec![];
        let mut differencess: Vec<Vec<BinDecomp16Bit<F>>> = vec![];

        for tree in &pqtrees.trees {
            // get the path
            let path = tree.get_path(&values);
            
            // derive data from decision path
            let mut decision_path = vec![];
            let mut permuted_sample: Sample<F> = vec![];
            let mut attribute_visits = vec![0; sample_length];
            let mut differences = vec![];
            for node in &path[..path.len() - 1] {
                if let Node::Internal {
                    id,
                    feature_index,
                    threshold,
                    ..
                } = node
                {
                    decision_path.push(DecisionNode {
                        node_id: F::from(id.unwrap()),
                        attr_id: F::from(*feature_index as u32),
                        threshold: F::from(*threshold),
                    });
                    // calculate the bit decompositions of the differences
                    let difference = (values[*feature_index] as i32) - (*threshold as i32);
                    differences.push(build_signed_bit_decomposition::<F>(difference).unwrap());
                    // accumulate the multiplicities for this tree
                    multiplicities[id.unwrap() as usize] += 1;
                    // build up the permuted sample
                    permuted_sample.push(sample[*feature_index]);
                    attribute_visits[*feature_index] += 1;
                } else {
                    panic!("All Nodes in the path must be internal, except the last");
                }
            }
            // populate the remaining attributes of the permuted sample
            for idx in 0..sample_length {
                if attribute_visits[idx] == 0 {
                    permuted_sample.push(sample[idx]);
                }
            }

            permuted_samples.push(permuted_sample);
            decision_paths.push(decision_path);
            differencess.push(differences);
            
            // build the leaf node
            if let Node::Leaf { id, value } = path[path.len() - 1] {
                path_ends.push(LeafNode {
                    node_id: F::from(id.unwrap()),
                    node_val: i32_to_field(*value)
                });
                // accumulate multiplicity for leaf node
                multiplicities[id.unwrap() as usize] += 1;
            } else {
                panic!("Last item in path should be a Node::Leaf");
            }
        }
        permuted_sampless.push(permuted_samples);
        decision_pathss.push(decision_paths);
        path_endss.push(path_ends);
        differencesss.push(differencess);
    }

    // calculate the bit decompositions of the visit counts
    let multiplicities: Vec<BinDecomp16Bit<F>> = multiplicities
        .into_iter()
        .map(build_unsigned_bit_decomposition)
        .map(|option| option.unwrap())
        .collect();

    CircuitizedSamples {
        samples: samples,
        permuted_samples: permuted_sampless,
        decision_paths: decision_pathss,
        differences: differencesss,
        path_ends: path_endss,
        multiplicities: multiplicities
    }
}

/// Circuitized trees use flat (i.e. non-recursive) structs for the decision and leaf nodes and
/// represent all integers using the field.
/// Circuitized trees have the same properties as PaddedQuantizedTree, except that they include an
/// extra "dummy" decision node so that the number of decision nodes is a power of two (equal to
/// the number of leaf nodes).
/// The dummy decision node has node id 2^depth - 1.
struct CircuitizedTrees<F: FieldExt> {
    decision_nodes: Vec<Vec<DecisionNode<F>>>, // indexed by tree, then by node (sorted by node id)
    leaf_nodes: Vec<Vec<LeafNode<F>>>, // indexed by tree, then by node (sorted by node id)
    depth: u32,
    scaling: f64,
}

impl<F: FieldExt> From<&PaddedQuantizedTrees> for CircuitizedTrees<F> {
    /// Extract the DecisionNode and LeafNode instances from the PaddedQuantizedTrees instance to
    /// obtain a CircuitizedTrees.
    fn from(pqtrees: &PaddedQuantizedTrees) -> Self {
        // extract, sort & pad the decision nodes
        let mut decision_nodes = vec![];
        let dummy_node = DecisionNode {
            node_id: F::from(2_u32.pow(pqtrees.depth) - 1),
            attr_id: F::from(0_u32),
            threshold: F::from(0_u32)
        };
        for tree in &pqtrees.trees {
            let mut tree_decision_nodes = tree.extract_decision_nodes();
            tree_decision_nodes.sort_by_key(|node| node.node_id);
            // add a dummy node to make length a power of two
            tree_decision_nodes.push(dummy_node.clone());
            decision_nodes.push(tree_decision_nodes);
        }
        // extract and sort the leaf nodes
        let mut leaf_nodes = vec![];
        for tree in &pqtrees.trees {
            let mut tree_leaf_nodes = tree.extract_leaf_nodes();
            tree_leaf_nodes.sort_by_key(|node| node.node_id);
            leaf_nodes.push(tree_leaf_nodes);
        }
        
        CircuitizedTrees {
            decision_nodes: decision_nodes,
            leaf_nodes: leaf_nodes,
            depth: pqtrees.depth,
            scaling: pqtrees.scaling,
        }
    }
}


/// Represents the trees model as it appears in the input JSON.
#[derive(Debug, Serialize, Deserialize)]
struct TreesModelInput {
    trees: Vec<Node<f64>>,
    bias: f64,
    scale: f64,
    n_features: usize,
}

struct PaddedQuantizedTrees {
    trees: Vec<Node<i32>>,
    depth: u32,
    scaling: f64,
}

impl From<&TreesModelInput> for PaddedQuantizedTrees {
    /// Given a TreesModelInput object representing a decision tree model operating on u32 samples, prepare the model for circuitization:
    /// 1. scale and bias are folded into the leaf values;
    /// 2. leaf values are symmetrically quantized to i32;
    /// 3. all trees are padded such that they are all perfect and of uniform depth (without modifying
    ///    the predictions of any tree) where the uniform depth is chosen to be 2^l + 1 for minimal l >= 0;
    /// 4. ids are assigned to all nodes (as per assign_id());
    /// 5. the feature indexes are transformed according to
    ///      idx -> (depth_of_node - 1) * trees_info.n_features + idx
    ///    thereby ensuring that each feature index occurs only once on each descent path.
    /// The resulting PaddedQuantizedTrees incorporates all the tree instances, the (uniform) depth
    /// of the trees, and the scaling factor to approximately undo the quantization (via division)
    /// after aggregating the scores.
    fn from(trees_info: &TreesModelInput) -> Self {
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
            tree.offset_feature_indices(trees_info.n_features);
        }

        PaddedQuantizedTrees {
            trees: qtrees,
            depth: target_depth,
            scaling: rescaling,
        }
    }
}

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
    fn new_leaf(id: Option<u32>, value: T) -> Self {
        Node::Leaf { id, value }
    }

    fn new_internal(id: Option<u32>, feature_index: usize, threshold: u16, left: Node<T>, right: Node<T>) -> Self {
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
    fn new_constant_tree(depth: u32, feature_index: usize, threshold: u16, value: T) -> Node<T> {
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
    fn offset_feature_indices(&mut self, multiplier: usize) {
        self.offset_feature_indices_for_depth(multiplier, 1);
    }

    /// Helper to offset_feature_indices.
    fn offset_feature_indices_for_depth(&mut self, multiplier: usize, depth: usize) {
        if let Node::Internal { feature_index, left, right, .. } = self {
            *feature_index = (depth - 1) * multiplier + *feature_index;
            left.offset_feature_indices_for_depth(multiplier, depth + 1);
            right.offset_feature_indices_for_depth(multiplier, depth + 1);
        }
    }

    /// Return the path traced by the specified sample down this tree.
    /// Pre: sample.len() > node.feature_index for this node and all descendents.
    fn get_path<'a>(&'a self, sample: &Vec<u16>) -> Vec<&'a Node<T>> {
        let mut path = Vec::new();
        self.append_path(sample, &mut path);
        path
    }

    /// Helper function to get_path.
    /// Appends self to path, then, if internal, calls this function on the appropriate child node.
    fn append_path<'a>(&'a self, sample: &Vec<u16>, path_to_here: &mut Vec<&'a Node<T>>) {
        path_to_here.push(self);
        if let Node::Internal {
            left,
            right,
            feature_index,
            threshold,
            ..
        } = self {
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
                attr_id: F::from(*feature_index as u32),
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
                    node_val: i32_to_field(*value)
                });
            }
            Node::Internal { left, right, .. } => {
                left.append_leaf_nodes(leaf_nodes);
                right.append_leaf_nodes(leaf_nodes);
            }
        }
    }
}

/// Helper function for conversion to field elements, handling negative values.
fn i32_to_field<F: FieldExt>(value: i32) -> F {
    if value >= 0 {
        F::from(value as u32)
    } else {
        F::from(value.abs() as u32).neg()
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
        let trees_info: TreesModelInput = serde_json::from_reader(file).expect(&format!(
            "'{}' should be valid TreesModelInput JSON.",
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

    #[test]
    fn test_numpy_loading() {
        let filename = String::from("src/zkdt/test_samples_10x6.npy");
        let input_arr: Array2<u16> = read_npy(filename).unwrap();
        let samples: Vec<Vec<u16>> = input_arr
            .outer_iter().map(|row| row.to_vec()).collect();
    }

    #[test]
    fn test_circuitize_samples() {
        let samples = vec![vec![0_u16; 5], vec![1_u16; 5]];
        let mut tree = build_small_tree();
        let trees_info = TreesModelInput {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: 5,
        };
        let pqtrees: PaddedQuantizedTrees = (&trees_info).into();
        let csamples = circuitize_samples::<Fr>(&samples, &pqtrees);
        // check size of outer dimensions
        assert_eq!(csamples.samples.len(), samples.len());
        assert_eq!(csamples.permuted_samples.len(), samples.len());
        assert_eq!(csamples.decision_paths.len(), samples.len());
        assert_eq!(csamples.differences.len(), samples.len());
        assert_eq!(csamples.path_ends.len(), samples.len());
        assert_eq!(csamples.multiplicities.len(), 8);
        // FIXME requires thorough inspection of inner dimensions
    }

    #[test]
    fn test_get_path() {
        let mut tree = quantize_and_perfect(build_small_tree(), 3);
        tree.assign_id(0);
        let path = tree.get_path(&vec![0_u16, 2_u16]);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].get_id(), Some(0));
        assert_eq!(path[1].get_id(), Some(1));
        assert_eq!(path[2].get_id(), Some(4));
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
    fn test_build_sample() {
        let result = build_sample::<Fr>(&vec![3_u16, 1_u16, 0_u16]);
        assert_eq!(result.len(), 3);
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
        assert_eq!(build_signed_bit_decomposition::<Fr>(-1 * 2_i32.pow(30)), None);
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
    
    #[test]
    fn test_padded_quantized_trees_from() {
        let mut tree = build_small_tree();
        let trees_info = TreesModelInput {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: 11,
        };
        let pqtrees: PaddedQuantizedTrees = (&trees_info).into();
        assert_eq!(pqtrees.trees.len(), 2);
        assert_eq!(pqtrees.depth, 3);
        // check trees are as claimed
        for tree in &pqtrees.trees {
            assert_eq!(tree.depth(std::cmp::max), 3);
            assert!(tree.is_perfect());
            assert_eq!(tree.get_id().unwrap(), 0);
        }
    }

    #[test]
    fn test_circuitized_trees_from() {
        let mut tree = build_small_tree();
        let trees_info = TreesModelInput {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: 11,
        };
        let pqtrees: PaddedQuantizedTrees = (&trees_info).into();
        let ctrees: CircuitizedTrees<Fr> = (&pqtrees).into();
        // check decision nodes
        for tree_dns in &ctrees.decision_nodes {
            assert_eq!(tree_dns.len(), 4);
            for (node_id, node) in tree_dns
                .iter().take(tree_dns.len() - 1).enumerate() {
                assert_eq!(node.node_id, Fr::from(node_id as u32));
            }
            assert_eq!(tree_dns[tree_dns.len() - 1].node_id, Fr::from(7 as u32));
        }
        // check leaf nodes
        for tree_lns in &ctrees.leaf_nodes {
            assert_eq!(tree_lns.len(), 4);
            for (node_id, node) in tree_lns
                .iter().enumerate() {
                assert_eq!(node.node_id, Fr::from((4 - 1) + node_id as u32));
            }
        }
        // accumulate score by taking the value of the first leaf node
        let mut acc_score: Fr = Fr::from(0);
        acc_score += ctrees.leaf_nodes[0][0].node_val;
        acc_score += ctrees.leaf_nodes[1][0].node_val;
        // check that the quantized scores accumulated as expected
        let expected_score = trees_info.scale * (0.1 + 3.0) + trees_info.bias;
        let quant_score = (expected_score * ctrees.scaling) as i32;
        let f_quant_score = if quant_score >= 0 {
            Fr::from(quant_score)
        } else {
            -Fr::from(quant_score.abs() as u32)
        };
        // just check that's it's close
        assert_eq!(f_quant_score, acc_score + Fr::from(1));
    }
}
