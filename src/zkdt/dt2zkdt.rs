extern crate serde;
extern crate serde_json;

use crate::zkdt::structs::{BinDecomp16Bit, DecisionNode, InputAttribute, LeafNode};
use crate::zkdt::trees::*;
use crate::FieldExt;
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde::{Deserialize, Serialize};
use std::iter::repeat;


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
    decomposition.bits[BIT_DECOMPOSITION_LENGTH - 1] =
        if value >= 0 { F::zero() } else { F::one() };
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
fn build_sample<F: FieldExt>(values: &[u16]) -> Sample<F> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| InputAttribute {
            attr_id: F::from(index as u16),
            attr_val: F::from(*value),
        })
        .collect()
}

/// Repeat the items of the provided slice `repetitions` times, before padding with the minimal
/// number of zeros such that the length is a power of two.
fn repeat_and_pad<T: Clone>(values: &[T], repetitions: usize, padding: T) -> Vec<T> {
    // repeat
    let repeated_length = repetitions * values.len();
    let repeated_iter = values.into_iter().cycle().take(repeated_length);
    // pad to nearest power of two
    let padding_length = next_power_of_two(repeated_length).unwrap() - repeated_length;
    let padding_iter = repeat(&padding).take(padding_length);
    // chain together and convert to a vector
    repeated_iter.chain(padding_iter).cloned().collect()
}

struct CircuitizedSamples<F: FieldExt> {
    samples: Vec<Sample<F>>,                        // indexed by samples
    permuted_samples: Vec<Vec<Sample<F>>>,          // indexed by trees, samples
    decision_paths: Vec<Vec<Vec<DecisionNode<F>>>>, // indexed by trees, samples, steps in path
    differences: Vec<Vec<Vec<BinDecomp16Bit<F>>>>,  // indexed by trees, samples, steps in path
    path_ends: Vec<Vec<LeafNode<F>>>,               // indexed by trees, samples
    multiplicities: Vec<BinDecomp16Bit<F>>,         // indexed by tree node indices
}

/// TODO describe return values
/// multiplicities is 2 ** depth in size.
/// length of each sample and permuted sample is next_power_of_two((depth - 1) * sample length)
/// Little endian, sign bit at end.
/// Pre: values_array is not empty
fn circuitize_samples<F: FieldExt>(
    values_array: &[Vec<u16>],
    pqtrees: &PaddedQuantizedTrees,
) -> CircuitizedSamples<F> {
    // repeat and pad the attributes of the sample
    let values_array: Vec<Vec<u16>> = values_array
        .iter()
        .map(|x| repeat_and_pad(x, pqtrees.depth - 1, 0_u16))
        .collect();
    let sample_length = values_array[0].len();

    let mut samples: Vec<Sample<F>> = vec![];
    let mut permuted_samples: Vec<Vec<Sample<F>>> = vec![];
    let mut decision_paths: Vec<Vec<Vec<DecisionNode<F>>>> = vec![];
    let mut path_ends: Vec<Vec<LeafNode<F>>> = vec![];
    let mut differences: Vec<Vec<Vec<BinDecomp16Bit<F>>>> = vec![];

    // initialize the node visit counts "multiplicities"
    let mut multiplicities: Vec<u32> = vec![0_u32; 2_usize.pow(pqtrees.depth as u32)];

    // build the samples array
    for values in &values_array {
        samples.push(build_sample(&values));
    }

    for tree in &pqtrees.trees {
        let mut permuted_samples_for_tree: Vec<Sample<F>> = vec![];
        let mut decision_paths_for_tree: Vec<Vec<DecisionNode<F>>> = vec![];
        let mut path_ends_for_tree: Vec<LeafNode<F>> = vec![];
        let mut differences_for_tree: Vec<Vec<BinDecomp16Bit<F>>> = vec![];

        for (values, sample) in values_array.iter().zip(samples.iter()) {
            // get the path
            let path = tree.get_path(&values);

            // derive data from decision path
            let mut decision_path = vec![];
            let mut permuted_sample: Sample<F> = vec![];
            let mut attribute_visits = vec![0; sample_length];
            let mut differences_for_tree_and_sample = vec![];
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
                    differences_for_tree_and_sample
                        .push(build_signed_bit_decomposition::<F>(difference).unwrap());
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

            permuted_samples_for_tree.push(permuted_sample);
            decision_paths_for_tree.push(decision_path);
            differences_for_tree.push(differences_for_tree_and_sample);

            // build the leaf node
            if let Node::Leaf { id, value } = path[path.len() - 1] {
                path_ends_for_tree.push(LeafNode {
                    node_id: F::from(id.unwrap()),
                    node_val: i32_to_field(*value),
                });
                // accumulate multiplicity for leaf node
                multiplicities[id.unwrap() as usize] += 1;
            } else {
                panic!("Last item in path should be a Node::Leaf");
            }
        }
        permuted_samples.push(permuted_samples_for_tree);
        decision_paths.push(decision_paths_for_tree);
        path_ends.push(path_ends_for_tree);
        differences.push(differences_for_tree);
    }

    // calculate the bit decompositions of the visit counts
    let multiplicities: Vec<BinDecomp16Bit<F>> = multiplicities
        .into_iter()
        .map(build_unsigned_bit_decomposition)
        .map(|option| option.unwrap())
        .collect();

    CircuitizedSamples {
        samples: samples,
        permuted_samples: permuted_samples,
        decision_paths: decision_paths,
        differences: differences,
        path_ends: path_ends,
        multiplicities: multiplicities,
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
    leaf_nodes: Vec<Vec<LeafNode<F>>>,         // indexed by tree, then by node (sorted by node id)
    depth: usize,
    scaling: f64,
}

impl<F: FieldExt> From<&PaddedQuantizedTrees> for CircuitizedTrees<F> {
    /// Extract the DecisionNode and LeafNode instances from the PaddedQuantizedTrees instance to
    /// obtain a CircuitizedTrees.
    fn from(pqtrees: &PaddedQuantizedTrees) -> Self {
        // extract, sort & pad the decision nodes
        let mut decision_nodes = vec![];
        let dummy_node = DecisionNode {
            node_id: F::from(2_u32.pow(pqtrees.depth as u32) - 1),
            attr_id: F::from(0_u32),
            threshold: F::from(0_u32),
        };
        for tree in &pqtrees.trees {
            let mut tree_decision_nodes = extract_decision_nodes(&tree);
            tree_decision_nodes.sort_by_key(|node| node.node_id);
            // add a dummy node to make length a power of two
            tree_decision_nodes.push(dummy_node.clone());
            decision_nodes.push(tree_decision_nodes);
        }
        // extract and sort the leaf nodes
        let mut leaf_nodes = vec![];
        for tree in &pqtrees.trees {
            let mut tree_leaf_nodes = extract_leaf_nodes(&tree);
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

/// Return a Vec containing a DecisionNode for each Node::Internal appearing in this tree, in arbitrary order.
/// Pre: if `node` is any descendent of this Node then `node.get_id()` is not None.
pub(crate) fn extract_decision_nodes<T: Copy, F: FieldExt>(tree: &Node<T>) -> Vec<DecisionNode<F>> {
    let mut decision_nodes = Vec::new();
    append_decision_nodes(tree, &mut decision_nodes);
    decision_nodes
}

/// Helper function to extract_decision_nodes.
fn append_decision_nodes<T: Copy, F: FieldExt>(tree: &Node<T>, decision_nodes: &mut Vec<DecisionNode<F>>) {
    if let Node::Internal {
        id,
        left,
        right,
        feature_index,
        threshold,
    } = tree
    {
        decision_nodes.push(DecisionNode {
            node_id: F::from(id.unwrap()),
            attr_id: F::from(*feature_index as u32),
            threshold: F::from(*threshold),
        });
        append_decision_nodes(left, decision_nodes);
        append_decision_nodes(right, decision_nodes);
    }
}


/// Return a Vec containing a LeafNode for each Node::Leaf appearing in this tree, in order of
/// id, where the ids are allocated as in extract_decision_nodes().
fn extract_leaf_nodes<F: FieldExt>(tree: &Node<i32>) -> Vec<LeafNode<F>> {
    let mut leaf_nodes = Vec::new();
    append_leaf_nodes(tree, &mut leaf_nodes);
    leaf_nodes
}

/// Helper function for extract_leaf_nodes.
fn append_leaf_nodes<F: FieldExt>(tree: &Node<i32>, leaf_nodes: &mut Vec<LeafNode<F>>) {
    match tree {
        Node::Leaf { id, value } => {
            leaf_nodes.push(LeafNode {
                node_id: F::from(id.unwrap()),
                node_val: i32_to_field(*value),
            });
        }
        Node::Internal { left, right, .. } => {
            append_leaf_nodes(left, leaf_nodes);
            append_leaf_nodes(right, leaf_nodes);
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
    depth: usize,
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
        let leaf_expander = |depth: usize, value: i32| Node::new_constant_tree(depth, 0, 0, value);
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

/// Helper function for conversion to field elements, handling negative values.
fn i32_to_field<F: FieldExt>(value: i32) -> F {
    if value >= 0 {
        F::from(value as u32)
    } else {
        F::from(value.abs() as u32).neg()
    }
}

/// Return the first power of two that is greater than or equal to the argument, or None if this
/// would exceed the range of a u32.
fn next_power_of_two(n: usize) -> Option<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use std::fs::File;

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

    /// Helper function for testing.
    fn quantize_and_perfect(tree_f64: Node<f64>, depth: usize) -> Node<i32> {
        let leaf_expander = |depth: usize, value: i32| Node::new_constant_tree(depth, 0, 0, value);
        tree_f64
            .map(&|x| x as i32)
            .perfect_to_depth(depth, &leaf_expander)
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
        assert_eq!(decision_nodes[0].node_id, Fr::from(root_id));
        assert_eq!(decision_nodes[0].attr_id, Fr::from(1));
        assert_eq!(decision_nodes[1].node_id, Fr::from(2 * root_id + 1));
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
        let mut tree = build_small_tree().map(&|x| x as i32);
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
        let _trees_info: TreesModelInput = serde_json::from_reader(file).expect(&format!(
            "'{}' should be valid TreesModelInput JSON.",
            TEST_TREES_INFO
        ));
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
    fn test_numpy_loading() {
        let filename = String::from("src/zkdt/test_samples_10x6.npy");
        let input_arr: Array2<u16> = read_npy(filename).unwrap();
        let _samples: Vec<Vec<u16>> = input_arr.outer_iter().map(|row| row.to_vec()).collect();
    }

    #[test]
    fn test_circuitize_samples() {
        let sample_length = 5;
        let samples = vec![
            vec![0_u16; sample_length],
            vec![2_u16, 0_u16, 0_u16, 0_u16, 0_u16],
            vec![2_u16; sample_length],
        ];
        let tree = build_small_tree();
        let trees_info = TreesModelInput {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: sample_length,
        };
        let pqtrees: PaddedQuantizedTrees = (&trees_info).into();
        let csamples = circuitize_samples::<Fr>(&samples, &pqtrees);
        // check size of outer dimensions
        let n_trees = trees_info.trees.len();
        let repeated_sample_length =
            next_power_of_two((pqtrees.depth - 1) * sample_length).unwrap();
        assert_eq!(csamples.samples.len(), samples.len());

        // check dimensions of permuted samples
        assert_eq!(csamples.permuted_samples.len(), n_trees);
        for permuted_samples_for_tree in &csamples.permuted_samples {
            assert_eq!(permuted_samples_for_tree.len(), samples.len());
            for permuted_sample in permuted_samples_for_tree {
                assert_eq!(permuted_sample.len(), repeated_sample_length);
            }
        }
        // check the contents of the permuted samples for the non-trivial tree (the 0th)
        let permuted_sample = &csamples.permuted_samples[0][0];
        assert_eq!(permuted_sample[0].attr_id, Fr::from(1));
        // ... sample travels left down the tree
        assert_eq!(
            permuted_sample[1].attr_id,
            Fr::from(sample_length as u32 + 0)
        );

        // check the dimension of the decision paths
        assert_eq!(csamples.decision_paths.len(), n_trees);
        for decision_paths_for_tree in &csamples.decision_paths {
            assert_eq!(decision_paths_for_tree.len(), samples.len());
            for decision_path in decision_paths_for_tree {
                assert_eq!(decision_path.len(), pqtrees.depth - 1);
            }
        }
        // check decision path contents for one combination
        let decision_path = &csamples.decision_paths[0][2];
        assert_eq!(decision_path[0].node_id, Fr::from(0));
        // ... sample travels right down the tree
        assert_eq!(decision_path[1].node_id, Fr::from(2));

        // check the dimension of the path_ends
        assert_eq!(csamples.path_ends.len(), n_trees);
        for path_ends_for_tree in &csamples.path_ends {
            assert_eq!(path_ends_for_tree.len(), samples.len());
        }
        // check the contents
        let path_ends_for_tree = &csamples.path_ends[0];
        assert_eq!(path_ends_for_tree[0].node_id, Fr::from(3));
        assert_eq!(path_ends_for_tree[1].node_id, Fr::from(4));

        // check the dimensions of the differences
        assert_eq!(csamples.differences.len(), n_trees);
        for differences_for_tree in &csamples.differences {
            assert_eq!(differences_for_tree.len(), samples.len());
            for differences in differences_for_tree {
                assert_eq!(differences.len(), pqtrees.depth - 1);
            }
        }
        // check contents
        let differences = &csamples.differences[0][0];
        // should have the bit decomposition of -2
        assert_eq!(differences[1].bits[0], Fr::from(0));
        assert_eq!(differences[1].bits[1], Fr::from(1));
        assert_eq!(differences[1].bits[2], Fr::from(0));
        assert_eq!(differences[1].bits[15], Fr::from(1));

        // check the multiplicities
        assert_eq!(
            csamples.multiplicities.len(),
            2_usize.pow(pqtrees.depth as u32)
        );
        // root node id has multiplicity n_trees * samples.len() = 6
        let multiplicity = &csamples.multiplicities[0];
        assert_eq!(multiplicity.bits[0], Fr::from(0));
        assert_eq!(multiplicity.bits[1], Fr::from(1));
        assert_eq!(multiplicity.bits[2], Fr::from(1));
        // dummy node id has multiplicity 0
        let multiplicity = &csamples.multiplicities[2_usize.pow(pqtrees.depth as u32) - 1];
        for bit in multiplicity.bits {
            assert_eq!(bit, Fr::from(0));
        }
    }

    #[test]
    fn test_get_path() {
        let mut tree = quantize_and_perfect(build_small_tree(), 3);
        tree.assign_id(0);
        let path = tree.get_path(&vec![2_u16, 0_u16]);
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

    #[test]
    fn test_padded_quantized_trees_from() {
        let tree = build_small_tree();
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
        let tree = build_small_tree();
        let trees_info = TreesModelInput {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: 11,
        };
        let pqtrees: PaddedQuantizedTrees = (&trees_info).into();
        let ctrees: CircuitizedTrees<Fr> = (&pqtrees).into();
        assert_eq!(ctrees.depth, pqtrees.depth);
        // check decision nodes
        for tree_dns in &ctrees.decision_nodes {
            assert_eq!(tree_dns.len(), 4);
            for (node_id, node) in tree_dns.iter().take(tree_dns.len() - 1).enumerate() {
                assert_eq!(node.node_id, Fr::from(node_id as u32));
            }
            assert_eq!(tree_dns[tree_dns.len() - 1].node_id, Fr::from(7 as u32));
        }
        // check leaf nodes
        for tree_lns in &ctrees.leaf_nodes {
            assert_eq!(tree_lns.len(), 4);
            for (node_id, node) in tree_lns.iter().enumerate() {
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
