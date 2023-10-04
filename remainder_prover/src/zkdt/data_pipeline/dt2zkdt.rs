//! Conversion from a decision trees model to a circuit ready form.
//!
//! # Circuitizing tree models (c.f. [`CircuitizedTrees`])
//!
//! ```
//! use ark_bn254::Fr;
//!
//! let n_trees = 3;
//! let n_features = 4;
//! let depth = 6;
//! // start with some random (probably not perfect) trees with f64 leaf values
//! // start with some random (probably not perfect) trees with f64 leaf values
//! let raw_trees_model = generate_raw_trees_model(n_trees, depth, n_features, 0.5);
//! // pad the trees, assign ids and quantize the leaf values
//! let trees_model: TreesModel = (&raw_trees_model).into();
//! // circuitize the trees (converts to DecisionNode<F>, LeafNode<F>)
//! let ctrees: CircuitizedTrees<Fr> = (&trees_model).into();
//! ```
//!
//! # Circuitizing samples (c.f. [`CircuitizedSamples`])
//!
//! Continuing the above example:
//!
//! ```
//! let n_samples = 10;
//! let raw_samples = generate_raw_samples(n_samples, n_features);
//! let samples = to_samples(&raw_samples, &trees_model);
//! // notice: circuitize_samples takes trees_model, not ctrees!
//! let csamples = circuitize_samples::<Fr>(&samples, &trees_model);
//! ```
//!
//! # Generating trees and samples for benchmarking
//!
//! Use the [`generate_raw_trees_model`] and [`generate_raw_samples`] functions (see above for example
//! usage).
//!
//! # Loading existing tree models & their samples
//!
//! E.g. to load tree models and samples resulting from the Python-based Catboost model processing
//! pipeline.
//!
//! ```
//! let raw_trees_model: RawTreesModel = load_raw_trees_model("src/zkdt/data_pipeline/test_qtrees.json");
//! let raw_samples: RawSamples = load_raw_samples("src/zkdt/data_pipeline/test_samples_10x6.npy");
//! ```

extern crate serde;
extern crate serde_json;

use crate::layer::LayerId;
use crate::mle::{Mle, MleRef};
use crate::mle::dense::DenseMle;
use crate::prover::input_layer::combine_input_layers::InputLayerBuilder;
use crate::prover::input_layer::ligero_input_layer::LigeroInputLayer;
use crate::utils::file_exists;
use crate::zkdt::constants::{get_cached_batched_mles_filepath_with_exp_size, get_tree_commitment_filepath_for_tree_number};
use crate::zkdt::helpers::*;
use crate::zkdt::input_data_to_circuit_adapter::ZKDTCircuitData;
use crate::zkdt::structs::{BinDecomp16Bit, BinDecomp4Bit, DecisionNode, InputAttribute, LeafNode};
use crate::zkdt::data_pipeline::trees::*;


use ark_std::log2;
use remainder_ligero::ligero_commit::{remainder_ligero_commit_prove};
use remainder_shared_types::FieldExt;
use ndarray::Array2;
use ndarray_npy::read_npy;
use rand::Rng;
use remainder_shared_types::transcript::poseidon_transcript::PoseidonTranscript;
use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use serde::{Deserialize, Serialize};
use serde_json::to_writer;
use tracing::instrument;
use std::fs::{File, self};
use std::path::Path;

/// The trees model resulting from the Python pipeline.
/// This struct is used for parsing JSON.
#[derive(Debug, Serialize, Deserialize)]
pub struct RawTreesModel {
    trees: Vec<Node<f64>>,
    bias: f64,
    scale: f64,
    n_features: usize,
}

/// Used for deriving CircuitizedTrees and CircuitizedSamples (given samples to circuitize).
/// For properties, see TreesModel.from().
pub struct TreesModel {
    trees: Vec<Node<i64>>,
    depth: usize,
    scaling: f64,
}

/// Circuitized trees use flat (i.e. non-recursive) structs for the decision and leaf nodes and
/// represent all integers using the field.
/// Circuitized trees have the same properties as PaddedQuantizedTree, except that they include an
/// extra "dummy" decision node so that the number of decision nodes is a power of two (equal to
/// the number of leaf nodes).
/// The dummy decision node has node id 2^depth - 1.
pub struct CircuitizedTrees<F: FieldExt> {
    pub decision_nodes: Vec<Vec<DecisionNode<F>>>, // indexed by tree, then by node (sorted by node id)
    pub leaf_nodes: Vec<Vec<LeafNode<F>>>,         // indexed by tree, then by node (sorted by node id)
    pub depth: usize,
    pub scaling: f64,
}

/// Represents the circuitization of a batch of samples with respect to a TreesModel.
/// * Bit decompositions are little endian.
/// * `differences` is a signed decomposition, with the sign bit at the end.
/// * Each vector in `node_multiplicities` has length `2.pow(trees_model.depth)`; it is indexed by
/// node_id for decision nodes, and by node id + 1 for leaf nodes (TODO, in discussion with Ende,
/// break up this into decision_node_multiplicities and leaf_node_multiplicities).
pub struct CircuitizedSamples<F: FieldExt> {
    pub samples: Vec<Vec<InputAttribute<F>>>,                       // indexed by samples
    pub decision_paths: Vec<Vec<Vec<DecisionNode<F>>>>,             // indexed by trees, samples, steps in path
    pub attributes_on_paths: Vec<Vec<Vec<InputAttribute<F>>>>,      // indexed by trees, samples, steps in path
    pub differences: Vec<Vec<Vec<BinDecomp16Bit<F>>>>,              // indexed by trees, samples, steps in path
    pub path_ends: Vec<Vec<LeafNode<F>>>,                           // indexed by trees, samples
    pub attribute_multiplicities: Vec<Vec<Vec<BinDecomp4Bit<F>>>>,  // indexed by trees, samples, then by attribute index
    pub node_multiplicities: Vec<Vec<BinDecomp16Bit<F>>>,           // indexed by trees, tree nodes
}

/// Output of [`to_samples`], input to [`circuitize_samples`].
pub struct Samples {
    pub values: Vec<Vec<u16>>,
    pub sample_length: usize
}

/// Prepare the provided RawSamples for processing by the TreesModel by padding the raw sample
/// values such that the length of each individual sample is a power of two, and that the number of
/// samples is also a power of two.
/// Pre: raw_samples.values.len() > 0.
pub fn to_samples(raw_samples: &RawSamples) -> Samples {
    let mut samples: Vec<Vec<u16>> = vec![];
    let sample_length = next_power_of_two(raw_samples.values[0].len()).unwrap();
    for raw_sample in &raw_samples.values {
        let mut sample = raw_sample.clone();
        sample.resize(sample_length, 0);
        samples.push(sample);
    }
    let target_sample_count = next_power_of_two(raw_samples.values.len()).unwrap();
    for i in raw_samples.values.len()..target_sample_count {
        samples.push(vec![0_u16; sample_length]);
    }
    Samples {
        values: samples,
        sample_length
    }
}

/// Circuitize the provided batch of samples using the specified TreesModel instance,
/// returning a CircuitizedSamples instance.
/// See documentation of CircuitizedSamples.
/// The length of each sample (in `samples`) is
/// `next_power_of_two((trees_model.depth - 1) * values_array[0].len())`
/// Pre: `values_array.len() > 0`.
pub fn circuitize_samples<F: FieldExt>(
    samples_in: &Samples,
    trees_model: &TreesModel,
) -> CircuitizedSamples<F> {
    let mut samples: Vec<Vec<InputAttribute<F>>> = vec![];
    let mut attributes_on_paths: Vec<Vec<Vec<InputAttribute<F>>>> = vec![];
    let mut decision_paths: Vec<Vec<Vec<DecisionNode<F>>>> = vec![];
    let mut path_ends: Vec<Vec<LeafNode<F>>> = vec![];
    let mut differences: Vec<Vec<Vec<BinDecomp16Bit<F>>>> = vec![];
    let mut attribute_multiplicities: Vec<Vec<Vec<BinDecomp4Bit<F>>>> = vec![];
    let mut node_multiplicities: Vec<Vec<BinDecomp16Bit<F>>> = vec![];

    // convert the samples to field elements
    for values_row in &samples_in.values {
        samples.push(
            values_row
                .iter()
                .enumerate()
                .map(|(index, value)| InputAttribute {
                    attr_id: F::from(index as u64),
                    attr_val: F::from(*value as u64),
                })
                .collect(),
        );
    }

    for tree in &trees_model.trees {
        // initialize the node visit counts "node_multiplicities"
        let mut node_multiplicities_for_tree: Vec<u32> = vec![0_u32; 2_usize.pow(trees_model.depth as u32)];
        let mut attribute_multiplicities_for_tree: Vec<Vec<u32>> = vec![];
        let mut attributes_on_paths_for_tree: Vec<Vec<InputAttribute<F>>> = vec![];
        let mut decision_paths_for_tree: Vec<Vec<DecisionNode<F>>> = vec![];
        let mut path_ends_for_tree: Vec<LeafNode<F>> = vec![];
        let mut differences_for_tree: Vec<Vec<BinDecomp16Bit<F>>> = vec![];

        for (values_row, sample) in samples_in.values.iter().zip(samples.iter()) {
            // get the path
            let path = tree.get_path(values_row);

            // derive data from decision path
            let mut decision_path = vec![];
            let mut attributes_on_path_for_tree_and_sample: Vec<InputAttribute<F>> = vec![];
            let mut differences_for_tree_and_sample = vec![];
            let mut attribute_multiplicities_for_tree_and_sample = vec![0_u32; sample.len()];
            for node in &path[..path.len() - 1] {
                if let Node::Internal {
                    id,
                    feature_index,
                    threshold,
                    ..
                } = node
                {
                    decision_path.push(DecisionNode {
                        node_id: F::from(id.unwrap() as u64),
                        attr_id: F::from(*feature_index as u64),
                        threshold: F::from(*threshold as u64),
                    });
                    // calculate the bit decompositions of the differences
                    let difference = (values_row[*feature_index] as i32) - (*threshold as i32);
                    let bits = build_signed_bit_decomposition(difference, 16).unwrap();
                    differences_for_tree_and_sample.push(BinDecomp16Bit::<F>::from(bits));
                    // accumulate the node_multiplicities for this tree
                    node_multiplicities_for_tree[id.unwrap() as usize] += 1;
                    // accumulate the attribute multiplicities
                    attribute_multiplicities_for_tree_and_sample[*feature_index] += 1;
                    // build up the attributes on path
                    attributes_on_path_for_tree_and_sample.push(sample[*feature_index]);
                } else {
                    panic!("All Nodes in the path must be internal, except the last");
                }
            }

            attributes_on_paths_for_tree.push(attributes_on_path_for_tree_and_sample);
            decision_paths_for_tree.push(decision_path);
            differences_for_tree.push(differences_for_tree_and_sample);
            attribute_multiplicities_for_tree.push(attribute_multiplicities_for_tree_and_sample);

            // build the leaf node
            if let Node::Leaf { id, value } = path[path.len() - 1] {
                path_ends_for_tree.push(LeafNode {
                    node_id: F::from(id.unwrap() as u64),
                    node_val: i64_to_field(*value),
                });
                // accumulate multiplicity for leaf node
                // index using node id + 1 (since it's a leaf node)
                node_multiplicities_for_tree[id.unwrap() as usize + 1] += 1;
            } else {
                panic!("Last item in path should be a Node::Leaf");
            }
        }
        attributes_on_paths.push(attributes_on_paths_for_tree);
        decision_paths.push(decision_paths_for_tree);
        path_ends.push(path_ends_for_tree);
        differences.push(differences_for_tree);
        // calculate the bit decompositions of the attribute multiplicities
        attribute_multiplicities.push(
            attribute_multiplicities_for_tree
                .into_iter()
                .map(|mults| mults
                     .into_iter()
                     .map(|mult| build_unsigned_bit_decomposition(mult, 4).unwrap())
                     .map(BinDecomp4Bit::<F>::from)
                     .collect())
                .collect()
        );
        // calculate the bit decompositions of the node multiplicities
        node_multiplicities.push(
            node_multiplicities_for_tree
                .into_iter()
                .map(|mult| build_unsigned_bit_decomposition(mult, 16).unwrap())
                .map(BinDecomp16Bit::<F>::from)
                .collect(),
        );
    }

    CircuitizedSamples {
        samples,
        decision_paths,
        attributes_on_paths,
        differences,
        path_ends,
        attribute_multiplicities,
        node_multiplicities,
    }
}

impl<F: FieldExt> From<&TreesModel> for CircuitizedTrees<F> {
    /// Extract the DecisionNode and LeafNode instances from the TreesModel instance to
    /// obtain a CircuitizedTrees.
    fn from(trees_model: &TreesModel) -> Self {
        // extract, sort & pad the decision nodes
        let mut decision_nodes = vec![];
        let dummy_node = DecisionNode {
            node_id: F::from(2_u64.pow(trees_model.depth as u32) - 1),
            attr_id: F::from(0_u64),
            threshold: F::from(0_u64),
        };
        for tree in &trees_model.trees {
            let mut tree_decision_nodes = extract_decision_nodes(tree);
            tree_decision_nodes.sort_by_key(|node| node.node_id);
            // add a dummy node to make length a power of two
            tree_decision_nodes.push(dummy_node);
            decision_nodes.push(tree_decision_nodes);
        }
        // extract and sort the leaf nodes
        let mut leaf_nodes = vec![];
        for tree in &trees_model.trees {
            let mut tree_leaf_nodes = extract_leaf_nodes(tree);
            tree_leaf_nodes.sort_by_key(|node| node.node_id);
            leaf_nodes.push(tree_leaf_nodes);
        }

        CircuitizedTrees {
            decision_nodes,
            leaf_nodes,
            depth: trees_model.depth,
            scaling: trees_model.scaling,
        }
    }
}

impl From<&RawTreesModel> for TreesModel {
    /// Given a RawTreesModel object representing a decision tree model operating on u32 samples, prepare the model for circuitization:
    /// 1. scale and bias are folded into the leaf values;
    /// 2. leaf values are symmetrically quantized to i64;
    /// 3. all trees are padded such that they are all perfect and of uniform depth (without modifying
    ///    the predictions of any tree) where the uniform depth is chosen to be 2^l + 1 for minimal l >= 0;
    /// 4. ids are assigned to all nodes (as per assign_id());
    /// The resulting TreesModel incorporates all the tree instances, the (uniform) depth
    /// of the trees, and the scaling factor to approximately undo the quantization (via division)
    /// after aggregating the scores.
    fn from(raw_trees_model: &RawTreesModel) -> Self {
        let mut trees_f64 = raw_trees_model.trees.clone();
        // fold scale into all trees
        for tree in &mut trees_f64 {
            tree.transform_values(&|value| raw_trees_model.scale * value);
        }
        // fold bias into first tree
        trees_f64[0].transform_values(&|value| value + raw_trees_model.bias);

        // quantize the leaf values
        let (qtrees, rescaling) = quantize_leaf_values(&trees_f64);

        // pad the trees for perfection
        let max_depth = qtrees
            .iter()
            .map(|tree: &Node<i64>| tree.depth(std::cmp::max))
            .max()
            .unwrap();
        let target_depth = next_power_of_two(max_depth - 1).unwrap() + 1;
        // insert DecisionNodes where needed to perfect each tree
        let mut qtrees: Vec<Node<i64>> = qtrees
            .iter()
            .map(|tree: &Node<i64>| tree.perfect_to_depth(target_depth))
            .collect();
        // assign ids to all nodes
        for tree in &mut qtrees {
            tree.assign_id(0);
        }

        TreesModel {
            trees: qtrees,
            depth: target_depth,
            scaling: rescaling,
        }
    }
}

/// Generate a RawTreesModel as specified.  Meaning of arguments as per generate_trees().
/// Scale and bias are chosen randomly.
pub fn generate_raw_trees_model(
    n_trees: usize,
    target_depth: usize,
    n_features: usize,
    premature_leaf_proba: f64,
) -> RawTreesModel {
    let mut rng = rand::thread_rng();
    RawTreesModel {
        trees: (0..n_trees)
            .map(|_| generate_tree(target_depth, n_features, premature_leaf_proba))
            .collect(),
        bias: rng.gen(),
        scale: rng.gen(),
        n_features,
    }
}

/// Output of load_raw_samples, for conversion (using a TreesModel) to a Samples instance.
/// Values are less than or equal to [`SIGNED_DECOMPOSITION_MAX_ARG_ABS`] (for the benefit of
/// [`build_signed_bit_decomposition`]).
#[derive(Clone)]
pub struct RawSamples {
    pub values: Vec<Vec<u16>>,
    pub sample_length: usize
}

/// Generate an array of samples for input into the trees model.  For demonstration purposes.
pub fn generate_raw_samples(n_samples: usize, n_features: usize) -> RawSamples {
    let mut rng = rand::thread_rng();
    let values = (0..n_samples)
        .map(|_| {
            (0..n_features)
                .map(|_| rng.gen_range(0..(2_u32.pow(15) + 1) as u16))
                .collect()
        })
        .collect();
    RawSamples {
        values,
        sample_length: n_features
    }
}

/// Load a 2d array of samples from a `.npy` file.
pub fn load_raw_samples(filename: &Path) -> RawSamples {
    let input_arr: Array2<u16> = read_npy(filename).unwrap();
    RawSamples {
        values: input_arr.outer_iter().map(|row| row.to_vec()).collect(),
        sample_length: input_arr.shape()[1]
    }
}

/// Load a trees model from a JSON file.
/// WARNING: note the pre-condition.  No checks are performed.
/// Pre: all threshold values fit in a 16 bit signed bit decomposition.
pub fn load_raw_trees_model(filename: &Path) -> RawTreesModel {
    let file = File::open(filename).unwrap_or_else(|_| panic!("'{:?}' should be available.", filename.to_path_buf()));
    let raw_trees_model: RawTreesModel = serde_json::from_reader(file)
        .unwrap_or_else(|_| panic!("'{:?}' should be valid RawTreesModel JSON.", filename.to_path_buf()));
    raw_trees_model
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
    fn test_raw_trees_model_loading() {
        let filename = "src/zkdt/data_pipeline/test_qtrees.json";
        let _raw_trees_model = load_raw_trees_model(Path::new(filename));
    }

    #[test]
    fn test_numpy_loading() {
        let filename = "src/zkdt/data_pipeline/test_samples_10x6.npy";
        let raw_samples = load_raw_samples(Path::new(filename));
        assert_eq!(raw_samples.values.len(), 10);
    }

    #[test]
    fn test_to_samples() {
        let sample_length = 5;
        let values = vec![
            vec![0_u16; sample_length],
            vec![2_u16, 0_u16, 0_u16, 0_u16, 0_u16],
            vec![2_u16; sample_length],
        ];
        let raw_samples = RawSamples {
            values,
            sample_length
        };
        // let tree = build_small_tree();
        // let raw_trees_model = RawTreesModel {
        //     trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
        //     bias: 1.1,
        //     scale: 6.6,
        //     n_features: sample_length,
        // };
        // let trees_model: TreesModel = (&raw_trees_model).into();
        let samples = to_samples(&raw_samples);
        // check the number of samples
        assert_eq!(samples.sample_length, next_power_of_two(raw_samples.sample_length).unwrap());
        // check length of individual samples
        assert_eq!(samples.values.len(), next_power_of_two(raw_samples.values.len()).unwrap());
        for sample in &samples.values {
            assert_eq!(sample.len(), samples.sample_length);
        }
    }

    #[test]
    fn test_circuitize_samples() {
        let sample_length = 5;
        let values = vec![
            vec![0_u16; sample_length],
            vec![2_u16, 0_u16, 0_u16, 0_u16, 0_u16],
            vec![2_u16; sample_length],
        ];
        let raw_samples = RawSamples {
            values,
            sample_length
        };
        let tree = build_small_tree();
        let raw_trees_model = RawTreesModel {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: sample_length,
        };
        let trees_model: TreesModel = (&raw_trees_model).into();
        let samples = to_samples(&raw_samples);
        let csamples = circuitize_samples::<Fr>(&samples, &trees_model);
        // check size of outer dimensions
        let n_trees = raw_trees_model.trees.len();
        assert_eq!(csamples.samples.len(), samples.values.len());

        // check dimensions of permuted samples
        assert_eq!(csamples.attributes_on_paths.len(), n_trees);
        for attributes_on_paths_for_tree in &csamples.attributes_on_paths {
            assert_eq!(attributes_on_paths_for_tree.len(), samples.values.len());
            for attributes_on_path in attributes_on_paths_for_tree {
                assert_eq!(attributes_on_path.len(), trees_model.depth - 1);
            }
        }
        // check the contents of the permuted samples for the non-trivial tree (the 0th)
        let attributes_on_path = &csamples.attributes_on_paths[0][0];
        assert_eq!(attributes_on_path[0].attr_id, Fr::from(1));
        // ... sample travels left down the tree
        assert_eq!(attributes_on_path[1].attr_id, Fr::from(0));

        // check the dimension of the decision paths
        assert_eq!(csamples.decision_paths.len(), n_trees);
        for decision_paths_for_tree in &csamples.decision_paths {
            assert_eq!(decision_paths_for_tree.len(), samples.values.len());
            for decision_path in decision_paths_for_tree {
                assert_eq!(decision_path.len(), trees_model.depth - 1);
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
            assert_eq!(path_ends_for_tree.len(), samples.values.len());
        }
        // check the contents
        let path_ends_for_tree = &csamples.path_ends[0];
        assert_eq!(path_ends_for_tree[0].node_id, Fr::from(3));
        assert_eq!(path_ends_for_tree[1].node_id, Fr::from(4));

        // check the dimensions of the differences
        assert_eq!(csamples.differences.len(), n_trees);
        for differences_for_tree in &csamples.differences {
            assert_eq!(differences_for_tree.len(), samples.values.len());
            for differences in differences_for_tree {
                assert_eq!(differences.len(), trees_model.depth - 1);
            }
        }
        // check contents
        let differences = &csamples.differences[0][0];
        // should have the bit decomposition of -2
        assert_eq!(differences[1].bits[0], Fr::from(0));
        assert_eq!(differences[1].bits[1], Fr::from(1));
        assert_eq!(differences[1].bits[2], Fr::from(0));
        assert_eq!(differences[1].bits[15], Fr::from(1));

        // check the node_multiplicities
        let node_multiplicities = csamples.node_multiplicities;
        let n_nodes = 2_usize.pow(trees_model.depth as u32); // includes dummy node
        assert_eq!(node_multiplicities.len(), n_trees);
        for node_multiplicities_for_tree in &node_multiplicities {
            assert_eq!(node_multiplicities_for_tree.len(), n_nodes);
            // root node id has multiplicity samples.values.len() = 4 (not 3, since post padding!)
            let multiplicity = &node_multiplicities_for_tree[0];
            assert_eq!(multiplicity.bits[0], Fr::from(0));
            assert_eq!(multiplicity.bits[1], Fr::from(0));
            assert_eq!(multiplicity.bits[2], Fr::from(1));
            assert_eq!(multiplicity.bits[3], Fr::from(0));
            // dummy node id has multiplicity 0
            // TODO!(ende): because of the plus 1 above, changes here from `n_nodes - 1` to `n_nodes / 2 - 1`
            let multiplicity = &node_multiplicities_for_tree[n_nodes / 2 - 1];
            for bit in multiplicity.bits {
                assert_eq!(bit, Fr::from(0));
            }
        }
        // FIXME add a better test.
    }

    #[test]
    fn test_trees_model_from() {
        let tree = build_small_tree();
        let raw_trees_model = RawTreesModel {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: 11,
        };
        let trees_model: TreesModel = (&raw_trees_model).into();
        assert_eq!(trees_model.trees.len(), 2);
        assert_eq!(trees_model.depth, 3);
        // check trees are as claimed
        for tree in &trees_model.trees {
            assert_eq!(tree.depth(std::cmp::max), 3);
            assert!(tree.is_perfect());
            assert_eq!(tree.get_id().unwrap(), 0);
        }
    }

    #[test]
    fn test_circuitized_trees_from() {
        let tree = build_small_tree();
        let raw_trees_model = RawTreesModel {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: 11,
        };
        let trees_model: TreesModel = (&raw_trees_model).into();
        let ctrees: CircuitizedTrees<Fr> = (&trees_model).into();
        assert_eq!(ctrees.depth, trees_model.depth);
        // check decision nodes
        for tree_dns in &ctrees.decision_nodes {
            assert_eq!(tree_dns.len(), 4);
            for (node_id, node) in tree_dns.iter().take(tree_dns.len() - 1).enumerate() {
                assert_eq!(node.node_id, Fr::from(node_id as u64));
            }
            assert_eq!(tree_dns[tree_dns.len() - 1].node_id, Fr::from(7_u64));
        }
        // check leaf nodes
        for tree_lns in &ctrees.leaf_nodes {
            assert_eq!(tree_lns.len(), 4);
            for (node_id, node) in tree_lns.iter().enumerate() {
                assert_eq!(node.node_id, Fr::from((4 - 1) + node_id as u64));
            }
        }
        // accumulate score by taking the value of the first leaf node
        let mut acc_score: Fr = Fr::from(0);
        acc_score += ctrees.leaf_nodes[0][0].node_val;
        acc_score += ctrees.leaf_nodes[1][0].node_val;
        // check that the quantized scores accumulated as expected
        let expected_score = raw_trees_model.scale * (0.1 + 3.0) + raw_trees_model.bias;
        let quant_score = (expected_score * ctrees.scaling) as i64;
        let f_quant_score = if quant_score >= 0 {
            Fr::from(quant_score as u64)
        } else {
            -Fr::from(quant_score.unsigned_abs())
        };
        // just check that's it's close
        assert_eq!(f_quant_score, acc_score);
    }

    #[test]
    fn test_generate_raw_trees_model() {
        let target_depth = 3;
        let n_features = 6;
        let n_trees = 11;
        let raw_trees_model = generate_raw_trees_model(n_trees, target_depth, n_features, 0.5);
        assert_eq!(raw_trees_model.n_features, n_features);
        assert_eq!(raw_trees_model.trees.len(), n_trees);
        for tree in &raw_trees_model.trees {
            assert!(tree.depth(std::cmp::max) <= target_depth);
        }
    }

    #[test]
    fn test_generate_raw_samples() {
        let raw_samples = generate_raw_samples(10, 3);
        assert_eq!(raw_samples.values.len(), 10);
        for row in &raw_samples.values {
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn test_documentation() {
        // TODO remove once the doctests are being run
        let n_trees = 3;
        let n_features = 4;
        let depth = 6;
        // start with some random (probably not perfect) trees with f64 leaf values
        let raw_trees_model = generate_raw_trees_model(n_trees, depth, n_features, 0.5);
        // pad the trees, assign ids and quantize the leaf values
        let trees_model: TreesModel = (&raw_trees_model).into();
        // circuitize the trees (converts to DecisionNode<F>, LeafNode<F>)
        let _ctrees: CircuitizedTrees<Fr> = (&trees_model).into();
        // .. continued
        // generate some samples to play with
        let n_samples = 10;
        let raw_samples = generate_raw_samples(n_samples, n_features);
        let samples = to_samples(&raw_samples);
        // notice: circuitize_samples takes trees_model, not ctrees!
        let _csamples = circuitize_samples::<Fr>(&samples, &trees_model);
        // .. continued
        let _raw_trees_model: RawTreesModel = load_raw_trees_model(Path::new("src/zkdt/data_pipeline/test_qtrees.json"));
        let _raw_samples: RawSamples = load_raw_samples(Path::new("src/zkdt/data_pipeline/test_samples_10x6.npy"));
    }

    #[test]
    #[ignore]
    fn test_upshot_loading_and_circuitization() {
        // for this to work, those files need to be in place (not stored on the repo):
        // 1. remainder_prover/upshot_data/upshot-quantized-samples.npy
        // 2. remainder_prover/upshot_data/quantized-upshot-model.json
        let raw_trees_model: RawTreesModel = load_raw_trees_model(Path::new("upshot_data/quantized-upshot-model.json"));
        let mut raw_samples: RawSamples = load_raw_samples(Path::new("upshot_data/upshot-quantized-samples.npy"));
        // use just a small batch
        raw_samples.values = raw_samples.values[0..4].to_vec();

        let trees_model: TreesModel = (&raw_trees_model).into();
        let samples = to_samples(&raw_samples);

        let _ctrees: CircuitizedTrees<Fr> = (&trees_model).into();
        let _csamples = circuitize_samples::<Fr>(&samples, &trees_model);
    }
}
