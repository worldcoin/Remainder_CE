//! Functions for loading and converting ZKDT data into a form-factor
//! ready for input into the ZKDT circuit

use std::path::Path;

use ark_std::log2;
use itertools::Itertools;
use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::structs::{
    BinDecomp16BitMle, BinDecomp8BitMle, DecisionNodeMle, InputAttributeMle, LeafNodeMle,
};

use super::{
    data_pipeline::dt2zkdt::{
        circuitize_auxiliaries, load_raw_samples, load_raw_trees_model, CircuitizedSamples,
        CircuitizedTrees, RawSamples, RawTreesModel, Samples, TreesModel,
    },
    structs::{BinDecomp16Bit, BinDecomp8Bit, DecisionNode, InputAttribute, LeafNode},
};
use remainder::mle::circuit_mle::to_flat_mles;
use remainder_shared_types::layer::LayerId;

#[derive(Clone)]
pub struct BatchedZKDTCircuitMles<F: FieldExt> {
    pub input_samples_mle_vec: Vec<InputAttributeMle<F>>,
    pub permuted_input_samples_mle_vec: Vec<InputAttributeMle<F>>,
    pub decision_node_paths_mle_vec: Vec<DecisionNodeMle<F>>,
    pub leaf_node_paths_mle_vec: Vec<LeafNodeMle<F>>,
    pub binary_decomp_diffs_mle_vec: Vec<BinDecomp16BitMle<F>>,
    pub multiplicities_bin_decomp_mle_decision: BinDecomp16BitMle<F>,
    pub multiplicities_bin_decomp_mle_leaf: BinDecomp16BitMle<F>,
    pub decision_nodes_mle: DecisionNodeMle<F>,
    pub leaf_nodes_mle: LeafNodeMle<F>,
    pub multiplicities_bin_decomp_mle_input: Vec<BinDecomp8BitMle<F>>,
}

#[derive(Clone)]
pub struct BatchedZKDTCircuitMlesMultiTree<F: FieldExt> {
    pub input_samples_mle_vec: Vec<InputAttributeMle<F>>,
    pub permuted_input_samples_mle_vec_vec: Vec<Vec<InputAttributeMle<F>>>,
    pub decision_node_paths_mle_vec_vec: Vec<Vec<DecisionNodeMle<F>>>,
    pub leaf_node_paths_mle_vec_vec: Vec<Vec<LeafNodeMle<F>>>,
    pub binary_decomp_diffs_mle_vec_vec: Vec<Vec<BinDecomp16BitMle<F>>>,
    pub multiplicities_bin_decomp_mle_decision_vec: Vec<BinDecomp16BitMle<F>>,
    pub multiplicities_bin_decomp_mle_leaf_vec: Vec<BinDecomp16BitMle<F>>,
    pub decision_nodes_mle_vec: Vec<DecisionNodeMle<F>>,
    pub leaf_nodes_mle_vec: Vec<LeafNodeMle<F>>,
    pub multiplicities_bin_decomp_mle_input: Vec<BinDecomp8BitMle<F>>,
}

#[derive(Serialize, Deserialize)]
pub struct ZKDTCircuitData<F> {
    pub input_data: Vec<Vec<InputAttribute<F>>>, // Input attributes
    pub permuted_input_data: Vec<Vec<InputAttribute<F>>>, // Permuted input attributes
    pub decision_node_paths: Vec<Vec<DecisionNode<F>>>, // Paths (decision node part only)
    pub leaf_node_paths: Vec<LeafNode<F>>,       // Paths (leaf node part only)
    pub binary_decomp_diffs: Vec<Vec<BinDecomp16Bit<F>>>, // Binary decomp of differences
    pub multiplicities_bin_decomp_decision: Vec<BinDecomp16Bit<F>>, // Binary decomp of multiplicities for decision nodes
    pub multiplicities_bin_decomp_leaf: Vec<BinDecomp16Bit<F>>, // Binary decomp of multiplicities for leaf nodes
    pub decision_nodes: Vec<DecisionNode<F>>,                   // Actual tree decision nodes
    pub leaf_nodes: Vec<LeafNode<F>>,                           // Actual tree leaf nodes
    pub multiplicities_bin_decomp_input: Vec<Vec<BinDecomp8Bit<F>>>, // Binary decomp of multiplicities, of input
}

impl<F: FieldExt> ZKDTCircuitData<F> {
    /// Constructor
    pub fn new(
        input_data: Vec<Vec<InputAttribute<F>>>,
        permuted_input_data: Vec<Vec<InputAttribute<F>>>,
        decision_node_paths: Vec<Vec<DecisionNode<F>>>,
        leaf_node_paths: Vec<LeafNode<F>>,
        binary_decomp_diffs: Vec<Vec<BinDecomp16Bit<F>>>,
        multiplicities_bin_decomp_decision: Vec<BinDecomp16Bit<F>>,
        multiplicities_bin_decomp_leaf: Vec<BinDecomp16Bit<F>>,
        decision_nodes: Vec<DecisionNode<F>>,
        leaf_nodes: Vec<LeafNode<F>>,
        multiplicities_bin_decomp_input: Vec<Vec<BinDecomp8Bit<F>>>,
    ) -> ZKDTCircuitData<F> {
        ZKDTCircuitData {
            input_data,
            permuted_input_data,
            decision_node_paths,
            leaf_node_paths,
            binary_decomp_diffs,
            multiplicities_bin_decomp_decision,
            multiplicities_bin_decomp_leaf,
            decision_nodes,
            leaf_nodes,
            multiplicities_bin_decomp_input,
        }
    }
}

#[instrument(skip(zkdt_circuit_data))]
pub fn convert_zkdt_circuit_data_multi_tree_into_mles<F: FieldExt>(
    zkdt_circuit_data: Vec<ZKDTCircuitData<F>>,
) -> BatchedZKDTCircuitMlesMultiTree<F> {
    let (
        input_data_vec,
        permuted_input_data_vec,
        decision_node_paths_vec,
        leaf_node_paths_vec,
        binary_decomp_diffs_vec,
        multiplicities_bin_decomp_decision_vec,
        multiplicities_bin_decomp_leaf_vec,
        decision_nodes_vec,
        leaf_nodes_vec,
        multiplicities_bin_decomp_input_vec,
    ): (
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
    ) = zkdt_circuit_data
        .into_iter()
        .map(|zkdt_circuit_data| {
            (
                zkdt_circuit_data.input_data,
                zkdt_circuit_data.permuted_input_data,
                zkdt_circuit_data.decision_node_paths,
                zkdt_circuit_data.leaf_node_paths,
                zkdt_circuit_data.binary_decomp_diffs,
                zkdt_circuit_data.multiplicities_bin_decomp_decision,
                zkdt_circuit_data.multiplicities_bin_decomp_leaf,
                zkdt_circuit_data.decision_nodes,
                zkdt_circuit_data.leaf_nodes,
                zkdt_circuit_data.multiplicities_bin_decomp_input,
            )
        })
        .multiunzip();

    // --- Generate MLEs for each ---

    let multiplicities_bin_decomp_mle_decision_vec = multiplicities_bin_decomp_decision_vec
        .into_iter()
        .map(|multiplicities_bin_decomp_decision| {
            let multiplicities_bin_decomp_decision = multiplicities_bin_decomp_decision
                .into_iter()
                .map(|x| x.bits)
                .collect_vec();
            let multiplicities_bin_decomp_decision =
                to_flat_mles(multiplicities_bin_decomp_decision);

            BinDecomp16BitMle::<F>::new_from_raw(
                multiplicities_bin_decomp_decision,
                LayerId::Input(0),
                None,
            )
        })
        .collect_vec();

    let multiplicities_bin_decomp_mle_leaf_vec = multiplicities_bin_decomp_leaf_vec
        .into_iter()
        .map(|multiplicities_bin_decomp_leaf| {
            let multiplicities_bin_decomp_leaf = multiplicities_bin_decomp_leaf
                .into_iter()
                .map(|x| x.bits)
                .collect_vec();
            let multiplicities_bin_decomp_leaf = to_flat_mles(multiplicities_bin_decomp_leaf);

            BinDecomp16BitMle::<F>::new_from_raw(
                multiplicities_bin_decomp_leaf,
                LayerId::Input(0),
                None,
            )
        })
        .collect_vec();

    let input_samples_mle_vec: Vec<InputAttributeMle<F>> = input_data_vec[0]
        .clone()
        .into_iter()
        .map(|input| {
            let input = input
                .into_iter()
                .map(|x| [x.attr_id, x.attr_val])
                .collect_vec();
            let input = to_flat_mles(input);
            InputAttributeMle::<F>::new_from_raw(input, LayerId::Input(0), None)
        })
        .collect_vec();

    let permuted_input_samples_mle_vec_vec: Vec<Vec<InputAttributeMle<F>>> =
        permuted_input_data_vec
            .into_iter()
            .map(|permuted_input_data| {
                permuted_input_data
                    .iter()
                    .map(|datum| {
                        let datum = datum
                            .into_iter()
                            .map(|x| [x.attr_id, x.attr_val])
                            .collect_vec();
                        let datum = to_flat_mles(datum);
                        InputAttributeMle::<F>::new_from_raw(datum, LayerId::Input(0), None)
                    })
                    .collect()
            })
            .collect_vec();

    let decision_node_paths_mle_vec_vec: Vec<Vec<DecisionNodeMle<F>>> = decision_node_paths_vec
        .into_iter()
        .map(|decision_node_paths| {
            decision_node_paths
                .iter()
                .map(|path| {
                    let path = path
                        .into_iter()
                        .map(|x| [x.node_id, x.attr_id, x.threshold])
                        .collect_vec();
                    let path = to_flat_mles(path);
                    DecisionNodeMle::<F>::new_from_raw(path, LayerId::Input(0), None)
                })
                .collect()
        })
        .collect_vec();

    let leaf_node_paths_mle_vec_vec: Vec<Vec<LeafNodeMle<F>>> = leaf_node_paths_vec
        .into_iter()
        .map(|leaf_node_paths| {
            leaf_node_paths
                .into_iter()
                .map(|path| {
                    let path = [path]
                        .into_iter()
                        .map(|x| [x.node_id, x.node_val])
                        .collect_vec();
                    let path = to_flat_mles(path);
                    LeafNodeMle::<F>::new_from_raw(path, LayerId::Input(0), None)
                })
                .collect()
        })
        .collect_vec();

    let binary_decomp_diffs_mle_vec_vec: Vec<Vec<BinDecomp16BitMle<F>>> = binary_decomp_diffs_vec
        .into_iter()
        .map(|binary_decomp_diffs| {
            binary_decomp_diffs
                .iter()
                .map(|binary_decomp_diff| {
                    let binary_decomp_diff = binary_decomp_diff
                        .into_iter()
                        .map(|bin_decomp| bin_decomp.bits)
                        .collect_vec();

                    // converts a [Vec<F; N>] into a [Vec<F>; N]
                    let binary_decomp_diff = to_flat_mles(binary_decomp_diff);

                    BinDecomp16BitMle::<F>::new_from_raw(
                        binary_decomp_diff,
                        LayerId::Input(0),
                        None,
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    let decision_nodes_mle_vec = decision_nodes_vec
        .into_iter()
        .map(|decision_nodes| {
            let decision_nodes = decision_nodes
                .into_iter()
                .map(|x| [x.node_id, x.attr_id, x.threshold])
                .collect_vec();
            let decision_nodes = to_flat_mles(decision_nodes);
            DecisionNodeMle::<F>::new_from_raw(decision_nodes, LayerId::Input(0), None)
        })
        .collect_vec();

    let leaf_nodes_mle_vec = leaf_nodes_vec
        .into_iter()
        .map(|leaf_nodes| {
            let leaf_nodes = leaf_nodes
                .into_iter()
                .map(|x| [x.node_id, x.node_val])
                .collect_vec();
            let leaf_nodes = to_flat_mles(leaf_nodes);
            LeafNodeMle::<F>::new_from_raw(leaf_nodes, LayerId::Input(0), None)
        })
        .collect_vec();

    let multiplicities_bin_decomp_mle_input = multiplicities_bin_decomp_input_vec[0]
        .iter()
        .map(|datum| {
            let datum = datum
                .into_iter()
                .map(|bin_decomp| bin_decomp.bits)
                .collect_vec();

            // converts a [Vec<F; N>] into a [Vec<F>; N]
            let datum = to_flat_mles(datum);

            BinDecomp8BitMle::<F>::new_from_raw(datum, LayerId::Input(0), None)
        })
        .collect_vec();

    BatchedZKDTCircuitMlesMultiTree {
        input_samples_mle_vec,
        permuted_input_samples_mle_vec_vec,
        decision_node_paths_mle_vec_vec,
        leaf_node_paths_mle_vec_vec,
        binary_decomp_diffs_mle_vec_vec,
        multiplicities_bin_decomp_mle_decision_vec,
        multiplicities_bin_decomp_mle_leaf_vec,
        decision_nodes_mle_vec,
        leaf_nodes_mle_vec,
        multiplicities_bin_decomp_mle_input,
    }
}

/// Takes the output from presumably something like [`read_upshot_data_single_tree_branch_from_filepath`]
/// and converts it into `BatchedCatboostMles<F>`, i.e. the input to the circuit.
#[instrument(skip(zkdt_circuit_data))]
pub fn convert_zkdt_circuit_data_into_mles<F: FieldExt>(
    zkdt_circuit_data: ZKDTCircuitData<F>,
    tree_height: usize,
    input_len: usize,
) -> BatchedZKDTCircuitMles<F> {
    // --- Unpacking ---
    let ZKDTCircuitData {
        input_data,
        permuted_input_data,
        decision_node_paths,
        leaf_node_paths,
        binary_decomp_diffs,
        multiplicities_bin_decomp_decision,
        multiplicities_bin_decomp_leaf,
        decision_nodes,
        leaf_nodes,
        multiplicities_bin_decomp_input,
    } = zkdt_circuit_data;

    // --- Generate MLEs for each ---
    let input_samples_mle_vec = input_data
        .into_iter()
        .map(|input| {
            let input = input
                .into_iter()
                .map(|x| [x.attr_id, x.attr_val])
                .collect_vec();
            let input = to_flat_mles(input);
            InputAttributeMle::<F>::new_from_raw(input, LayerId::Input(0), None)
        })
        .collect_vec();
    let permuted_input_samples_mle_vec = permuted_input_data
        .iter()
        .map(|datum| {
            let datum = datum
                .into_iter()
                .map(|x| [x.attr_id, x.attr_val])
                .collect_vec();
            let datum = to_flat_mles(datum);
            InputAttributeMle::<F>::new_from_raw(datum, LayerId::Input(0), None)
        })
        .collect();
    let decision_node_paths_mle_vec: Vec<DecisionNodeMle<F>> = decision_node_paths
        .iter()
        .map(|path| {
            let path = path
                .into_iter()
                .map(|x| [x.node_id, x.attr_id, x.threshold])
                .collect_vec();
            let path = to_flat_mles(path);
            DecisionNodeMle::<F>::new_from_raw(path, LayerId::Input(0), None)
        })
        .collect();
    let leaf_node_paths_mle_vec = leaf_node_paths
        .into_iter()
        .map(|path| {
            let path = [path]
                .into_iter()
                .map(|x| [x.node_id, x.node_val])
                .collect_vec();
            let path = to_flat_mles(path);
            LeafNodeMle::<F>::new_from_raw(path, LayerId::Input(0), None)
        })
        .collect();
    let binary_decomp_diffs_mle_vec = binary_decomp_diffs
        .iter()
        .map(|binary_decomp_diff| {
            let binary_decomp_diff = binary_decomp_diff
                .into_iter()
                .map(|bin_decomp| bin_decomp.bits)
                .collect_vec();

            // converts a [Vec<F; N>] into a [Vec<F>; N]
            let binary_decomp_diff = to_flat_mles(binary_decomp_diff);

            BinDecomp16BitMle::<F>::new_from_raw(binary_decomp_diff, LayerId::Input(0), None)
        })
        .collect_vec();

    let multiplicities_bin_decomp_decision = multiplicities_bin_decomp_decision
        .into_iter()
        .map(|bin_decomp| bin_decomp.bits)
        .collect_vec();
    let multiplicities_bin_decomp_mle_decision = BinDecomp16BitMle::<F>::new_from_raw(
        to_flat_mles(multiplicities_bin_decomp_decision),
        LayerId::Input(0),
        None,
    );

    let multiplicities_bin_decomp_leaf = multiplicities_bin_decomp_leaf
        .into_iter()
        .map(|bin_decomp| bin_decomp.bits)
        .collect_vec();
    let multiplicities_bin_decomp_mle_leaf = BinDecomp16BitMle::<F>::new_from_raw(
        to_flat_mles(multiplicities_bin_decomp_leaf),
        LayerId::Input(0),
        None,
    );

    let decision_nodes = decision_nodes
        .into_iter()
        .map(|x| [x.node_id, x.attr_id, x.threshold])
        .collect_vec();
    let decision_nodes = to_flat_mles(decision_nodes);

    let decision_nodes_mle =
        DecisionNodeMle::<F>::new_from_raw(decision_nodes, LayerId::Input(0), None);

    let leaf_nodes = leaf_nodes
        .into_iter()
        .map(|x| [x.node_id, x.node_val])
        .collect_vec();
    let leaf_nodes = to_flat_mles(leaf_nodes);
    let leaf_nodes_mle = LeafNodeMle::<F>::new_from_raw(leaf_nodes, LayerId::Input(0), None);

    let multiplicities_bin_decomp_mle_input = multiplicities_bin_decomp_input
        .iter()
        .map(|datum| {
            let datum = datum
                .into_iter()
                .map(|bin_decomp| bin_decomp.bits)
                .collect_vec();

            // converts a [Vec<F; N>] into a [Vec<F>; N]
            let datum = to_flat_mles(datum);

            BinDecomp8BitMle::<F>::new_from_raw(datum, LayerId::Input(0), None)
        })
        .collect_vec();

    BatchedZKDTCircuitMles {
        input_samples_mle_vec,
        permuted_input_samples_mle_vec,
        decision_node_paths_mle_vec,
        leaf_node_paths_mle_vec,
        binary_decomp_diffs_mle_vec,
        multiplicities_bin_decomp_mle_decision,
        multiplicities_bin_decomp_mle_leaf,
        decision_nodes_mle,
        leaf_nodes_mle,
        multiplicities_bin_decomp_mle_input,
    }
}

/// Specifies exactly which minibatch to use within a sample.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MinibatchData {
    /// log_2 of the minibatch size
    pub log_sample_minibatch_size: usize,
    /// Minibatch index within the bigger batch
    pub sample_minibatch_number: usize,
    /// Tree batch size (notice it's not log_2)
    pub tree_batch_size: usize,
    /// the index of the tree batch
    pub tree_batch_number: usize,
}

/// Gives all batched data associated with tree number `tree_idx`.
///
/// ## Arguments
/// * `maybe_minibatch_data` - The minibatch to grab data for, including minibatch size and index.
/// * `tree_idx` - The tree number we are generating witnesses for.
/// * `raw_trees_model_path` - Path to the JSON file representing the quantized version of the model
///     (as output by the Python preprocessing)
/// * `raw_samples_path` - Path to the NumPy file representing the quantized version of the samples
///     (again, as output by the Python preprocessing)
///
/// ## Returns
/// * `zkdt_circuit_data` - Data which is ready to be thrown into circuit
/// * `tree_height` - Length of every path within the given tree
/// * `input_len` - Padded number of features within each input
/// * `log_minibatch_size` - log_2 of the size of the minibatch, i.e. number of inputs being loaded.
/// * `minibatch_number` - Minibatch index we are processing
///
/// ## Notes
/// Note that `raw_samples.values.len()` is currently 4573! This means we can go
/// up to 8192 (padded) in terms of batch sizes which are powers of 2
///
/// ## TODOs
/// * Throw an error if `sample_minibatch_number` causes us to go out of bounds!
#[instrument]
pub fn load_upshot_data_single_tree_batch<F: FieldExt>(
    maybe_minibatch_data: Option<MinibatchData>,
    tree_idx: usize,
    raw_trees_model_path: &Path,
    raw_samples_path: &Path,
) -> ZKDTCircuitData<F> {
    // --- Grab trees + raw samples ---
    let raw_trees_model: RawTreesModel = load_raw_trees_model(raw_trees_model_path);
    let mut raw_samples: RawSamples = load_raw_samples(raw_samples_path);

    // --- Grab sample minibatch ---
    let minibatch_data = match maybe_minibatch_data {
        Some(param_minibatch_data) => param_minibatch_data,
        None => MinibatchData {
            log_sample_minibatch_size: log2(raw_samples.values.len() as usize) as usize,
            sample_minibatch_number: 0,
            tree_batch_size: log2(raw_trees_model.trees.len()) as usize,
            tree_batch_number: 0,
        },
    };
    let sample_minibatch_size = 2_usize.pow(minibatch_data.log_sample_minibatch_size as u32);
    let minibatch_start_idx = minibatch_data.sample_minibatch_number * sample_minibatch_size;
    raw_samples.values = raw_samples.values
        [minibatch_start_idx..(minibatch_start_idx + sample_minibatch_size)]
        .to_vec();

    // --- Conversions ---
    let full_trees_model: TreesModel = (&raw_trees_model).into();
    let single_tree = full_trees_model.slice(tree_idx..tree_idx + 1);
    let samples: Samples = (&raw_samples).into();
    let ctrees: CircuitizedTrees<F> = (&single_tree).into();

    // --- Compute actual witnesses ---
    let csamples: CircuitizedSamples<F> = (&samples).into();
    let caux = circuitize_auxiliaries(&samples, &single_tree);
    let tree_height = ctrees.depth;
    let decision_len = 2_usize.pow(tree_height as u32 - 1);

    // --- Sanitycheck ---
    debug_assert_eq!(caux.attributes_on_paths.len(), 1);
    debug_assert_eq!(caux.decision_paths.len(), 1);
    debug_assert_eq!(caux.path_ends.len(), 1);
    debug_assert_eq!(caux.differences.len(), 1);
    debug_assert_eq!(caux.node_multiplicities.len(), 1);
    debug_assert_eq!(ctrees.decision_nodes.len(), 1);
    debug_assert_eq!(ctrees.leaf_nodes.len(), 1);
    debug_assert_eq!(caux.attribute_multiplicities.len(), 1);

    let mut multiplicities_bin_decomp = caux.node_multiplicities[0].clone();
    let multiplicities_bin_decomp_leaf = multiplicities_bin_decomp.split_off(decision_len);
    let multiplicities_bin_decomp_decision = multiplicities_bin_decomp;

    // --- Grab only the slice of witnesses which are relevant to the target `tree_number` ---
    ZKDTCircuitData::new(
        csamples,
        caux.attributes_on_paths[0].clone(),
        caux.decision_paths[0].clone(),
        caux.path_ends[0].clone(),
        caux.differences[0].clone(),
        multiplicities_bin_decomp_decision,
        multiplicities_bin_decomp_leaf,
        ctrees.decision_nodes[0].clone(),
        ctrees.leaf_nodes[0].clone(),
        caux.attribute_multiplicities_per_sample.clone(),
    )
}

#[instrument]
pub fn load_upshot_data_multi_tree_batch<F: FieldExt>(
    maybe_minibatch_data: Option<MinibatchData>,
    raw_trees_model_path: &Path,
    raw_samples_path: &Path,
) -> Vec<ZKDTCircuitData<F>> {
    // --- Grab trees + raw samples ---
    let raw_trees_model: RawTreesModel = load_raw_trees_model(raw_trees_model_path);
    let mut raw_samples: RawSamples = load_raw_samples(raw_samples_path);

    // --- Grab sample minibatch ---
    let minibatch_data = match maybe_minibatch_data {
        Some(param_minibatch_data) => param_minibatch_data,
        None => MinibatchData {
            log_sample_minibatch_size: log2(raw_samples.values.len() as usize) as usize,
            sample_minibatch_number: 0,
            tree_batch_size: log2(raw_trees_model.trees.len()) as usize,
            tree_batch_number: 0,
        },
    };

    // --- Do the truncation if needed ---
    let sample_minibatch_size = 2_usize.pow(minibatch_data.log_sample_minibatch_size as u32);
    if sample_minibatch_size < raw_samples.values.len() {
        let minibatch_start_idx = minibatch_data.sample_minibatch_number * sample_minibatch_size;
        raw_samples.values = raw_samples.values
            [minibatch_start_idx..(minibatch_start_idx + sample_minibatch_size)]
            .to_vec();
    }

    let tree_batch_size = minibatch_data.tree_batch_size;
    let tree_batch_number = minibatch_data.tree_batch_number;

    // --- Conversions ---
    let full_trees_model: TreesModel = (&raw_trees_model).into();
    let mut tree_batch = full_trees_model
        .slice((tree_batch_size * tree_batch_number)..(tree_batch_size * (tree_batch_number + 1)));
    tree_batch.pad_tree_count_to_multiple(tree_batch_size);
    // let samples: Samples = (&raw_samples).into();
    let samples: Samples =
        raw_samples.to_samples_with_target_larger_minibatch_size(sample_minibatch_size);
    let ctrees: CircuitizedTrees<F> = (&tree_batch).into();

    // --- Compute actual witnesses ---
    let csamples: CircuitizedSamples<F> = (&samples).into();
    let caux = circuitize_auxiliaries(&samples, &tree_batch);
    let tree_height = ctrees.depth;
    let decision_len = 2_usize.pow(tree_height as u32 - 1);

    // --- Sanitycheck ---
    debug_assert_eq!(caux.attributes_on_paths.len(), tree_batch_size);
    debug_assert_eq!(caux.decision_paths.len(), tree_batch_size);
    debug_assert_eq!(caux.path_ends.len(), tree_batch_size);
    debug_assert_eq!(caux.differences.len(), tree_batch_size);
    debug_assert_eq!(caux.node_multiplicities.len(), tree_batch_size);
    debug_assert_eq!(ctrees.decision_nodes.len(), tree_batch_size);
    debug_assert_eq!(ctrees.leaf_nodes.len(), tree_batch_size);
    // debug_assert_eq!(caux.attribute_multiplicities.len(), tree_batch_number);

    // --- Grab only the slice of witnesses which are relevant to the target `tree_number` ---

    let circuitdata_vec = caux
        .attributes_on_paths
        .clone()
        .into_iter()
        .enumerate()
        .map(|(idx, _)| {
            let mut multiplicities_bin_decomp = caux.node_multiplicities[idx].clone();
            let multiplicities_bin_decomp_leaf = multiplicities_bin_decomp.split_off(decision_len);
            let multiplicities_bin_decomp_decision = multiplicities_bin_decomp;

            ZKDTCircuitData::new(
                csamples.clone(),
                caux.attributes_on_paths[idx].clone(),
                caux.decision_paths[idx].clone(),
                caux.path_ends[idx].clone(),
                caux.differences[idx].clone(),
                multiplicities_bin_decomp_decision,
                multiplicities_bin_decomp_leaf,
                ctrees.decision_nodes[idx].clone(),
                ctrees.leaf_nodes[idx].clone(),
                caux.attribute_multiplicities_per_sample.clone(),
            )
        })
        .collect_vec();
    circuitdata_vec
}
