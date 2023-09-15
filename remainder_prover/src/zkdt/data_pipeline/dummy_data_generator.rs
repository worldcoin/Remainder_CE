use crate::layer::LayerId;
use crate::mle::MleRef;
use crate::mle::dense::DenseMle;
use crate::utils::file_exists;
use crate::zkdt::constants::get_cached_batched_mles_filename_with_exp_size;
use crate::zkdt::data_pipeline::dt2zkdt::generate_upshot_data_all_batch_sizes;
use remainder_shared_types::FieldExt;
use serde::{Serialize, Deserialize};
use serde_json::{to_writer, from_reader};

use super::super::constants::CACHED_BATCHED_MLES_FILE;
use super::dt2zkdt::load_upshot_data_single_tree_batch;
use super::super::structs::*;

use ark_std::test_rng;
use itertools::{repeat_n, Itertools};
use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::iter::zip;
use std::path::Path;

/*
What's our plan here?
- First create some dummy data
- Then create MLEs and MLERefs using that dummy data
- Then create expressions over those which are representative
- Hmm sounds like we might need to do intermediate stuff
*/

// --- Constants ---
pub const DUMMY_INPUT_LEN: usize = 1 << 6; // was 1 << 5
pub const NUM_DUMMY_INPUTS: usize = 8;
pub const TREE_HEIGHT: usize = 9; // was 9 // was 8
const NUM_DECISION_NODES: u64 = 2_u64.pow(TREE_HEIGHT as u32 - 1) - 1;
const NUM_LEAF_NODES: u64 = NUM_DECISION_NODES + 1;

#[derive(Debug, Clone)]
struct PathAndPermutation<F: FieldExt> {
    path_decision_nodes: Vec<DecisionNode<F>>,
    ret_leaf_node: LeafNode<F>,
    diffs: Vec<F>,
    used_input_attributes: Vec<InputAttribute<F>>,
}

/// - First element is the decision path nodes
/// - Second element is the final predicted leaf node
/// - Third element is the difference between the current attribute value and the node threshold
/// - Fourth element is the input attributes which were used during inference
/// - Third element is the attributes used (deprecated)
fn generate_correct_path_and_permutation<F: FieldExt>(
    decision_nodes: &[DecisionNode<F>],
    leaf_nodes: &[LeafNode<F>],
    input_datum: &[InputAttribute<F>],
) -> PathAndPermutation<F> {
    // --- Keep track of the path and permutation ---
    let mut path_decision_nodes: Vec<DecisionNode<F>> = vec![];
    let mut used_input_attributes: Vec<InputAttribute<F>> = vec![];
    // let mut permuted_access_indices: Vec<F> = vec![];

    // --- Keep track of how many times each attribute ID was used (to get the corresponding attribute idx) ---
    // Key: Attribute ID
    // Value: Multiplicity
    let mut attr_num_hits: HashMap<F, u64> = HashMap::new();

    // --- Keep track of the differences for path nodes ---
    let mut diffs: Vec<F> = vec![];

    // --- Go through the decision nodes ---
    let mut current_node_idx = 0;
    while current_node_idx < decision_nodes.len() {
        // --- Add to path; grab the appropriate attribute index ---
        path_decision_nodes.push(decision_nodes[current_node_idx]);
        let attr_id = decision_nodes[current_node_idx].attr_id.get_lower_128() as usize;

        // --- Stores the current input attribute which is being used ---
        used_input_attributes.push(input_datum[attr_id]);

        // --- Assume that repeats are basically layered one after another ---
        // let num_repeats = attr_num_hits.get(&decision_nodes[current_node_idx].attr_id).unwrap_or(&0);
        // let offset = *num_repeats * (DUMMY_INPUT_LEN as u64);
        // permuted_access_indices.push(decision_nodes[current_node_idx].attr_id + F::from(offset));

        // --- Adds a hit to the current attribute ID ---
        let num_attr_id_hits = attr_num_hits
            .entry(decision_nodes[current_node_idx].attr_id)
            .or_insert(0);
        *num_attr_id_hits += 1;

        // --- Compute the difference ---
        let diff = input_datum[attr_id].attr_val - decision_nodes[current_node_idx].threshold;
        diffs.push(diff);

        // --- Check if we should go left or right ---
        if input_datum[attr_id].attr_val > decision_nodes[current_node_idx].threshold {
            current_node_idx = current_node_idx * 2 + 2;
        } else {
            current_node_idx = current_node_idx * 2 + 1;
        }
    }

    // --- Leaf node indices are offset by 2^{TREE_HEIGHT} ---
    let ret_leaf_node = leaf_nodes[current_node_idx - NUM_DECISION_NODES as usize];

    // assert!(path_decision_nodes.len() == TREE_HEIGHT - 1);

    // (path_decision_nodes, ret_leaf_node, permuted_access_indices, diffs)
    PathAndPermutation {
        path_decision_nodes,
        ret_leaf_node,
        diffs,
        used_input_attributes,
    }
}

fn generate_16_bit_signed_decomp<F: FieldExt>(value: F) -> BinDecomp16Bit<F> {
    let upper_bound = F::from(2_u64.pow(16) - 1);

    // --- Compute the sign bit ---
    let sign_bit = if value <= upper_bound {
        F::zero()
    } else {
        F::one()
    };

    // --- Convert to positive ---
    let abs_value = if value <= upper_bound {
        value
    } else {
        value.neg()
    };

    // --- Grab the unsigned representation... ---
    let mut unsigned_bit_decomp = generate_16_bit_unsigned_decomp(abs_value);

    // --- The first bit should be zero (i.e. unsigned version is decomposable in 15 bits) ---
    assert!(unsigned_bit_decomp.bits[0] == F::zero());

    // --- Set the sign bit in the first slot ---
    unsigned_bit_decomp.bits[0] = sign_bit;
    unsigned_bit_decomp
}

fn generate_16_bit_unsigned_decomp<F: FieldExt>(value: F) -> BinDecomp16Bit<F> {
    // --- Ensure we can decompose in (positive) 16 bits ---
    let upper_bound = F::from(2_u64.pow(16) - 1);
    assert!(value >= F::zero());
    assert!(value <= upper_bound);

    // --- Grab the string repr ---
    let binary_repr = format!("{:0>16b}", value.get_lower_128());

    // --- Length must be 16, then parse as array of length 16 ---
    let mut binary_repr_arr = [F::zero(); 16];
    for (idx, item) in binary_repr_arr.iter_mut().enumerate() {
        let char_repr = binary_repr.chars().nth(idx).unwrap();
        debug_assert!(char_repr == '0' || char_repr == '1');
        *item = if char_repr == '0' {
            F::zero()
        } else {
            F::one()
        }
    }

    BinDecomp16Bit {
        bits: binary_repr_arr,
    }
}

/// dummydata input form factor for circuit inputs
#[derive(Serialize, Deserialize)]
pub struct ZKDTDummyCircuitData<F> {
    dummy_input_data: Vec<Vec<InputAttribute<F>>>, // Input attributes
    dummy_permuted_input_data: Vec<Vec<InputAttribute<F>>>, // Permuted input attributes
    dummy_decision_node_paths: Vec<Vec<DecisionNode<F>>>, // Paths (decision node part only)
    dummy_leaf_node_paths: Vec<LeafNode<F>>,       // Paths (leaf node part only)
    dummy_binary_decomp_diffs: Vec<Vec<BinDecomp16Bit<F>>>, // Binary decomp of differences
    dummy_multiplicities_bin_decomp: Vec<BinDecomp16Bit<F>>, // Binary decomp of multiplicities
    dummy_decision_nodes: Vec<DecisionNode<F>>,    // Actual tree decision nodes
    dummy_leaf_nodes: Vec<LeafNode<F>>,            // Actual tree leaf nodes
}

impl<F: FieldExt> ZKDTDummyCircuitData<F> {
    /// creates new dummydata
    pub fn new(
        dummy_input_data: Vec<Vec<InputAttribute<F>>>,
        dummy_permuted_input_data: Vec<Vec<InputAttribute<F>>>,
        dummy_decision_node_paths: Vec<Vec<DecisionNode<F>>>,
        dummy_leaf_node_paths: Vec<LeafNode<F>>,
        dummy_binary_decomp_diffs: Vec<Vec<BinDecomp16Bit<F>>>,
        dummy_multiplicities_bin_decomp: Vec<BinDecomp16Bit<F>>,
        dummy_decision_nodes: Vec<DecisionNode<F>>,
        dummy_leaf_nodes: Vec<LeafNode<F>>,
    ) -> ZKDTDummyCircuitData<F> {
        ZKDTDummyCircuitData {
            dummy_input_data,
            dummy_permuted_input_data,
            dummy_decision_node_paths,
            dummy_leaf_node_paths,
            dummy_binary_decomp_diffs,
            dummy_multiplicities_bin_decomp,
            dummy_decision_nodes,
            dummy_leaf_nodes,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ZKDTCircuitData<F> {
    input_data: Vec<Vec<InputAttribute<F>>>, // Input attributes
    permuted_input_data: Vec<Vec<InputAttribute<F>>>, // Permuted input attributes
    decision_node_paths: Vec<Vec<DecisionNode<F>>>, // Paths (decision node part only)
    leaf_node_paths: Vec<LeafNode<F>>,       // Paths (leaf node part only)
    binary_decomp_diffs: Vec<Vec<BinDecomp16Bit<F>>>, // Binary decomp of differences
    multiplicities_bin_decomp: Vec<BinDecomp16Bit<F>>, // Binary decomp of multiplicities
    decision_nodes: Vec<DecisionNode<F>>,    // Actual tree decision nodes
    leaf_nodes: Vec<LeafNode<F>>,            // Actual tree leaf nodes
    multiplicities_bin_decomp_input: Vec<Vec<BinDecomp4Bit<F>>>, // Binary decomp of multiplicities, of input
}

impl<F: FieldExt> ZKDTCircuitData<F> {
    /// creates new dummydata
    pub fn new(
        input_data: Vec<Vec<InputAttribute<F>>>,
        permuted_input_data: Vec<Vec<InputAttribute<F>>>,
        decision_node_paths: Vec<Vec<DecisionNode<F>>>,
        leaf_node_paths: Vec<LeafNode<F>>,
        binary_decomp_diffs: Vec<Vec<BinDecomp16Bit<F>>>,
        multiplicities_bin_decomp: Vec<BinDecomp16Bit<F>>,
        decision_nodes: Vec<DecisionNode<F>>,
        leaf_nodes: Vec<LeafNode<F>>,
        multiplicities_bin_decomp_input: Vec<Vec<BinDecomp4Bit<F>>>,
    ) -> ZKDTCircuitData<F> {
        ZKDTCircuitData {
            input_data,
            permuted_input_data,
            decision_node_paths,
            leaf_node_paths,
            binary_decomp_diffs,
            multiplicities_bin_decomp,
            decision_nodes,
            leaf_nodes,
            multiplicities_bin_decomp_input
        }
    }
}

/// Need to generate dummy circuit inputs, starting with the input data
/// Then get the path data and binary decomp stuff
/// TODO!(ryancao): add the attribute index field to `InputAttribute<F>`
/// -- Actually, scratch the above: we might be getting rid of `attr_id`s
/// altogether and replacing with `attr_idx` everywhere (as suggested by Ben!)
fn generate_dummy_data<F: FieldExt>() -> ZKDTDummyCircuitData<F> {
    // --- Get the RNG ---
    let mut rng = test_rng();

    // --- Generate dummy input data ---
    let mut dummy_input_data: Vec<Vec<InputAttribute<F>>> = vec![];
    // let mut dummy_permuted_input_data: Vec<Vec<InputAttribute<F>>> = vec![];

    // --- Generate dummy input index data ---
    let dummy_attr_idx_data = (0..(DUMMY_INPUT_LEN * (TREE_HEIGHT - 1)))
        .map(|x| F::from(x as u64))
        .collect_vec();
    let _dummy_attr_idx_data = repeat_n(dummy_attr_idx_data, NUM_DUMMY_INPUTS).collect_vec();

    // --- Populate (note that we have to permute later) ---
    for _ in 0..NUM_DUMMY_INPUTS {
        // --- Generate a single copy of all the attributes ---
        let mut single_attribute_copy = vec![];
        // let mut single_permuted_attribute_copy = vec![];
        for attr_id in 0..DUMMY_INPUT_LEN {
            let input_attribute = InputAttribute {
                attr_id: F::from(attr_id as u64),
                attr_val: F::from(rng.gen_range(0..(2_u64.pow(12)))),
            };
            single_attribute_copy.push(input_attribute);
            // single_permuted_attribute_copy.push(input_attribute);
        }

        // --- Have to repeat attributes TREE_HEIGHT - 1 times ---
        let dummy_input_datum = single_attribute_copy
            .clone()
            .into_iter()
            .cycle()
            .take(single_attribute_copy.len() * (TREE_HEIGHT - 1))
            .collect_vec();
        // let dummy_permuted_input_datum = single_permuted_attribute_copy.clone().into_iter().cycle().take(single_permuted_attribute_copy.len() * (TREE_HEIGHT - 1)).collect_vec();

        // --- Add to final list ---
        dummy_input_data.push(dummy_input_datum);
        // dummy_permuted_input_data.push(dummy_permuted_input_datum);
    }

    // --- Generate a dummy tree ---
    let mut dummy_decision_nodes: Vec<DecisionNode<F>> = vec![];
    let mut dummy_leaf_nodes: Vec<LeafNode<F>> = vec![];

    // --- Populate decision nodes ---
    // Note that attr_id can only be in [0, DUMMY_INPUT_LEN)
    for idx in 0..NUM_DECISION_NODES {
        let decision_node = DecisionNode {
            node_id: F::from(idx as u64),
            attr_id: F::from(rng.gen_range(0..DUMMY_INPUT_LEN as u64)),
            threshold: F::from(rng.gen_range(0..(2_u64.pow(12)))),
        };
        dummy_decision_nodes.push(decision_node);
    }

    // --- Populate leaf nodes ---
    for idx in NUM_DECISION_NODES..(NUM_DECISION_NODES + NUM_LEAF_NODES) {
        let leaf_node = LeafNode {
            node_id: F::from(idx as u64),
            node_val: F::from(rng.gen::<u64>()),
        };
        dummy_leaf_nodes.push(leaf_node);
    }

    // --- Generate auxiliaries ---
    let dummy_auxiliaries = dummy_input_data
        .clone()
        .into_iter()
        .map(|dummy_attrs| {
            generate_correct_path_and_permutation(
                &dummy_decision_nodes,
                &dummy_leaf_nodes,
                &dummy_attrs,
            )
        })
        .collect_vec();

    // --- Collect correct paths ---
    let dummy_decision_node_paths = dummy_auxiliaries
        .clone()
        .into_iter()
        .map(
            |PathAndPermutation {
                 path_decision_nodes: x,
                 ..
             }| x,
        )
        .collect_vec();

    // --- Collect correct leaf nodes ---
    let dummy_leaf_node_paths = dummy_auxiliaries
        .clone()
        .into_iter()
        .map(
            |PathAndPermutation {
                 ret_leaf_node: x, ..
             }| x,
        )
        .collect_vec();

    // --- Collect correct permutation indices ---
    // let dummy_permutation_indices = dummy_auxiliaries
    //     .clone()
    //     .into_iter()
    //     .map(|(_, _, x, _)| {
    //         x
    //     }).collect_vec();

    // --- Compute the actual permutations ---
    let all_used_input_attributes = dummy_auxiliaries
        .clone()
        .into_iter()
        .map(
            |PathAndPermutation {
                 used_input_attributes,
                 ..
             }| used_input_attributes,
        )
        .collect_vec();
    let dummy_permuted_input_data = zip(all_used_input_attributes, dummy_input_data.clone())
        .map(|(used_input_attributes, original_input_attributes)| {
            let mut used_input_attributes_clone = used_input_attributes.clone();

            // --- Basically need to create a new vector with input attributes ---
            let ret = used_input_attributes
                .into_iter()
                .chain(original_input_attributes.clone().into_iter().filter(|x| {
                    // --- Filter by duplicates, but remove them from the containing set ---
                    if let Some(index) = used_input_attributes_clone.iter().position(|&y| y == *x) {
                        used_input_attributes_clone.remove(index);
                        return false;
                    }
                    true
                }))
                .collect_vec();

            assert_eq!(ret.len(), original_input_attributes.len());

            ret
        })
        .collect_vec();

    // --- Compute multiplicities: just add the ones that are given in the returned map ---
    // TODO!(ryancao): Just use the paths already! The decision nodes are there!
    let multiplicities: Vec<F> = vec![F::zero(); (2_u64.pow(TREE_HEIGHT as u32) - 1) as usize];
    let dummy_multiplicities_bin_decomp = dummy_auxiliaries
        .clone()
        .into_iter()
        .fold(
            multiplicities,
            |prev_multiplicities,
             PathAndPermutation {
                 path_decision_nodes,
                 ret_leaf_node: path_leaf_node,
                 ..
             }| {
                // --- TODO!(ryancao): This is so bad lol ---
                let mut new_multiplicities: Vec<F> = prev_multiplicities;

                // --- Just grab the node IDs from each decision node and add them to the multiplicities ---
                path_decision_nodes.into_iter().for_each(|decision_node| {
                    let node_id = decision_node.node_id.get_lower_128() as usize;
                    new_multiplicities[node_id] += F::one();
                });

                // --- Count the leaf node as well! ---
                let node_id = path_leaf_node.node_id.get_lower_128() as usize;
                new_multiplicities[node_id] += F::one();

                new_multiplicities
            },
        )
        .into_iter()
        .map(|multiplicity| {
            // --- Grab the binary decomp ---
            generate_16_bit_unsigned_decomp(multiplicity)
        })
        .collect_vec();

    // --- Compute the binary decompositions of the differences ---
    let dummy_binary_decomp_diffs = dummy_auxiliaries
        .into_iter()
        .map(|PathAndPermutation { diffs, .. }| {
            diffs
                .into_iter()
                .map(|diff| {
                    let ret = generate_16_bit_signed_decomp(diff);
                    check_signed_recomposition(diff, ret);
                    ret
                })
                .collect_vec()
        })
        .collect_vec();

    ZKDTDummyCircuitData {
        dummy_input_data,
        dummy_permuted_input_data,
        dummy_decision_node_paths,
        dummy_leaf_node_paths,
        dummy_binary_decomp_diffs,
        dummy_multiplicities_bin_decomp,
        dummy_decision_nodes,
        dummy_leaf_nodes,
    }
}

/// Gets the sign bit (0 for positive, 1 for negative) and abs value
fn get_sign_bit_and_abs_value<F: FieldExt>(value: F) -> (F, F) {
    let upper_bound = F::from(2_u64.pow(16) - 1);
    let sign_bit = if value > upper_bound {
        F::one()
    } else {
        F::zero()
    };
    let abs_value = if value > upper_bound {
        value.neg()
    } else {
        value
    };
    (sign_bit, abs_value)
}

/// Computes the recomposition of the bits within `decomp` and checks
fn check_signed_recomposition<F: FieldExt>(actual_value: F, decomp: BinDecomp16Bit<F>) -> bool {
    let (sign_bit, _) = get_sign_bit_and_abs_value(actual_value);
    let mut total = F::zero();

    // --- Perform recomposition of non-sign bits ---
    for bit_idx in 1..16 {
        let base = F::from(2_u64.pow((16 - (bit_idx + 1)) as u32));
        total += base * decomp.bits[bit_idx];
    }
    total = if sign_bit == F::one() {
        total.neg()
    } else {
        total
    };
    if total != actual_value {
        // dbg!(
        //     "RIP: Total = {:?}, actual_value = {:?}",
        //     total,
        //     actual_value
        // );
        panic!();
        // return false;
    }
    true
}

pub struct BatchedDummyMles<F: FieldExt> {
    pub dummy_input_data_mle: Vec<DenseMle<F, InputAttribute<F>>>,
    pub dummy_permuted_input_data_mle: Vec<DenseMle<F, InputAttribute<F>>>,
    pub dummy_decision_node_paths_mle: Vec<DenseMle<F, DecisionNode<F>>>,
    pub dummy_leaf_node_paths_mle: Vec<DenseMle<F, LeafNode<F>>>,
    pub dummy_binary_decomp_diffs_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    pub dummy_multiplicities_bin_decomp_mle: DenseMle<F, BinDecomp16Bit<F>>,
    pub dummy_decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
    pub dummy_leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
}

// #[derive(Serialize, Deserialize)]
#[derive(Clone)]
pub struct BatchedCatboostMles<F: FieldExt> {
    pub input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    pub permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    pub decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>,
    pub leaf_node_paths_mle_vec: Vec<DenseMle<F, LeafNode<F>>>,
    pub binary_decomp_diffs_mle_vec: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    pub multiplicities_bin_decomp_mle_decision: DenseMle<F, BinDecomp16Bit<F>>,
    pub multiplicities_bin_decomp_mle_leaf: DenseMle<F, BinDecomp16Bit<F>>,
    pub decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
    pub leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
    pub multiplicities_bin_decomp_mle_input_vec: Vec<DenseMle<F, BinDecomp4Bit<F>>>,
}

/// Writes the results of the [`load_upshot_data_single_tree_batch`] function call
/// to a file for ease of reading (i.e. faster testing, mostly lol)
pub fn write_mles_batch_catboost_single_tree<F: FieldExt>() {
    let loaded_zkdt_circuit_data = load_upshot_data_single_tree_batch::<F>(None, None);
    let mut f = fs::File::create(CACHED_BATCHED_MLES_FILE).unwrap();
    to_writer(&mut f, &loaded_zkdt_circuit_data).unwrap();
}

/// Reads the cached results from [`load_upshot_data_single_tree_batch`] and returns them.
pub fn read_upshot_data_single_tree_branch_from_file<F: FieldExt>() -> (ZKDTCircuitData<F>, (usize, usize)) {
    let file = std::fs::File::open(CACHED_BATCHED_MLES_FILE).unwrap();
    from_reader(&file).unwrap()
}

/// Loads a result from [`generate_upshot_data_all_batch_sizes`].
pub fn read_upshot_data_single_tree_branch_from_file_with_batch_exp<F: FieldExt>(
    exp_batch_size: usize,
    upshot_data_dir_path: &Path
) -> (ZKDTCircuitData<F>, (usize, usize)) {

    // --- Sanitychecks ---
    debug_assert!(exp_batch_size >= 1);
    debug_assert!(exp_batch_size <= 12);

    // --- Load ---
    let file = std::fs::File::open(get_cached_batched_mles_filename_with_exp_size(exp_batch_size, upshot_data_dir_path)).unwrap();
    from_reader(&file).unwrap()
}

/// Generates circuit data in batched form for a single Catboost tree
/// 
/// ## Arguments
/// * `exp_batch_size` - 2^{`exp_batch_size`} is the actual batch size that we want.
///     Note that this value must be between 1 and 12, inclusive!
pub fn generate_mles_batch_catboost_single_tree<F: FieldExt>(exp_batch_size: usize, upshot_data_dir_path: &Path) -> (BatchedCatboostMles<F>, (usize, usize)) {

    // --- Sanitychecks ---
    debug_assert!(exp_batch_size >= 1);
    debug_assert!(exp_batch_size <= 12);

    // --- Check to see if the cached file exists ---
    let cached_file_path = get_cached_batched_mles_filename_with_exp_size(exp_batch_size, upshot_data_dir_path);

    // --- If no cached file exists, run the entire cache thingy ---
    if !file_exists(&cached_file_path) {
        generate_upshot_data_all_batch_sizes::<F>(None, upshot_data_dir_path);
    }

    // --- First generate the dummy data ---
    let (ZKDTCircuitData {
        // dummy_attr_idx_data,
        input_data,
        // permutation_indices,
        permuted_input_data,
        decision_node_paths,
        leaf_node_paths,
        binary_decomp_diffs,
        mut multiplicities_bin_decomp,
        decision_nodes,
        leaf_nodes,
        multiplicities_bin_decomp_input,
    }, (tree_height, input_len)) = read_upshot_data_single_tree_branch_from_file::<F>();

    // println!("input_data {:?}", input_data[0]);
    // println!("permuted_input_data {:?}", permuted_input_data[0]);
    // println!("multiplicities_bin_decomp_input {:?}", multiplicities_bin_decomp_input[0]);

    let decision_len = 2_usize.pow(tree_height as u32 - 1);
    let multiplicities_bin_decomp_leaf = multiplicities_bin_decomp.split_off(decision_len);
    let multiplicities_bin_decomp_decision = multiplicities_bin_decomp;

    // --- Generate MLEs for each ---
    // TODO!(ryancao): Change this into batched form
    // let attr_idx_data_mle = DenseMle::<_, F>::new(attr_idx_data[0].clone());
    let input_data_mle_vec = input_data.into_iter().map(|input| DenseMle::new_from_iter(input
        .clone()
        .into_iter()
        .map(InputAttribute::from), LayerId::Input(0), None)).collect_vec();
    // let permutation_indices_mle = DenseMle::<_, F>::new(permutation_indices[0].clone());
    let permuted_input_data_mle_vec = permuted_input_data
        .iter().map(|datum| DenseMle::new_from_iter(datum
            .clone()
            .into_iter()
            .map(InputAttribute::from), LayerId::Input(0), None)).collect();
    let decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>> = decision_node_paths
        .iter()
        .map(|path|
            DenseMle::new_from_iter(path
            .clone()
            .into_iter(), LayerId::Input(0), None))
        .collect();
    let leaf_node_paths_mle_vec = leaf_node_paths
        .into_iter()
        .map(|path| DenseMle::new_from_iter([path].into_iter(), LayerId::Input(0), None))
        .collect();
    let binary_decomp_diffs_mle_vec = binary_decomp_diffs
        .iter()
        .map(|binary_decomp_diff|
            DenseMle::new_from_iter(binary_decomp_diff
                .clone()
                .into_iter()
                .map(BinDecomp16Bit::from), LayerId::Input(0), None))
        .collect_vec();
    let multiplicities_bin_decomp_mle_decision = DenseMle::new_from_iter(multiplicities_bin_decomp_decision
        .clone()
        .into_iter()
        .map(BinDecomp16Bit::from), LayerId::Input(0), None);
    let multiplicities_bin_decomp_mle_leaf = DenseMle::new_from_iter(multiplicities_bin_decomp_leaf
        .clone()
        .into_iter()
        .map(BinDecomp16Bit::from), LayerId::Input(0), None);
    let decision_nodes_mle = DenseMle::new_from_iter(decision_nodes
        .clone()
        .into_iter()
        .map(DecisionNode::from), LayerId::Input(0), None);
    let leaf_nodes_mle = DenseMle::new_from_iter(leaf_nodes
        .clone()
        .into_iter()
        .map(LeafNode::from), LayerId::Input(0), None);
    let multiplicities_bin_decomp_mle_input = multiplicities_bin_decomp_input
        .iter().map(|datum|
            DenseMle::new_from_iter(datum
            .clone()
            .into_iter()
            .map(BinDecomp4Bit::from), LayerId::Input(0), None))
        .collect_vec();


    (BatchedCatboostMles {
        input_data_mle_vec,
        permuted_input_data_mle_vec,
        decision_node_paths_mle_vec,
        leaf_node_paths_mle_vec,
        binary_decomp_diffs_mle_vec,
        multiplicities_bin_decomp_mle_decision,
        multiplicities_bin_decomp_mle_leaf,
        decision_nodes_mle,
        leaf_nodes_mle,
        multiplicities_bin_decomp_mle_input_vec: multiplicities_bin_decomp_mle_input
    }, (tree_height, input_len))
}

pub fn generate_dummy_mles_batch<F: FieldExt>() -> BatchedDummyMles<F> {
    // --- First generate the dummy data ---
    let ZKDTDummyCircuitData {
        // dummy_attr_idx_data,
        dummy_input_data,
        // dummy_permutation_indices,
        dummy_permuted_input_data,
        dummy_decision_node_paths,
        dummy_leaf_node_paths,
        dummy_binary_decomp_diffs,
        dummy_multiplicities_bin_decomp,
        dummy_decision_nodes,
        dummy_leaf_nodes,
    } = generate_dummy_data::<F>();

    // --- Generate MLEs for each ---
    // TODO!(ryancao): Change this into batched form
    // let dummy_attr_idx_data_mle = DenseMle::<_, F>::new(dummy_attr_idx_data[0].clone());
    let dummy_input_data_mle = dummy_input_data.into_iter().map(|input| DenseMle::new_from_iter(input
        .clone()
        .into_iter()
        .map(InputAttribute::from), LayerId::Input(0), None)).collect_vec();
    // let dummy_permutation_indices_mle = DenseMle::<_, F>::new(dummy_permutation_indices[0].clone());
    let dummy_permuted_input_data_mle = dummy_permuted_input_data
        .iter().map(|datum| DenseMle::new_from_iter(datum
            .clone()
            .into_iter()
            .map(InputAttribute::from), LayerId::Input(0), None)).collect();
    let dummy_decision_node_paths_mle = dummy_decision_node_paths
        .iter()
        .map(|path|
            DenseMle::new_from_iter(path
            .clone()
            .into_iter(), LayerId::Input(0), None))
        .collect();
    let dummy_leaf_node_paths_mle = dummy_leaf_node_paths
        .into_iter()
        .map(|path| DenseMle::new_from_iter([path].into_iter(), LayerId::Input(0), None))
        .collect();
    let dummy_binary_decomp_diffs_mle = dummy_binary_decomp_diffs
        .iter()
        .map(|dummy_binary_decomp_diff|
            DenseMle::new_from_iter(dummy_binary_decomp_diff
                .clone()
                .into_iter()
                .map(BinDecomp16Bit::from), LayerId::Input(0), None))
        .collect_vec();
    let dummy_multiplicities_bin_decomp_mle = DenseMle::new_from_iter(dummy_multiplicities_bin_decomp
        .clone()
        .into_iter()
        .map(BinDecomp16Bit::from), LayerId::Input(0), None);
    let dummy_decision_nodes_mle = DenseMle::new_from_iter(dummy_decision_nodes
        .clone()
        .into_iter()
        .map(DecisionNode::from), LayerId::Input(0), None);
    let dummy_leaf_nodes_mle = DenseMle::new_from_iter(dummy_leaf_nodes
        .clone()
        .into_iter()
        .map(LeafNode::from), LayerId::Input(0), None);

    BatchedDummyMles {
        dummy_input_data_mle,
        dummy_permuted_input_data_mle,
        dummy_decision_node_paths_mle,
        dummy_leaf_node_paths_mle,
        dummy_binary_decomp_diffs_mle,
        dummy_multiplicities_bin_decomp_mle,
        dummy_decision_nodes_mle,
        dummy_leaf_nodes_mle,
    }
}

pub(crate) struct DummyMles<F: FieldExt> {
    pub(crate) dummy_input_data_mle: DenseMle<F, InputAttribute<F>>,
    pub(crate) dummy_permuted_input_data_mle: DenseMle<F, InputAttribute<F>>,
    pub(crate) dummy_decision_node_paths_mle: DenseMle<F, DecisionNode<F>>,
    pub(crate) dummy_leaf_node_paths_mle: DenseMle<F, LeafNode<F>>,
    pub(crate) dummy_binary_decomp_diffs_mle: DenseMle<F, BinDecomp16Bit<F>>,
    pub(crate) dummy_multiplicities_bin_decomp_mle: DenseMle<F, BinDecomp16Bit<F>>,
    pub(crate) dummy_decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
    pub(crate) dummy_leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
}

/// Takes the above dummy data from `generate_dummy_data()` and converts
/// into MLE form factor.
pub(crate) fn generate_dummy_mles<F: FieldExt>() -> DummyMles<F> {
    // --- First generate the dummy data ---
    let ZKDTDummyCircuitData {
        dummy_input_data,
        dummy_permuted_input_data,
        dummy_decision_node_paths,
        dummy_leaf_node_paths,
        dummy_binary_decomp_diffs,
        dummy_multiplicities_bin_decomp,
        dummy_decision_nodes,
        dummy_leaf_nodes,
    } = generate_dummy_data::<F>();

    // --- Generate MLEs for each ---
    // TODO!(ryancao): Change this into batched form
    // let dummy_attr_idx_data_mle = DenseMle::<_, F>::new(dummy_attr_idx_data[0].clone());
    let dummy_input_data_mle = DenseMle::new_from_iter(
        dummy_input_data[0]
            .clone()
            .into_iter()
            .map(InputAttribute::from),
        LayerId::Input(0),
        None,
    ); // let dummy_permutation_indices_mle = DenseMle::<_, F>::new(dummy_permutation_indices[0].clone());
    let dummy_permuted_input_data_mle = DenseMle::new_from_iter(
        dummy_permuted_input_data[0]
            .clone()
            .into_iter()
            .map(InputAttribute::from),
        LayerId::Input(0),
        None,
    );
    let dummy_decision_node_paths_mle = DenseMle::new_from_iter(
        dummy_decision_node_paths[0]
            .clone()
            .into_iter()
            .map(DecisionNode::from),
        LayerId::Input(0),
        None,
    );
    let dummy_leaf_node_paths_mle = DenseMle::new_from_iter(
        vec![dummy_leaf_node_paths[0]]
            .into_iter()
            .map(LeafNode::from),
        LayerId::Input(0),
        None,
    );
    let dummy_binary_decomp_diffs_mle = DenseMle::new_from_iter(
        dummy_binary_decomp_diffs[0]
            .clone()
            .into_iter()
            .map(BinDecomp16Bit::from),
        LayerId::Input(0),
        None,
    );
    let dummy_multiplicities_bin_decomp_mle = DenseMle::new_from_iter(
        dummy_multiplicities_bin_decomp
            .into_iter()
            .map(BinDecomp16Bit::from),
        LayerId::Input(0),
        None,
    );
    let dummy_decision_nodes_mle = DenseMle::new_from_iter(
        dummy_decision_nodes.into_iter().map(DecisionNode::from),
        LayerId::Input(0),
        None,
    );
    let dummy_leaf_nodes_mle = DenseMle::new_from_iter(
        dummy_leaf_nodes.into_iter().map(LeafNode::from),
        LayerId::Input(0),
        None,
    );

    DummyMles {        // dummy_attr_idx_data_mle,
        dummy_input_data_mle,
        dummy_permuted_input_data_mle,
        dummy_decision_node_paths_mle,
        dummy_leaf_node_paths_mle,
        dummy_binary_decomp_diffs_mle,
        dummy_multiplicities_bin_decomp_mle,
        dummy_decision_nodes_mle,
        dummy_leaf_nodes_mle,
    }
}

// --- Create expressions using... testing modules? ---
#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        expression::ExpressionStandard,
        layer::{Claim, LayerId},
        mle::{beta::BetaTable, dense::DenseMle, dense::DenseMleRef, MleRef},
        sumcheck::{
            compute_sumcheck_message, get_round_degree,
            tests::{dummy_sumcheck, get_dummy_expression_eval, verify_sumcheck_messages},
            Evals,
        },
    };
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use ark_std::test_rng;

    /// Literally just calls the [`write_mles_batch_catboost_single_tree`] function
    /// to write the preprocessed stuff to file so we can load it in later
    #[test]
    fn test_write_mles_batch_catboost_single_tree() {
        write_mles_batch_catboost_single_tree::<Fr>();
    }

    /// Checks that bits within the diff binary decomp and the multiplicity
    /// binary decomp are all either 0 or 1
    #[test]
    fn dummy_bits_are_binary_test() {
        // --- First generate the dummy data ---
        let ZKDTDummyCircuitData {
            dummy_binary_decomp_diffs,
            dummy_multiplicities_bin_decomp,
            ..
        } = generate_dummy_data::<Fr>();

        // --- Checks that all the (diff) bits are either zero or one ---
        dummy_binary_decomp_diffs
            .into_iter()
            .for_each(|per_input_dummy_binary_decomp_diffs| {
                // --- We should have exactly TREE_HEIGHT - 1 diffs/decomps ---
                assert!(per_input_dummy_binary_decomp_diffs.len() == TREE_HEIGHT - 1);

                per_input_dummy_binary_decomp_diffs.into_iter().for_each(
                    |dummy_binary_decomp_diff| {
                        dummy_binary_decomp_diff.bits.into_iter().for_each(|bit| {
                            assert!(bit == Fr::zero() || bit == Fr::one());
                        })
                    },
                );
            });

        // --- Checks the same for the multiplicity binary decompositions ---
        assert!(
            dummy_multiplicities_bin_decomp.len()
                == (NUM_DECISION_NODES + NUM_LEAF_NODES) as usize
        );
        dummy_multiplicities_bin_decomp
            .into_iter()
            .for_each(|multiplicity_bit_decomp| {
                multiplicity_bit_decomp.bits.into_iter().for_each(|bit| {
                    assert!(bit == Fr::zero() || bit == Fr::one());
                })
            })
    }

    /// Basic "bits are binary" test (for the diffs), but in circuit!
    #[test]
    fn circuit_dummy_bits_are_binary_test_diff() {
        let mut rng = test_rng();
        let layer_claim: Claim<Fr> = (
            vec![
                Fr::from(rng.gen::<u64>()),
                Fr::from(rng.gen::<u64>()),
                Fr::from(rng.gen::<u64>()),
                Fr::from(rng.gen::<u64>()),
            ],
            Fr::zero(),
        );
        let mut beta = BetaTable::new(layer_claim.clone()).unwrap();
        beta.table.index_mle_indices(0);

        let DummyMles {
            dummy_binary_decomp_diffs_mle,
            ..
        } = generate_dummy_mles::<Fr>();

        // --- Grab the bin decomp MLE ---
        let first_bin_decomp_bit_mle: Vec<DenseMleRef<Fr>> =
            dummy_binary_decomp_diffs_mle.mle_bit_refs();
        let first_bin_decomp_bit_expr =
            ExpressionStandard::Mle(first_bin_decomp_bit_mle[0].clone());

        // --- Do b * (1 - b) = b - b^2 ---
        let b_squared = ExpressionStandard::Product(vec![
            first_bin_decomp_bit_mle[0].clone(),
            first_bin_decomp_bit_mle[0].clone(),
        ]);
        // dbg!(&b_squared);
        // dbg!(&first_bin_decomp_bit_mle[0]);
        let mut b_minus_b_squared = first_bin_decomp_bit_expr - b_squared;
        // dbg!(&b_minus_b_squared);

        // --- Evaluating at V(0, 0, 0) --> 0 ---
        let _dummy_claim = (vec![Fr::from(1); 3], Fr::zero());
        let mut b_minus_b_squared_clone = b_minus_b_squared.clone();
        b_minus_b_squared.index_mle_indices(0);
        // b_minus_b_squared.init_beta_tables(dummy_claim.clone());

        // idk if this is actually how we should do this
        let round_degree = get_round_degree(&b_minus_b_squared, 0);
        // dbg!(round_degree);
        let res =
            compute_sumcheck_message(&mut b_minus_b_squared.clone(), 0, round_degree, &mut beta);

        // --- Only first two values need to be zeros ---
        let Evals::<Fr>(vec) = res.unwrap();
        assert_eq!(vec[0], Fr::zero());
        assert_eq!(vec[1], Fr::zero());

        let layer_claims = get_dummy_expression_eval(&b_minus_b_squared_clone, &mut rng);

        let res_messages =
            dummy_sumcheck(&mut b_minus_b_squared_clone, &mut rng, layer_claims.clone());
        let verify_res = verify_sumcheck_messages(
            res_messages,
            b_minus_b_squared_clone,
            layer_claims,
            &mut rng,
        );
        assert!(verify_res.is_ok());
    }

    /// basic "bits are binary" test (for multiplicities), but in circuit!
    #[test]
    fn circuit_dummy_bits_are_binary_test_multiplicities() {
        let mut rng = test_rng();
        let layer_claim: Claim<Fr> = (vec![Fr::from(rng.gen::<u64>()); 12], Fr::zero());
        let mut beta = BetaTable::new(layer_claim.clone()).unwrap();
        beta.table.index_mle_indices(0);

        let DummyMles {
            dummy_multiplicities_bin_decomp_mle,
            ..
        } = generate_dummy_mles::<Fr>();

        // --- Grab the bin decomp MLE ---
        let first_bin_decomp_bit_mle: Vec<DenseMleRef<Fr>> =
            dummy_multiplicities_bin_decomp_mle.mle_bit_refs();
        let first_bin_decomp_bit_expr =
            ExpressionStandard::Mle(first_bin_decomp_bit_mle[0].clone());

        // --- Do b * (1 - b) = b - b^2 ---
        let b_squared = ExpressionStandard::Product(vec![
            first_bin_decomp_bit_mle[0].clone(),
            first_bin_decomp_bit_mle[0].clone(),
        ]);
        let mut b_minus_b_squared = first_bin_decomp_bit_expr - b_squared;

        // --- We should get all zeros ---x
        let all_zeros: Vec<Fr> = vec![Fr::zero()]
            .repeat(2_u64.pow(first_bin_decomp_bit_mle[0].num_vars() as u32) as usize);
        let all_zeros_mle = DenseMle::<Fr, _>::new_from_raw(all_zeros, LayerId::Input(0), None);
        let _all_zeros_mle_expr = ExpressionStandard::Mle(all_zeros_mle.mle_ref());

        // --- Evaluating at V(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) --> 0 ---
        let _dummy_claim = (vec![Fr::one(); 3 + 9], Fr::zero());

        // --- Initialize beta tables manually ---
        let mut b_minus_b_squared_clone = b_minus_b_squared.clone();
        b_minus_b_squared.index_mle_indices(0);
        // b_minus_b_squared.init_beta_tables(dummy_claim.clone());

        let first_round_deg = get_round_degree(&b_minus_b_squared, 0);

        // --- The first two elements in the sumcheck message should both be zero ---
        // Afterwards there are no guarantees since we're doing a potentially non-linear interpolation
        let res = compute_sumcheck_message(
            &mut b_minus_b_squared.clone(),
            1,
            first_round_deg,
            &mut beta,
        );
        let Evals::<Fr>(vec) = res.unwrap();
        assert_eq!(vec[0], Fr::zero());
        assert_eq!(vec[1], Fr::zero());

        let layer_claims = get_dummy_expression_eval(&b_minus_b_squared_clone, &mut rng);

        let res_messages =
            dummy_sumcheck(&mut b_minus_b_squared_clone, &mut rng, layer_claims.clone());
        let verify_res = verify_sumcheck_messages(
            res_messages,
            b_minus_b_squared_clone,
            layer_claims,
            &mut rng,
        );
        assert!(verify_res.is_ok());
    }

    /// Binary recomposition test (out of circuit)
    #[test]
    fn dummy_binary_recomp_test() {
        // --- First generate the dummy data ---
        let ZKDTDummyCircuitData {
            dummy_permuted_input_data,
            dummy_decision_node_paths,
            dummy_binary_decomp_diffs,
            ..
        } = generate_dummy_data::<Fr>();

        // --- Grab the attr vals from the permuted inputs ---
        let permuted_attr_vals = dummy_permuted_input_data
            .into_iter()
            .map(|input_attributes| {
                // Dummy inputs should always have length `original_len * tree_height - 1`
                // from duplication
                assert!(input_attributes.len() == DUMMY_INPUT_LEN * (TREE_HEIGHT - 1));

                // Just extract the attribute vals
                input_attributes
                    .into_iter()
                    .map(|input_attribute| input_attribute.attr_val)
                    .collect_vec()
            })
            .collect_vec();

        // --- Grab the thresholds from the path nodes ---
        let decision_node_thresholds = dummy_decision_node_paths
            .into_iter()
            .map(|dummy_decision_node_path| {
                // Paths should always be length TREE_HEIGHT - 1
                assert!(dummy_decision_node_path.len() == TREE_HEIGHT - 1);

                dummy_decision_node_path
                    .into_iter()
                    .map(|dummy_decision_node| dummy_decision_node.threshold)
                    .collect_vec()
            })
            .collect_vec();

        // --- Slice the permuted inputs to match the path node length ---
        let permuted_attr_vals = permuted_attr_vals
            .into_iter()
            .map(|single_input_attr_vals| single_input_attr_vals[..TREE_HEIGHT - 1].to_vec())
            .collect_vec();

        // --- Compute diffs ---
        assert!(decision_node_thresholds.len() == permuted_attr_vals.len());
        let all_diffs = zip(decision_node_thresholds, permuted_attr_vals)
            .map(
                |(input_decision_node_thresholds, input_permuted_attr_vals)| {
                    assert!(input_decision_node_thresholds.len() == input_permuted_attr_vals.len());
                    zip(input_decision_node_thresholds, input_permuted_attr_vals)
                        .map(|(decision_node_threshold, permuted_attr_val)| {
                            permuted_attr_val - decision_node_threshold
                        })
                        .collect_vec()
                },
            )
            .collect_vec();

        // --- Now time to compute binary recompositions ---
        // Just do a zip between the decomps and `all_diffs` above
        zip(dummy_binary_decomp_diffs, all_diffs).for_each(
            |(input_binary_decomp_diffs, input_diffs)| {
                assert!(input_binary_decomp_diffs.len() == input_diffs.len());
                zip(input_binary_decomp_diffs, input_diffs).for_each(
                    |(binary_decomp_diff, diff)| {
                        check_signed_recomposition(diff, binary_decomp_diff);
                    },
                );
            },
        );
    }

    /// Binary recomposition test (showing that the binary recomposition of the
    /// difference recomposes to equal the differences)
    /// The original expression: (1 - b_s)(diff - abs_recomp) + b_s(diff + abs_recomp) = 0
    /// The simplified expression: (diff - abs_recomp) + 2b_s(abs_recomp) = 0
    /// abs_recomp = \sum_{i = 1}^{15} b_i 2^{16 - i - 1}
    #[test]
    fn circuit_dummy_binary_recomp_test() {
        let mut rng = test_rng();
        let layer_claim: Claim<Fr> = (
            vec![
                Fr::from(rng.gen::<u64>()),
                Fr::from(rng.gen::<u64>()),
                Fr::from(rng.gen::<u64>()),
                Fr::from(rng.gen::<u64>()),
            ],
            Fr::zero(),
        );
        let mut beta = BetaTable::new(layer_claim).unwrap();
        beta.table.index_mle_indices(0);
        let DummyMles {
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle,
            dummy_binary_decomp_diffs_mle,
            ..
        } = generate_dummy_mles::<Fr>();

        // --- Grab the bin decomp MLEs and associated expressions ---
        let bin_decomp_mles: Vec<DenseMleRef<Fr>> = dummy_binary_decomp_diffs_mle.mle_bit_refs();

        // --- Grab the things necessary to compute the diff (the permuted input and thresholds) ---
        let threshold_mle: DenseMleRef<Fr> = dummy_decision_node_paths_mle.threshold();
        let threshold_mle_expr = ExpressionStandard::Mle(threshold_mle.clone());
        let permuted_input_values_mle: DenseMleRef<Fr> =
            dummy_permuted_input_data_mle.attr_val(Some(threshold_mle.num_vars()));
        let permuted_input_values_mle_expr = ExpressionStandard::Mle(permuted_input_values_mle);

        // --- For debugging ---
        // let threshold_mle_expr_eval = evaluate_expr(&mut threshold_mle_expr.clone(), 1, 2);
        // dbg!(threshold_mle_expr_eval);
        // let permuted_input_values_mle_expr_eval = evaluate_expr(&mut permuted_input_values_mle_expr.clone(), 1, 2);
        // dbg!(permuted_input_values_mle_expr_eval);

        // --- Need to just get diff ---
        // dbg!(permuted_input_values_mle.num_vars()); // Should be 3
        // dbg!(threshold_mle.num_vars()); // Should be 3
        let diff_expr = permuted_input_values_mle_expr - threshold_mle_expr;
        // let permuted_input_values_mle_expr_eval = compute_sumcheck_message(&mut permuted_input_values_mle_expr.clone(), 1, 2);
        // let threshold_mle_expr_eval = compute_sumcheck_message(&mut threshold_mle_expr.clone(), 1, 2);
        // dbg!(permuted_input_values_mle_expr_eval);
        // dbg!(threshold_mle_expr_eval);

        // --- We need `abs_recomp` and `b_s * abs_recomp` ---
        let b_s_initial_acc = ExpressionStandard::Constant(Fr::zero());
        let sign_bit_mle = bin_decomp_mles[0].clone();
        let bin_decomp_mles_clone = bin_decomp_mles.clone();

        // --- Time for iterators... sigh ---
        let b_s_times_abs_recomp_expr = bin_decomp_mles.into_iter().enumerate().skip(1).fold(
            b_s_initial_acc,
            |acc_expr, (bit_idx, bin_decomp_mle)| {
                // --- First compute b_s * coeff ---
                let b_s_times_coeff =
                    ExpressionStandard::Product(vec![bin_decomp_mle, sign_bit_mle.clone()]);

                let b_s_times_coeff_ptr = Box::new(b_s_times_coeff);

                // --- Then compute (b_s * coeff) * 2^{bit_idx} ---
                let base = Fr::from(2_u64.pow((16 - (bit_idx + 1)) as u32));
                let b_s_times_coeff_times_base =
                    ExpressionStandard::Scaled(b_s_times_coeff_ptr, base);

                // Debugging
                // let b_i_expr = ExpressionStandard::Mle(bin_decomp_mle.clone());
                // let b_i_expr_eval = evaluate_expr(&mut b_i_expr.clone(), 1, 1);
                // let b_s_times_coeff_times_base_eval = evaluate_expr(&mut b_s_times_coeff_times_base.clone(), 1, 2);
                // dbg!(bit_idx);
                // dbg!(bin_decomp_mle.clone().num_vars());
                // dbg!(b_i_expr_eval);
                // dbg!(b_s_times_coeff_times_base_eval);

                acc_expr + b_s_times_coeff_times_base
            },
        );

        let abs_recomp_initial_acc = ExpressionStandard::Constant(Fr::zero());
        let abs_recomp_expr = bin_decomp_mles_clone.into_iter().enumerate().skip(1).fold(
            abs_recomp_initial_acc,
            |acc_expr, (bit_idx, bin_decomp_mle)| {
                // --- Compute just coeff * 2^{bit_idx} ---
                let base = Fr::from(2_u64.pow((16 - (bit_idx + 1)) as u32));
                let coeff_expr = ExpressionStandard::Mle(bin_decomp_mle);
                let coeff_expr_ptr = Box::new(coeff_expr);
                let coeff_times_base = ExpressionStandard::Scaled(coeff_expr_ptr, base);

                // Debugging
                let _coeff_times_base_eval =
                    compute_sumcheck_message(&mut coeff_times_base.clone(), 1, 2, &mut beta);

                acc_expr + coeff_times_base
            },
        );

        // --- Subtract the two, and (TODO!(ryancao)) ensure they have the same number of variables ---
        let mut final_expr = diff_expr - abs_recomp_expr
            + b_s_times_abs_recomp_expr.clone()
            + b_s_times_abs_recomp_expr;

        // --- Let's just see what the expressions give us... ---
        // Debugging
        // let diff_result = compute_sumcheck_message(&mut diff_expr, 1, 2);
        // let abs_recomp_expr_result = compute_sumcheck_message(&mut abs_recomp_expr, 1, 2);
        // let b_s_times_abs_recomp_expr_result = compute_sumcheck_message(&mut b_s_times_abs_recomp_expr, 1, 2);
        // dbg!(diff_result);
        // dbg!(abs_recomp_expr_result);
        // dbg!(b_s_times_abs_recomp_expr_result);

        // let dummy_claim = (vec![Fr::one(); 3], Fr::zero());

        let _final_expr_clone = final_expr.clone();

        final_expr.index_mle_indices(0);
        // final_expr.init_beta_tables(dummy_claim.clone());

        // --- Only the first two evals should be zeros ---
        let res = compute_sumcheck_message(&mut final_expr, 1, 3, &mut beta);
        let Evals::<Fr>(vec) = res.unwrap();
        assert_eq!(vec[0], Fr::zero());
        assert_eq!(vec[1], Fr::zero());

        // let res_messages = dummy_sumcheck(final_expr_clone, &mut rng, dummy_claim);
        // let verify_res = verify_sumcheck_messages(res_messages);
        // assert!(verify_res.is_ok());
    }

    /// Permutation test showing that the characteristic polynomial of the
    /// initial inputs is equivalent to that of the permuted inputs.
    /// What's the expression we actually need, and what are the terms?
    /// - We need the packed inputs first, (attr_idx + r_1 * attr_id + r_2 * attr_val)
    /// - Then we need the characteristic polynomial terms evaluated at `r_3`: r_3 - (attr_idx + r_1 * attr_id + r_2 * attr_val)
    /// - Then we need to multiply all of them together in a binary product tree
    #[test]
    fn dummy_permutation_test() {
        let DummyMles {
            dummy_input_data_mle,
            ..
        } = generate_dummy_mles::<Fr>();

        let mut rng = test_rng();

        // --- Get packed inputs first ---
        let _r1: Fr = Fr::from(rng.gen::<u64>());
        let _r2: Fr = Fr::from(rng.gen::<u64>());

        // --- Multiply to do packing ---
        let dummy_attribute_id_mleref = dummy_input_data_mle.attr_id(None);
        let _dummy_attribute_id_mleref_expr = ExpressionStandard::Mle(dummy_attribute_id_mleref);
        let dummy_attribute_val_mleref = dummy_input_data_mle.attr_val(None);
        let _dummy_attribute_val_mleref_expr = ExpressionStandard::Mle(dummy_attribute_val_mleref);

        // ---
    }
}
