use crate::mle::dense::DenseMle;
use crate::FieldExt;

use super::structs::*;

use ark_std::test_rng;
use itertools::{repeat_n, Itertools};
use rand::Rng;
use std::collections::HashMap;
use std::iter::zip;

/*
What's our plan here?
- First create some dummy data
- Then create MLEs and MLERefs using that dummy data
- Then create expressions over those which are representative
- Hmm sounds like we might need to do intermediate stuff
*/

// --- Constants ---
const DUMMY_INPUT_LEN: usize = 1 << 5;
const NUM_DUMMY_INPUTS: usize = 1 << 8;
const TREE_HEIGHT: usize = 9;
const NUM_DECISION_NODES: u32 = 2_u32.pow(TREE_HEIGHT as u32 - 1) - 1;
const NUM_LEAF_NODES: u32 = NUM_DECISION_NODES + 1;

/// - First element is the decision path nodes
/// - Second element is the final predicted leaf node
/// - Third element is the difference between the current attribute value and the node threshold
/// - Fourth element is the input attributes which were used during inference
/// - Third element is the attributes used (deprecated)
fn generate_correct_path_and_permutation<F: FieldExt>(
    decision_nodes: &Vec<DecisionNode<F>>,
    leaf_nodes: &Vec<LeafNode<F>>,
    input_datum: &Vec<InputAttribute<F>>,
) -> (
    Vec<DecisionNode<F>>,
    LeafNode<F>,
    Vec<F>,
    Vec<InputAttribute<F>>,
) {
    // --- Keep track of the path and permutation ---
    let mut path_decision_nodes: Vec<DecisionNode<F>> = vec![];
    let mut used_input_attributes: Vec<InputAttribute<F>> = vec![];
    // let mut permuted_access_indices: Vec<F> = vec![];

    // --- Keep track of how many times each attribute ID was used (to get the corresponding attribute idx) ---
    // Key: Attribute ID
    // Value: Multiplicity
    let mut attr_num_hits: HashMap<F, u32> = HashMap::new();

    // --- Keep track of the differences for path nodes ---
    let mut diffs: Vec<F> = vec![];

    // --- Go through the decision nodes ---
    let mut current_node_idx = 0;
    while current_node_idx < decision_nodes.len() {
        // --- Add to path; grab the appropriate attribute index ---
        path_decision_nodes.push(decision_nodes[current_node_idx]);
        let attr_id = (decision_nodes[current_node_idx].attr_id.into_bigint()).as_ref()[0] as usize;

        // --- Stores the current input attribute which is being used ---
        used_input_attributes.push(input_datum[attr_id]);

        // --- Assume that repeats are basically layered one after another ---
        // let num_repeats = attr_num_hits.get(&decision_nodes[current_node_idx].attr_id).unwrap_or(&0);
        // let offset = *num_repeats * (DUMMY_INPUT_LEN as u32);
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
    (
        path_decision_nodes,
        ret_leaf_node,
        diffs,
        used_input_attributes,
    )
}

fn generate_16_bit_signed_decomp<F: FieldExt>(value: F) -> BinDecomp16Bit<F> {
    let upper_bound = F::from(2_u32.pow(16) - 1);

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
    let upper_bound = F::from(2_u32.pow(16) - 1);
    assert!(value >= F::zero());
    assert!(value <= upper_bound);

    // --- Grab the string repr ---
    let binary_repr = format!("{:0>16b}", value.into_bigint().as_ref()[0]);

    // --- Length must be 16, then parse as array of length 16 ---
    let mut binary_repr_arr = [F::zero(); 16];
    for idx in 0..16 {
        let char_repr = binary_repr.chars().nth(idx).unwrap();
        assert!(char_repr == '0' || char_repr == '1');
        binary_repr_arr[idx] = if char_repr == '0' {
            F::zero()
        } else {
            F::one()
        }
    }

    BinDecomp16Bit {
        bits: binary_repr_arr,
    }
}

/// Need to generate dummy circuit inputs, starting with the input data
/// Then get the path data and binary decomp stuff
/// TODO!(ryancao): add the attribute index field to `InputAttribute<F>`
/// -- Actually, scratch the above: we might be getting rid of `attr_id`s
/// altogether and replacing with `attr_idx` everywhere (as suggested by Ben!)
fn generate_dummy_data<F: FieldExt>() -> (
    // Vec<Vec<F>>,                    // Input attribute indices
    Vec<Vec<InputAttribute<F>>>, // Input attributes
    // Vec<Vec<F>>,                    // Permuted input attribute indices
    Vec<Vec<InputAttribute<F>>>, // Permuted input attributes
    Vec<Vec<DecisionNode<F>>>,   // Paths (decision node part only)
    Vec<LeafNode<F>>,            // Paths (leaf node part only)
    Vec<Vec<BinDecomp16Bit<F>>>, // Binary decomp of differences
    Vec<BinDecomp16Bit<F>>,      // Binary decomp of multiplicities
    Vec<DecisionNode<F>>,        // Actual tree decision nodes
    Vec<LeafNode<F>>,            // Actual tree leaf nodes
) {
    // --- Get the RNG ---
    let mut rng = test_rng();

    // --- Generate dummy input data ---
    let mut dummy_input_data: Vec<Vec<InputAttribute<F>>> = vec![];
    // let mut dummy_permuted_input_data: Vec<Vec<InputAttribute<F>>> = vec![];

    // --- Generate dummy input index data ---
    let dummy_attr_idx_data = (0..(DUMMY_INPUT_LEN * (TREE_HEIGHT - 1)))
        .map(|x| F::from(x as u16))
        .collect_vec();
    let dummy_attr_idx_data = repeat_n(dummy_attr_idx_data, NUM_DUMMY_INPUTS).collect_vec();

    // --- Populate (note that we have to permute later) ---
    for _ in 0..NUM_DUMMY_INPUTS {
        // --- Generate a single copy of all the attributes ---
        let mut single_attribute_copy = vec![];
        // let mut single_permuted_attribute_copy = vec![];
        for attr_id in 0..DUMMY_INPUT_LEN {
            let input_attribute = InputAttribute {
                attr_id: F::from(attr_id as u16),
                attr_val: F::from(rng.gen_range(0..(2_u16.pow(12)) as u16)),
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
            node_id: F::from(idx as u16),
            attr_id: F::from(rng.gen_range(0..DUMMY_INPUT_LEN as u16)),
            threshold: F::from(rng.gen_range(0..(2_u16.pow(12)))),
        };
        dummy_decision_nodes.push(decision_node);
    }

    // --- Populate leaf nodes ---
    for idx in NUM_DECISION_NODES..(NUM_DECISION_NODES + NUM_LEAF_NODES) {
        let leaf_node = LeafNode {
            node_id: F::from(idx as u16),
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
        .map(|(x, _, _, _)| x)
        .collect_vec();

    // --- Collect correct leaf nodes ---
    let dummy_leaf_node_paths = dummy_auxiliaries
        .clone()
        .into_iter()
        .map(|(_, x, _, _)| x)
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
        .map(|(_, _, _, used_input_attributes)| used_input_attributes)
        .collect_vec();
    let dummy_permuted_input_data = zip(all_used_input_attributes, dummy_input_data.clone())
        .map(|(used_input_attributes, original_input_attributes)| {
            let mut used_input_attributes_clone = used_input_attributes.clone();

            // --- Basically need to create a new vector with input attributes ---
            let ret = used_input_attributes
                .clone()
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

            return ret;
        })
        .collect_vec();

    // --- Compute multiplicities: just add the ones that are given in the returned map ---
    // TODO!(ryancao): Just use the paths already! The decision nodes are there!
    let multiplicities: Vec<F> = vec![F::zero(); (2_u32.pow(TREE_HEIGHT as u32) - 1) as usize];
    let dummy_multiplicities_bin_decomp = dummy_auxiliaries
        .clone()
        .into_iter()
        .fold(
            multiplicities,
            |prev_multiplicities, (path_decision_nodes, path_leaf_node, _, _)| {
                // --- TODO!(ryancao): This is so bad lol ---
                let mut new_multiplicities: Vec<F> = prev_multiplicities.clone();

                // --- Just grab the node IDs from each decision node and add them to the multiplicities ---
                path_decision_nodes.into_iter().for_each(|decision_node| {
                    let node_id = decision_node.node_id.into_bigint().as_ref()[0] as usize;
                    new_multiplicities[node_id] += F::one();
                });

                // --- Count the leaf node as well! ---
                let node_id = path_leaf_node.node_id.into_bigint().as_ref()[0] as usize;
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
        .clone()
        .into_iter()
        .map(|(_, _, diffs, _)| {
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

    (
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
    )
}

/// Gets the sign bit (0 for positive, 1 for negative) and abs value
fn get_sign_bit_and_abs_value<F: FieldExt>(value: F) -> (F, F) {
    let upper_bound = F::from(2_u32.pow(16) - 1);
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
    return (sign_bit, abs_value);
}

/// Computes the recomposition of the bits within `decomp` and checks
fn check_signed_recomposition<F: FieldExt>(actual_value: F, decomp: BinDecomp16Bit<F>) -> bool {
    let (sign_bit, _) = get_sign_bit_and_abs_value(actual_value);
    let mut total = F::zero();

    // --- Perform recomposition of non-sign bits ---
    for bit_idx in 1..16 {
        let base = F::from(2_u32.pow((16 - (bit_idx + 1)) as u32));
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

/// Takes the above dummy data from `generate_dummy_data()` and converts
/// into MLE form factor.
fn generate_dummy_mles<F: FieldExt>() -> (
    // DenseMle<F, F>,
    DenseMle<F, InputAttribute<F>>,
    // DenseMle<F, F>,
    DenseMle<F, InputAttribute<F>>,
    DenseMle<F, DecisionNode<F>>,
    DenseMle<F, LeafNode<F>>,
    DenseMle<F, BinDecomp16Bit<F>>,
    DenseMle<F, BinDecomp16Bit<F>>,
    DenseMle<F, DecisionNode<F>>,
    DenseMle<F, LeafNode<F>>,
) {
    // --- First generate the dummy data ---
    let (
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
    ) = generate_dummy_data::<F>();

    // --- Generate MLEs for each ---
    // TODO!(ryancao): Change this into batched form
    // let dummy_attr_idx_data_mle = DenseMle::<_, F>::new(dummy_attr_idx_data[0].clone());
    let dummy_input_data_mle = dummy_input_data[0]
        .clone()
        .into_iter()
        .map(InputAttribute::from)
        .collect::<DenseMle<F, InputAttribute<F>>>();
    // let dummy_permutation_indices_mle = DenseMle::<_, F>::new(dummy_permutation_indices[0].clone());
    let dummy_permuted_input_data_mle = dummy_permuted_input_data[0]
        .clone()
        .into_iter()
        .map(InputAttribute::from)
        .collect::<DenseMle<F, InputAttribute<F>>>();
    let dummy_decision_node_paths_mle = dummy_decision_node_paths[0]
        .clone()
        .into_iter()
        .map(DecisionNode::from)
        .collect::<DenseMle<F, DecisionNode<F>>>();
    let dummy_leaf_node_paths_mle = vec![dummy_leaf_node_paths[0]]
        .into_iter()
        .map(LeafNode::from)
        .collect::<DenseMle<F, LeafNode<F>>>();
    let dummy_binary_decomp_diffs_mle = dummy_binary_decomp_diffs[0]
        .clone()
        .into_iter()
        .map(BinDecomp16Bit::from)
        .collect::<DenseMle<F, BinDecomp16Bit<F>>>();
    let dummy_multiplicities_bin_decomp_mle = dummy_multiplicities_bin_decomp
        .clone()
        .into_iter()
        .map(BinDecomp16Bit::from)
        .collect::<DenseMle<F, BinDecomp16Bit<F>>>();
    let dummy_decision_nodes_mle = dummy_decision_nodes
        .clone()
        .into_iter()
        .map(DecisionNode::from)
        .collect::<DenseMle<F, DecisionNode<F>>>();
    let dummy_leaf_nodes_mle = dummy_leaf_nodes
        .clone()
        .into_iter()
        .map(LeafNode::from)
        .collect::<DenseMle<F, LeafNode<F>>>();

    (
        // dummy_attr_idx_data_mle,
        dummy_input_data_mle,
        // dummy_permutation_indices_mle,
        dummy_permuted_input_data_mle,
        dummy_decision_node_paths_mle,
        dummy_leaf_node_paths_mle,
        dummy_binary_decomp_diffs_mle,
        dummy_multiplicities_bin_decomp_mle,
        dummy_decision_nodes_mle,
        dummy_leaf_nodes_mle,
    )
}

// --- Create expressions using... testing modules? ---
#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        expression::{Expression, ExpressionStandard},
        layer::Claim,
        mle::{beta::BetaTable, dense::DenseMle, dense::DenseMleRef, Mle, MleRef},
        sumcheck::{
            compute_sumcheck_message, get_round_degree,
            tests::{dummy_sumcheck, verify_sumcheck_messages},
            Evals,
        },
    };
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use ark_std::One;
    use ark_std::UniformRand;
    use ark_std::Zero;

    /// Checks that bits within the diff binary decomp and the multiplicity
    /// binary decomp are all either 0 or 1
    #[test]
    fn dummy_bits_are_binary_test() {
        // --- First generate the dummy data ---
        let (
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
        ) = generate_dummy_data::<Fr>();

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
            dummy_multiplicities_bin_decomp.len() == (NUM_DECISION_NODES + NUM_LEAF_NODES) as usize
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
                Fr::rand(&mut rng),
                Fr::rand(&mut rng),
                Fr::rand(&mut rng),
                Fr::rand(&mut rng),
            ],
            Fr::zero(),
        );
        let mut beta = BetaTable::new(layer_claim.clone()).unwrap();
        beta.table.index_mle_indices(0);

        let (
            // dummy_attr_idx_data_mle,
            dummy_input_data_mle,
            // dummy_permutation_indices_mle,
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle,
            dummy_binary_decomp_diffs_mle,
            dummy_multiplicities_bin_decomp_mle,
            dummy_decision_nodes_mle,
            dummy_leaf_nodes_mle,
        ) = generate_dummy_mles::<Fr>();

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
        let dummy_claim = (vec![Fr::from(1); 3], Fr::zero());
        let b_minus_b_squared_clone = b_minus_b_squared.clone();
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

        let res_messages = dummy_sumcheck(
            &mut b_minus_b_squared_clone.clone(),
            &mut rng,
            layer_claim.clone(),
        );
        let verify_res =
            verify_sumcheck_messages(res_messages, b_minus_b_squared_clone, layer_claim, &mut rng);
        assert!(verify_res.is_ok());
    }

    /// basic "bits are binary" test (for multiplicities), but in circuit!
    #[test]
    fn circuit_dummy_bits_are_binary_test_multiplicities() {
        let mut rng = test_rng();
        let layer_claim: Claim<Fr> = (vec![Fr::rand(&mut rng); 12], Fr::zero());
        let mut beta = BetaTable::new(layer_claim.clone()).unwrap();
        beta.table.index_mle_indices(0);

        let (
            // dummy_attr_idx_data_mle,
            dummy_input_data_mle,
            // dummy_permutation_indices_mle,
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle,
            dummy_binary_decomp_diffs_mle,
            dummy_multiplicities_bin_decomp_mle,
            dummy_decision_nodes_mle,
            dummy_leaf_nodes_mle,
        ) = generate_dummy_mles::<Fr>();

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
            .repeat(2_u32.pow(first_bin_decomp_bit_mle[0].num_vars() as u32) as usize);
        let all_zeros_mle = DenseMle::new(all_zeros);
        let all_zeros_mle_expr = ExpressionStandard::Mle(all_zeros_mle.mle_ref());

        // --- Evaluating at V(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) --> 0 ---
        let dummy_claim = (vec![Fr::one(); 3 + 9], Fr::zero());

        // --- Initialize beta tables manually ---
        let b_minus_b_squared_clone = b_minus_b_squared.clone();
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

        let res_messages = dummy_sumcheck(
            &mut b_minus_b_squared_clone.clone(),
            &mut rng,
            layer_claim.clone(),
        );
        let verify_res =
            verify_sumcheck_messages(res_messages, b_minus_b_squared_clone, layer_claim, &mut rng);
        assert!(verify_res.is_ok());
    }

    /// Binary recomposition test (out of circuit)
    #[test]
    fn dummy_binary_recomp_test() {
        // --- First generate the dummy data ---
        let (
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
        ) = generate_dummy_data::<Fr>();

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
                Fr::rand(&mut rng),
                Fr::rand(&mut rng),
                Fr::rand(&mut rng),
                Fr::rand(&mut rng),
            ],
            Fr::zero(),
        );
        let mut beta = BetaTable::new(layer_claim).unwrap();
        beta.table.index_mle_indices(0);
        let (
            // dummy_attr_idx_data_mle,
            dummy_input_data_mle,
            // dummy_permutation_indices_mle,
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle,
            dummy_binary_decomp_diffs_mle,
            dummy_multiplicities_bin_decomp_mle,
            dummy_decision_nodes_mle,
            dummy_leaf_nodes_mle,
        ) = generate_dummy_mles::<Fr>();

        // --- Grab the bin decomp MLEs and associated expressions ---
        let bin_decomp_mles: Vec<DenseMleRef<Fr>> = dummy_binary_decomp_diffs_mle.mle_bit_refs();

        // --- Grab the things necessary to compute the diff (the permuted input and thresholds) ---
        let threshold_mle: DenseMleRef<Fr> = dummy_decision_node_paths_mle.threshold();
        let threshold_mle_expr = ExpressionStandard::Mle(threshold_mle.clone());
        let permuted_input_values_mle: DenseMleRef<Fr> =
            dummy_permuted_input_data_mle.attr_val(Some(threshold_mle.num_vars()));
        let permuted_input_values_mle_expr =
            ExpressionStandard::Mle(permuted_input_values_mle.clone());

        // --- For debugging ---
        // let threshold_mle_expr_eval = evaluate_expr(&mut threshold_mle_expr.clone(), 1, 2);
        // dbg!(threshold_mle_expr_eval);
        // let permuted_input_values_mle_expr_eval = evaluate_expr(&mut permuted_input_values_mle_expr.clone(), 1, 2);
        // dbg!(permuted_input_values_mle_expr_eval);

        // --- Need to just get diff ---
        // dbg!(permuted_input_values_mle.num_vars); // Should be 3
        // dbg!(threshold_mle.num_vars); // Should be 3
        let mut diff_expr = permuted_input_values_mle_expr.clone() - threshold_mle_expr.clone();
        // let permuted_input_values_mle_expr_eval = compute_sumcheck_message(&mut permuted_input_values_mle_expr.clone(), 1, 2);
        // let threshold_mle_expr_eval = compute_sumcheck_message(&mut threshold_mle_expr.clone(), 1, 2);
        // dbg!(permuted_input_values_mle_expr_eval);
        // dbg!(threshold_mle_expr_eval);

        // --- We need `abs_recomp` and `b_s * abs_recomp` ---
        let b_s_initial_acc = ExpressionStandard::Constant(Fr::zero());
        let sign_bit_mle = bin_decomp_mles[0].clone();
        let bin_decomp_mles_clone = bin_decomp_mles.clone();

        // --- Time for iterators... sigh ---
        let mut b_s_times_abs_recomp_expr = bin_decomp_mles.into_iter().enumerate().skip(1).fold(
            b_s_initial_acc,
            |acc_expr, (bit_idx, bin_decomp_mle)| {
                // --- First compute b_s * coeff ---
                let b_s_times_coeff =
                    ExpressionStandard::Product(vec![bin_decomp_mle.clone(), sign_bit_mle.clone()]);

                let b_s_times_coeff_ptr = Box::new(b_s_times_coeff);

                // --- Then compute (b_s * coeff) * 2^{bit_idx} ---
                let base = Fr::from(2_u32.pow((16 - (bit_idx + 1)) as u32));
                let b_s_times_coeff_times_base =
                    ExpressionStandard::Scaled(b_s_times_coeff_ptr, base);

                // Debugging
                // let b_i_expr = ExpressionStandard::Mle(bin_decomp_mle.clone());
                // let b_i_expr_eval = evaluate_expr(&mut b_i_expr.clone(), 1, 1);
                // let b_s_times_coeff_times_base_eval = evaluate_expr(&mut b_s_times_coeff_times_base.clone(), 1, 2);
                // dbg!(bit_idx);
                // dbg!(bin_decomp_mle.clone().num_vars);
                // dbg!(b_i_expr_eval);
                // dbg!(b_s_times_coeff_times_base_eval);

                acc_expr + b_s_times_coeff_times_base
            },
        );

        let abs_recomp_initial_acc = ExpressionStandard::Constant(Fr::zero());
        let mut abs_recomp_expr = bin_decomp_mles_clone.into_iter().enumerate().skip(1).fold(
            abs_recomp_initial_acc,
            |acc_expr, (bit_idx, bin_decomp_mle)| {
                // --- Compute just coeff * 2^{bit_idx} ---
                let base = Fr::from(2_u32.pow((16 - (bit_idx + 1)) as u32));
                let coeff_expr = ExpressionStandard::Mle(bin_decomp_mle);
                let coeff_expr_ptr = Box::new(coeff_expr);
                let coeff_times_base = ExpressionStandard::Scaled(coeff_expr_ptr, base);

                // Debugging
                let coeff_times_base_eval =
                    compute_sumcheck_message(&mut coeff_times_base.clone(), 1, 2, &mut beta);

                acc_expr + coeff_times_base
            },
        );

        // --- Subtract the two, and (TODO!(ryancao)) ensure they have the same number of variables ---
        let mut final_expr = diff_expr.clone() - abs_recomp_expr.clone()
            + b_s_times_abs_recomp_expr.clone()
            + b_s_times_abs_recomp_expr.clone();

        // --- Let's just see what the expressions give us... ---
        // Debugging
        // let diff_result = compute_sumcheck_message(&mut diff_expr, 1, 2);
        // let abs_recomp_expr_result = compute_sumcheck_message(&mut abs_recomp_expr, 1, 2);
        // let b_s_times_abs_recomp_expr_result = compute_sumcheck_message(&mut b_s_times_abs_recomp_expr, 1, 2);
        // dbg!(diff_result);
        // dbg!(abs_recomp_expr_result);
        // dbg!(b_s_times_abs_recomp_expr_result);

        // let dummy_claim = (vec![Fr::one(); 3], Fr::zero());

        let final_expr_clone = final_expr.clone();

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
        let (
            // dummy_attr_idx_data_mle,
            dummy_input_data_mle,
            // dummy_permutation_indices_mle,
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle,
            dummy_binary_decomp_diffs_mle,
            dummy_multiplicities_bin_decomp_mle,
            dummy_decision_nodes_mle,
            dummy_leaf_nodes_mle,
        ) = generate_dummy_mles::<Fr>();

        let mut rng = test_rng();

        // --- Get packed inputs first ---
        let r1: Fr = rng.gen();
        let r2: Fr = rng.gen();

        // --- Multiply to do packing ---
        let dummy_attribute_id_mleref = dummy_input_data_mle.attr_id(None);
        let dummy_attribute_id_mleref_expr = ExpressionStandard::Mle(dummy_attribute_id_mleref);
        let dummy_attribute_val_mleref = dummy_input_data_mle.attr_val(None);
        let dummy_attribute_val_mleref_expr = ExpressionStandard::Mle(dummy_attribute_val_mleref);

        // ---
    }
}
