use crate::FieldExt;
use crate::mle::dense::DenseMle;
use super::structs::*;

use itertools::{Itertools, repeat_n};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
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

/// - First element is the decision path nodes
/// - Second element is the final predicted leaf node
/// - Third element is the attributes used
/// - Fourth element is the multiplicities of the tree nodes which were hit
/// - Fifth element is the difference between the current attribute value and the node threshold
fn generate_correct_path_and_permutation<F: FieldExt>(
    decision_nodes: &Vec<DecisionNode<F>>,
    leaf_nodes: &Vec<LeafNode<F>>,
    input_datum: &Vec<InputAttribute<F>>
) -> (Vec<DecisionNode<F>>, LeafNode<F>, Vec<F>, HashMap<F, u32>, Vec<F>) {

    // --- Keep track of the path and permutation ---
    let mut path_decision_nodes: Vec<DecisionNode<F>> = vec![];
    let mut permuted_access_indices: Vec<F> = vec![];

    // --- Keep track of how many times each tree node was used ---
    // Key: Tree node index (decision OR leaf node)
    // Value: Multiplicity
    let mut attr_num_hits: HashMap<F, u32> = HashMap::new();

    // --- Keep track of the differences for path nodes ---
    let mut diffs: Vec<F> = vec![];

    // --- Go through the decision nodes ---
    let mut current_node_idx = 0;
    while current_node_idx < decision_nodes.len() {

        // --- Add to path; grab the appropriate attribute index ---
        path_decision_nodes.push(decision_nodes[current_node_idx]);
        let attr_idx = (decision_nodes[current_node_idx].attr_id.into_bigint()).as_ref()[0] as usize;

        // --- Assume that repeats are basically layered one after another ---
        let num_repeats = attr_num_hits.get(&decision_nodes[current_node_idx].attr_id).unwrap_or(&0);
        let offset = *num_repeats * (DUMMY_INPUT_LEN as u32);
        permuted_access_indices.push(decision_nodes[current_node_idx].attr_id * F::from(offset));

        // --- Adds a hit to the current node ---
        let cur_multiplicity = attr_num_hits.entry(decision_nodes[current_node_idx].node_id).or_insert(0);
        *cur_multiplicity += 1;

        // --- Compute the difference ---
        let diff = input_datum[attr_idx].attr_val - decision_nodes[current_node_idx].threshold;
        diffs.push(diff);

        // --- Check if we should go left or right ---
        if input_datum[attr_idx].attr_val > decision_nodes[current_node_idx].threshold {
            current_node_idx = current_node_idx * 2 + 2;
        } else {
            current_node_idx = current_node_idx * 2 + 1;
        }
    }

    // --- Leaf node indices are offset by 2^{TREE_HEIGHT} ---
    let ret_leaf_node = leaf_nodes[current_node_idx - 2_u32.pow(TREE_HEIGHT as u32) as usize];

    (path_decision_nodes, ret_leaf_node, permuted_access_indices, attr_num_hits, diffs)

}

fn generate_16_bit_signed_decomp<F: FieldExt>(value: F) -> BinDecomp16Bit<F> {

    // --- Compute the sign bit ---
    let sign_bit = if value >= F::zero() {F::zero()} else {F::one()};

    // --- Convert to positive ---
    let abs_value = if value >= F::zero() {value} else {value.neg()};

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
    assert!(value >= F::zero());
    assert!(value <= F::from(2_u32.pow(16) - 1));

    // --- Grab the string repr ---
    let binary_repr = format!("{:b}", value.into_bigint().as_ref()[0]);

    // --- Length must be 16, then parse as array of length 16 ---
    dbg!(binary_repr);
    assert!(binary_repr.clone().len() == 16);
    let mut binary_repr_arr = [F::zero(); 16];
    for idx in 0..16 {
        let char_repr = binary_repr.chars().nth(idx).unwrap();
        assert!(char_repr == '0' || char_repr == '1');
        binary_repr_arr[idx] = if char_repr == '0' {F::zero()} else {F::one()}
    }

    BinDecomp16Bit { bits: binary_repr_arr }
}

/// Need to generate dummy circuit inputs, starting with the input data
/// Then get the path data and binary decomp stuff
fn generate_dummy_data<F: FieldExt>() -> (
    Vec<Vec<F>>,                    // Input attribute indices
    Vec<Vec<InputAttribute<F>>>,    // Input attributes
    Vec<Vec<F>>,                    // Permuted input attribute indices
    Vec<Vec<InputAttribute<F>>>,    // Permuted input attributes
    Vec<Vec<DecisionNode<F>>>,      // Paths (decision node part only)
    Vec<LeafNode<F>>,               // Paths (leaf node part only)
    Vec<Vec<BinDecomp16Bit<F>>>,    // Binary decomp of differences
    Vec<BinDecomp16Bit<F>>,         // Binary decomp of multiplicities
    Vec<DecisionNode<F>>,           // Actual tree decision nodes
    Vec<LeafNode<F>>,               // Actual tree leaf nodes
) {

    // --- Grabbing the RNG with seed for deterministic outputs ---
    let seed: [u8; 32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    // --- Generate dummy input data ---
    let mut dummy_input_data: Vec<Vec<InputAttribute<F>>> = vec![];
    let mut dummy_permuted_input_data: Vec<Vec<InputAttribute<F>>> = vec![];

    // --- Generate dummy input index data ---
    let dummy_attr_idx_data = (0..(DUMMY_INPUT_LEN * (TREE_HEIGHT - 1)))
        .map(|x| {
            F::from(x as u16)
        })
        .collect_vec();
    let dummy_attr_idx_data = repeat_n(dummy_attr_idx_data, NUM_DUMMY_INPUTS).collect_vec();

    // --- Populate (NOTE that we have to permute later) ---
    for dummy_input in 0..NUM_DUMMY_INPUTS {

        // --- Generate a single copy of all the attributes ---
        let mut single_attribute_copy = vec![];
        let mut single_permuted_attribute_copy = vec![];
        for _ in 0..DUMMY_INPUT_LEN {
            let input_attribute = InputAttribute {
                attr_id: F::from(dummy_input as u16),
                attr_val: F::from(rng.gen::<u16>()),
            };
            single_attribute_copy.push(input_attribute);
            single_permuted_attribute_copy.push(input_attribute);
        }

        // --- Have to repeat attributes TREE_HEIGHT - 1 times ---
        let dummy_input_datum = single_attribute_copy.clone().into_iter().cycle().take(single_attribute_copy.len() * (TREE_HEIGHT - 1)).collect_vec();
        let dummy_permuted_input_datum = single_permuted_attribute_copy.clone().into_iter().cycle().take(single_permuted_attribute_copy.len() * (TREE_HEIGHT - 1)).collect_vec();

        // --- Add to final list ---
        dummy_input_data.push(dummy_input_datum);
        dummy_permuted_input_data.push(dummy_permuted_input_datum);
    }

    // --- Generate a dummy tree ---
    let mut dummy_decision_nodes: Vec<DecisionNode<F>> = vec![];
    let mut dummy_leaf_nodes: Vec<LeafNode<F>> = vec![];

    // --- Populate decision nodes ---
    // Note that attr_id can only be in [0, DUMMY_INPUT_LEN)
    for idx in 0..(2_u32.pow(TREE_HEIGHT as u32)) - 1 {
        let decision_node = DecisionNode {
            node_id: F::from(idx as u16),
            attr_id: F::from(rng.gen_range(0..DUMMY_INPUT_LEN as u16)),
            threshold: F::from(rng.gen::<u16>()),
        };
        dummy_decision_nodes.push(decision_node);
    }

    // --- Populate leaf nodes ---
    for idx in 2_u32.pow(TREE_HEIGHT as u32)..2_u32.pow(TREE_HEIGHT as u32 + 1) {
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
            generate_correct_path_and_permutation(&dummy_decision_nodes, &dummy_leaf_nodes, &dummy_attrs)
        }).collect_vec();

    // --- Collect correct paths ---
    let dummy_decision_node_paths = dummy_auxiliaries
        .clone()
        .into_iter()
        .map(|(x, _, _, _, _)| {
            x
        }).collect_vec();

    // --- Collect correct leaf nodes ---
    let dummy_leaf_node_paths = dummy_auxiliaries
        .clone()
        .into_iter()
        .map(|(_, x, _, _, _)| {
            x
        }).collect_vec();

    // --- Collect correct permutation indices ---
    let dummy_permutation_indices = dummy_auxiliaries
        .clone()
        .into_iter()
        .map(|(_, _, x, _, _)| {
            x
        }).collect_vec();

    // --- Compute multiplicities: just add the ones that are given in the returned map ---
    // TODO!(ryancao): Compute the binary decompositions
    let multiplicities: Vec<F> = vec![F::zero(); 2_u32.pow(TREE_HEIGHT as u32 + 1) as usize];
    let dummy_multiplicities_bin_decomp = dummy_auxiliaries
        .clone()
        .into_iter()
        .fold(multiplicities, |prev_multiplicities, (_, _, _, new_multiplicities_map, _)| {

            let mut new_multiplicities: Vec<F> = prev_multiplicities.clone();

            // --- Just increment the current vector of multiplicities by the map at that index ---
            new_multiplicities_map.into_iter().for_each(|(tree_node_idx, multiplicity)| {
                let usize_tree_node_idx = tree_node_idx.into_bigint().as_ref()[0] as usize;
                new_multiplicities[usize_tree_node_idx] += F::from(multiplicity);
            });

            // --- Just add the corresponding terms ---
            // let updated_multiplicities = zip(prev_multiplicities, new_multiplicities)
            //     .map(|(a, b)| {
            //         a + b
            //     })
            //     .collect_vec();

            new_multiplicities
        }).into_iter()
        .map(|multiplicity| {
            // --- Grab the binary decomp ---
            generate_16_bit_unsigned_decomp(multiplicity)
        }).collect_vec();

    // --- Compute the binary decompositions of the differences ---
    let dummy_binary_decomp_diffs = dummy_auxiliaries
        .clone()
        .into_iter()
        .map(|(_, _, _, _, diffs)| {
            diffs.into_iter().map(|diff| {
                generate_16_bit_signed_decomp(diff)
            }).collect_vec()
        }).collect_vec();

    (
        dummy_attr_idx_data,
        dummy_input_data,
        dummy_permutation_indices,
        dummy_permuted_input_data,
        dummy_decision_node_paths,
        dummy_leaf_node_paths,
        dummy_binary_decomp_diffs,
        dummy_multiplicities_bin_decomp,
        dummy_decision_nodes,
        dummy_leaf_nodes
    )
}

/// Takes the above dummy data from `generate_dummy_data()` and converts
/// into MLE form factor.
fn generate_dummy_mles<F: FieldExt>() -> (
    DenseMle<F, F>,
    DenseMle<F, InputAttribute<F>>,
    DenseMle<F, F>,
    DenseMle<F, InputAttribute<F>>,
    DenseMle<F, DecisionNode<F>>,
    DenseMle<F, LeafNode<F>>,
    DenseMle<F, BinDecomp16Bit<F>>,
    DenseMle<F, BinDecomp16Bit<F>>,
    DenseMle<F, DecisionNode<F>>,
    DenseMle<F, LeafNode<F>>
){

    // --- First generate the dummy data ---
    let (
        dummy_attr_idx_data,
        dummy_input_data,
        dummy_permutation_indices,
        dummy_permuted_input_data,
        dummy_decision_node_paths,
        dummy_leaf_node_paths,
        dummy_binary_decomp_diffs,
        dummy_multiplicities_bin_decomp,
        dummy_decision_nodes,
        dummy_leaf_nodes
    ) = generate_dummy_data::<F>();

    // --- Generate MLEs for each ---
    // TODO!(ryancao): Change this into batched form
    let dummy_attr_idx_data_mle = DenseMle::<_, F>::new(dummy_attr_idx_data[0].clone());
    let dummy_input_data_mle = dummy_input_data[0]
        .clone()
        .into_iter()
        .map(InputAttribute::from)
        .collect::<DenseMle<F, InputAttribute<F>>>();
    let dummy_permutation_indices_mle = DenseMle::<_, F>::new(dummy_permutation_indices[0].clone());
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
    let dummy_multiplicities_bin_decomp_mle = vec![dummy_multiplicities_bin_decomp[0]]
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
        dummy_attr_idx_data_mle,
        dummy_input_data_mle,
        dummy_permutation_indices_mle,
        dummy_permuted_input_data_mle,
        dummy_decision_node_paths_mle,
        dummy_leaf_node_paths_mle,
        dummy_binary_decomp_diffs_mle,
        dummy_multiplicities_bin_decomp_mle,
        dummy_decision_nodes_mle,
        dummy_leaf_nodes_mle
    )
}

// --- Create expressions using... testing modules? ---
#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        expression::{ExpressionStandard},
        mle::{dense::DenseMle, Mle, dense::DenseMleRef},
        sumcheck::{evaluate_expr, SumOrEvals},
    };
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use ark_std::One;
    use ark_std::Zero;

    /// Basic binary decomposition test
    #[test]
    fn eval_expr_nums() {

    let (
        dummy_attr_idx_data_mle,
        dummy_input_data_mle,
        dummy_permutation_indices_mle,
        dummy_permuted_input_data_mle,
        dummy_decision_node_paths_mle,
        dummy_leaf_node_paths_mle,
        dummy_binary_decomp_diffs_mle,
        dummy_multiplicities_bin_decomp_mle,
        dummy_decision_nodes_mle,
        dummy_leaf_nodes_mle
    ) = generate_dummy_mles::<Fr>();

        // --- Grab the bin decomp MLE ---
        let first_bin_decomp_bit_mle: Vec<DenseMleRef<Fr>> = dummy_binary_decomp_diffs_mle.mle_bit_refs();
        let first_bin_decomp_bit_expr = ExpressionStandard::Mle(first_bin_decomp_bit_mle[0].clone());

        // --- Do b * (1 - b) = b - b^2 ---
        let b_squared = ExpressionStandard::Product(vec![first_bin_decomp_bit_mle[0].clone(), first_bin_decomp_bit_mle[0].clone()]);
        let mut b_minus_b_squared = first_bin_decomp_bit_expr - b_squared;

        // --- We should get all zeros ---
        let all_zeros: Vec<Fr> = vec![Fr::zero()].repeat(2_u32.pow(first_bin_decomp_bit_mle[0].num_vars as u32) as usize);
        let all_zeros_mle = DenseMle::new(all_zeros);
        let mut all_zeros_mle_expr = ExpressionStandard::Mle(all_zeros_mle.mle_ref());

        let res = evaluate_expr(&mut b_minus_b_squared, 1, 2);
        let other_res = evaluate_expr(&mut all_zeros_mle_expr, 1, 2);
        assert_eq!(res.unwrap(), other_res.unwrap());
        // let exp = SumOrEvals::Sum(Fr::from(7));
        // assert_eq!(res.unwrap(), exp);
    }
}