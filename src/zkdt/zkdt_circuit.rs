use crate::FieldExt;
use crate::mle::dense::DenseMle;
use super::structs::*;

use itertools::{Itertools, repeat_n};
use rand::Rng;
use std::collections::HashMap;
use std::iter::zip;
use ark_std::test_rng;

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
const TREE_HEIGHT: usize = 8;

/// - First element is the decision path nodes
/// - Second element is the final predicted leaf node
/// - Third element is the attributes used
/// - Fourth element is the difference between the current attribute value and the node threshold
fn generate_correct_path_and_permutation<F: FieldExt>(
    decision_nodes: &Vec<DecisionNode<F>>,
    leaf_nodes: &Vec<LeafNode<F>>,
    input_datum: &Vec<InputAttribute<F>>
) -> (Vec<DecisionNode<F>>, LeafNode<F>, Vec<F>, Vec<F>) {

    // --- Keep track of the path and permutation ---
    let mut path_decision_nodes: Vec<DecisionNode<F>> = vec![];
    let mut permuted_access_indices: Vec<F> = vec![];

    // --- Keep track of how many times each attribute ID was used (to get the corresponding attribute idx) ---
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
        permuted_access_indices.push(decision_nodes[current_node_idx].attr_id + F::from(offset));

        // --- Adds a hit to the current attribute ID ---
        let num_attr_id_hits = attr_num_hits.entry(decision_nodes[current_node_idx].attr_id).or_insert(0);
        *num_attr_id_hits += 1;

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
    let ret_leaf_node = leaf_nodes[current_node_idx - (2_u32.pow(TREE_HEIGHT as u32) - 1) as usize];

    (path_decision_nodes, ret_leaf_node, permuted_access_indices, diffs)

}

fn generate_16_bit_signed_decomp<F: FieldExt>(value: F) -> BinDecomp16Bit<F> {

    let upper_bound = F::from(2_u32.pow(16) - 1);

    // --- Compute the sign bit ---
    let sign_bit = if value <= upper_bound {F::zero()} else {F::one()};

    // --- Convert to positive ---
    let abs_value = if value <= upper_bound {value} else {value.neg()};

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
        binary_repr_arr[idx] = if char_repr == '0' {F::zero()} else {F::one()}
    }

    BinDecomp16Bit { bits: binary_repr_arr }
}

/// Need to generate dummy circuit inputs, starting with the input data
/// Then get the path data and binary decomp stuff
/// TODO!(ryancao): add the attribute index field to `InputAttribute<F>`
/// -- Actually, scratch the above: we might be getting rid of `attr_id`s
/// altogether and replacing with `attr_idx` everywhere (as suggested by Ben!)
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
    for _ in 0..NUM_DUMMY_INPUTS {

        // --- Generate a single copy of all the attributes ---
        let mut single_attribute_copy = vec![];
        let mut single_permuted_attribute_copy = vec![];
        for attr_id in 0..DUMMY_INPUT_LEN {
            let input_attribute = InputAttribute {
                attr_id: F::from(attr_id as u16),
                // Val can be anything from 0 to 1023
                attr_val: F::from(test_rng().gen_range(0..(2_u16.pow(12)) as u16)),
                // attr_val: F::from(dummy_input as u16), // TODO!(ryancao): For debugging purposes only!
                // attr_val: F::one(),
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
    for idx in 0..(2_u32.pow(TREE_HEIGHT as u32) - 1) {
        let decision_node = DecisionNode {
            node_id: F::from(idx as u16),
            attr_id: F::from(test_rng().gen_range(0..DUMMY_INPUT_LEN as u16)),
            // Same here; threshold is anything from 0 to 1023
            threshold: F::from(test_rng().gen_range(0..(2_u16.pow(12)))),
            // threshold: F::from(0), // TODO!(ryancao): For debugging purposes only!
        };
        dummy_decision_nodes.push(decision_node);
    }

    // --- Populate leaf nodes ---
    for idx in 2_u32.pow(TREE_HEIGHT as u32)..2_u32.pow(TREE_HEIGHT as u32 + 1) {
        let leaf_node = LeafNode {
            node_id: F::from(idx as u16),
            node_val: F::from(test_rng().gen::<u64>()),
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
        .map(|(x, _, _, _)| {
            x
        }).collect_vec();

    // --- Collect correct leaf nodes ---
    let dummy_leaf_node_paths = dummy_auxiliaries
        .clone()
        .into_iter()
        .map(|(_, x, _, _)| {
            x
        }).collect_vec();

    // --- Collect correct permutation indices ---
    let dummy_permutation_indices = dummy_auxiliaries
        .clone()
        .into_iter()
        .map(|(_, _, x, _)| {
            x
        }).collect_vec();

    // --- Compute multiplicities: just add the ones that are given in the returned map ---
    // TODO!(ryancao): Just use the paths already! The decision nodes are there!
    let multiplicities: Vec<F> = vec![F::zero(); (2_u32.pow(TREE_HEIGHT as u32 + 1) - 1) as usize];
    let dummy_multiplicities_bin_decomp = dummy_auxiliaries
        .clone()
        .into_iter()
        .fold(multiplicities, |prev_multiplicities, (path_decision_nodes, _, _, _)| {

            // --- TODO!(ryancao): This is so bad lol ---
            let mut new_multiplicities: Vec<F> = prev_multiplicities.clone();

            // --- Just grab the node IDs from each decision node and add them to the multiplicities ---
            path_decision_nodes.into_iter().for_each(|decision_node| {
                let node_id = decision_node.node_id.into_bigint().as_ref()[0] as usize;
                new_multiplicities[node_id] += F::one();
            });

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
        .map(|(_, _, _, diffs)| {
            diffs.into_iter().map(|diff| {
                let ret = generate_16_bit_signed_decomp(diff);
                check_signed_recomposition(diff, ret);
                ret
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

/// Gets the sign bit (0 for positive, 1 for negative) and abs value
fn get_sign_bit_and_abs_value<F: FieldExt>(value: F) -> (F, F) {
    let upper_bound = F::from(2_u32.pow(16) - 1);
    let sign_bit = if value > upper_bound {F::one()} else {F::zero()};
    let abs_value = if value > upper_bound {value.neg()} else {value};
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
    total = if sign_bit == F::one() {total.neg()} else {total};
    if total != actual_value {
        dbg!("RIP: Total = {:?}, actual_value = {:?}", total, actual_value);
        panic!();
        // return false;
    }
    true
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
        expression::{ExpressionStandard, Expression},
        mle::{dense::DenseMle, Mle, dense::DenseMleRef},
        sumcheck::{evaluate_expr, SumOrEvals},
    };
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use ark_std::One;
    use ark_std::Zero;

    /// Basic "bits are binary" test
    #[test]
    fn dummy_bits_are_binary_test() {

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

        // --- TODO!(ryancao): This is jank in the sense that we're just evaluating the first ---
        // --- prover message and just ensuring that both of them are zero, but really we should ---
        // --- be showing that all the evaluations match ---
        let res = evaluate_expr(&mut b_minus_b_squared, 1, 2);
        let other_res = evaluate_expr(&mut all_zeros_mle_expr, 1, 2);
        assert_eq!(res.unwrap(), other_res.unwrap());

        // --- TODO!(ryancao): Actually sumchecking over all of these expressions ---
    }

    /// Binary recomposition test (showing that the binary recomposition of the
    /// difference recomposes to equal the differences)
    /// The original expression: (1 - b_s)(diff - abs_recomp) + b_s(diff + abs_recomp) = 0
    /// The simplified expression: (diff - abs_recomp) + 2b_s(abs_recomp) = 0
    #[test]
    fn dummy_binary_recomp_test() {

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

        // --- Grab the bin decomp MLEs and associated expressions ---
        let bin_decomp_mles: Vec<DenseMleRef<Fr>> = dummy_binary_decomp_diffs_mle.mle_bit_refs();
        let mut bin_decomp_mle_exprs = vec![];
        for bit_idx in 0..16 {
            let cur_bit_mle_expr = ExpressionStandard::Mle(bin_decomp_mles[bit_idx].clone());
            bin_decomp_mle_exprs.push(cur_bit_mle_expr);
            dbg!(bin_decomp_mles[bit_idx].num_vars); // Should be 3 for path length/tree height
        }

        // --- Grab the things necessary to compute the diff (the permuted input and thresholds) ---
        let threshold_mle: DenseMleRef<Fr> = dummy_decision_node_paths_mle.threshold();
        let threshold_mle_expr = ExpressionStandard::Mle(threshold_mle.clone());
        let permuted_input_values_mle: DenseMleRef<Fr> = dummy_permuted_input_data_mle.attr_val(Some(threshold_mle.num_vars));
        let permuted_input_values_mle_expr = ExpressionStandard::Mle(permuted_input_values_mle.clone());

        // --- Need to just get diff ---
        dbg!(permuted_input_values_mle.num_vars); // Should be 3
        dbg!(threshold_mle.num_vars); // Should be 3
        let diff_expr = permuted_input_values_mle_expr - threshold_mle_expr;

        // --- We need `abs_recomp` and `b_s * abs_recomp` ---
        let mut abs_recomp_expr_list = vec![];
        let mut b_s_times_abs_recomp_expr_list = vec![];

        for bit_idx in 1..16 {
            // --- First compute b_s * coeff ---
            let b_s_times_coeff = ExpressionStandard::Product(vec![bin_decomp_mles[bit_idx].clone(), bin_decomp_mles[0].clone()]);
            let b_s_times_coeff_ptr = Box::new(b_s_times_coeff);

            // --- Then compute (b_s * coeff) * 2^{bit_idx} ---
            let base = Fr::from(2_u32.pow((16 - (bit_idx + 1)) as u32));
            let b_s_times_coeff_times_base = ExpressionStandard::Scaled(b_s_times_coeff_ptr, base);
            b_s_times_abs_recomp_expr_list.push(b_s_times_coeff_times_base);

            // --- Also compute just coeff * 2^{bit_idx} ---
            let coeff_expr = ExpressionStandard::Mle(bin_decomp_mles[bit_idx].clone());
            let coeff_expr_ptr = Box::new(coeff_expr);
            let coeff_times_base = ExpressionStandard::Scaled(coeff_expr_ptr, base);
            abs_recomp_expr_list.push(coeff_times_base);
        }

        assert!(abs_recomp_expr_list.len() == 15);
        assert!(b_s_times_abs_recomp_expr_list.len() == 15);

        // --- Combine all the elements in accumulation fashion ---
        let abs_recomp_expr = abs_recomp_expr_list
            .into_iter()
            .reduce(|a, b| {
                a + b
            })
            .unwrap();
        let b_s_times_abs_recomp_expr = b_s_times_abs_recomp_expr_list
            .into_iter()
            .reduce(|a, b| {
                a + b
            })
            .unwrap();

        // --- Subtract the two, and (TODO!(ryancao)) ensure they have the same number of variables ---
        let mut final_expr = 
            diff_expr - abs_recomp_expr + b_s_times_abs_recomp_expr.clone() + b_s_times_abs_recomp_expr;

        // --- We should get all zeros ---
        let all_zeros: Vec<Fr> = vec![Fr::zero()].repeat(2_u32.pow(permuted_input_values_mle.num_vars as u32) as usize);
        let all_zeros_mle = DenseMle::new(all_zeros);
        let mut all_zeros_mle_expr = ExpressionStandard::Mle(all_zeros_mle.mle_ref());

        let res = evaluate_expr(&mut final_expr, 1, 2);
        let other_res = evaluate_expr(&mut all_zeros_mle_expr, 1, 2);
        assert_eq!(res.unwrap(), other_res.unwrap());
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

        // --- Get packed inputs first ---
        let r1: Fr = test_rng().gen();
        let r2: Fr = test_rng().gen();

        // --- Multiply to do packing ---
        let dummy_attribute_id_mleref = dummy_input_data_mle.attr_id(None);
        let dummy_attribute_id_mleref_expr = ExpressionStandard::Mle(dummy_attribute_id_mleref);
        let dummy_attribute_val_mleref = dummy_input_data_mle.attr_val(None);
        let dummy_attribute_val_mleref_expr = ExpressionStandard::Mle(dummy_attribute_val_mleref);

        // --- 

    }
}