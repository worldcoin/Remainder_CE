use ark_std::{log2};
use itertools::{Itertools, repeat_n};

use remainder::{mle::{dense::DenseMle, MleRef, Mle, MleIndex}, layer::{LayerBuilder, empty_layer::EmptyLayer, batched::{BatchedLayer, unbatch_mles}, LayerId, Padding}, prover::{input_layer::{ligero_input_layer::LigeroInputLayer, combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer, MleInputLayer, enum_input_layer::InputLayerEnum, random_input_layer::RandomInputLayer}}};
use crate::builders::{BitExponentiationBuilderCatBoost, FSDecisionPackingBuilder, FSLeafPackingBuilder, FSRMinusXBuilder};
use remainder::{prover::{GKRCircuit, Layers, Witness, GKRError}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::super::{builders::{SplitProductBuilder, EqualityCheck, DecisionPackingBuilder, LeafPackingBuilder, RMinusXBuilder, SquaringBuilder, ProductBuilder}, structs::{DecisionNode, LeafNode, BinDecomp16Bit}};

use remainder::prover::input_layer::enum_input_layer::CommitmentEnum;

/// Dataparallel version of the "path multiset circuit" described in the audit spec.
/// 
/// Recall that this circuit computes the characteristic polynomial of two multisets,
/// one which "packs" the nodes involved in all $\pathx$ nodes by combining the `node_id`
/// and `threshold` via `r_packings[0]`, then multiplying all of the nodes together via
/// a huge product tree, and the other which again "packs" the nodes within the decision
/// tree (both decision and leaf nodes), exponentiates them by their multiplicites (this is
/// provided in 16-bit unsigned binary decomposition form factor by the prover), and multiplies
/// all those exponentiated values together. 
/// 
/// See spec for more high-level details!
pub(crate) struct MultiSetCircuit<F: FieldExt> {
    decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
    leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
    multiplicities_bin_decomp_mle_decision: DenseMle<F, BinDecomp16Bit<F>>,
    multiplicities_bin_decomp_mle_leaf: DenseMle<F, BinDecomp16Bit<F>>,
    decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>, // batched
    leaf_node_paths_mle_vec: Vec<DenseMle<F, LeafNode<F>>>,         // batched
    r: F,
    r_packings: (F, F),
}

impl<F: FieldExt> MultiSetCircuit<F> {
    pub fn new(
        decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
        leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
        multiplicities_bin_decomp_mle_decision: DenseMle<F, BinDecomp16Bit<F>>,
        multiplicities_bin_decomp_mle_leaf: DenseMle<F, BinDecomp16Bit<F>>,
        decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>, // batched
        leaf_node_paths_mle_vec: Vec<DenseMle<F, LeafNode<F>>>,         // batched
        r: F,
        r_packings: (F, F),
    ) -> Self {
        Self {
            decision_nodes_mle,
            leaf_nodes_mle,
            multiplicities_bin_decomp_mle_decision,
            multiplicities_bin_decomp_mle_leaf,
            decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            r,
            r_packings
        }
    }
}

impl<F: FieldExt> GKRCircuit<F> for MultiSetCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        let tree_height = (1 << (self.decision_node_paths_mle_vec[0].num_iterated_vars() - 2)) + 1;
        
        // --- Grab raw combined versions of the dataparallel MLEs and merge them for the input layer ---
        let mut dummy_decision_node_paths_mle_vec_combined = DenseMle::<F, DecisionNode<F>>::combine_mle_batch(self.decision_node_paths_mle_vec.clone());
        let mut dummy_leaf_node_paths_mle_vec_combined = DenseMle::<F, LeafNode<F>>::combine_mle_batch(self.leaf_node_paths_mle_vec.clone());

        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut self.decision_nodes_mle),
            Box::new(&mut self.leaf_nodes_mle),
            Box::new(&mut self.multiplicities_bin_decomp_mle_decision),
            Box::new(&mut self.multiplicities_bin_decomp_mle_leaf),
            Box::new(&mut dummy_decision_node_paths_mle_vec_combined),
            Box::new(&mut dummy_leaf_node_paths_mle_vec_combined),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer.to_input_layer();

        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        // --- Layer 0: Compute the "packed" version of the decision and leaf tree nodes ---
        // Note that this also "evaluates" each packed entry at the random characteristic polynomial
        // evaluation challenge point `self.r`.
        let mut dummy_decision_nodes_mle = self.decision_nodes_mle.clone();
        dummy_decision_nodes_mle.set_prefix_bits(self.decision_nodes_mle.get_prefix_bits());
        let decision_packing_builder = DecisionPackingBuilder::new(
            dummy_decision_nodes_mle, self.r, self.r_packings);

        let mut dummy_leaf_nodes_mle = self.leaf_nodes_mle.clone();
        dummy_leaf_nodes_mle.set_prefix_bits(self.leaf_nodes_mle.get_prefix_bits());
        let leaf_packing_builder = LeafPackingBuilder::new(
            dummy_leaf_nodes_mle, self.r, self.r_packings.0
        );

        let packing_builders = decision_packing_builder.concat(leaf_packing_builder);
        let (decision_packed, leaf_packed) = layers.add_gkr(packing_builders);

        let mut dummy_multiplicities_bin_decomp_mle_decision = self.multiplicities_bin_decomp_mle_decision.clone();
        dummy_multiplicities_bin_decomp_mle_decision.set_prefix_bits(self.multiplicities_bin_decomp_mle_decision.get_prefix_bits());

        // --- Layer 2, part 1: computes (r - x) * b_ij + (1 - b_ij) ---
        // Note that this is for the actual exponentiation computation:
        // we have that (r - x)^c_i = \prod_{j = 0}^{15} (r - x)^{2^{b_ij}} * b_{ij} + (1 - b_ij)
        // where \sum_{j = 0}^{15} 2^j b_{ij} = c_i.
        let prev_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
            dummy_multiplicities_bin_decomp_mle_decision.clone(),
            0,
            decision_packed.clone()
        );

        let mut dummy_multiplicities_bin_decomp_mle_leaf = self.multiplicities_bin_decomp_mle_leaf.clone();
        dummy_multiplicities_bin_decomp_mle_leaf.set_prefix_bits(self.multiplicities_bin_decomp_mle_leaf.get_prefix_bits());
        let prev_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
            dummy_multiplicities_bin_decomp_mle_leaf.clone(),
            0,
            leaf_packed.clone()
        );
        let pre_prod_builders = prev_prod_builder_decision.concat(prev_prod_builder_leaf);

        // --- Layer 2, part 2: (r - x)^2 ---
        // Note that we need to compute (r - x)^{2^0}, ..., (r - x)^{2^{15}}
        // We do this via repeated squaring of the previous power.
        let r_minus_x_square_builder_decision = SquaringBuilder::new(
            decision_packed
        );
        let r_minus_x_square_builder_leaf = SquaringBuilder::new(
            leaf_packed
        );
        let r_minus_x_square_builders = r_minus_x_square_builder_decision.concat(r_minus_x_square_builder_leaf);

        // --- For number-of-layers efficiency, we combine the above layers into a single GKR layer ---
        let layer_2_builders = pre_prod_builders.concat(r_minus_x_square_builders);
        let ((mut prev_prod_decision, mut prev_prod_leaf), (r_minus_x_power_decision, r_minus_x_power_leaf)) = layers.add_gkr(layer_2_builders);

        // --- Layer 3, part 1: (r - x)^2 * b_ij + (1 - b_ij) ---
        let prev_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
            dummy_multiplicities_bin_decomp_mle_decision.clone(),
            1,
            r_minus_x_power_decision.clone()
        );
        let prev_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
            dummy_multiplicities_bin_decomp_mle_leaf.clone(),
            1,
            r_minus_x_power_leaf.clone()
        );
        let pre_prod_builders = prev_prod_builder_decision.concat(prev_prod_builder_leaf);

        // --- Layer 3, part 2: (r - x)^4 ---
        let r_minus_x_square_builder_decision = SquaringBuilder::new(
            r_minus_x_power_decision
        );
        let r_minus_x_square_builder_leaf = SquaringBuilder::new(
            r_minus_x_power_leaf
        );
        let r_minus_x_square_builders = r_minus_x_square_builder_decision.concat(r_minus_x_square_builder_leaf);

        let layer_3_builders = pre_prod_builders.concat(r_minus_x_square_builders);
        let ((mut curr_prod_decision, mut curr_prod_leaf), (mut r_minus_x_power_decision, mut r_minus_x_power_leaf)) = layers.add_gkr(layer_3_builders);

        // need to square from (r - x)^(2^2) to (r - x)^(2^15),
        // so needs 13 more iterations
        // in each iteration, we get the following:
        // (r - x)^(2^(i+1)), (r - x)^(2^i) * b_ij + (1 - b_ij), PROD ALL[(r - x)^(2^(i-1)) * b_ij + (1 - b_ij)]
        for i in 2..15 {

            // layer 4, part 1
            let r_minus_x_square_builder_decision = SquaringBuilder::new(
                r_minus_x_power_decision.clone()
            );
            let r_minus_x_square_builder_leaf = SquaringBuilder::new(
                r_minus_x_power_leaf.clone()
            );
            let r_minus_x_square_builders = r_minus_x_square_builder_decision.concat(r_minus_x_square_builder_leaf);

            // layer 4, part 2
            let curr_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
                dummy_multiplicities_bin_decomp_mle_decision.clone(),
                i,
                r_minus_x_power_decision.clone()
            );

            let curr_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
                dummy_multiplicities_bin_decomp_mle_leaf.clone(),
                i,
                r_minus_x_power_leaf.clone()
            );

            let curr_prod_builders = curr_prod_builder_decision.concat(curr_prod_builder_leaf);

            // layer 4, part 3
            let prod_builder_decision = ProductBuilder::new(
                curr_prod_decision,
                prev_prod_decision
            );

            let prod_builder_leaf = ProductBuilder::new(
                curr_prod_leaf,
                prev_prod_leaf
            );

            let prod_builders = prod_builder_decision.concat(prod_builder_leaf);

            let layer_i_builders = r_minus_x_square_builders.concat(curr_prod_builders).concat_with_padding(prod_builders, Padding::Right(1));

            (((r_minus_x_power_decision, r_minus_x_power_leaf),
                    (curr_prod_decision, curr_prod_leaf)),
                    (prev_prod_decision, prev_prod_leaf)) = layers.add_gkr(layer_i_builders);
        }

        // --- At this point we have all of the following ---
        // - (r - x)^(2^0), ..., (r - x)^(2^{15})
        // - (r - x)^(2^0) * b_{i0} + (1 - b_{i0}), ..., (r - x)^(2^{14}) * b_{i{14}} + (1 - b_{i{14}})
        // - \prod_{j = 0}^{13} [(r - x)^(2^j) * b_ij + (1 - b_ij)]
        // --- We are computing \prod_{j = 0}^{15} [(r - x)^(2^j) * b_ij + (1 - b_ij)] ---
        // Therefore we need the following:
        // - (r - x)^(2^{15}) * b_{i{15}} + (1 - b_{i{15}}) (`BitExponentiate`)
        // - \prod_{j = 0}^{14} [(r - x)^(2^j) * b_ij + (1 - b_ij)] (`ProductBuilder`)
        // - \prod_{j = 0}^{15} [(r - x)^(2^j) * b_ij + (1 - b_ij)] (`ProductBuilder`)

        // layer 17, part 1
        let curr_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
            dummy_multiplicities_bin_decomp_mle_decision.clone(),
            15,
            r_minus_x_power_decision
        );

        let curr_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
            dummy_multiplicities_bin_decomp_mle_leaf.clone(),
            15,
            r_minus_x_power_leaf
        );

        let curr_prod_builders = curr_prod_builder_decision.concat(curr_prod_builder_leaf);

        // layer 17, part 2
        let prod_builder_decision = ProductBuilder::new(
            curr_prod_decision,
            prev_prod_decision
        );

        let prod_builder_leaf = ProductBuilder::new(
            curr_prod_leaf,
            prev_prod_leaf
        );
        let prod_builders = prod_builder_decision.concat(prod_builder_leaf);

        let layer_17_builders = curr_prod_builders.concat(prod_builders);
        let ((curr_prod_decision, curr_prod_leaf),
            (prev_prod_decision, prev_prod_leaf)) = layers.add_gkr(layer_17_builders);

        // layer 18
        let prod_builder_decision = ProductBuilder::new(
            curr_prod_decision,
            prev_prod_decision
        );

        let prod_builder_leaf = ProductBuilder::new(
            curr_prod_leaf,
            prev_prod_leaf
        );
        let prod_builders = prod_builder_decision.concat(prod_builder_leaf);

        let (prev_prod_decision, prev_prod_leaf) = layers.add_gkr(prod_builders);

        // --- These are our (raw) final answers for the exponentiation component of the circuit ---
        // In other words, \prod_{j = 0}^{15} [(r - x)^(2^j) * b_ij + (1 - b_ij)] for both decision and leaf nodes
        let mut exponentiated_decision = prev_prod_decision;
        let mut exponentiated_leaf = prev_prod_leaf;

        // --- Here is the part where we product across all `i` ---
        // In other words, computing \prod_{i = 0}^n (\prod_{j = 0}^{15} [(r - x)^(2^j) * b_ij + (1 - b_ij)])
        for _ in 0..tree_height-1 {

            // layer 20, or i+20
            let prod_builder_decision = SplitProductBuilder::new(
                exponentiated_decision
            );
            let prod_builder_leaf = SplitProductBuilder::new(
                exponentiated_leaf
            );

            let prod_builders = prod_builder_decision.concat(prod_builder_leaf);

            (exponentiated_decision, exponentiated_leaf) = layers.add_gkr(prod_builders);
        }

        let prod_builder_nodes = ProductBuilder::new(
            exponentiated_decision,
            exponentiated_leaf
        );

        // --- This is our final answer \prod_{i = 0}^n (\prod_{j = 0}^{15} [(r - x)^(2^j) * b_ij + (1 - b_ij)]) for the exponentiated characteristic polynomial
        // evaluated at a random challenge point ---
        let exponentiated_nodes = layers.add::<_, EmptyLayer<F, Self::Transcript>>(prod_builder_nodes);
        
        // **** above is nodes exponentiated ****
        // **** below is all decision nodes on the path multiplied ****
        println!("Nodes exponentiated, number of layers {:?}", layers.next_layer_id());

        let bit_difference = self.decision_node_paths_mle_vec[0].num_iterated_vars() - self.leaf_node_paths_mle_vec[0].num_iterated_vars();

        // --- Layer 0: packing (exactly the same as the packing builder above, but using the path decision
        // and leaf nodes rather than those from the tree) ---
        let batch_bits = log2(self.decision_node_paths_mle_vec.len()) as usize;

        let decision_path_packing_builder = BatchedLayer::new(
            self.decision_node_paths_mle_vec.iter().map(
                |decision_node_mle| {
                    let mut decision_node_mle = decision_node_mle.clone();
                    decision_node_mle.set_prefix_bits(Some(dummy_decision_node_paths_mle_vec_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    DecisionPackingBuilder::new(
                        decision_node_mle.clone(),
                        self.r,
                        self.r_packings
                    )
                }
            ).collect_vec());

        let leaf_path_packing_builder = BatchedLayer::new(
            self.leaf_node_paths_mle_vec.iter().map(
                |leaf_node_mle| {
                    let mut leaf_node_mle = leaf_node_mle.clone();
                    leaf_node_mle.set_prefix_bits(Some(dummy_leaf_node_paths_mle_vec_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    LeafPackingBuilder::new(
                        leaf_node_mle.clone(),
                        self.r,
                        self.r_packings.0
                    )
                }
            ).collect_vec());

        let path_packing_builders = decision_path_packing_builder.concat_with_padding(leaf_path_packing_builder, Padding::Right(bit_difference - 1));
        let (decision_path_packed, leaf_path_packed) = layers.add_gkr(path_packing_builders);

        // --- `unbatch_mles` takes the dataparallel version of the MLEs (i.e. vector form) and combines
        // them into a single MLE ---
        let mut vector_x_decision = unbatch_mles(decision_path_packed);
        let mut vector_x_leaf = unbatch_mles(leaf_path_packed);

        // --- Create a multiplication tree across the entire batch (decision nodes) ---
        for _ in 0..vector_x_decision.num_iterated_vars() {
            let curr_num_vars = vector_x_decision.num_iterated_vars();
            let layer = SplitProductBuilder::new(vector_x_decision);
            vector_x_decision = if curr_num_vars != 1 {
                layers.add_gkr(layer)
            } else {
                layers.add::<_, EmptyLayer<_, _>>(layer)
            }
        }

        // --- Create a multiplication tree across the entire batch (leaf nodes) ---
        for _ in 0..vector_x_leaf.num_iterated_vars() {
            let curr_num_vars = vector_x_leaf.num_iterated_vars();
            let layer = SplitProductBuilder::new(vector_x_leaf);
            vector_x_leaf = if curr_num_vars != 1 {
                layers.add_gkr(layer)
            } else {
                layers.add::<_, EmptyLayer<_, _>>(layer)
            }
        }

        // --- Multiply the two together ---
        let path_product_builder = ProductBuilder::new(
            vector_x_decision,
            vector_x_leaf
        );

        // --- This is our final answer \prod_{i = 0}^n (r - x_i) for the path node characteristic polynomial
        // evaluated at the same random challenge point ---
        let path_product = layers.add::<_, EmptyLayer<F, Self::Transcript>>(path_product_builder);

        let difference_builder = EqualityCheck::new(
            exponentiated_nodes,
            path_product
        );

        let circuit_output = layers.add::<_, EmptyLayer<F, Self::Transcript>>(difference_builder);

        println!("Multiset circuit finished, number of layers {:?}", layers.next_layer_id());

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![input_layer.to_enum()],
        }
    }
}