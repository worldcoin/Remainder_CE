use ark_std::{log2};
use itertools::{Itertools, repeat_n, multizip};

use remainder::{mle::{dense::DenseMle, MleRef, Mle, MleIndex}, layer::{LayerBuilder, empty_layer::EmptyLayer, batched::{BatchedLayer, unbatch_mles, combine_zero_mle_ref}, LayerId, Padding}, prover::{input_layer::{ligero_input_layer::LigeroInputLayer, combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer, MleInputLayer, enum_input_layer::InputLayerEnum, random_input_layer::RandomInputLayer}}};
use remainder::{prover::{GKRCircuit, Layers, Witness, GKRError}};
use crate::builders::{BitExponentiationBuilderCatBoost, FSDecisionPackingBuilder, FSLeafPackingBuilder, FSRMinusXBuilder};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::{super::{builders::{SplitProductBuilder, EqualityCheck, DecisionPackingBuilder, LeafPackingBuilder, RMinusXBuilder, SquaringBuilder, ProductBuilder}, structs::{DecisionNode, LeafNode, BinDecomp16Bit}}, legacy_circuits::MultiSetCircuit};

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
pub(crate) struct FSMultiSetCircuitMultiTree<F: FieldExt> {
    pub decision_nodes_mle_tree: Vec<DenseMle<F, DecisionNode<F>>>,
    pub leaf_nodes_mle_tree: Vec<DenseMle<F, LeafNode<F>>>,
    pub multiplicities_bin_decomp_mle_decision_tree: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    pub multiplicities_bin_decomp_mle_leaf_tree: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    pub decision_node_paths_mle_vec_tree: Vec<Vec<DenseMle<F, DecisionNode<F>>>>, // batched
    pub leaf_node_paths_mle_vec_tree: Vec<Vec<DenseMle<F, LeafNode<F>>>>,         // batched
    pub r_mle: DenseMle<F, F>,
    pub r_packing_mle: DenseMle<F, F>,
    pub r_packing_another_mle: DenseMle<F, F>,
}

impl<F: FieldExt> FSMultiSetCircuitMultiTree<F> {
    pub fn new(
        decision_nodes_mle_tree: Vec<DenseMle<F, DecisionNode<F>>>,
        leaf_nodes_mle_tree: Vec<DenseMle<F, LeafNode<F>>>,
        multiplicities_bin_decomp_mle_decision_tree: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
        multiplicities_bin_decomp_mle_leaf_tree: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
        decision_node_paths_mle_vec_tree: Vec<Vec<DenseMle<F, DecisionNode<F>>>>, // batched
        leaf_node_paths_mle_vec_tree: Vec<Vec<DenseMle<F, LeafNode<F>>>>,         // batched
        r_mle: DenseMle<F, F>,
        r_packing_mle: DenseMle<F, F>,
        r_packing_another_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            decision_nodes_mle_tree,
            leaf_nodes_mle_tree,
            multiplicities_bin_decomp_mle_decision_tree,
            multiplicities_bin_decomp_mle_leaf_tree,
            decision_node_paths_mle_vec_tree,
            leaf_node_paths_mle_vec_tree,
            r_mle,
            r_packing_mle,
            r_packing_another_mle,
        }
    }
}

impl<F: FieldExt> GKRCircuit<F> for FSMultiSetCircuitMultiTree<F> {
    type Transcript = PoseidonTranscript<F>;
    
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }

    fn synthesize_and_commit(
            &mut self,
            transcript: &mut Self::Transcript,
        ) -> Result<(Witness<F, Self::Transcript>, Vec<CommitmentEnum<F>>), GKRError> {
            unimplemented!()
    }
}

impl<F: FieldExt> FSMultiSetCircuitMultiTree<F> {
    pub fn yield_sub_circuit(&mut self) -> Witness<F, PoseidonTranscript<F>> {

        let mut layers: Layers<_, PoseidonTranscript<F>> = Layers::new();
        let num_dataparallel_bits = log2(self.decision_node_paths_mle_vec_tree[0].len()) as usize;
        let tree_bits = log2(self.decision_node_paths_mle_vec_tree.len()) as usize;

        self.decision_nodes_mle_tree.iter_mut().for_each(
            |mle| {
                mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, tree_bits)).collect_vec()));
            }
        );
        self.leaf_nodes_mle_tree.iter_mut().for_each(
            |mle| {
                mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, tree_bits)).collect_vec()));
            }
        );
        self.multiplicities_bin_decomp_mle_decision_tree.iter_mut().for_each(
            |mle| {
                mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, tree_bits)).collect_vec()));
            }
        );
        self.multiplicities_bin_decomp_mle_leaf_tree.iter_mut().for_each(
            |mle| {
                mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, tree_bits)).collect_vec()));
            }
        );

        self.decision_node_paths_mle_vec_tree.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each(|mle| {
                mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits + tree_bits)).collect_vec()));
            })
        });

        self.leaf_node_paths_mle_vec_tree.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each(|mle| {
                mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits + tree_bits)).collect_vec()));
            })
        });

        // --- Layer 0: Compute the "packed" version of the decision and leaf tree nodes ---
        // Note that this also "evaluates" each packed entry at the random characteristic polynomial
        // evaluation challenge point `self.r`.

        let (decision_packing_builder_vec, leaf_packing_builder_vec): (Vec<_>, Vec<_>) = self.decision_nodes_mle_tree.clone().into_iter().zip(self.leaf_nodes_mle_tree.clone().into_iter()).map(
            |(decision_nodes_mle, leaf_nodes_mle)| {
                let decision_packing_builder = FSDecisionPackingBuilder::new(
                    decision_nodes_mle,
                    self.r_mle.clone(),
                    self.r_packing_mle.clone(),
                    self.r_packing_another_mle.clone(),
                );

                let leaf_packing_builder = FSLeafPackingBuilder::new(
                    leaf_nodes_mle,
                    self.r_mle.clone(),
                    self.r_packing_mle.clone(),
                );

                // let packing_builders = decision_packing_builder.concat(leaf_packing_builder);
                (decision_packing_builder, leaf_packing_builder)
            }).unzip();

        
        
        let (decision_packed_vec, leaf_packed_vec): (Vec<_>, Vec<_>)= layers.add_gkr(BatchedLayer::new(decision_packing_builder_vec).concat(BatchedLayer::new(leaf_packing_builder_vec)));

        // let mut multiplicities_bin_decomp_mle_decision = self.multiplicities_bin_decomp_mle_decision.clone();
        // multiplicities_bin_decomp_mle_decision.set_prefix_bits(self.multiplicities_bin_decomp_mle_decision.get_prefix_bits());

        // --- Layer 2, part 1: computes (r - x) * b_ij + (1 - b_ij) ---
        // Note that this is for the actual exponentiation computation:
        // we have that (r - x)^c_i = \prod_{j = 0}^{15} (r - x)^{2^{b_ij}} * b_{ij} + (1 - b_ij)
        // where \sum_{j = 0}^{15} 2^j b_{ij} = c_i.

        let (prev_prod_builder_decision_vec, prev_prod_builder_leaf_vec, r_minus_x_square_builder_decision_vec, r_minus_x_square_builder_leaf_vec): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = self.multiplicities_bin_decomp_mle_decision_tree.clone().into_iter().zip(self.multiplicities_bin_decomp_mle_leaf_tree.clone().into_iter()).zip(decision_packed_vec.into_iter().zip(leaf_packed_vec.into_iter())).map(
            |((multiplicities_bin_decomp_mle_decision, multiplicities_bin_decomp_mle_leaf), (decision_packed, leaf_packed))| {
                let prev_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
                    multiplicities_bin_decomp_mle_decision.clone(),
                    0,
                    decision_packed.clone()
                );
        
                let prev_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
                    multiplicities_bin_decomp_mle_leaf.clone(),
                    0,
                    leaf_packed.clone()
                );
                // let pre_prod_builders = prev_prod_builder_decision.concat(prev_prod_builder_leaf);

                // --- Layer 2, part 2: (r - x)^2 ---
                // Note that we need to compute (r - x)^{2^0}, ..., (r - x)^{2^{15}}
                // We do this via repeated squaring of the previous power.
                let r_minus_x_square_builder_decision = SquaringBuilder::new(
                    decision_packed
                );
                let r_minus_x_square_builder_leaf = SquaringBuilder::new(
                    leaf_packed
                );
                // let r_minus_x_square_builders = r_minus_x_square_builder_decision.concat(r_minus_x_square_builder_leaf);

                // let layer_2_builders = pre_prod_builders.concat(r_minus_x_square_builders);
                
                (prev_prod_builder_decision, prev_prod_builder_leaf, r_minus_x_square_builder_decision, r_minus_x_square_builder_leaf)
        }).multiunzip();

        let concat_builder_layer_2 = (BatchedLayer::new(prev_prod_builder_decision_vec).concat(BatchedLayer::new(prev_prod_builder_leaf_vec))).concat(BatchedLayer::new(r_minus_x_square_builder_decision_vec).concat(BatchedLayer::new(r_minus_x_square_builder_leaf_vec)));

        // --- For number-of-layers efficiency, we combine the above layers into a single GKR layer ---
        
        let ((mut prev_prod_decision_vec, mut prev_prod_leaf_vec), (r_minus_x_power_decision_vec, r_minus_x_power_leaf_vec)): ((Vec<_>, Vec<_>), (Vec<_>, Vec<_>)) = layers.add_gkr(concat_builder_layer_2);

        
        let (prev_prod_builder_decision_vec, prev_prod_builder_leaf_vec, r_minus_x_square_builder_decision_vec, r_minus_x_square_builder_leaf_vec): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = multizip((self.multiplicities_bin_decomp_mle_decision_tree.iter(), self.multiplicities_bin_decomp_mle_leaf_tree.iter(), r_minus_x_power_decision_vec.iter(), r_minus_x_power_leaf_vec.iter())).map(
            |(multiplicities_bin_decomp_mle_decision, multiplicities_bin_decomp_mle_leaf, r_minus_x_power_decision, r_minus_x_power_leaf)| {
                // --- Layer 3, part 1: (r - x)^2 * b_ij + (1 - b_ij) ---
                let prev_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
                    multiplicities_bin_decomp_mle_decision.clone(),
                    1,
                    r_minus_x_power_decision.clone()
                );

                let prev_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
                    multiplicities_bin_decomp_mle_leaf.clone(),
                    1,
                    r_minus_x_power_leaf.clone()
                );
                // let pre_prod_builders = prev_prod_builder_decision.concat(prev_prod_builder_leaf);

                // --- Layer 3, part 2: (r - x)^4 ---
                let r_minus_x_square_builder_decision = SquaringBuilder::new(
                    r_minus_x_power_decision.clone()
                );
                let r_minus_x_square_builder_leaf = SquaringBuilder::new(
                    r_minus_x_power_leaf.clone()
                );
                // let r_minus_x_square_builders = r_minus_x_square_builder_decision.concat(r_minus_x_square_builder_leaf);

                // let layer_3_builders = pre_prod_builders.concat(r_minus_x_square_builders);
                // layer_3_builders

                (prev_prod_builder_decision, prev_prod_builder_leaf, r_minus_x_square_builder_decision, r_minus_x_square_builder_leaf)
                    
        }).multiunzip();

        let layer_3_builders_concat = BatchedLayer::new(prev_prod_builder_decision_vec).concat(BatchedLayer::new(prev_prod_builder_leaf_vec)).concat(BatchedLayer::new(r_minus_x_square_builder_decision_vec).concat(BatchedLayer::new(r_minus_x_square_builder_leaf_vec)));
        
        let ((mut curr_prod_decision_vec, mut curr_prod_leaf_vec), (mut r_minus_x_power_decision_vec, mut r_minus_x_power_leaf_vec)): ((Vec<_>, Vec<_>), (Vec<_>, Vec<_>)) = layers.add_gkr(layer_3_builders_concat);

        // need to square from (r - x)^(2^2) to (r - x)^(2^15),
        // so needs 13 more iterations
        // in each iteration, get the following:
        // (r - x)^(2^(i+1)), (r - x)^(2^i) * b_ij + (1 - b_ij), PROD ALL[(r - x)^(2^(i-1)) * b_ij + (1 - b_ij)]

       

                for i in 2..15 {
                    let (r_minus_x_square_builder_decision_vec,
                        r_minus_x_square_builder_leaf_vec,
                        curr_prod_builder_decision_vec,
                        curr_prod_builder_leaf_vec,
                        prod_builder_decision_vec,
                        prod_builder_leaf_vec): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) = multizip((r_minus_x_power_decision_vec.iter(), r_minus_x_power_leaf_vec.iter(), curr_prod_decision_vec.into_iter(), curr_prod_leaf_vec.into_iter(), prev_prod_decision_vec.into_iter(), prev_prod_leaf_vec.into_iter(), self.multiplicities_bin_decomp_mle_decision_tree.iter(), self.multiplicities_bin_decomp_mle_leaf_tree.iter())).map(
                        |(r_minus_x_power_decision, r_minus_x_power_leaf, curr_prod_decision, curr_prod_leaf, prev_prod_decision, prev_prod_leaf, multiplicities_bin_decomp_mle_decision, multiplicities_bin_decomp_mle_leaf)| {

                    // layer 4, part 1
                    let r_minus_x_square_builder_decision = SquaringBuilder::new(
                        r_minus_x_power_decision.clone()
                    );
                    let r_minus_x_square_builder_leaf = SquaringBuilder::new(
                        r_minus_x_power_leaf.clone()
                    );
                    // let r_minus_x_square_builders = r_minus_x_square_builder_decision.concat(r_minus_x_square_builder_leaf);
        
                    // layer 4, part 2
                    let curr_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
                        multiplicities_bin_decomp_mle_decision.clone(),
                        i,
                        r_minus_x_power_decision.clone()
                    );
        
                    let curr_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
                        multiplicities_bin_decomp_mle_leaf.clone(),
                        i,
                        r_minus_x_power_leaf.clone()
                    );
        
                    // let curr_prod_builders = curr_prod_builder_decision.concat(curr_prod_builder_leaf);
        
                    // layer 4, part 3
                    let prod_builder_decision = ProductBuilder::new(
                        curr_prod_decision,
                        prev_prod_decision
                    );
        
                    let prod_builder_leaf = ProductBuilder::new(
                        curr_prod_leaf,
                        prev_prod_leaf
                    );
        
                    // let prod_builders = prod_builder_decision.concat(prod_builder_leaf);

                    (r_minus_x_square_builder_decision, r_minus_x_square_builder_leaf, curr_prod_builder_decision, curr_prod_builder_leaf, prod_builder_decision, prod_builder_leaf)
        
                    // let layer_i_builders = r_minus_x_square_builders.concat(curr_prod_builders).concat_with_padding(prod_builders, Padding::Right(1));

                    // layer_i_builders

                    }).multiunzip();

                    let r_minus_x_square_builders_vec = BatchedLayer::new(r_minus_x_square_builder_decision_vec).concat(BatchedLayer::new(r_minus_x_square_builder_leaf_vec));
                    let curr_prod_builders_vec = BatchedLayer::new(curr_prod_builder_decision_vec).concat(BatchedLayer::new(curr_prod_builder_leaf_vec));
                    let prod_builders_vec = BatchedLayer::new(prod_builder_decision_vec).concat(BatchedLayer::new(prod_builder_leaf_vec));
                    let layer_i_builders_vec = r_minus_x_square_builders_vec.concat(curr_prod_builders_vec).concat_with_padding(prod_builders_vec, Padding::Right(1));
        
                    (((r_minus_x_power_decision_vec, r_minus_x_power_leaf_vec),
                            (curr_prod_decision_vec, curr_prod_leaf_vec)),
                            (prev_prod_decision_vec, prev_prod_leaf_vec)) = layers.add_gkr(layer_i_builders_vec);
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

        let (curr_prod_builder_decision_vec, curr_prod_builder_leaf_vec, prod_builder_decision_vec, prod_builder_leaf_vec): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = multizip((r_minus_x_power_decision_vec.into_iter(), r_minus_x_power_leaf_vec.into_iter(), self.multiplicities_bin_decomp_mle_decision_tree.iter(), self.multiplicities_bin_decomp_mle_leaf_tree.iter(), curr_prod_decision_vec.into_iter(), curr_prod_leaf_vec.into_iter(), prev_prod_decision_vec.into_iter(), prev_prod_leaf_vec.into_iter())).map(
            |(r_minus_x_power_decision, r_minus_x_power_leaf, multiplicities_bin_decomp_mle_decision, multiplicities_bin_decomp_mle_leaf, curr_prod_decision, curr_prod_leaf, prev_prod_decision, prev_prod_leaf)| {
                 // layer 17, part 1
                let curr_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
                    multiplicities_bin_decomp_mle_decision.clone(),
                    15,
                    r_minus_x_power_decision
                );

                let curr_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
                    multiplicities_bin_decomp_mle_leaf.clone(),
                    15,
                    r_minus_x_power_leaf
                );

                // let curr_prod_builders = curr_prod_builder_decision.concat(curr_prod_builder_leaf);

                // layer 17, part 2
                let prod_builder_decision = ProductBuilder::new(
                    curr_prod_decision,
                    prev_prod_decision
                );

                let prod_builder_leaf = ProductBuilder::new(
                    curr_prod_leaf,
                    prev_prod_leaf
                );
                // let prod_builders = prod_builder_decision.concat(prod_builder_leaf);

                // let layer_17_builders = curr_prod_builders.concat(prod_builders);

                // layer_17_builders
                (curr_prod_builder_decision, curr_prod_builder_leaf, prod_builder_decision, prod_builder_leaf)
                
        }).multiunzip();

        let curr_prod_builders = BatchedLayer::new(curr_prod_builder_decision_vec).concat(BatchedLayer::new(curr_prod_builder_leaf_vec));
        let prod_builders = BatchedLayer::new(prod_builder_decision_vec).concat(BatchedLayer::new(prod_builder_leaf_vec));

        let ((curr_prod_decision_vec, curr_prod_leaf_vec),
                    (prev_prod_decision_vec, prev_prod_leaf_vec)): ((Vec<_>, Vec<_>), (Vec<_>, Vec<_>)) = layers.add_gkr(curr_prod_builders.concat(prod_builders));

       

        // layer 18

        let (prod_builder_decision_vec, prod_builder_leaf_vec): (Vec<_>, Vec<_>) = multizip((prev_prod_decision_vec, prev_prod_leaf_vec, curr_prod_decision_vec, curr_prod_leaf_vec)).map(
            |(prev_prod_decision, prev_prod_leaf, curr_prod_decision, curr_prod_leaf)| {
                let prod_builder_decision = ProductBuilder::new(
                    curr_prod_decision,
                    prev_prod_decision
                );
        
                let prod_builder_leaf = ProductBuilder::new(
                    curr_prod_leaf,
                    prev_prod_leaf
                );
                // let prod_builders = prod_builder_decision.concat(prod_builder_leaf);

                // prod_builders
                (prod_builder_decision, prod_builder_leaf)
            }
        ).unzip();


        let (prev_prod_decision_vec, prev_prod_leaf_vec): (Vec<_>, Vec<_>) = layers.add_gkr(BatchedLayer::new(prod_builder_decision_vec).concat(BatchedLayer::new(prod_builder_leaf_vec))) ;

        // --- These are our (raw) final answers for the exponentiation component of the circuit ---
        // In other words, \prod_{j = 0}^{15} [(r - x)^(2^j) * b_ij + (1 - b_ij)] for both decision and leaf nodes
        let mut exponentiated_decision_vec = prev_prod_decision_vec;
        let mut exponentiated_leaf_vec = prev_prod_leaf_vec;




                let tree_height = (1 << (self.decision_node_paths_mle_vec_tree[0][0].num_iterated_vars() - 2)) + 1;

                // --- Here is the part where we product across all `i` ---
                // In other words, computing \prod_{i = 0}^n (\prod_{j = 0}^{15} [(r - x)^(2^j) * b_ij + (1 - b_ij)])
                for _ in 0..tree_height-1 {

                    let (prod_builders_decision_vec, prod_builders_leaf_vec): (Vec<_>, Vec<_>) = exponentiated_decision_vec.into_iter().zip(exponentiated_leaf_vec).map(
                        |(exponentiated_decision, exponentiated_leaf)| {
                            // layer 20, or i+20
                            let prod_builder_decision = SplitProductBuilder::new(
                                exponentiated_decision
                            );
                            let prod_builder_leaf = SplitProductBuilder::new(
                                exponentiated_leaf
                            );

                            // let prod_builders = prod_builder_decision.concat(prod_builder_leaf);
                            (prod_builder_decision, prod_builder_leaf)
                        }
                    ).unzip();
                    
                    // (prod_builder_decision, prod_builder_leaf)

                    (exponentiated_decision_vec, exponentiated_leaf_vec) = layers.add_gkr(BatchedLayer::new(prod_builders_decision_vec).concat(BatchedLayer::new(prod_builders_leaf_vec)));
                }
            
        
        let prod_builder_nodes_vec = exponentiated_decision_vec.into_iter().zip(exponentiated_leaf_vec.into_iter()).map(
            |(exponentiated_decision, exponentiated_leaf)| {
                let prod_builder_nodes = ProductBuilder::new(
                    exponentiated_decision,
                    exponentiated_leaf
                );
                prod_builder_nodes
            }
        ).collect_vec();

        

        // --- This is our final answer \prod_{i = 0}^n (\prod_{j = 0}^{15} [(r - x)^(2^j) * b_ij + (1 - b_ij)]) for the exponentiated characteristic polynomial
        // evaluated at a random challenge point ---
        let exponentiated_nodes_vec = layers.add_gkr(BatchedLayer::new(prod_builder_nodes_vec));

        // **** above is nodes exponentiated ****
        // **** below is all decision nodes on the path multiplied ****
        println!("Nodes exponentiated, number of layers {:?}", layers.next_layer_id());

        let bit_difference = self.decision_node_paths_mle_vec_tree[0][0].num_iterated_vars() - self.leaf_node_paths_mle_vec_tree[0][0].num_iterated_vars();

        // --- Layer 0: packing (exactly the same as the packing builder above, but using the path decision
        // and leaf nodes rather than those from the tree) ---
       


        let (decision_path_packing_builder_vec, leaf_path_packing_builder_vec): (Vec<_>, Vec<_>) = self.decision_node_paths_mle_vec_tree.clone().into_iter().zip(self.leaf_node_paths_mle_vec_tree.clone().into_iter()).map(
            |(decision_node_paths_mle_vec, leaf_node_paths_mle_vec)| {
                let decision_path_packing_builder = BatchedLayer::new(
                    decision_node_paths_mle_vec.iter().map(
                        |decision_node_mle| {
                            let mut decision_node_mle = decision_node_mle.clone();
                            FSDecisionPackingBuilder::new(
                                decision_node_mle.clone(),
                                self.r_mle.clone(),
                                self.r_packing_mle.clone(),
                                self.r_packing_another_mle.clone()
                            )
                        }
                    ).collect_vec());
        
                let leaf_path_packing_builder = BatchedLayer::new(
                    leaf_node_paths_mle_vec.iter().map(
                        |leaf_node_mle| {
                            let mut leaf_node_mle = leaf_node_mle.clone();
                            FSLeafPackingBuilder::new(
                                leaf_node_mle.clone(),
                                self.r_mle.clone(),
                                self.r_packing_mle.clone()
                            )
                        }
                    ).collect_vec());
                
                // let path_packing_builders = decision_path_packing_builder.concat_with_padding(leaf_path_packing_builder, Padding::Right(bit_difference - 1));
                // path_packing_builders
                (decision_path_packing_builder, leaf_path_packing_builder)
            }
        ).unzip();

        

        let (decision_path_packed_vec, leaf_path_packed_vec) : (Vec<_>, Vec<_>)= layers.add_gkr(BatchedLayer::new(decision_path_packing_builder_vec).concat_with_padding(BatchedLayer::new(leaf_path_packing_builder_vec), Padding::Right(bit_difference - 1)));

        // --- `unbatch_mles` takes the dataparallel version of the MLEs (i.e. vector form) and combines
        // them into a single MLE ---


        let mut vector_x_decision_vec = decision_path_packed_vec.into_iter().map(|decision_path_packed| unbatch_mles(decision_path_packed)).collect_vec();
        let mut vector_x_leaf_vec = leaf_path_packed_vec.into_iter().map(|leaf_path_packed| unbatch_mles(leaf_path_packed)).collect_vec();




        for _ in 0..vector_x_decision_vec[0].num_iterated_vars() {
            vector_x_decision_vec = layers.add_gkr(BatchedLayer::new(vector_x_decision_vec.into_iter().map(
                |vector_x_decision| {
                    SplitProductBuilder::new(vector_x_decision)
                }
            ).collect_vec()));
        }

        for _ in 0..vector_x_leaf_vec[0].num_iterated_vars() {
            vector_x_leaf_vec = layers.add_gkr(BatchedLayer::new(vector_x_leaf_vec.into_iter().map(
                |vector_x_leaf| {
                    SplitProductBuilder::new(vector_x_leaf)
                }
            ).collect_vec()));
        }


        let path_product_builder_vec = vector_x_decision_vec.into_iter().zip(vector_x_leaf_vec.into_iter()).map(
            |(vector_x_decision, vector_x_leaf)| {
                ProductBuilder::new(
                    vector_x_decision,
                    vector_x_leaf
                )
            }
        ).collect_vec();

        let path_product_vec = layers.add_gkr(BatchedLayer::new(path_product_builder_vec));


        // --- This is our final answer \prod_{i = 0}^n (r - x_i) for the path node characteristic polynomial
        // evaluated at the same random challenge point ---


        let difference_builder_vec = exponentiated_nodes_vec.into_iter().zip(path_product_vec.into_iter()).map(
            |(exponentiated_nodes, path_product)| {
                let difference_builder = EqualityCheck::new(
                    exponentiated_nodes,
                    path_product
                );
                difference_builder
            }
        ).collect_vec();

        let circuit_output_vec = layers.add::<_, EmptyLayer<F, PoseidonTranscript<F>>>(BatchedLayer::new(difference_builder_vec));
        let circuit_output = combine_zero_mle_ref(circuit_output_vec);

        println!("Multiset circuit finished, number of layers {:?}", layers.next_layer_id());

        println!("# layers -- multiset: {:?}", layers.next_layer_id());

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![],
        }
    }
}