use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::get_default_poseidon_parameters_internal;
use ark_ff::BigInteger;
use rand::Rng;
use rayon::{iter::Split, vec};
use tracing_subscriber::fmt::layer;
use std::{io::Empty, iter};

use ark_std::{log2, test_rng};
use itertools::{Itertools, repeat_n};

use crate::{mle::{dense::DenseMle, MleRef, beta::BetaTable, Mle, MleIndex}, layer::{LayerBuilder, empty_layer::EmptyLayer, batched::{BatchedLayer, combine_zero_mle_ref, unbatch_mles}, LayerId, Padding}, sumcheck::{compute_sumcheck_message, Evals, get_round_degree}, zkdt::zkdt_layer::{BitExponentiationBuilderCatBoost, IdentityBuilder, AttributeConsistencyBuilderZeroRef}, prover::{input_layer::{ligero_input_layer::LigeroInputLayer, combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer, MleInputLayer, enum_input_layer::InputLayerEnum}, combine_layers::combine_layers}};
use crate::{prover::{GKRCircuit, Layers, Witness, GKRError}, mle::{mle_enum::MleEnum}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::{zkdt_layer::{InputPackingBuilder, SplitProductBuilder, EqualityCheck, AttributeConsistencyBuilder, DecisionPackingBuilder, LeafPackingBuilder, ConcatBuilder, RMinusXBuilder, BitExponentiationBuilder, SquaringBuilder, ProductBuilder}, structs::{InputAttribute, DecisionNode, LeafNode, BinDecomp16Bit}, binary_recomp_circuit::circuit_builders::{BinaryRecompBuilder, NodePathDiffBuilder, BinaryRecompCheckerBuilder, PartialBitsCheckerBuilder}, zkdt_helpers::{BatchedCatboostMles, generate_mles_batch_catboost_single_tree}};

use crate::prover::input_layer::enum_input_layer::CommitmentEnum;

pub struct FSPermutationCircuit<F: FieldExt> {
    pub input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,               // batched
    pub permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,      // batched
    pub r: F,
    pub r_packing: F,
}

impl<F: FieldExt> GKRCircuit<F> for FSPermutationCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }

    fn synthesize_and_commit(
            &mut self,
            transcript: &mut Self::Transcript,
        ) -> Result<(Witness<F, Self::Transcript>, Vec<CommitmentEnum<F>>), GKRError> {
            let mut dummy_input_data_mle_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(self.input_data_mle_vec.clone());
            let mut dummy_permuted_input_data_mle_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(self.permuted_input_data_mle_vec.clone());
    
            let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
                Box::new(&mut dummy_input_data_mle_combined),
                Box::new(&mut dummy_permuted_input_data_mle_combined),
            ];
            let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
            let input_prefix_bits = input_layer.fetch_prefix_bits();
            let input_layer: PublicInputLayer<F, Self::Transcript> = input_layer.to_input_layer();
            // TODO!(ende) change back to ligero
    
            let mut layers: Layers<_, Self::Transcript> = Layers::new();
    
            let batch_bits = log2(self.input_data_mle_vec.len()) as usize;
        
        
            let input_packing_builder = BatchedLayer::new(
                self.input_data_mle_vec.iter().map(
                    |input_data_mle| {
                        let mut input_data_mle = input_data_mle.clone();
                        // TODO!(ende) fix this atrocious fixed(false)
                        input_data_mle.add_prefix_bits(Some(dummy_input_data_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                        InputPackingBuilder::new(
                            input_data_mle,
                            self.r,
                            self.r_packing
                        )
                    }).collect_vec());
    
            let input_permuted_packing_builder = BatchedLayer::new(
                self.permuted_input_data_mle_vec.iter().map(
                    |input_data_mle| {
                        let mut input_data_mle = input_data_mle.clone();
                        // TODO!(ende) fix this atrocious fixed(true)
                        input_data_mle.add_prefix_bits(Some(dummy_permuted_input_data_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                        InputPackingBuilder::new(
                            input_data_mle,
                            self.r,
                            self.r_packing
                        )
                    }).collect_vec());
    
            let packing_builders = input_packing_builder.concat(input_permuted_packing_builder);
    
            let (mut input_packed, mut input_permuted_packed) = layers.add_gkr(packing_builders);
    
            let input_len = 1 << (self.input_data_mle_vec[0].num_iterated_vars() - 1);
            for _ in 0..log2(input_len) {
                let prod_builder = BatchedLayer::new(
                    input_packed.into_iter().map(
                        |input_packed| SplitProductBuilder::new(input_packed)
                    ).collect());
                let prod_permuted_builder = BatchedLayer::new(
                    input_permuted_packed.into_iter().map(
                        |input_permuted_packed| SplitProductBuilder::new(input_permuted_packed)
                    ).collect());
                let split_product_builders = prod_builder.concat(prod_permuted_builder);
                (input_packed, input_permuted_packed) = layers.add_gkr(split_product_builders);
            }
    
            let difference_builder = EqualityCheck::new_batched(
                input_packed,
                input_permuted_packed,
            );
    
            let difference_mle = layers.add_gkr(difference_builder);
    
            let circuit_output = combine_zero_mle_ref(difference_mle);
    
            Witness {
                layers,
                output_layers: vec![circuit_output.get_enum()],
                input_layers: vec![input_layer.to_enum()],
            };
            todo!()
    }
}

pub struct PermutationCircuit<F: FieldExt> {
    pub input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,               // batched
    pub permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,      // batched
    pub r: F,
    pub r_packing: F,
}

impl<F: FieldExt> GKRCircuit<F> for PermutationCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        let mut dummy_input_data_mle_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(self.input_data_mle_vec.clone());
        let mut dummy_permuted_input_data_mle_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(self.permuted_input_data_mle_vec.clone());

        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut dummy_input_data_mle_combined),
            Box::new(&mut dummy_permuted_input_data_mle_combined),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let input_prefix_bits = input_layer.fetch_prefix_bits();
        let input_layer: PublicInputLayer<F, Self::Transcript> = input_layer.to_input_layer();
        // TODO!(ende) change back to ligero

        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        let batch_bits = log2(self.input_data_mle_vec.len()) as usize;
    
    
        let input_packing_builder = BatchedLayer::new(
            self.input_data_mle_vec.iter().map(
                |input_data_mle| {
                    let mut input_data_mle = input_data_mle.clone();
                    // TODO!(ende) fix this atrocious fixed(false)
                    input_data_mle.add_prefix_bits(Some(dummy_input_data_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    InputPackingBuilder::new(
                        input_data_mle,
                        self.r,
                        self.r_packing
                    )
                }).collect_vec());

        let input_permuted_packing_builder = BatchedLayer::new(
            self.permuted_input_data_mle_vec.iter().map(
                |input_data_mle| {
                    let mut input_data_mle = input_data_mle.clone();
                    // TODO!(ende) fix this atrocious fixed(true)
                    input_data_mle.add_prefix_bits(Some(dummy_permuted_input_data_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    InputPackingBuilder::new(
                        input_data_mle,
                        self.r,
                        self.r_packing
                    )
                }).collect_vec());

        let packing_builders = input_packing_builder.concat(input_permuted_packing_builder);

        let (mut input_packed, mut input_permuted_packed) = layers.add_gkr(packing_builders);

        let input_len = 1 << (self.input_data_mle_vec[0].num_iterated_vars() - 1);
        for _ in 0..log2(input_len) {
            let prod_builder = BatchedLayer::new(
                input_packed.into_iter().map(
                    |input_packed| SplitProductBuilder::new(input_packed)
                ).collect());
            let prod_permuted_builder = BatchedLayer::new(
                input_permuted_packed.into_iter().map(
                    |input_permuted_packed| SplitProductBuilder::new(input_permuted_packed)
                ).collect());
            let split_product_builders = prod_builder.concat(prod_permuted_builder);
            (input_packed, input_permuted_packed) = layers.add_gkr(split_product_builders);
        }

        let difference_builder = EqualityCheck::new_batched(
            input_packed,
            input_permuted_packed,
        );

        let difference_mle = layers.add_gkr(difference_builder);

        let circuit_output = combine_zero_mle_ref(difference_mle);

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![input_layer.to_enum()],
        }
    }
}

/// permutation circuit, non batched version
pub struct NonBatchedPermutationCircuit<F: FieldExt> {
    input_data_mle: DenseMle<F, InputAttribute<F>>,
    permuted_input_data_mle: DenseMle<F, InputAttribute<F>>,
    r: F,
    r_packing: F,
    input_len: usize,
}

impl<F: FieldExt> GKRCircuit<F> for NonBatchedPermutationCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        // layer 0: packing
        let input_packing_builder: InputPackingBuilder<F> = InputPackingBuilder::new(
            self.input_data_mle.clone(),
            self.r,
            self.r_packing);

        let input_permuted_packing_builder: InputPackingBuilder<F> = InputPackingBuilder::new(
            self.permuted_input_data_mle.clone(),
            self.r,
            self.r_packing);

        let packing_builders = input_packing_builder.concat(input_permuted_packing_builder);

        let (mut input_packed, mut input_permuted_packed) = layers.add_gkr(packing_builders);

        for _ in 0..log2(self.input_len) {
            let prod_builder = SplitProductBuilder::new(
                input_packed
            );
            let permuted_prod_builder = SplitProductBuilder::new(
                input_permuted_packed
            );
            let split_product_builders = prod_builder.concat(permuted_prod_builder);
            (input_packed, input_permuted_packed) = layers.add_gkr(split_product_builders);
        }

        let difference_builder = EqualityCheck::new(
            input_packed,
            input_permuted_packed,
        );

        let difference_mle = layers.add::<_, EmptyLayer<F, Self::Transcript>>(difference_builder);

        todo!()
    }
}

struct NonBatchedAttributeConsistencyCircuit<F: FieldExt> {
    permuted_input_data_mle_vec: DenseMle<F, InputAttribute<F>>,
    decision_node_paths_mle_vec: DenseMle<F, DecisionNode<F>>,
    tree_height: usize,
}

impl<F: FieldExt> GKRCircuit<F> for NonBatchedAttributeConsistencyCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        let attribute_consistency_builder = AttributeConsistencyBuilder::new(
            self.permuted_input_data_mle_vec.clone(),
            self.decision_node_paths_mle_vec.clone(),
            self.tree_height
        );

        let difference_mle = layers.add_gkr(attribute_consistency_builder);

        todo!()
    }
}

struct AttributeConsistencyCircuit<F: FieldExt> {
    permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>,
    tree_height: usize,
}

impl<F: FieldExt> GKRCircuit<F> for AttributeConsistencyCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        let mut dummy_permuted_input_data_mle_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(self.permuted_input_data_mle_vec.clone());
        let mut dummy_decision_node_paths_mle_combined = DenseMle::<F, DecisionNode<F>>::combine_mle_batch(self.decision_node_paths_mle_vec.clone());

        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut dummy_permuted_input_data_mle_combined),
            Box::new(&mut dummy_decision_node_paths_mle_combined),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let input_prefix_bits = input_layer.fetch_prefix_bits();
        let input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer.to_input_layer();

        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        let batch_bits = log2(self.permuted_input_data_mle_vec.len()) as usize;
    
        let attribute_consistency_builder = BatchedLayer::new(

            self.permuted_input_data_mle_vec
                    .iter()
                    .zip(self.decision_node_paths_mle_vec.iter())
                    .map(|(input_data_mle, decision_path_mle)| {

                        let mut input_data_mle = input_data_mle.clone();
                        input_data_mle.add_prefix_bits(Some(dummy_permuted_input_data_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));

                        let mut decision_path_mle = decision_path_mle.clone();
                        decision_path_mle.add_prefix_bits(Some(dummy_decision_node_paths_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));

                        AttributeConsistencyBuilderZeroRef::new(
                            input_data_mle,
                            decision_path_mle,
                            self.tree_height
                        )

        }).collect_vec());

        let difference_mle = layers.add_gkr(attribute_consistency_builder);
        let circuit_output = combine_zero_mle_ref(difference_mle);

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![input_layer.to_enum()],
        }
    }
}

struct MultiSetCircuit<F: FieldExt> {
    decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
    leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
    multiplicities_bin_decomp_mle_decision: DenseMle<F, BinDecomp16Bit<F>>,
    multiplicities_bin_decomp_mle_leaf: DenseMle<F, BinDecomp16Bit<F>>,
    decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>, // batched
    leaf_node_paths_mle_vec: Vec<DenseMle<F, LeafNode<F>>>,         // batched
    r: F,
    r_packings: (F, F),
    tree_height: usize,
}

impl<F: FieldExt> GKRCircuit<F> for MultiSetCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        
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
        let input_prefix_bits = input_layer.fetch_prefix_bits();
        let input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer.to_input_layer();

        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        // layer 0: x
        let mut dummy_decision_nodes_mle = self.decision_nodes_mle.clone();
        dummy_decision_nodes_mle.add_prefix_bits(self.decision_nodes_mle.get_prefix_bits());
        let decision_packing_builder = DecisionPackingBuilder::new(
            dummy_decision_nodes_mle, self.r, self.r_packings);

        let mut dummy_leaf_nodes_mle = self.leaf_nodes_mle.clone();
        dummy_leaf_nodes_mle.add_prefix_bits(self.leaf_nodes_mle.get_prefix_bits());
        let leaf_packing_builder = LeafPackingBuilder::new(
            dummy_leaf_nodes_mle, self.r, self.r_packings.0
        );

        let packing_builders = decision_packing_builder.concat(leaf_packing_builder);
        let (decision_packed, leaf_packed) = layers.add_gkr(packing_builders);

        // layer 1: (r - x)
        let r_minus_x_builder_decision =  RMinusXBuilder::new(
            decision_packed, self.r_packings.0
        );
        let r_minus_x_builder_leaf =  RMinusXBuilder::new(
            leaf_packed, self.r_packings.0
        );
        let r_minus_x_builders = r_minus_x_builder_decision.concat(r_minus_x_builder_leaf);
        let (r_minus_x_power_decision, r_minus_x_power_leaf) = layers.add_gkr(r_minus_x_builders);

        let mut dummy_multiplicities_bin_decomp_mle_decision = self.multiplicities_bin_decomp_mle_decision.clone();
        dummy_multiplicities_bin_decomp_mle_decision.add_prefix_bits(self.multiplicities_bin_decomp_mle_decision.get_prefix_bits());
        // layer 2, part 1: (r - x) * b_ij + (1 - b_ij)
        let prev_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
            dummy_multiplicities_bin_decomp_mle_decision.clone(),
            0,
            r_minus_x_power_decision.clone()
        );

        let mut dummy_multiplicities_bin_decomp_mle_leaf = self.multiplicities_bin_decomp_mle_leaf.clone();
        dummy_multiplicities_bin_decomp_mle_leaf.add_prefix_bits(self.multiplicities_bin_decomp_mle_leaf.get_prefix_bits());
        let prev_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
            dummy_multiplicities_bin_decomp_mle_leaf.clone(),
            0,
            r_minus_x_power_leaf.clone()
        );
        let pre_prod_builders = prev_prod_builder_decision.concat(prev_prod_builder_leaf);

        // layer 2, part 2: (r - x)^2
        let r_minus_x_square_builder_decision = SquaringBuilder::new(
            r_minus_x_power_decision
        );
        let r_minus_x_square_builder_leaf = SquaringBuilder::new(
            r_minus_x_power_leaf
        );
        let r_minus_x_square_builders = r_minus_x_square_builder_decision.concat(r_minus_x_square_builder_leaf);

        let layer_2_builders = pre_prod_builders.concat(r_minus_x_square_builders);
        let ((mut prev_prod_decision, mut prev_prod_leaf), (r_minus_x_power_decision, r_minus_x_power_leaf)) = layers.add_gkr(layer_2_builders);

        // layer 3, part 1: (r - x)^2 * b_ij + (1 - b_ij)
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

        // layer 3, part 2: (r - x)^4
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
        // in each iteration, get the following:
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

        // at this point we have
        // (r - x)^(2^15), (r - x)^(2^14) * b_ij + (1 - b_ij), PROD ALL[(r - x)^(2^13) * b_ij + (1 - b_ij)]
        // need to BitExponentiate 1 time
        // and PROD w prev_prod 2 times

        // layer 17, part 1
        let curr_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
            dummy_multiplicities_bin_decomp_mle_decision.clone(),
            15,
            r_minus_x_power_decision.clone()
        );

        let curr_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
            dummy_multiplicities_bin_decomp_mle_leaf.clone(),
            15,
            r_minus_x_power_leaf.clone()
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

        let mut exponentiated_decision = prev_prod_decision;
        let mut exponentiated_leaf = prev_prod_leaf;

        for _ in 0..self.tree_height-1 {

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

        let exponentiated_nodes = layers.add::<_, EmptyLayer<F, Self::Transcript>>(prod_builder_nodes);
        
        // **** above is nodes exponentiated ****
        // **** below is all decision nodes on the path multiplied ****
        println!("Nodes exponentiated, number of layers {:?}", layers.next_layer_id());

        let bit_difference = self.decision_node_paths_mle_vec[0].num_iterated_vars() - self.leaf_node_paths_mle_vec[0].num_iterated_vars();

        // layer 0: packing

        let batch_bits = log2(self.decision_node_paths_mle_vec.len()) as usize;

        let decision_path_packing_builder = BatchedLayer::new(
            self.decision_node_paths_mle_vec.iter().map(
                |decision_node_mle| {
                    let mut decision_node_mle = decision_node_mle.clone();
                    decision_node_mle.add_prefix_bits(Some(dummy_decision_node_paths_mle_vec_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
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
                    leaf_node_mle.add_prefix_bits(Some(dummy_leaf_node_paths_mle_vec_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    LeafPackingBuilder::new(
                        leaf_node_mle.clone(),
                        self.r,
                        self.r_packings.0
                    )
                }
            ).collect_vec());

        let path_packing_builders = decision_path_packing_builder.concat_with_padding(leaf_path_packing_builder, Padding::Right(bit_difference - 1));
        let (decision_path_packed, leaf_path_packed) = layers.add_gkr(path_packing_builders);

        // layer 2: r - x

        let bit_difference = decision_path_packed[0].num_iterated_vars() - leaf_path_packed[0].num_iterated_vars();

        let r_minus_x_path_builder_decision = BatchedLayer::new(
            decision_path_packed.iter().map(|x| RMinusXBuilder::new(
                x.clone(),
                self.r_packings.0
            )).collect_vec());

        let r_minus_x_path_builder_leaf = BatchedLayer::new(
            leaf_path_packed.iter().map(|x| RMinusXBuilder::new(
                x.clone(),
                self.r_packings.0
            )).collect_vec());

        let r_minus_x_path_builders = r_minus_x_path_builder_decision.concat_with_padding(r_minus_x_path_builder_leaf, Padding::Right(bit_difference));

        let (r_minus_x_path_decision, r_minus_x_path_leaf) = layers.add_gkr(r_minus_x_path_builders);

        let mut vector_x_decision = unbatch_mles(r_minus_x_path_decision);
        let mut vector_x_leaf = unbatch_mles(r_minus_x_path_leaf);

        // product all the batches together
        for _ in 0..vector_x_decision.num_iterated_vars() {
            let curr_num_vars = vector_x_decision.num_iterated_vars();
            let layer = SplitProductBuilder::new(vector_x_decision);
            vector_x_decision = if curr_num_vars != 1 {
                layers.add_gkr(layer)
            } else {
                layers.add::<_, EmptyLayer<_, _>>(layer)
            }
        }

        for _ in 0..vector_x_leaf.num_iterated_vars() {
            let curr_num_vars = vector_x_leaf.num_iterated_vars();
            let layer = SplitProductBuilder::new(vector_x_leaf);
            vector_x_leaf = if curr_num_vars != 1 {
                layers.add_gkr(layer)
            } else {
                layers.add::<_, EmptyLayer<_, _>>(layer)
            }
        }

        let path_product_builder = ProductBuilder::new(
            vector_x_decision,
            vector_x_leaf
        );

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

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use ark_std::{test_rng, UniformRand};
    use itertools::Itertools;
    use rand::Rng;

    use crate::{zkdt::{zkdt_helpers::{DummyMles, generate_dummy_mles, NUM_DUMMY_INPUTS, DUMMY_INPUT_LEN, TREE_HEIGHT, generate_dummy_mles_batch, BatchedDummyMles, BatchedCatboostMles, generate_mles_batch_catboost_single_tree}, zkdt_circuit_parts::NonBatchedPermutationCircuit, structs::{InputAttribute, DecisionNode, LeafNode}, binary_recomp_circuit::circuits::{PartialBitsCheckerCircuit, BinaryRecompCircuit}}, prover::{GKRCircuit, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer}}, mle::{dense::DenseMle, MleRef, Mle}, layer::LayerId};
    use remainder_shared_types::transcript::{Transcript, poseidon_transcript::PoseidonTranscript};
    use crate::prover::tests::test_circuit;

    use super::{PermutationCircuit, NonBatchedAttributeConsistencyCircuit, MultiSetCircuit, AttributeConsistencyCircuit};

    #[test]
    fn test_permutation_circuit_catboost_non_batched() {
        let mut rng = test_rng();

        let (BatchedCatboostMles {
            input_data_mle_vec,
            permuted_input_data_mle_vec, ..
        }, (_tree_height, input_len)) = generate_mles_batch_catboost_single_tree::<Fr>();

        let mut circuit = NonBatchedPermutationCircuit {
            input_data_mle: input_data_mle_vec[0].clone(),
            permuted_input_data_mle: permuted_input_data_mle_vec[0].clone(),
            r: Fr::from(rng.gen::<u64>()),
            r_packing: Fr::from(rng.gen::<u64>()),
            input_len,
        };

        let mut transcript = PoseidonTranscript::new("Permutation Circuit Prover Transcript");

        let now = Instant::now();

        let proof = circuit.prove(&mut transcript);

        println!("Proof generated!: Took {} seconds", now.elapsed().as_secs_f32());

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Permutation Circuit Verifier Transcript");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            },
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }

    #[test]
    fn test_permutation_circuit_catboost_batched() {
        let mut rng = test_rng();

        let (BatchedCatboostMles {
            input_data_mle_vec,
            permuted_input_data_mle_vec, ..
        }, (_tree_height, input_len)) = generate_mles_batch_catboost_single_tree::<Fr>();

        let dummy_input_len = input_data_mle_vec.len();

        let mut circuit = PermutationCircuit {
            input_data_mle_vec,
            permuted_input_data_mle_vec,
            r: Fr::from(rng.gen::<u64>()),
            r_packing: Fr::from(rng.gen::<u64>()),
        };

        test_circuit(circuit, None);
    }

    #[test]
    fn test_permutation_circuit_dummy_batched() {
        let mut rng = test_rng();

        let BatchedDummyMles {
            dummy_input_data_mle,
            dummy_permuted_input_data_mle, ..
        } = generate_dummy_mles_batch();

        let mut circuit = PermutationCircuit {
            input_data_mle_vec: dummy_input_data_mle,
            permuted_input_data_mle_vec: dummy_permuted_input_data_mle,
            r: Fr::from(rng.gen::<u64>()),
            r_packing: Fr::from(rng.gen::<u64>()),
        };

        let mut transcript = PoseidonTranscript::new("Permutation Circuit Prover Transcript");

        let now = Instant::now();

        let proof = circuit.prove(&mut transcript);

        println!("Proof generated!: Took {} seconds", now.elapsed().as_secs_f32());

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Permutation Circuit Verifier Transcript");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            },
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }

    #[test]
    fn test_attribute_consistency_circuit_dummy_non_batched() {

        let DummyMles::<Fr> {
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle, ..
        } = generate_dummy_mles();

        let mut circuit = NonBatchedAttributeConsistencyCircuit {
            permuted_input_data_mle_vec: dummy_permuted_input_data_mle,
            decision_node_paths_mle_vec: dummy_decision_node_paths_mle,
            tree_height: TREE_HEIGHT,
        };

        let mut transcript = PoseidonTranscript::new("Attribute Consistency Circuit Prover Transcript");

        let proof = circuit.prove(&mut transcript);

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Attribute Consistency Circuit Verifier Transcript");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            },
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }

    #[test]
    fn test_attribute_consistency_circuit_catboost_non_batched() {

        let (BatchedCatboostMles {
            permuted_input_data_mle_vec,
            decision_node_paths_mle_vec, ..
        }, (tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>();

        let mut circuit = NonBatchedAttributeConsistencyCircuit {
            permuted_input_data_mle_vec: permuted_input_data_mle_vec[0].clone(),
            decision_node_paths_mle_vec: decision_node_paths_mle_vec[0].clone(),
            tree_height: tree_height,
        };

        let mut transcript = PoseidonTranscript::new("Attribute Consistency Circuit Prover Transcript");

        let proof = circuit.prove(&mut transcript);

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Attribute Consistency Circuit Verifier Transcript");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            },
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }

    #[test]
    fn test_attribute_consistency_circuit_catboost_batched() {

        let (BatchedCatboostMles {
            permuted_input_data_mle_vec,
            decision_node_paths_mle_vec, ..
        }, (tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>();

        let mut circuit = AttributeConsistencyCircuit {
            permuted_input_data_mle_vec,
            decision_node_paths_mle_vec,
            tree_height
        };

        test_circuit(circuit, None);

    }

    #[test]
    fn test_multiset_circuit_dummy_batched() {

        let mut rng = test_rng();

        let BatchedDummyMles::<Fr> {
            dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle,
            dummy_multiplicities_bin_decomp_mle,
            dummy_decision_nodes_mle,
            dummy_leaf_nodes_mle, ..
        } = generate_dummy_mles_batch();

        // let mut circuit = MultiSetCircuit {
        //     dummy_decision_nodes_mle,
        //     dummy_leaf_nodes_mle,
        //     dummy_multiplicities_bin_decomp_mle,
        //     dummy_decision_node_paths_mle_vec: dummy_decision_node_paths_mle,
        //     dummy_leaf_node_paths_mle_vec: dummy_leaf_node_paths_mle,
        //     r: Fr::rand(&mut rng),
        //     r_packings: (Fr::rand(&mut rng), Fr::rand(&mut rng)),
        //     tree_height: TREE_HEIGHT,
        // };

        // let mut transcript = PoseidonTranscript::new("Multiset Circuit Prover Transcript");

        // let proof = circuit.prove(&mut transcript);

        // match proof {
        //     Ok(proof) => {
        //         let mut transcript = PoseidonTranscript::new("Multiset Circuit Verifier Transcript");
        //         let result = circuit.verify(&mut transcript, proof);
        //         if let Err(err) = result {
        //             println!("{}", err);
        //             panic!();
        //         }
        //     },
        //     Err(err) => {
        //         println!("{}", err);
        //         panic!();
        //     }
        // }
    }

    #[test]
    fn test_multiset_circuit_catboost_batched() {

        let mut rng = test_rng();

        let (BatchedCatboostMles {decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            multiplicities_bin_decomp_mle_decision,
            multiplicities_bin_decomp_mle_leaf,
            decision_nodes_mle,
            leaf_nodes_mle, ..}, (tree_height, input_len)) = generate_mles_batch_catboost_single_tree::<Fr>();

        let mut circuit = MultiSetCircuit {
            decision_nodes_mle,
            leaf_nodes_mle,
            multiplicities_bin_decomp_mle_decision,
            multiplicities_bin_decomp_mle_leaf,
            decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            r: Fr::from(rng.gen::<u64>()),
            r_packings: (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())),
            tree_height,
        };

        test_circuit(circuit, None);
    }
}