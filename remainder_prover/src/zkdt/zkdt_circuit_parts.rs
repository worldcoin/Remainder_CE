use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::get_default_poseidon_parameters_internal;
use ark_ff::BigInteger;
use rayon::{iter::Split, vec};
use tracing_subscriber::fmt::layer;
use std::io::Empty;

use ark_std::log2;
use itertools::{Itertools, repeat_n};

use crate::{mle::{dense::DenseMle, MleRef, beta::BetaTable, Mle, MleIndex}, layer::{LayerBuilder, empty_layer::EmptyLayer, batched::{BatchedLayer, combine_zero_mle_ref, unbatch_mles}, LayerId, Padding}, sumcheck::{compute_sumcheck_message, Evals, get_round_degree}, zkdt::zkdt_layer::{BitExponentiationBuilderCatBoost, IdentityBuilder, AttributeConsistencyBuilderZeroRef}, prover::input_layer::{ligero_input_layer::LigeroInputLayer, combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer}};
use crate::{prover::{GKRCircuit, Layers, Witness}, mle::{mle_enum::MleEnum}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::{zkdt_layer::{InputPackingBuilder, SplitProductBuilder, EqualityCheck, AttributeConsistencyBuilder, DecisionPackingBuilder, LeafPackingBuilder, ConcatBuilder, RMinusXBuilder, BitExponentiationBuilder, SquaringBuilder, ProductBuilder, BinaryRecompBuilder, NodePathDiffBuilder, BinaryRecompCheckerBuilder, PartialBitsCheckerBuilder}, structs::{InputAttribute, DecisionNode, LeafNode, BinDecomp16Bit}};

pub struct PermutationCircuit<F: FieldExt> {
    pub dummy_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,               // batched
    pub dummy_permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,      // batched
    pub r: F,
    pub r_packing: F,
    pub input_len: usize,
    pub num_inputs: usize
}

impl<F: FieldExt> GKRCircuit<F> for PermutationCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        let batch_bits = log2(self.dummy_input_data_mle_vec.len()) as usize;
    
    
        let input_packing_builder = BatchedLayer::new(self.dummy_input_data_mle_vec.iter().map(|input_data_mle| {
            let mut input_data_mle = input_data_mle.clone();
            input_data_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
            InputPackingBuilder::new(
                input_data_mle,
                self.r,
                self.r_packing
            )
        }).collect_vec());

        let input_permuted_packing_builder = BatchedLayer::new(self.dummy_permuted_input_data_mle_vec.iter().map(|input_data_mle| {
            let mut input_data_mle = input_data_mle.clone();
            input_data_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
            InputPackingBuilder::new(
                input_data_mle,
                self.r,
                self.r_packing
            )
        }).collect_vec());

        let packing_builders = input_packing_builder.concat(input_permuted_packing_builder);
        // let mut expression: crate::expression::ExpressionStandard<F> = packing_builders.build_expression();
        // let num_vars = expression.index_mle_indices(0);
        // let degree = get_round_degree(&expression, 0);    
        // let (_, next) = packing_builders.next_layer(LayerId::Layer(0), None);

        // let eval = {
        //     let mut beta = BetaTable::new((vec![F::zero(), F::one(), F::zero(), F::zero(), F::zero(), F::zero(), F::zero(), F::zero(), F::zero()], F::zero())).unwrap();
        //     beta.table.index_mle_indices(0);
        //     let eval = compute_sumcheck_message(&expression, 0, degree, &beta).unwrap();
        //     let Evals(evals) = eval;
        //     evals[0] + evals[1]
        // };

        // let next = next.mle_ref();
        // let val = next.bookkeeping_table()[1];
        // dbg!(eval, val);

        // assert_eq!(eval, val);

        let (mut input_packed, mut input_permuted_packed) = layers.add_gkr(packing_builders);

        for _ in 0..log2(self.input_len) {
            let prod_builder = BatchedLayer::new(input_packed.into_iter().map(|input_packed| SplitProductBuilder::new(
                input_packed
            )).collect());
            let prod_permuted_builder = BatchedLayer::new(input_permuted_packed.into_iter().map(|input_permuted_packed| SplitProductBuilder::new(
                input_permuted_packed
            )).collect());
            let split_product_builders = prod_builder.concat(prod_permuted_builder);
            (input_packed, input_permuted_packed) = layers.add_gkr(split_product_builders);
        }

        let difference_builder = EqualityCheck::new_batched(
            input_packed,
            input_permuted_packed,
        );

        let difference_mle = layers.add_gkr(difference_builder);

        let difference_mle = combine_zero_mle_ref(difference_mle);

        todo!()
    }
}

struct TestCircuit<F: FieldExt> {
    dummy_decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
    dummy_leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
    dummy_multiplicities_bin_decomp_mle_decision: DenseMle<F, BinDecomp16Bit<F>>,
    dummy_multiplicities_bin_decomp_mle_leaf: DenseMle<F, BinDecomp16Bit<F>>,
    dummy_decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>, // batched
    dummy_leaf_node_paths_mle_vec: Vec<DenseMle<F, LeafNode<F>>>,         // batched
    r: F,
    r_packings: (F, F),
    tree_height: usize,
}

impl<F: FieldExt> GKRCircuit<F> for TestCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        // let dummy_decision = DecisionNode{
        //     node_id: F::from(4 as u64),
        //     attr_id: F::from(3 as u64),
        //     threshold: F::from(2 as u64),
        // };

        // let dummy_leaf = LeafNode{
        //     node_id: F::from(4 as u64),
        //     node_val: F::from(2 as u64),
        // };

        // let dummy_decision_path = vec![dummy_decision; 8];
        // let dummy_leaf_path = vec![dummy_leaf; 1];

        // let decision_mle = DenseMle::new_from_iter(dummy_decision_path
        //     .clone()
        //     .into_iter()
        //     .map(DecisionNode::from), LayerId::Input(0), None);

        // let leaf_mle = DenseMle::new_from_iter(dummy_leaf_path
        //     .clone()
        //     .into_iter()
        //     .map(LeafNode::from), LayerId::Input(0), None);

        // let decision_mle_vec = [decision_mle.clone(), decision_mle.clone()];
        // let leaf_mle_vec = [leaf_mle.clone(), leaf_mle.clone()];


        let bit_difference = self.dummy_decision_node_paths_mle_vec[0].num_iterated_vars() - self.dummy_leaf_node_paths_mle_vec[0].num_iterated_vars();

        // layer 0: packing

        let batch_bits = log2(self.dummy_decision_node_paths_mle_vec.len()) as usize;

        let decision_path_packing_builder = BatchedLayer::new(
            self.dummy_decision_node_paths_mle_vec.iter().map(
                |decision_node_mle| {
                    let mut decision_node_mle = decision_node_mle.clone();
                    decision_node_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
                    DecisionPackingBuilder::new(
                        decision_node_mle.clone(),
                        self.r,
                        self.r_packings
                    )
                }
            ).collect_vec());

        let leaf_path_packing_builder = BatchedLayer::new(
            self.dummy_leaf_node_paths_mle_vec.iter().map(
                |leaf_node_mle| {
                    let mut leaf_node_mle = leaf_node_mle.clone();
                    leaf_node_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
                    LeafPackingBuilder::new(
                        leaf_node_mle.clone(),
                        self.r,
                        self.r_packings.0
                    )
                }
            ).collect_vec());

        // let path_packing_builders = decision_path_packing_builder.concat(leaf_path_packing_builder);
        let path_packing_builders = decision_path_packing_builder.concat_with_padding(leaf_path_packing_builder, Padding::Right(bit_difference - 1));
        let (decision_path_packed, leaf_path_packed) = layers.add_gkr(path_packing_builders);
        // let decision_path_packed = layers.add_gkr(decision_path_packing_builder);

        let mut vector_x_decision = unbatch_mles(decision_path_packed);
        let mut vector_x_leaf = unbatch_mles(leaf_path_packed);

        for _ in 0..vector_x_decision.num_iterated_vars() {
            let curr_num_vars = vector_x_decision.num_iterated_vars();
            let layer = SplitProductBuilder::new(vector_x_decision);
            vector_x_decision = if curr_num_vars != 1 {
                layers.add_gkr(layer)
            } else {
                layers.add::<_, EmptyLayer<_, _>>(layer)
            };
        }

        for _ in 0..vector_x_leaf.num_iterated_vars() {
            let curr_num_vars = vector_x_leaf.num_iterated_vars();
            let layer = SplitProductBuilder::new(vector_x_leaf);
            vector_x_leaf = if curr_num_vars != 1 {
                layers.add_gkr(layer)
            } else {
                layers.add::<_, EmptyLayer<_, _>>(layer)
            };
        }

        let path_product_builder = ProductBuilder::new(
            vector_x_decision,
            vector_x_leaf
        );

        let path_product = layers.add::<_, EmptyLayer<F, Self::Transcript>>(path_product_builder);

        // vector_x should have length 1

        let final_val = path_product.clone().mle_ref().bookkeeping_table[0].to_bytes_le();
        let exponentiated_nodes = DenseMle::new_from_raw(vec![F::from_bytes_le(&final_val)], LayerId::Input(0), None);

        let difference_builder = EqualityCheck::new(
            exponentiated_nodes,
            path_product
        );

        let difference_mle = layers.add::<_, EmptyLayer<F, Self::Transcript>>(difference_builder);

        // return (layers, vec![Box::new(difference_mle)]);

        // return (layers, vec![Box::new(difference_mle)]);

        let one_mle_large: DenseMle<F, F> = DenseMle::one(8, LayerId::Input(0), None);
        let one_mle_small: DenseMle<F, F> = DenseMle::one(1, LayerId::Input(0), None);

        let one_mle_large_vec = vec![one_mle_large.clone(), one_mle_large.clone(), one_mle_large.clone(), one_mle_large.clone()];
        let one_mle_small_vec = vec![one_mle_small.clone(), one_mle_small.clone(), one_mle_small.clone(), one_mle_small.clone()];

        let bit_difference = one_mle_large.num_iterated_vars() - one_mle_small.num_iterated_vars();
        let batch_bits = log2(one_mle_large_vec.len()) as usize;

        let identity_large = BatchedLayer::new(
            one_mle_large_vec.iter().map(|mle_large| {
                let mut mle_large = mle_large.clone();
                mle_large.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
                IdentityBuilder::new(
                    mle_large
                )
            }).collect_vec()
        );

        let identity_small = BatchedLayer::new(
            one_mle_small_vec.iter().map(|mle_small| {
                let mut mle_small = mle_small.clone();
                mle_small.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
                IdentityBuilder::new(
                    mle_small
                )
            }).collect_vec()
        );

        let identity_builders = identity_large.concat_with_padding(identity_small, Padding::Right(bit_difference));

        let (mut one_large, one_small) = layers.add_gkr(identity_builders);

        let mut one_large = unbatch_mles(one_large);
        let mut one_small = unbatch_mles(one_small);
        dbg!(&one_small);

        for _ in 0..one_large.num_iterated_vars() {
            let curr_num_vars = one_large.num_iterated_vars();
            let layer = SplitProductBuilder::new(one_large);
            one_large = if curr_num_vars != 1 {
                layers.add_gkr(layer)
            } else {
                layers.add::<_, EmptyLayer<_, _>>(layer)
            }
        }

        for _ in 0..one_small.num_iterated_vars() {
            let curr_num_vars = one_small.num_iterated_vars();
            let layer = SplitProductBuilder::new(one_small);
            one_small = if curr_num_vars != 1 {
                layers.add_gkr(layer)
            } else {
                layers.add::<_, EmptyLayer<_, _>>(layer)
            }
        }

        let difference_builder = EqualityCheck::new(
            one_large,
            one_small
        );

        let difference_mle = layers.add::<_, EmptyLayer<_, _>>(difference_builder);
        // return (layers, vec![Box::new(difference_mle)]);


        // layer 0: packing

        let batch_bits = log2(self.dummy_decision_node_paths_mle_vec.len()) as usize;

        let decision_path_packing_builder = BatchedLayer::new(
            self.dummy_decision_node_paths_mle_vec.iter().map(
                |decision_node_mle| {
                    let mut decision_node_mle = decision_node_mle.clone();
                    decision_node_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
                    DecisionPackingBuilder::new(
                        decision_node_mle.clone(),
                        self.r,
                        self.r_packings
                    )
                }
            ).collect_vec());

        let decision_path_packed = layers.add_gkr(decision_path_packing_builder);


        let r_minus_x_path_builder_decision = BatchedLayer::new(
            decision_path_packed.iter().map(|x| RMinusXBuilder::new(
                x.clone(),
                self.r_packings.0
            )).collect_vec());

        let r_minus_x_path_decision = layers.add_gkr(r_minus_x_path_builder_decision);

        // let batch_size = log2(r_minus_x_path.len()) as usize;

        let mut vector_x_decision = unbatch_mles(r_minus_x_path_decision);

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


        // vector_x should have length 1
        let final_val = vector_x_decision.clone().mle_ref().bookkeeping_table[0].to_bytes_le();
        let exponentiated_nodes = DenseMle::new_from_raw(vec![F::from_bytes_le(&final_val)], LayerId::Input(0), None);

        let difference_builder = EqualityCheck::new(
            vector_x_decision,
            exponentiated_nodes
        );

        let difference_mle = layers.add::<_, EmptyLayer<F, Self::Transcript>>(difference_builder);

        // return (layers, vec![Box::new(difference_mle)]);

        let one_mle_result: DenseMle<F, F> = DenseMle::one(1, LayerId::Input(0), None);
        let one_mle: DenseMle<F, F> = DenseMle::one(2, LayerId::Input(0), None);
        let batch_mle = vec![
            one_mle.clone(),
            one_mle.clone(),
            // one_mle.clone(),
            // one_mle
        ];

        let batch_bits = log2(batch_mle.len()) as usize;

        let r_minus_x_path_builder = BatchedLayer::new(
            batch_mle.iter().map(|x| {
                let mut x = x.clone();
                x.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
                RMinusXBuilder::new(
                    x.clone(),
                    self.r
                )}
            ).collect_vec());

        let r_minus_x_path = layers.add_gkr(r_minus_x_path_builder);

        let mut vector_x_decision = unbatch_mles(r_minus_x_path);

        for _ in 0..vector_x_decision.num_iterated_vars() {
            let curr_num_vars = vector_x_decision.num_iterated_vars();
            let layer = SplitProductBuilder::new(vector_x_decision);
            vector_x_decision = if curr_num_vars != 1 {
                layers.add_gkr(layer)
            } else {
                layers.add::<_, EmptyLayer<_, _>>(layer)
            }
        }

        let difference_builder = EqualityCheck::new(
            vector_x_decision,
            one_mle_result
        );

        let difference_mle = layers.add::<_, EmptyLayer<_, _>>(difference_builder);
        // return (layers, vec![Box::new(difference_mle)]);

        // TODO!(ende) clear this when debug is done
        // r: Fr::from(3),
        // r_packings: (Fr::from(5), Fr::from(4)),

        let bit_difference = self.dummy_decision_nodes_mle.num_iterated_vars() - self.dummy_leaf_nodes_mle.num_iterated_vars();

        // layer 0
        let decision_packing_builder = DecisionPackingBuilder::new(
            self.dummy_decision_nodes_mle.clone(), self.r, self.r_packings);

        let leaf_packing_builder = LeafPackingBuilder::new(
            self.dummy_leaf_nodes_mle.clone(), self.r, self.r_packings.0
        );
        let decision_packed = layers.add_gkr(decision_packing_builder);

        // let packing_builders = decision_packing_builder.concat_with_padding(leaf_packing_builder, Padding::Right(bit_difference));
        // let (decision_packed, leaf_packed) = layers.add_gkr(packing_builders);

        let decision_packing_builder_2 = DecisionPackingBuilder::new(
            self.dummy_decision_nodes_mle.clone(), self.r, self.r_packings);

        let leaf_packing_builder_2 = LeafPackingBuilder::new(
            self.dummy_leaf_nodes_mle.clone(), self.r, self.r_packings.0
        );

        let decision_packed_2 = layers.add_gkr(decision_packing_builder_2);

        // let packing_builders_2 = decision_packing_builder_2.concat_with_padding(leaf_packing_builder_2, Padding::Right(bit_difference));
        // let (decision_packed_2, leaf_packed_2) = layers.add_gkr(packing_builders_2);

        let difference_builder = EqualityCheck::new(
            decision_packed,
            decision_packed_2
        );

        let difference_mle = layers.add_gkr(difference_builder);

        // return (layers, vec![Box::new(difference_mle)]);

        let decision_packing_builder = DecisionPackingBuilder::new(
            self.dummy_decision_nodes_mle.clone(), self.r, self.r_packings);

        let packing_builders = decision_packing_builder.concat_with_padding(leaf_packing_builder, Padding::Right(bit_difference));
        let (decision_packed, leaf_packed) = layers.add_gkr(packing_builders);

        // layer 1
        let r_minus_x_builder_decision =  RMinusXBuilder::new(
            decision_packed, self.r_packings.0
        );
        let r_minus_x_builder_leaf =  RMinusXBuilder::new(
            leaf_packed, self.r_packings.0
        );
        let r_minus_x_builders = r_minus_x_builder_decision.concat(r_minus_x_builder_leaf);
        let (mut r_minus_x_decision, mut r_minus_x_leaf) = layers.add_gkr(r_minus_x_builders);

        // layer 2
        let prev_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
            self.dummy_multiplicities_bin_decomp_mle_decision.clone(),
            0,
            r_minus_x_decision.clone()
        );

        let prev_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
            self.dummy_multiplicities_bin_decomp_mle_leaf.clone(),
            0,
            r_minus_x_leaf.clone()
        );
        let pre_prod_builders = prev_prod_builder_decision.concat(prev_prod_builder_leaf);
        let (mut prev_prod_decision, mut prev_prod_leaf) = layers.add_gkr(pre_prod_builders);

        for i in 1..16 {

            // layer 3, or i + 2
            let r_minus_x_square_builder_decision = SquaringBuilder::new(
                r_minus_x_decision
            );
            let r_minus_x_square_builder_leaf = SquaringBuilder::new(
                r_minus_x_leaf
            );
            let r_minus_x_square_builders = r_minus_x_square_builder_decision.concat(r_minus_x_square_builder_leaf);
            let (r_minus_x_square_decision, r_minus_x_square_leaf) = layers.add_gkr(r_minus_x_square_builders);

            // layer 4, or i + 3
            let curr_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
                self.dummy_multiplicities_bin_decomp_mle_decision.clone(),
                i,
                r_minus_x_square_decision.clone()
            );

            let curr_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
                self.dummy_multiplicities_bin_decomp_mle_leaf.clone(),
                i,
                r_minus_x_square_leaf.clone()
            );

            let curr_prod_builders = curr_prod_builder_decision.concat(curr_prod_builder_leaf);

            let (curr_prod_decision, curr_prod_leaf) = layers.add_gkr(curr_prod_builders);

            // layer 5, or i + 4
            let prod_builder_decision = ProductBuilder::new(
                curr_prod_decision,
                prev_prod_decision
            );

            let prod_builder_leaf = ProductBuilder::new(
                curr_prod_leaf,
                prev_prod_leaf
            );

            let prod_builders = prod_builder_decision.concat(prod_builder_leaf);

            (prev_prod_decision, prev_prod_leaf) = layers.add_gkr(prod_builders);

            r_minus_x_decision = r_minus_x_square_decision;
            r_minus_x_leaf = r_minus_x_square_leaf;

        }

        let mut exponentiated_decision = prev_prod_decision;
        let mut exponentiated_leaf = prev_prod_leaf;

        for i in 0..self.tree_height-1 {

            // layer 20, or i+20
            let prod_builder_decision = SplitProductBuilder::new(
                exponentiated_decision
            );
            let prod_builder_leaf = SplitProductBuilder::new(
                exponentiated_leaf
            );

            let prod_builders = prod_builder_decision.concat(prod_builder_leaf);

            // if i == self.tree_height - 1 {
            //     exponentiated_decision = layers.add::<_, EmptyLayer<F, Self::Transcript>>(prod_builder);
            // } else {
            //     (exponentiated_decision, exponentiated_leaf) = layers.add_gkr(prod_builders);
            // }
            (exponentiated_decision, exponentiated_leaf) = layers.add_gkr(prod_builders);
        }

        let prod_builder_nodes = ProductBuilder::new(
            exponentiated_decision,
            exponentiated_leaf
        );

        let exponentiated_nodes = layers.add::<_, EmptyLayer<F, Self::Transcript>>(prod_builder_nodes);

        let difference_builder = EqualityCheck::new(
            exponentiated_nodes.clone(),
            exponentiated_nodes
        );

        let difference_mle = layers.add::<_, EmptyLayer<F, Self::Transcript>>(difference_builder);

        // return (layers, vec![Box::new(difference_mle)]);

        let mut layers = Layers::new();

        let mut vector_x: DenseMle<F, F> = DenseMle::one(8, LayerId::Layer(0), None);

        let pad_mle: DenseMle<F, F> = DenseMle::one(1, LayerId::Layer(0), None);

        let concat_builder = ConcatBuilder::new(
            vector_x,
            pad_mle.clone()
        );
        vector_x = layers.add_gkr(concat_builder);

        let bit_difference = vector_x.num_iterated_vars() - pad_mle.num_iterated_vars();

        let vector_x_builder = IdentityBuilder::new(
            vector_x.clone()
        );
        let pad_mle_builder = IdentityBuilder::new(
            pad_mle
        );

        let path_packing_builders = vector_x_builder.concat_with_padding(pad_mle_builder, Padding::Right(bit_difference));

        for _ in 0..vector_x.num_iterated_vars() {
            let curr_num_vars = vector_x.num_iterated_vars();
            let layer = SplitProductBuilder::new(vector_x);
            vector_x = if curr_num_vars != 1 {
                layers.add_gkr(layer)
            } else {
                layers.add::<_, EmptyLayer<_, _>>(layer)
            }
        }

        let difference_builder = EqualityCheck::new(
            vector_x.clone(),
            vector_x
        );

        // let difference_mle = layers.add_gkr(difference_builder);

        let difference_mle = layers.add::<_, EmptyLayer<F, Self::Transcript>>(difference_builder);

        // return (layers, vec![Box::new(difference_mle)]);


        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        let one_mle: DenseMle<F, F> = DenseMle::one(16, LayerId::Layer(0), None);
        let mut one_mle_vec = vec![];
        for _ in 0..4 {
            one_mle_vec.push(one_mle.clone());
        }

        let mut one_mle_longer = one_mle_vec.clone();
        for _ in 0..4 {
            one_mle_longer.push(one_mle.clone());
        }

        let one_concat_builder = BatchedLayer::new(
            one_mle_vec.iter()
            .zip(one_mle_longer.iter()).map(
                |(a, b)|
                ConcatBuilder::new(
                    a.clone(),
                    b.clone()
                )).collect_vec());

        let one_concat = layers.add_gkr(one_concat_builder);

        let difference_builder = EqualityCheck::new_batched(
            one_concat.clone(),
            one_concat,
        );

        let difference_mle = layers.add_gkr(difference_builder);

        let difference_mle = combine_zero_mle_ref(difference_mle);

        // return (layers, vec![Box::new(difference_mle)]);


        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        let batch_bits = log2(self.dummy_decision_node_paths_mle_vec.len()) as usize;

        let bit_difference = self.dummy_decision_node_paths_mle_vec[0].num_iterated_vars() - self.dummy_leaf_node_paths_mle_vec[0].num_iterated_vars();

        let decision_path_packing_builder = BatchedLayer::new(
            self.dummy_decision_node_paths_mle_vec.iter().map(
                |decision_node_mle| {
                    let mut decision_node_mle = decision_node_mle.clone();
                    decision_node_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
                    DecisionPackingBuilder::new(
                        decision_node_mle.clone(),
                        self.r,
                        self.r_packings
                    )
                }
                    
            ).collect_vec());

        let decision_path_packing_builder_another = BatchedLayer::new(
            self.dummy_decision_node_paths_mle_vec.iter().map(
                |decision_node_mle| {
                    let mut decision_node_mle = decision_node_mle.clone();
                    decision_node_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
                    DecisionPackingBuilder::new(
                        decision_node_mle.clone(),
                        self.r,
                        self.r_packings
                    )
                }
            ).collect_vec());

        let leaf_path_packing_builder = BatchedLayer::new(
            self.dummy_leaf_node_paths_mle_vec.iter().map(
                |leaf_node_mle| {
                    let mut leaf_node_mle = leaf_node_mle.clone();
                    leaf_node_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
                    LeafPackingBuilder::new(
                        leaf_node_mle.clone(),
                        self.r,
                        self.r_packings.0
                    )
                }
            ).collect_vec());

        // let path_packing_builders = decision_path_packing_builder.concat(leaf_path_packing_builder);
        let path_packing_builders = decision_path_packing_builder.concat_with_padding(leaf_path_packing_builder, Padding::Right(bit_difference));
        let (decision_path_packed, leaf_path_packed) = layers.add_gkr(path_packing_builders);
        let decision_path_packed_another = layers.add_gkr(decision_path_packing_builder_another);

        let difference_builder = EqualityCheck::new_batched(
            decision_path_packed,
            decision_path_packed_another,
        );

        let difference_mle = layers.add_gkr(difference_builder);

        let difference_mle = combine_zero_mle_ref(difference_mle);

        todo!()
    }
}

/// permutation circuit, non batched version
pub struct PermutationCircuitNonBatched<F: FieldExt> {
    dummy_input_data_mle: DenseMle<F, InputAttribute<F>>,
    dummy_permuted_input_data_mle: DenseMle<F, InputAttribute<F>>,
    r: F,
    r_packing: F,
    input_len: usize,
}

impl<F: FieldExt> GKRCircuit<F> for PermutationCircuitNonBatched<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        // layer 0: packing
        let input_packing_builder: InputPackingBuilder<F> = InputPackingBuilder::new(
            self.dummy_input_data_mle.clone(),
            self.r,
            self.r_packing);

        let input_permuted_packing_builder: InputPackingBuilder<F> = InputPackingBuilder::new(
            self.dummy_permuted_input_data_mle.clone(),
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

struct AttributeConsistencyCircuitNonBatched<F: FieldExt> {
    dummy_permuted_input_data_mle_vec: DenseMle<F, InputAttribute<F>>,
    dummy_decision_node_paths_mle_vec: DenseMle<F, DecisionNode<F>>,
    tree_height: usize,
}

impl<F: FieldExt> GKRCircuit<F> for AttributeConsistencyCircuitNonBatched<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        let attribute_consistency_builder = AttributeConsistencyBuilder::new(
            self.dummy_permuted_input_data_mle_vec.clone(),
            self.dummy_decision_node_paths_mle_vec.clone(),
            self.tree_height
        );

        let difference_mle = layers.add_gkr(attribute_consistency_builder);

        todo!()
    }
}

struct AttributeConsistencyCircuit<F: FieldExt> {
    dummy_permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    dummy_decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>,
    tree_height: usize,
}

impl<F: FieldExt> GKRCircuit<F> for AttributeConsistencyCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        let batch_bits = log2(self.dummy_permuted_input_data_mle_vec.len()) as usize;
    
        let attribute_consistency_builder = BatchedLayer::new(

            self.dummy_permuted_input_data_mle_vec
                    .iter()
                    .zip(self.dummy_decision_node_paths_mle_vec.iter())
                    .map(|(input_data_mle, decision_path_mle)| {

                        let mut input_data_mle = input_data_mle.clone();
                        input_data_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));

                        let mut decision_path_mle = decision_path_mle.clone();
                        decision_path_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));

                        AttributeConsistencyBuilderZeroRef::new(
                            input_data_mle,
                            decision_path_mle,
                            self.tree_height
                        )

        }).collect_vec());

        let difference_mle = layers.add_gkr(attribute_consistency_builder);
        let difference_mle = combine_zero_mle_ref(difference_mle);

        todo!()
    }
}

struct MultiSetCircuit<F: FieldExt> {
    dummy_decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
    dummy_leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
    dummy_multiplicities_bin_decomp_mle_decision: DenseMle<F, BinDecomp16Bit<F>>,
    dummy_multiplicities_bin_decomp_mle_leaf: DenseMle<F, BinDecomp16Bit<F>>,
    dummy_decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>, // batched
    dummy_leaf_node_paths_mle_vec: Vec<DenseMle<F, LeafNode<F>>>,         // batched
    r: F,
    r_packings: (F, F),
    tree_height: usize,
}

impl<F: FieldExt> GKRCircuit<F> for MultiSetCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        // layer 0: x
        let decision_packing_builder = DecisionPackingBuilder::new(
            self.dummy_decision_nodes_mle.clone(), self.r, self.r_packings);

        let leaf_packing_builder = LeafPackingBuilder::new(
            self.dummy_leaf_nodes_mle.clone(), self.r, self.r_packings.0
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

        // layer 2, part 1: (r - x) * b_ij + (1 - b_ij)
        let prev_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
            self.dummy_multiplicities_bin_decomp_mle_decision.clone(),
            0,
            r_minus_x_power_decision.clone()
        );

        let prev_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
            self.dummy_multiplicities_bin_decomp_mle_leaf.clone(),
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
            self.dummy_multiplicities_bin_decomp_mle_decision.clone(),
            1,
            r_minus_x_power_decision.clone()
        );

        let prev_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
            self.dummy_multiplicities_bin_decomp_mle_leaf.clone(),
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
                self.dummy_multiplicities_bin_decomp_mle_decision.clone(),
                i,
                r_minus_x_power_decision.clone()
            );

            let curr_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
                self.dummy_multiplicities_bin_decomp_mle_leaf.clone(),
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
            self.dummy_multiplicities_bin_decomp_mle_decision.clone(),
            15,
            r_minus_x_power_decision.clone()
        );

        let curr_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
            self.dummy_multiplicities_bin_decomp_mle_leaf.clone(),
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

        let bit_difference = self.dummy_decision_node_paths_mle_vec[0].num_iterated_vars() - self.dummy_leaf_node_paths_mle_vec[0].num_iterated_vars();

        // layer 0: packing

        let batch_bits = log2(self.dummy_decision_node_paths_mle_vec.len()) as usize;

        let decision_path_packing_builder = BatchedLayer::new(
            self.dummy_decision_node_paths_mle_vec.iter().map(
                |decision_node_mle| {
                    let mut decision_node_mle = decision_node_mle.clone();
                    decision_node_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
                    DecisionPackingBuilder::new(
                        decision_node_mle.clone(),
                        self.r,
                        self.r_packings
                    )
                }
            ).collect_vec());

        let leaf_path_packing_builder = BatchedLayer::new(
            self.dummy_leaf_node_paths_mle_vec.iter().map(
                |leaf_node_mle| {
                    let mut leaf_node_mle = leaf_node_mle.clone();
                    leaf_node_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
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

        let difference_mle = layers.add::<_, EmptyLayer<F, Self::Transcript>>(difference_builder);

        println!("Multiset circuit finished, number of layers {:?}", layers.next_layer_id());

        todo!()
    }
}

struct BinaryRecompCircuit<F: FieldExt> {
    decision_node_path_mle: DenseMle<F, DecisionNode<F>>,
    permuted_inputs_mle: DenseMle<F, InputAttribute<F>>,
    diff_signed_bin_decomp: DenseMle<F, BinDecomp16Bit<F>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinaryRecompCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- Inputs to the circuit are just these three MLEs ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.decision_node_path_mle), Box::new(&mut self.permuted_inputs_mle), Box::new(&mut self.diff_signed_bin_decomp)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- First we create the positive binary recomp ---
        let pos_bin_recomp_builder = BinaryRecompBuilder::new(self.diff_signed_bin_decomp.clone());
        let pos_bin_recomp_mle = layers.add_gkr(pos_bin_recomp_builder);

        // --- Next, we create the diff ---
        // TODO!(ryancao): Combine this and the above layer!!!
        let diff_builder = NodePathDiffBuilder::new(
            self.decision_node_path_mle.clone(),
            self.permuted_inputs_mle.clone()
        );
        let raw_diff_mle = layers.add_gkr(diff_builder);

        // --- Finally, we create the checker ---
        let recomp_checker_builder = BinaryRecompCheckerBuilder::new(
            raw_diff_mle,
            self.diff_signed_bin_decomp.clone(),
            pos_bin_recomp_mle,
        );
        let recomp_checker_mle = layers.add_gkr(recomp_checker_builder);

        // --- Create input layers ---
        let live_committed_input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();

        Witness { layers, output_layers: vec![recomp_checker_mle.get_enum()], input_layers: vec![live_committed_input_layer.to_enum()] }
    }

}

struct PartialBitsCheckerCircuit<F: FieldExt> {
    permuted_inputs_mle: DenseMle<F, InputAttribute<F>>,
    num_vars_to_grab: usize,
}
impl<F: FieldExt> GKRCircuit<F> for PartialBitsCheckerCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.permuted_inputs_mle)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

        let mut layers = Layers::new();
        let builder = PartialBitsCheckerBuilder::new(self.permuted_inputs_mle.clone(), self.num_vars_to_grab);
        let result = layers.add_gkr(builder);

        let input_layer: PublicInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();
        Witness { layers, output_layers: vec![result.get_enum()], input_layers: vec![input_layer.to_enum()] }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use ark_std::{test_rng, UniformRand};
    use rand::Rng;

    use crate::{zkdt::{zkdt_helpers::{DummyMles, generate_dummy_mles, NUM_DUMMY_INPUTS, DUMMY_INPUT_LEN, TREE_HEIGHT, generate_dummy_mles_batch, BatchedDummyMles, BatchedCatboostMles, generate_mles_batch_catboost_single_tree}, zkdt_circuit_parts::{PartialBitsCheckerCircuit, BinaryRecompCircuit, PermutationCircuitNonBatched}}, prover::GKRCircuit};
    use remainder_shared_types::transcript::{Transcript, poseidon_transcript::PoseidonTranscript};

    use super::{PermutationCircuit, AttributeConsistencyCircuitNonBatched, MultiSetCircuit, TestCircuit, AttributeConsistencyCircuit};
    #[test]
    fn test_partial_bits_checker() {
        let DummyMles { 
            dummy_permuted_input_data_mle,
            // dummy_decision_node_paths_mle,
            // dummy_binary_decomp_diffs_mle,
            ..
        } = generate_dummy_mles();

        let mut circuit = PartialBitsCheckerCircuit::<Fr> {
            permuted_inputs_mle: dummy_permuted_input_data_mle,
            num_vars_to_grab: 1,
        };

        let mut transcript = PoseidonTranscript::new("Bin Recomp Circuit Transcript");
        let now = Instant::now();
        let proof = circuit.prove(&mut transcript);
        println!("Proof generated!: Took {} seconds", now.elapsed().as_secs_f32());

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Bin Recomp Circuit Transcript");
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
    fn test_bin_recomp_circuit() {
        let DummyMles { 
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle,
            dummy_binary_decomp_diffs_mle,
            ..
        } = generate_dummy_mles();

        let mut circuit = BinaryRecompCircuit::<Fr> {
            decision_node_path_mle: dummy_decision_node_paths_mle,
            permuted_inputs_mle: dummy_permuted_input_data_mle,
            diff_signed_bin_decomp: dummy_binary_decomp_diffs_mle,
        };

        let mut transcript = PoseidonTranscript::new("Bin Recomp Circuit Transcript");
        let now = Instant::now();
        let proof = circuit.prove(&mut transcript);
        println!("Proof generated!: Took {} seconds", now.elapsed().as_secs_f32());

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Bin Recomp Circuit Transcript");
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
    fn test_permutation_circuit_catboost_non_batched() {
        let mut rng = test_rng();

        let (BatchedCatboostMles {
            dummy_input_data_mle,
            dummy_permuted_input_data_mle, ..
        }, (_tree_height, input_len)) = generate_mles_batch_catboost_single_tree::<Fr>();

        let mut circuit = PermutationCircuitNonBatched {
            dummy_input_data_mle: dummy_input_data_mle[0].clone(),
            dummy_permuted_input_data_mle: dummy_permuted_input_data_mle[0].clone(),
            r: Fr::from(rng.gen::<u64>()),
            r_packing: Fr::from(rng.gen::<u64>()),
            input_len: input_len,
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
            dummy_input_data_mle,
            dummy_permuted_input_data_mle, ..
        }, (_tree_height, input_len)) = generate_mles_batch_catboost_single_tree::<Fr>();

        let dummy_input_len = dummy_input_data_mle.len();

        let mut circuit = PermutationCircuit {
            dummy_input_data_mle_vec: dummy_input_data_mle,
            dummy_permuted_input_data_mle_vec: dummy_permuted_input_data_mle,
            r: Fr::from(rng.gen::<u64>()),
            r_packing: Fr::from(rng.gen::<u64>()),
            input_len: input_len,
            num_inputs: dummy_input_len,
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
    fn test_permutation_circuit_dummy_batched() {
        let mut rng = test_rng();

        let BatchedDummyMles {
            dummy_input_data_mle,
            dummy_permuted_input_data_mle, ..
        } = generate_dummy_mles_batch();

        let mut circuit = PermutationCircuit {
            dummy_input_data_mle_vec: dummy_input_data_mle,
            dummy_permuted_input_data_mle_vec: dummy_permuted_input_data_mle,
            r: Fr::from(rng.gen::<u64>()),
            r_packing: Fr::from(rng.gen::<u64>()),
            input_len: DUMMY_INPUT_LEN * (TREE_HEIGHT - 1),
            num_inputs: 1,
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

        let mut circuit = AttributeConsistencyCircuitNonBatched {
            dummy_permuted_input_data_mle_vec: dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle_vec: dummy_decision_node_paths_mle,
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
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle, ..
        }, (tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>();

        let mut circuit = AttributeConsistencyCircuitNonBatched {
            dummy_permuted_input_data_mle_vec: dummy_permuted_input_data_mle[0].clone(),
            dummy_decision_node_paths_mle_vec: dummy_decision_node_paths_mle[0].clone(),
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
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle, ..
        }, (tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>();

        let mut circuit = AttributeConsistencyCircuit {
            dummy_permuted_input_data_mle_vec: dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle_vec: dummy_decision_node_paths_mle,
            tree_height
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
        //     num_inputs: NUM_DUMMY_INPUTS,
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
    fn test_test_circuit() {

        let mut rng = test_rng();

        let (BatchedCatboostMles {dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle,
            dummy_multiplicities_bin_decomp_mle_decision,
            dummy_multiplicities_bin_decomp_mle_leaf,
            dummy_decision_nodes_mle,
            dummy_leaf_nodes_mle, ..}, (tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>();

        let mut circuit = TestCircuit {
            dummy_decision_nodes_mle,
            dummy_leaf_nodes_mle,
            dummy_multiplicities_bin_decomp_mle_decision,
            dummy_multiplicities_bin_decomp_mle_leaf,
            dummy_decision_node_paths_mle_vec: dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle_vec: dummy_leaf_node_paths_mle,
            r: Fr::from(2),
            r_packings: (Fr::from(5), Fr::from(4)),
            tree_height,
        };

        let mut transcript = PoseidonTranscript::new("Concat Batch Circuit Prover Transcript");

        let proof = circuit.prove(&mut transcript);

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Concat Batch Circuit Verifier Transcript");
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
    fn test_multiset_circuit_catboost_batched() {

        let mut rng = test_rng();

        let (BatchedCatboostMles {dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle,
            dummy_multiplicities_bin_decomp_mle_decision,
            dummy_multiplicities_bin_decomp_mle_leaf,
            dummy_decision_nodes_mle,
            dummy_leaf_nodes_mle, ..}, (tree_height, input_len)) = generate_mles_batch_catboost_single_tree::<Fr>();

        let mut circuit = MultiSetCircuit {
            dummy_decision_nodes_mle,
            dummy_leaf_nodes_mle,
            dummy_multiplicities_bin_decomp_mle_decision,
            dummy_multiplicities_bin_decomp_mle_leaf,
            dummy_decision_node_paths_mle_vec: dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle_vec: dummy_leaf_node_paths_mle,
            r: Fr::from(rng.gen::<u64>()),
            r_packings: (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())),
            tree_height,
        };
        // TODO!(ende) clear this when debug is done

        let mut transcript = PoseidonTranscript::new("Batched Multiset Circuit Prover Transcript");

        let proof = circuit.prove(&mut transcript);

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Batched Multiset Circuit Verifier Transcript");
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
}