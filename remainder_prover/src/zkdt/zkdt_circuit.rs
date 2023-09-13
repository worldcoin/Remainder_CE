use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::get_default_poseidon_parameters_internal;
use ark_ff::BigInteger;
use rand::Rng;
use rayon::{iter::Split, vec};
use tracing_subscriber::fmt::layer;
use std::{io::Empty, iter};

use ark_std::{log2, test_rng};
use itertools::{Itertools, repeat_n};

use crate::{mle::{dense::DenseMle, MleRef, beta::BetaTable, Mle, MleIndex}, layer::{LayerBuilder, empty_layer::EmptyLayer, batched::{BatchedLayer, combine_zero_mle_ref, unbatch_mles}, LayerId, Padding}, sumcheck::{compute_sumcheck_message, Evals, get_round_degree}, zkdt::builders::{BitExponentiationBuilderCatBoost, IdentityBuilder, AttributeConsistencyBuilderZeroRef}, prover::{input_layer::{ligero_input_layer::LigeroInputLayer, combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer, MleInputLayer, enum_input_layer::{InputLayerEnum, CommitmentEnum}, self, random_input_layer::RandomInputLayer}, combine_layers::combine_layers, GKRError}};
use crate::{prover::{GKRCircuit, Layers, Witness}, mle::{mle_enum::MleEnum}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::{builders::{FSInputPackingBuilder, SplitProductBuilder, EqualityCheck, AttributeConsistencyBuilder, FSDecisionPackingBuilder, FSLeafPackingBuilder, ConcatBuilder, FSRMinusXBuilder, BitExponentiationBuilder, SquaringBuilder, ProductBuilder}, structs::{InputAttribute, DecisionNode, LeafNode, BinDecomp16Bit}, binary_recomp_circuit::{circuit_builders::{BinaryRecompBuilder, NodePathDiffBuilder, BinaryRecompCheckerBuilder, PartialBitsCheckerBuilder}, circuits::BinaryRecompCircuitBatched}, data_pipeline::dummy_data_generator::{BatchedCatboostMles, generate_mles_batch_catboost_single_tree}, path_consistency_circuit::circuits::PathCheckCircuitBatchedMul};
use std::{marker::PhantomData, path::Path};


pub struct PermutationSubCircuit<F: FieldExt> {
    pub input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,               // batched
    pub permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,      // batched
    r_mle: DenseMle<F, F>,
    r_packing_mle: DenseMle<F, F>,
}

impl<F: FieldExt> PermutationSubCircuit<F> {
    fn yield_sub_circuit(&mut self) -> Witness<F, PoseidonTranscript<F>> {
        let mut layers: Layers<_, PoseidonTranscript<F>> = Layers::new();

        let batch_bits = log2(self.input_data_mle_vec.len()) as usize;


        let input_packing_builder = BatchedLayer::new(
            self.input_data_mle_vec.iter().map(
                |input_data_mle| {
                    let mut input_data_mle = input_data_mle.clone();
                    // TODO!(ende) fix this atrocious fixed(false)
                    input_data_mle.add_prefix_bits(Some(input_data_mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    FSInputPackingBuilder::new(
                        input_data_mle,
                        self.r_mle.clone(),
                        self.r_packing_mle.clone()
                    )
                }).collect_vec());

        let input_permuted_packing_builder = BatchedLayer::new(
            self.permuted_input_data_mle_vec.iter().map(
                |input_data_mle| {
                    let mut input_data_mle = input_data_mle.clone();
                    // TODO!(ende) fix this atrocious fixed(true)
                    input_data_mle.add_prefix_bits(Some(input_data_mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    FSInputPackingBuilder::new(
                        input_data_mle,
                        self.r_mle.clone(),
                        self.r_packing_mle.clone()
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
            input_layers: vec![],
        }
    }
}

pub struct AttributeConsistencySubCircuit<F: FieldExt> {
    permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>,
}

impl<F: FieldExt> AttributeConsistencySubCircuit<F> {
    fn yield_sub_circuit(&mut self) -> Witness<F, PoseidonTranscript<F>> {

        let tree_height = (1 << (self.decision_node_paths_mle_vec[0].num_iterated_vars() - 2)) + 1;

        let mut layers: Layers<_, PoseidonTranscript<F>> = Layers::new();

        let batch_bits = log2(self.permuted_input_data_mle_vec.len()) as usize;

        let attribute_consistency_builder = BatchedLayer::new(

            self.permuted_input_data_mle_vec
                    .iter()
                    .zip(self.decision_node_paths_mle_vec.iter())
                    .map(|(input_data_mle, decision_path_mle)| {

                        let mut input_data_mle = input_data_mle.clone();
                        input_data_mle.add_prefix_bits(Some(input_data_mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));

                        let mut decision_path_mle = decision_path_mle.clone();
                        decision_path_mle.add_prefix_bits(Some(decision_path_mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));

                        AttributeConsistencyBuilderZeroRef::new(
                            input_data_mle,
                            decision_path_mle,
                            tree_height
                        )

        }).collect_vec());

        let difference_mle = layers.add_gkr(attribute_consistency_builder);
        let circuit_output = combine_zero_mle_ref(difference_mle);

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![],
        }
    }
}

pub struct MultiSetSubCircuit<F: FieldExt> {
    decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
    leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
    multiplicities_bin_decomp_mle_decision: DenseMle<F, BinDecomp16Bit<F>>,
    multiplicities_bin_decomp_mle_leaf: DenseMle<F, BinDecomp16Bit<F>>,
    decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>, // batched
    leaf_node_paths_mle_vec: Vec<DenseMle<F, LeafNode<F>>>,         // batched
    r_mle: DenseMle<F, F>,
    r_mle_another: DenseMle<F, F>,
    r_packing_mle: DenseMle<F, F>,
    r_packing_another_mle: DenseMle<F, F>,
}

impl<F: FieldExt> MultiSetSubCircuit<F> {
    fn yield_sub_circuit(&mut self) -> Witness<F, PoseidonTranscript<F>> {

        let tree_height = (1 << (self.decision_node_paths_mle_vec[0].num_iterated_vars() - 2)) + 1;

        let mut layers: Layers<_, PoseidonTranscript<F>> = Layers::new();

        // layer 0: x
        let mut decision_nodes_mle = self.decision_nodes_mle.clone();
        decision_nodes_mle.add_prefix_bits(self.decision_nodes_mle.get_prefix_bits());
        let decision_packing_builder = FSDecisionPackingBuilder::new(
            decision_nodes_mle,
            self.r_mle.clone(),
            self.r_packing_mle.clone(),
            self.r_packing_another_mle.clone(),
        );

        let mut leaf_nodes_mle = self.leaf_nodes_mle.clone();
        leaf_nodes_mle.add_prefix_bits(self.leaf_nodes_mle.get_prefix_bits());
        let leaf_packing_builder = FSLeafPackingBuilder::new(
            leaf_nodes_mle,
            self.r_mle.clone(),
            self.r_packing_mle.clone(),
        );

        let packing_builders = decision_packing_builder.concat(leaf_packing_builder);
        let (decision_packed, leaf_packed) = layers.add_gkr(packing_builders);

        // layer 1: (r - x)
        let r_minus_x_builder_decision =  FSRMinusXBuilder::new(
            decision_packed,
            self.r_mle_another.clone()
        );
        let r_minus_x_builder_leaf =  FSRMinusXBuilder::new(
            leaf_packed,
            self.r_mle_another.clone()
        );
        let r_minus_x_builders = r_minus_x_builder_decision.concat(r_minus_x_builder_leaf);
        let (r_minus_x_power_decision, r_minus_x_power_leaf) = layers.add_gkr(r_minus_x_builders);

        let mut multiplicities_bin_decomp_mle_decision = self.multiplicities_bin_decomp_mle_decision.clone();
        multiplicities_bin_decomp_mle_decision.add_prefix_bits(self.multiplicities_bin_decomp_mle_decision.get_prefix_bits());
        // layer 2, part 1: (r - x) * b_ij + (1 - b_ij)
        let prev_prod_builder_decision = BitExponentiationBuilderCatBoost::new(
            multiplicities_bin_decomp_mle_decision.clone(),
            0,
            r_minus_x_power_decision.clone()
        );

        let mut multiplicities_bin_decomp_mle_leaf = self.multiplicities_bin_decomp_mle_leaf.clone();
        multiplicities_bin_decomp_mle_leaf.add_prefix_bits(self.multiplicities_bin_decomp_mle_leaf.get_prefix_bits());
        let prev_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
            multiplicities_bin_decomp_mle_leaf.clone(),
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
            multiplicities_bin_decomp_mle_decision.clone(),
            1,
            r_minus_x_power_decision.clone()
        );

        let prev_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
            multiplicities_bin_decomp_mle_leaf.clone(),
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
                multiplicities_bin_decomp_mle_decision.clone(),
                i,
                r_minus_x_power_decision.clone()
            );

            let curr_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
                multiplicities_bin_decomp_mle_leaf.clone(),
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
            multiplicities_bin_decomp_mle_decision.clone(),
            15,
            r_minus_x_power_decision.clone()
        );

        let curr_prod_builder_leaf = BitExponentiationBuilderCatBoost::new(
            multiplicities_bin_decomp_mle_leaf.clone(),
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

        let exponentiated_nodes = layers.add::<_, EmptyLayer<F, PoseidonTranscript<F>>>(prod_builder_nodes);

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
                    decision_node_mle.add_prefix_bits(Some(decision_node_mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    FSDecisionPackingBuilder::new(
                        decision_node_mle.clone(),
                        self.r_mle.clone(),
                        self.r_packing_mle.clone(),
                        self.r_packing_another_mle.clone()
                    )
                }
            ).collect_vec());

        let leaf_path_packing_builder = BatchedLayer::new(
            self.leaf_node_paths_mle_vec.iter().map(
                |leaf_node_mle| {
                    let mut leaf_node_mle = leaf_node_mle.clone();
                    leaf_node_mle.add_prefix_bits(Some(leaf_node_mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    FSLeafPackingBuilder::new(
                        leaf_node_mle.clone(),
                        self.r_mle.clone(),
                        self.r_packing_mle.clone()
                    )
                }
            ).collect_vec());

        let path_packing_builders = decision_path_packing_builder.concat_with_padding(leaf_path_packing_builder, Padding::Right(bit_difference - 1));
        let (decision_path_packed, leaf_path_packed) = layers.add_gkr(path_packing_builders);

        // layer 2: r - x

        let bit_difference = decision_path_packed[0].num_iterated_vars() - leaf_path_packed[0].num_iterated_vars();

        let r_minus_x_path_builder_decision = BatchedLayer::new(
            decision_path_packed.iter().map(|x| FSRMinusXBuilder::new(
                x.clone(),
                self.r_mle_another.clone()
            )).collect_vec());

        let r_minus_x_path_builder_leaf = BatchedLayer::new(
            leaf_path_packed.iter().map(|x| FSRMinusXBuilder::new(
                x.clone(),
                self.r_mle_another.clone()
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

        let path_product = layers.add::<_, EmptyLayer<F, PoseidonTranscript<F>>>(path_product_builder);

        let difference_builder = EqualityCheck::new(
            exponentiated_nodes,
            path_product
        );

        let circuit_output = layers.add::<_, EmptyLayer<F, PoseidonTranscript<F>>>(difference_builder);

        println!("Multiset circuit finished, number of layers {:?}", layers.next_layer_id());

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![],
        }
    }
}


pub struct CombinedCircuits<F: FieldExt> {
    pub batched_catboost_mles: BatchedCatboostMles<F>
}

impl<F: FieldExt> GKRCircuit<F> for CombinedCircuits<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }

    fn synthesize_and_commit(
            &mut self,
            transcript: &mut Self::Transcript,
        ) -> Result<(Witness<F, Self::Transcript>, Vec<input_layer::enum_input_layer::CommitmentEnum<F>>), crate::prover::GKRError> {
        let (
            // --- Actual circuit components ---
            mut permutation_circuit,
            mut attribute_consistency_circuit,
            mut multiset_circuit,
            mut binary_recomp_circuit_batched,
            mut path_consistency_circuit_batched,

            // --- Input layer ---
            input_layers,
            inpunt_layers_commits
        ) = self.create_sub_circuits(transcript).unwrap();

        let permutation_witness = permutation_circuit.yield_sub_circuit();
        let attribute_consistency_witness = attribute_consistency_circuit.yield_sub_circuit();
        let multiset_consistency_witness = multiset_circuit.yield_sub_circuit();
        let binary_recomp_circuit_batched_witness = binary_recomp_circuit_batched.yield_sub_circuit();

        let (mut combined_circuit_layers, mut combined_circuit_output_layers) = combine_layers(
            vec![
                permutation_witness.layers,
                attribute_consistency_witness.layers,
                multiset_consistency_witness.layers,
                binary_recomp_circuit_batched_witness.layers,
            ],
            vec![
                permutation_witness.output_layers,
                attribute_consistency_witness.output_layers,
                multiset_consistency_witness.output_layers,
                binary_recomp_circuit_batched_witness.output_layers,
            ],
        )
        .unwrap();

        // --- Manually add the layers and output layers from the circuit involving gate MLEs ---
        let updated_combined_output_layers = path_consistency_circuit_batched.add_subcircuit_layers_to_combined_layers(
            &mut combined_circuit_layers, 
            combined_circuit_output_layers);
        

        Ok((
            Witness {
                layers: combined_circuit_layers,
                output_layers: updated_combined_output_layers,
                input_layers,
            },
            inpunt_layers_commits,
        ))
    }
}

impl <F: FieldExt> CombinedCircuits<F> {
    fn create_sub_circuits(&mut self, transcript: &mut PoseidonTranscript<F>) -> Result<(
            PermutationSubCircuit<F>,
            AttributeConsistencySubCircuit<F>,
            MultiSetSubCircuit<F>,
            BinaryRecompCircuitBatched<F>,
            PathCheckCircuitBatchedMul<F>,
            Vec<InputLayerEnum<F, PoseidonTranscript<F>>>, // input layers, including random layers
            Vec<CommitmentEnum<F>> // input layers' commitments
        ), GKRError> {

        let BatchedCatboostMles {
            mut input_data_mle_vec,
            mut permuted_input_data_mle_vec,
            mut decision_node_paths_mle_vec,
            mut leaf_node_paths_mle_vec,
            mut multiplicities_bin_decomp_mle_decision,
            mut multiplicities_bin_decomp_mle_leaf,
            mut decision_nodes_mle,
            mut leaf_nodes_mle,
            mut binary_decomp_diffs_mle_vec,
        } = self.batched_catboost_mles.clone();

        // deal w input
        let mut input_data_mle_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(input_data_mle_vec.clone());
        let mut permuted_input_data_mle_vec_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(permuted_input_data_mle_vec.clone());
        let mut decision_node_paths_mle_vec_combined = DenseMle::<F, DecisionNode<F>>::combine_mle_batch(decision_node_paths_mle_vec.clone());
        let mut leaf_node_paths_mle_vec_combined = DenseMle::<F, LeafNode<F>>::combine_mle_batch(leaf_node_paths_mle_vec.clone());
        let mut combined_batched_diff_signed_bin_decomp_mle = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(binary_decomp_diffs_mle_vec.clone());

        // put them into input layer
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut input_data_mle_combined),
            Box::new(&mut permuted_input_data_mle_vec_combined),
            Box::new(&mut decision_node_paths_mle_vec_combined),
            Box::new(&mut leaf_node_paths_mle_vec_combined),
            Box::new(&mut multiplicities_bin_decomp_mle_decision),
            Box::new(&mut multiplicities_bin_decomp_mle_leaf),
            Box::new(&mut decision_nodes_mle),
            Box::new(&mut leaf_nodes_mle),
            Box::new(&mut combined_batched_diff_signed_bin_decomp_mle),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

        // add prefix bits to vectors
        for input_data_mle in input_data_mle_vec.iter_mut() {
            input_data_mle.add_prefix_bits(input_data_mle_combined.get_prefix_bits());
        }
        for permuted_input_data_mle in permuted_input_data_mle_vec.iter_mut() {
            permuted_input_data_mle.add_prefix_bits(permuted_input_data_mle_vec_combined.get_prefix_bits())
        }
        for decision_node_paths_mle in decision_node_paths_mle_vec.iter_mut() {
            decision_node_paths_mle.add_prefix_bits(decision_node_paths_mle_vec_combined.get_prefix_bits())
        }
        for leaf_node_paths_mle in leaf_node_paths_mle_vec.iter_mut() {
            leaf_node_paths_mle.add_prefix_bits(leaf_node_paths_mle_vec_combined.get_prefix_bits())
        }
        for binary_decomp_diffs_mle in binary_decomp_diffs_mle_vec.iter_mut() {
            binary_decomp_diffs_mle.add_prefix_bits(combined_batched_diff_signed_bin_decomp_mle.get_prefix_bits())
        }

        let input_layer: PublicInputLayer<F, PoseidonTranscript<F>> = input_layer.to_input_layer();
        let mut input_layer = input_layer.to_enum();

        let input_commit = input_layer
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;
            InputLayerEnum::append_commitment_to_transcript(&input_commit, transcript).unwrap();

        // FS
        let random_r = RandomInputLayer::new(transcript, 1, LayerId::Input(1));
        let r_mle = random_r.get_mle();
        let mut random_r = random_r.to_enum();
        let random_r_commit = random_r
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;

        let random_r_another = RandomInputLayer::new(transcript, 1, LayerId::Input(2));
        let r_mle_another = random_r_another.get_mle();
        let mut random_r_another = random_r_another.to_enum();
        let random_r_another_commit = random_r_another
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;

        let random_r_packing = RandomInputLayer::new(transcript, 1, LayerId::Input(3));
        let r_packing_mle = random_r_packing.get_mle();
        let mut random_r_packing = random_r_packing.to_enum();
        let random_r_packing_commit = random_r_packing
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;

        let random_r_packing_another = RandomInputLayer::new(transcript, 1, LayerId::Input(4));
        let r_packing_another_mle = random_r_packing_another.get_mle();
        let mut random_r_packing_another = random_r_packing_another.to_enum();
        let random_r_packing_another_commit = random_r_packing_another
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;

        // FS

        // construct the circuits
        let permutation_circuit = PermutationSubCircuit {
            input_data_mle_vec,
            permuted_input_data_mle_vec: permuted_input_data_mle_vec.clone(),
            r_mle: r_mle.clone(),
            r_packing_mle: r_packing_mle.clone(),
        };

        let attribute_consistency_circuit = AttributeConsistencySubCircuit {
            permuted_input_data_mle_vec: permuted_input_data_mle_vec.clone(),
            decision_node_paths_mle_vec: decision_node_paths_mle_vec.clone(),
        };

        let multiset_circuit = MultiSetSubCircuit {
            decision_nodes_mle,
            leaf_nodes_mle,
            multiplicities_bin_decomp_mle_decision,
            multiplicities_bin_decomp_mle_leaf,
            decision_node_paths_mle_vec: decision_node_paths_mle_vec.clone(),
            leaf_node_paths_mle_vec: leaf_node_paths_mle_vec.clone(),
            r_mle: r_mle.clone(),
            r_mle_another,
            r_packing_mle: r_packing_mle.clone(),
            r_packing_another_mle,
        };

        let binary_recomp_circuit_batched = BinaryRecompCircuitBatched::new(
            decision_node_paths_mle_vec.clone(),
            permuted_input_data_mle_vec.clone(),
            binary_decomp_diffs_mle_vec.clone(),
        );

        let path_consistency_circuit_batched = PathCheckCircuitBatchedMul::new(
            decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            binary_decomp_diffs_mle_vec
        );

        Ok((
            // --- Actual circuit components ---
            permutation_circuit, 
            attribute_consistency_circuit, 
            multiset_circuit, 
            binary_recomp_circuit_batched,
            path_consistency_circuit_batched,

            // --- Input layers ---
            vec![
                input_layer,
                random_r,
                random_r_another,
                random_r_packing,
                random_r_packing_another
            ],

            vec![
                input_commit,
                random_r_commit,
                random_r_another_commit,
                random_r_packing_commit,
                random_r_packing_another_commit
            ]
        ))
    }
}

/// GKRCircuit that proves inference for a single decision tree
pub struct ZKDTCircuit<F: FieldExt> {
    _marker: PhantomData<F>,
}

impl<F: FieldExt> ZKDTCircuit<F> {
    pub fn new(directory: &Path) -> Self {
        todo!()
    }
}

impl<F: FieldExt> GKRCircuit<F> for ZKDTCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use ark_std::{test_rng, UniformRand};
    use itertools::Itertools;
    use rand::Rng;

    use crate::zkdt::data_pipeline::dummy_data_generator::generate_mles_batch_catboost_single_tree;
    
    use crate::prover::tests::test_circuit;

    use super::CombinedCircuits;

    #[test]
    fn test_combine_circuits() {

        let (batched_catboost_mles, (_, _)) = generate_mles_batch_catboost_single_tree::<Fr>();

        let combined_circuit = CombinedCircuits {
            batched_catboost_mles
        };

        test_circuit(combined_circuit, None);
    }

}