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
use crate::{prover::{GKRCircuit, Layers, Witness}, mle::{mle_enum::MleEnum}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::{zkdt_layer::{InputPackingBuilder, SplitProductBuilder, EqualityCheck, AttributeConsistencyBuilder, DecisionPackingBuilder, LeafPackingBuilder, ConcatBuilder, RMinusXBuilder, BitExponentiationBuilder, SquaringBuilder, ProductBuilder}, structs::{InputAttribute, DecisionNode, LeafNode, BinDecomp16Bit}, binary_recomp_circuit::circuit_builders::{BinaryRecompBuilder, NodePathDiffBuilder, BinaryRecompCheckerBuilder, PartialBitsCheckerBuilder}, zkdt_helpers::{BatchedCatboostMles, generate_mles_batch_catboost_single_tree}};
use std::{marker::PhantomData, path::Path};


pub struct PermutationSubCircuit<F: FieldExt> {
    pub dummy_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,               // batched
    pub dummy_input_data_mle_combined: DenseMle<F, F>,
    pub dummy_permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,      // batched
    pub dummy_permuted_input_data_mle_combined: DenseMle<F, F>,
    pub r: F,
    pub r_packing: F,
    pub input_len: usize,
    pub num_inputs: usize
}

impl<F: FieldExt> PermutationSubCircuit<F> {
    fn yield_sub_circuit(&mut self) -> Witness<F, PoseidonTranscript<F>> {
        let mut layers: Layers<_, PoseidonTranscript<F>> = Layers::new();

        let batch_bits = log2(self.dummy_input_data_mle_vec.len()) as usize;
    
    
        let input_packing_builder = BatchedLayer::new(
            self.dummy_input_data_mle_vec.iter().map(
                |input_data_mle| {
                    let mut input_data_mle = input_data_mle.clone();
                    // TODO!(ende) fix this atrocious fixed(false)
                    input_data_mle.add_prefix_bits(Some(self.dummy_input_data_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    InputPackingBuilder::new(
                        input_data_mle,
                        self.r,
                        self.r_packing
                    )
                }).collect_vec());

        let input_permuted_packing_builder = BatchedLayer::new(
            self.dummy_permuted_input_data_mle_vec.iter().map(
                |input_data_mle| {
                    let mut input_data_mle = input_data_mle.clone();
                    // TODO!(ende) fix this atrocious fixed(true)
                    input_data_mle.add_prefix_bits(Some(self.dummy_permuted_input_data_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    InputPackingBuilder::new(
                        input_data_mle,
                        self.r,
                        self.r_packing
                    )
                }).collect_vec());

        let packing_builders = input_packing_builder.concat(input_permuted_packing_builder);

        let (mut input_packed, mut input_permuted_packed) = layers.add_gkr(packing_builders);

        for _ in 0..log2(self.input_len) {
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
    dummy_permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    dummy_permuted_input_data_mle_combined: DenseMle<F, F>,
    dummy_decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>,
    dummy_decision_node_paths_mle_combined: DenseMle<F, F>,
    tree_height: usize,
}

impl<F: FieldExt> AttributeConsistencySubCircuit<F> {
    fn yield_sub_circuit(&mut self) -> Witness<F, PoseidonTranscript<F>> {
        let mut layers: Layers<_, PoseidonTranscript<F>> = Layers::new();

        let batch_bits = log2(self.dummy_permuted_input_data_mle_vec.len()) as usize;
    
        let attribute_consistency_builder = BatchedLayer::new(

            self.dummy_permuted_input_data_mle_vec
                    .iter()
                    .zip(self.dummy_decision_node_paths_mle_vec.iter())
                    .map(|(input_data_mle, decision_path_mle)| {

                        let mut input_data_mle = input_data_mle.clone();
                        input_data_mle.add_prefix_bits(Some(self.dummy_permuted_input_data_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));

                        let mut decision_path_mle = decision_path_mle.clone();
                        decision_path_mle.add_prefix_bits(Some(self.dummy_decision_node_paths_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));

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
            input_layers: vec![],
        }
    }
}

pub struct Combine2Circuits<F: FieldExt> {
    batched_catboost_mles: (BatchedCatboostMles<F>, (usize, usize))
}

impl<F: FieldExt> GKRCircuit<F> for Combine2Circuits<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        let (mut permutation_circuit,
            mut attribute_consistency_circuit,
            input_layer) = self.create_sub_circuits();

        let permutation_witness = permutation_circuit.yield_sub_circuit();
        let attribute_consistency_witness = attribute_consistency_circuit.yield_sub_circuit();

        let (layers, output_layers) = combine_layers(
            vec![
                permutation_witness.layers,
                attribute_consistency_witness.layers
            ],
            vec![
                permutation_witness.output_layers,
                attribute_consistency_witness.output_layers
            ],
        )
        .unwrap();
    
        Witness {
            layers,
            output_layers,
            input_layers: vec![input_layer],
        }
    }
}

impl <F: FieldExt> Combine2Circuits<F> {
    fn create_sub_circuits(&mut self) -> (
            PermutationSubCircuit<F>,
            AttributeConsistencySubCircuit<F>,
            InputLayerEnum<F, PoseidonTranscript<F>>) {

        let mut rng = test_rng();

        let (BatchedCatboostMles {
            dummy_input_data_mle,
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle,
            mut dummy_multiplicities_bin_decomp_mle_decision,
            mut dummy_multiplicities_bin_decomp_mle_leaf,
            mut dummy_decision_nodes_mle,
            mut dummy_leaf_nodes_mle, ..}, (tree_height, input_len)) = generate_mles_batch_catboost_single_tree::<F>();
            
        
        // deal w input 
        let mut dummy_input_data_mle_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(dummy_input_data_mle.clone());
        let mut dummy_permuted_input_data_mle_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(dummy_permuted_input_data_mle.clone());
        let mut dummy_decision_node_paths_mle_combined = DenseMle::<F, DecisionNode<F>>::combine_mle_batch(dummy_decision_node_paths_mle.clone());
        let mut dummy_leaf_node_paths_mle_combined = DenseMle::<F, LeafNode<F>>::combine_mle_batch(dummy_leaf_node_paths_mle.clone());

        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut dummy_input_data_mle_combined),
            Box::new(&mut dummy_permuted_input_data_mle_combined),
            Box::new(&mut dummy_decision_node_paths_mle_combined),
            Box::new(&mut dummy_leaf_node_paths_mle_combined),
            Box::new(&mut dummy_multiplicities_bin_decomp_mle_decision),
            Box::new(&mut dummy_multiplicities_bin_decomp_mle_leaf),
            Box::new(&mut dummy_decision_nodes_mle),
            Box::new(&mut dummy_leaf_nodes_mle),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let input_prefix_bits = input_layer.fetch_prefix_bits(); // for debug purpose
        let input_layer: PublicInputLayer<F, PoseidonTranscript<F>> = input_layer.to_input_layer();

        // construct the circuits
        let dummy_input_len = dummy_input_data_mle.len();
        let mut permutation_circuit = PermutationSubCircuit {
            dummy_input_data_mle_vec: dummy_input_data_mle,
            dummy_input_data_mle_combined,
            dummy_permuted_input_data_mle_vec: dummy_permuted_input_data_mle.clone(),
            dummy_permuted_input_data_mle_combined: dummy_permuted_input_data_mle_combined.clone(),
            r: F::from(rng.gen::<u64>()),
            r_packing: F::from(rng.gen::<u64>()),
            input_len,
            num_inputs: dummy_input_len,
        };

        let mut attribute_consistency_circuit = AttributeConsistencySubCircuit {
            dummy_permuted_input_data_mle_vec: dummy_permuted_input_data_mle,
            dummy_permuted_input_data_mle_combined,
            dummy_decision_node_paths_mle_vec: dummy_decision_node_paths_mle,
            dummy_decision_node_paths_mle_combined,
            tree_height,
        };

        (permutation_circuit, attribute_consistency_circuit, input_layer.to_enum())
    }
}

///GKRCircuit that proves inference for a single decision tree
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

    use crate::{zkdt::{zkdt_helpers::{DummyMles, generate_dummy_mles, NUM_DUMMY_INPUTS, DUMMY_INPUT_LEN, TREE_HEIGHT, generate_dummy_mles_batch, BatchedDummyMles, BatchedCatboostMles, generate_mles_batch_catboost_single_tree}, zkdt_circuit_parts::NonBatchedPermutationCircuit, structs::{InputAttribute, DecisionNode, LeafNode}, binary_recomp_circuit::circuits::{PartialBitsCheckerCircuit, BinaryRecompCircuit}}, prover::{GKRCircuit, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer}}, mle::{dense::DenseMle, MleRef, Mle}, layer::LayerId};
    use remainder_shared_types::transcript::{Transcript, poseidon_transcript::PoseidonTranscript};
    use crate::prover::tests::test_circuit;

    use super::{Combine2Circuits, PermutationSubCircuit, AttributeConsistencySubCircuit};

    #[test]
    fn test_combine_2_circuit() {

        let batched_catboost_mles = generate_mles_batch_catboost_single_tree::<Fr>();

        let combined_circuit = Combine2Circuits {
            batched_catboost_mles
        };
    
        test_circuit(combined_circuit, None);
    }

}