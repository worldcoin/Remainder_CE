use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::get_default_poseidon_parameters_internal;
use ark_ff::BigInteger;
use rand::Rng;
use rayon::{iter::Split, vec};
use tracing_subscriber::fmt::layer;
use std::{io::Empty, iter};

use ark_std::{log2, test_rng};
use itertools::{Itertools, repeat_n};

use crate::{mle::{dense::DenseMle, MleRef, beta::BetaTable, Mle, MleIndex}, layer::{LayerBuilder, empty_layer::EmptyLayer, batched::{BatchedLayer, combine_zero_mle_ref, unbatch_mles}, LayerId, Padding}, sumcheck::{compute_sumcheck_message, Evals, get_round_degree}, zkdt::builders::{BitExponentiationBuilderCatBoost, IdentityBuilder, AttributeConsistencyBuilderZeroRef, FSInputPackingBuilder, FSDecisionPackingBuilder, FSLeafPackingBuilder, FSRMinusXBuilder}, prover::{input_layer::{ligero_input_layer::LigeroInputLayer, combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer, MleInputLayer, enum_input_layer::InputLayerEnum, self, random_input_layer::RandomInputLayer}, combine_layers::combine_layers}};
use crate::{prover::{GKRCircuit, Layers, Witness, GKRError}, mle::{mle_enum::MleEnum}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::super::{builders::{InputPackingBuilder, SplitProductBuilder, EqualityCheck, AttributeConsistencyBuilder, DecisionPackingBuilder, LeafPackingBuilder, ConcatBuilder, RMinusXBuilder, BitExponentiationBuilder, SquaringBuilder, ProductBuilder}, structs::{InputAttribute, DecisionNode, LeafNode, BinDecomp16Bit}, binary_recomp_circuit::circuit_builders::{BinaryRecompBuilder, NodePathDiffBuilder, BinaryRecompCheckerBuilder, PartialBitsCheckerBuilder}, data_pipeline::dummy_data_generator::{BatchedCatboostMles, generate_mles_batch_catboost_single_tree}};

use crate::prover::input_layer::enum_input_layer::CommitmentEnum;

pub(crate) struct AttributeConsistencyCircuit<F: FieldExt> {
    permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>,
}

impl<F: FieldExt> GKRCircuit<F> for AttributeConsistencyCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        let tree_height = (1 << (self.decision_node_paths_mle_vec[0].num_iterated_vars() - 2)) + 1;

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
                            tree_height
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

impl<F: FieldExt> AttributeConsistencyCircuit<F> {
    pub fn new(
        permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
        decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>,
    ) -> Self {
        Self {
            permuted_input_data_mle_vec,
            decision_node_paths_mle_vec
        }
    }
}