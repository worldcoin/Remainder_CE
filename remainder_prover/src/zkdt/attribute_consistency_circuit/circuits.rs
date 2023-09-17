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


pub(crate) struct NonBatchedAttributeConsistencyCircuit<F: FieldExt> {
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

impl<F: FieldExt> NonBatchedAttributeConsistencyCircuit<F> {
    pub fn new(
        permuted_input_data_mle_vec: DenseMle<F, InputAttribute<F>>,
        decision_node_paths_mle_vec: DenseMle<F, DecisionNode<F>>,
        tree_height: usize,
    ) -> Self {
        Self {
            permuted_input_data_mle_vec,
            decision_node_paths_mle_vec,
            tree_height,
        }
    }
}