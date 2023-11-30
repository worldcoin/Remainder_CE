use crate::{mle::{dense::DenseMle}, prover::{input_layer::{MleInputLayer}}};
use crate::{prover::{GKRCircuit, Layers, Witness}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::super::{builders::{AttributeConsistencyBuilder}, structs::{InputAttribute, DecisionNode}};

pub(crate) struct NonBatchedAttributeConsistencyCircuit<F: FieldExt> {
    permuted_input_data_mle_vec: DenseMle<F, InputAttribute<F>>,
    decision_node_paths_mle_vec: DenseMle<F, DecisionNode<F>>,
    tree_height: usize,
}

impl<F: FieldExt> GKRCircuit<F> for NonBatchedAttributeConsistencyCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<F> = Layers::new();

        let attribute_consistency_builder = AttributeConsistencyBuilder::new(
            self.permuted_input_data_mle_vec.clone(),
            self.decision_node_paths_mle_vec.clone(),
            self.tree_height
        );

        let _difference_mle = layers.add_gkr(attribute_consistency_builder);

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