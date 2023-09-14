#[cfg(test)]
mod tests {
    use std::time::Instant;

    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use ark_std::{test_rng, UniformRand};
    use itertools::Itertools;
    use rand::Rng;

    use crate::{zkdt::{data_pipeline::dummy_data_generator::{DummyMles, generate_dummy_mles, NUM_DUMMY_INPUTS, DUMMY_INPUT_LEN, TREE_HEIGHT, generate_dummy_mles_batch, BatchedDummyMles, BatchedCatboostMles, generate_mles_batch_catboost_single_tree}, structs::{InputAttribute, DecisionNode, LeafNode}, binary_recomp_circuit::circuits::{PartialBitsCheckerCircuit, BinaryRecompCircuit}}, prover::{GKRCircuit, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer}}, mle::{dense::DenseMle, MleRef, Mle}, layer::LayerId};
    use remainder_shared_types::transcript::{Transcript, poseidon_transcript::PoseidonTranscript};
    use crate::prover::tests::test_circuit;

    use super::super::circuits::{NonBatchedAttributeConsistencyCircuit, AttributeConsistencyCircuit};


    #[test]
    fn test_attribute_consistency_circuit_dummy_non_batched() {

        let DummyMles::<Fr> {
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle, ..
        } = generate_dummy_mles();

        let circuit = NonBatchedAttributeConsistencyCircuit::new(
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle,
            TREE_HEIGHT,
        );

        test_circuit(circuit, None);
    }

    #[test]
    fn test_attribute_consistency_circuit_catboost_non_batched() {

        let (BatchedCatboostMles {
            permuted_input_data_mle_vec,
            decision_node_paths_mle_vec, ..
        }, (tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1);

        let circuit = NonBatchedAttributeConsistencyCircuit::new(
            permuted_input_data_mle_vec[0].clone(),
            decision_node_paths_mle_vec[0].clone(),
            tree_height,
        );

        test_circuit(circuit, None);
    }

    #[test]
    fn test_attribute_consistency_circuit_catboost_batched() {

        let (BatchedCatboostMles {
            permuted_input_data_mle_vec,
            decision_node_paths_mle_vec, ..
        }, (tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1);

        let mut circuit = AttributeConsistencyCircuit::new(
            permuted_input_data_mle_vec,
            decision_node_paths_mle_vec,
        );

        test_circuit(circuit, None);

    }

}