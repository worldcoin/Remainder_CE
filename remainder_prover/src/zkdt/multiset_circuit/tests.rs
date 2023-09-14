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

    use super::super::circuits::{MultiSetCircuit, FSMultiSetCircuit};

    #[test]
    fn test_multiset_circuit_catboost_batched() {

        let mut rng = test_rng();

        let (BatchedCatboostMles {decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            multiplicities_bin_decomp_mle_decision,
            multiplicities_bin_decomp_mle_leaf,
            decision_nodes_mle,
            leaf_nodes_mle, ..}, (tree_height, input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1);

        let circuit = MultiSetCircuit::new(
            decision_nodes_mle,
            leaf_nodes_mle,
            multiplicities_bin_decomp_mle_decision,
            multiplicities_bin_decomp_mle_leaf,
            decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            Fr::from(rng.gen::<u64>()),
            (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())),
        );

        test_circuit(circuit, None);
    }



    #[test]
    fn test_fs_multiset_circuit_catboost_batched() {

        let (BatchedCatboostMles {decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            multiplicities_bin_decomp_mle_decision,
            multiplicities_bin_decomp_mle_leaf,
            decision_nodes_mle,
            leaf_nodes_mle, ..}, (tree_height, input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1);

        let circuit = FSMultiSetCircuit::new(
            decision_nodes_mle,
            leaf_nodes_mle,
            multiplicities_bin_decomp_mle_decision,
            multiplicities_bin_decomp_mle_leaf,
            decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
        );

        test_circuit(circuit, None);
    }

}