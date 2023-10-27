#[cfg(test)]
mod tests {
    use std::{path::Path};

    use remainder_shared_types::Fr;
    
    
    use ark_std::{start_timer, end_timer};
    

    
    use crate::{zkdt::{data_pipeline::dummy_data_generator::{DummyMles, generate_dummy_mles, TREE_HEIGHT}, attribute_consistency_circuit::dataparallel_circuits::AttributeConsistencyCircuit, cache_upshot_catboost_inputs_for_testing::generate_mles_batch_catboost_single_tree, input_data_to_circuit_adapter::BatchedZKDTCircuitMles}};
    use remainder_shared_types::transcript::{Transcript};
    use crate::prover::tests::test_circuit;

    use super::super::circuits::{NonBatchedAttributeConsistencyCircuit};


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

        let (BatchedZKDTCircuitMles {
            permuted_input_samples_mle_vec: permuted_input_data_mle_vec,
            decision_node_paths_mle_vec, ..
        }, (tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        let circuit = NonBatchedAttributeConsistencyCircuit::new(
            permuted_input_data_mle_vec[0].clone(),
            decision_node_paths_mle_vec[0].clone(),
            tree_height,
        );

        test_circuit(circuit, None);
    }

    #[test]
    fn test_attribute_consistency_circuit_catboost_batched() {


        let circuit_timer = start_timer!(|| "attr cons circuit");
        let wit_gen_timer = start_timer!(|| "wit gen");

        let (BatchedZKDTCircuitMles {
            permuted_input_samples_mle_vec: permuted_input_data_mle_vec,
            decision_node_paths_mle_vec, ..
        }, (_tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        end_timer!(wit_gen_timer);

        let circuit = AttributeConsistencyCircuit::new(
            permuted_input_data_mle_vec,
            decision_node_paths_mle_vec,
        );

        test_circuit(circuit, None);
        end_timer!(circuit_timer);

    }

}