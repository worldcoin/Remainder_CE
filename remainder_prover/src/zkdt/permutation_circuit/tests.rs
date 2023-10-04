#[cfg(test)]
mod tests {
    use std::{path::Path};

    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use ark_std::{test_rng};
    
    use rand::Rng;

    use crate::{zkdt::{data_pipeline::dummy_data_generator::{generate_dummy_mles_batch, BatchedDummyMles, BatchedCatboostMles, generate_mles_batch_catboost_single_tree}}};
    use remainder_shared_types::transcript::{Transcript};
    use crate::prover::tests::test_circuit;

    use super::super::circuits::{NonBatchedPermutationCircuit, PermutationCircuit, FSPermutationCircuit};


    #[test]
    fn test_permutation_circuit_catboost_non_batched() {
        let mut rng = test_rng();

        let (BatchedCatboostMles {
            input_samples_mle_vec: input_data_mle_vec,
            permuted_input_samples_mle_vec: permuted_input_data_mle_vec, ..
        }, (_tree_height, input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        let circuit = NonBatchedPermutationCircuit::new(
            input_data_mle_vec[0].clone(),
            permuted_input_data_mle_vec[0].clone(),
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
            input_len,
        );

        test_circuit(circuit, None);
    }

    #[test]
    fn test_permutation_circuit_catboost_batched() {
        let mut rng = test_rng();

        let (BatchedCatboostMles {
            input_samples_mle_vec: input_data_mle_vec,
            permuted_input_samples_mle_vec: permuted_input_data_mle_vec, ..
        }, (_tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        let circuit = PermutationCircuit::new(
            input_data_mle_vec,
            permuted_input_data_mle_vec,
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
        );

        test_circuit(circuit, None);
    }

    #[test]
    fn test_permutation_circuit_dummy_batched() {
        let mut rng = test_rng();

        let BatchedDummyMles {
            dummy_input_data_mle,
            dummy_permuted_input_data_mle, ..
        } = generate_dummy_mles_batch();

        let circuit = PermutationCircuit::new(
            dummy_input_data_mle,
            dummy_permuted_input_data_mle,
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
        );

        test_circuit(circuit, None);
    }

    #[test]
    fn test_fs_permutation_circuit_catboost_batched() {
        
        let (BatchedCatboostMles {
            input_samples_mle_vec: input_data_mle_vec,
            permuted_input_samples_mle_vec: permuted_input_data_mle_vec, ..
        }, (_tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        let circuit = FSPermutationCircuit::new(
            input_data_mle_vec,
            permuted_input_data_mle_vec,
        );

        test_circuit(circuit, None);
    }

}