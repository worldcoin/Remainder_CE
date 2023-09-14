#[cfg(test)]
mod tests {
    use std::time::Instant;

    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;

    use crate::{zkdt::{path_consistency_circuit::circuits::{PathCheckCircuit, PathCheckCircuitBatched, PathCheckCircuitBatchedMul}, data_pipeline::dummy_data_generator::{BatchedCatboostMles, generate_mles_batch_catboost_single_tree}}, prover::GKRCircuit};
    use remainder_shared_types::transcript::{Transcript, poseidon_transcript::PoseidonTranscript};
    

    #[test]
    fn test_path_circuit_catboost() {

        let (BatchedCatboostMles {
            decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            binary_decomp_diffs_mle_vec, ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>(1);

        // let DummyMles::<Fr> {
        //     dummy_decision_node_paths_mle,
        //     dummy_leaf_node_paths_mle, 
        //     dummy_binary_decomp_diffs_mle, ..
        // } = generate_dummy_mles();

        //let num_copy_bits = log2(dummy_decision_node_paths_mle.len());
        //let flattened_decision_node_paths_mle = combine_mles(dummy_decision_node_paths_mle, num_copy_bits as usize);

        let mut circuit = PathCheckCircuit::new(
            decision_node_paths_mle_vec[0].clone(), 
            leaf_node_paths_mle_vec[0].clone(),
            binary_decomp_diffs_mle_vec[0].clone(),
            0,
        );
        let now = Instant::now();
        let mut transcript = PoseidonTranscript::new("Permutation Circuit Prover Transcript");
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
    fn test_path_mul_circuit_catboost() {

        let (BatchedCatboostMles {
            decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            binary_decomp_diffs_mle_vec, ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>(1);

        // let DummyMles::<Fr> {
        //     dummy_decision_node_paths_mle,
        //     dummy_leaf_node_paths_mle, 
        //     dummy_binary_decomp_diffs_mle, ..
        // } = generate_dummy_mles();

        //let num_copy_bits = log2(dummy_decision_node_paths_mle.len());
        //let flattened_decision_node_paths_mle = combine_mles(dummy_decision_node_paths_mle, num_copy_bits as usize);

        let mut circuit = PathCheckCircuit::new(
            decision_node_paths_mle_vec[0].clone(), 
            leaf_node_paths_mle_vec[0].clone(),
            binary_decomp_diffs_mle_vec[0].clone(),
            0,
        );
        let now = Instant::now();
        let mut transcript = PoseidonTranscript::new("Permutation Circuit Prover Transcript");
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
    fn test_path_circuit_catboost_batched() {

        let (BatchedCatboostMles {
            decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            binary_decomp_diffs_mle_vec, ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>(1);

        // let DummyMles::<Fr> {
        //     dummy_decision_node_paths_mle,
        //     dummy_leaf_node_paths_mle, 
        //     dummy_binary_decomp_diffs_mle, ..
        // } = generate_dummy_mles();

        // let num_copy_bits = log2(dummy_decision_node_paths_mle.len());
        // let flattened_decision_node_paths_mle = combine_mles(dummy_decision_node_paths_mle, num_copy_bits as usize);

        let mut circuit = PathCheckCircuitBatched::new(
            decision_node_paths_mle_vec.clone(), 
            leaf_node_paths_mle_vec.clone(),
            binary_decomp_diffs_mle_vec.clone(),
        );
        let now = Instant::now();
        let mut transcript = PoseidonTranscript::new("Permutation Circuit Prover Transcript");
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
    fn test_path_circuit_catboost_batched_mulgate() {

        let (BatchedCatboostMles {
            decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            binary_decomp_diffs_mle_vec, ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>(1);

        // let DummyMles::<Fr> {
        //     dummy_decision_node_paths_mle,
        //     dummy_leaf_node_paths_mle, 
        //     dummy_binary_decomp_diffs_mle, ..
        // } = generate_dummy_mles();

        // let num_copy_bits = log2(dummy_decision_node_paths_mle.len());
        // let flattened_decision_node_paths_mle = combine_mles(dummy_decision_node_paths_mle, num_copy_bits as usize);

        let mut circuit = PathCheckCircuitBatchedMul::new(
            decision_node_paths_mle_vec.clone(), 
            leaf_node_paths_mle_vec.clone(),
            binary_decomp_diffs_mle_vec.clone(),
        );
        let now = Instant::now();
        let mut transcript = PoseidonTranscript::new("Permutation Circuit Prover Transcript");
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

}