#[cfg(test)]
mod tests {
    use std::time::Instant;

    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;

    use crate::{zkdt::{zkdt_helpers::{BatchedCatboostMles, generate_mles_batch_catboost_single_tree}, path_consistency_circuit::circuits::{PathCheckCircuit, PathCheckCircuitBatched}}, prover::GKRCircuit};
    use remainder_shared_types::transcript::{Transcript, poseidon_transcript::PoseidonTranscript};
    

    #[test]
    fn test_path_circuit_catboost() {

        let (BatchedCatboostMles {
            dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle,
            dummy_binary_decomp_diffs_mle, ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>();

        // let DummyMles::<Fr> {
        //     dummy_decision_node_paths_mle,
        //     dummy_leaf_node_paths_mle, 
        //     dummy_binary_decomp_diffs_mle, ..
        // } = generate_dummy_mles();

        //let num_copy_bits = log2(dummy_decision_node_paths_mle.len());
        //let flattened_decision_node_paths_mle = combine_mles(dummy_decision_node_paths_mle, num_copy_bits as usize);

        let mut circuit = PathCheckCircuit {
            decision_node_paths_mle: dummy_decision_node_paths_mle[0].clone(), 
            leaf_node_paths_mle: dummy_leaf_node_paths_mle[0].clone(),
            bin_decomp_diff_mle: dummy_binary_decomp_diffs_mle[0].clone(),
            num_copy: 0,
        };
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
            dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle,
            dummy_binary_decomp_diffs_mle, ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>();

        // let DummyMles::<Fr> {
        //     dummy_decision_node_paths_mle,
        //     dummy_leaf_node_paths_mle, 
        //     dummy_binary_decomp_diffs_mle, ..
        // } = generate_dummy_mles();

        // let num_copy_bits = log2(dummy_decision_node_paths_mle.len());
        // let flattened_decision_node_paths_mle = combine_mles(dummy_decision_node_paths_mle, num_copy_bits as usize);

        let mut circuit = PathCheckCircuitBatched {
            batched_decision_node_paths_mle: dummy_decision_node_paths_mle.clone(), 
            batched_leaf_node_paths_mle: dummy_leaf_node_paths_mle.clone(),
            batched_bin_decomp_diff_mle: dummy_binary_decomp_diffs_mle.clone(),
        };
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