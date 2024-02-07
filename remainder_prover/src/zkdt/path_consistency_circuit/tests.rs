#[cfg(test)]
mod tests {
    use std::{path::Path, time::Instant};

    use itertools::Itertools;
    use remainder_shared_types::Fr;

    use crate::{
        prover::{helpers::test_circuit, GKRCircuit},
        zkdt::{
            cache_upshot_catboost_inputs_for_testing::generate_mles_batch_catboost_single_tree,
            input_data_to_circuit_adapter::{
                convert_zkdt_circuit_data_into_mles, load_upshot_data_single_tree_batch,
                BatchedZKDTCircuitMles, MinibatchData,
            },
            path_consistency_circuit::{
                circuits::{PathCheckCircuit, PathMulCheckCircuit},
                dataparallel_circuits::PathCheckCircuitBatched,
                multitree_circuits::PathCheckCircuitBatchedNoMulMultiTree,
            },
        },
    };
    use remainder_shared_types::transcript::{poseidon_transcript::PoseidonSponge, Transcript};

    #[test]
    fn test_path_circuit_catboost() {
        let (
            BatchedZKDTCircuitMles {
                decision_node_paths_mle_vec,
                leaf_node_paths_mle_vec,
                binary_decomp_diffs_mle_vec,
                ..
            },
            (_tree_height, _),
        ) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

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
        let mut transcript = PoseidonSponge::new("Permutation Circuit Prover Transcript");
        let proof = circuit.prove(&mut transcript);
        println!(
            "Proof generated!: Took {} seconds",
            now.elapsed().as_secs_f32()
        );

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonSponge::new("Permutation Circuit Verifier Transcript");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            }
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }

    #[test]
    fn test_path_mul_circuit_catboost() {
        let (
            BatchedZKDTCircuitMles {
                decision_node_paths_mle_vec,
                leaf_node_paths_mle_vec,
                binary_decomp_diffs_mle_vec,
                ..
            },
            (_tree_height, _),
        ) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        // let DummyMles::<Fr> {
        //     dummy_decision_node_paths_mle,
        //     dummy_leaf_node_paths_mle,
        //     dummy_binary_decomp_diffs_mle, ..
        // } = generate_dummy_mles();

        //let num_copy_bits = log2(dummy_decision_node_paths_mle.len());
        //let flattened_decision_node_paths_mle = combine_mles(dummy_decision_node_paths_mle, num_copy_bits as usize);

        let mut circuit = PathMulCheckCircuit::new(
            decision_node_paths_mle_vec[0].clone(),
            leaf_node_paths_mle_vec[0].clone(),
            binary_decomp_diffs_mle_vec[0].clone(),
        );
        let now = Instant::now();
        let mut transcript = PoseidonSponge::new("Permutation Circuit Prover Transcript");
        let proof = circuit.prove(&mut transcript);
        println!(
            "Proof generated!: Took {} seconds",
            now.elapsed().as_secs_f32()
        );

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonSponge::new("Permutation Circuit Verifier Transcript");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            }
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }

    #[test]
    fn test_path_circuit_catboost_batched_mulgate() {
        let (
            BatchedZKDTCircuitMles {
                decision_node_paths_mle_vec,
                leaf_node_paths_mle_vec,
                binary_decomp_diffs_mle_vec,
                ..
            },
            (_tree_height, _),
        ) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        // let DummyMles::<Fr> {
        //     dummy_decision_node_paths_mle,
        //     dummy_leaf_node_paths_mle,
        //     dummy_binary_decomp_diffs_mle, ..
        // } = generate_dummy_mles();

        // let num_copy_bits = log2(dummy_decision_node_paths_mle.len());
        // let flattened_decision_node_paths_mle = combine_mles(dummy_decision_node_paths_mle, num_copy_bits as usize);

        let mut circuit = PathCheckCircuitBatched::new(
            decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            binary_decomp_diffs_mle_vec,
        );
        let now = Instant::now();
        let mut transcript = PoseidonSponge::new("Permutation Circuit Prover Transcript");
        let proof = circuit.prove(&mut transcript);
        println!(
            "Proof generated!: Took {} seconds",
            now.elapsed().as_secs_f32()
        );

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonSponge::new("Permutation Circuit Verifier Transcript");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            }
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }

    #[test]
    fn test_path_consistency_circuit_catboost_multitree() {
        let minibatch_data = MinibatchData {
            log_sample_minibatch_size: 2,
            sample_minibatch_number: 2,
        };
        let trees_batched_data: Vec<BatchedZKDTCircuitMles<Fr>> = (0..1)
            .map(|tree_num| {
                // --- Read in the Upshot data from file ---
                let (zkdt_circuit_data, (tree_height, input_len), _) =
                    load_upshot_data_single_tree_batch::<Fr>(
                        Some(minibatch_data.clone()),
                        tree_num,
                        Path::new(&"upshot_data/quantized-upshot-model.json".to_string()),
                        Path::new(&"upshot_data/upshot-quantized-samples.npy".to_string()),
                    );
                let (batched_catboost_mles, (_, _)) =
                    convert_zkdt_circuit_data_into_mles(zkdt_circuit_data, tree_height, input_len);
                batched_catboost_mles
            })
            .collect_vec();

        let (
            mut leaf_node_paths_mle_vec,
            mut decision_node_paths_vecs,
            mut binary_decomp_diffs_mle_vec,
        ): (Vec<_>, Vec<_>, Vec<_>) = trees_batched_data
            .into_iter()
            .map(|batched_mle| {
                (
                    batched_mle.leaf_node_paths_mle_vec,
                    batched_mle.decision_node_paths_mle_vec,
                    batched_mle.binary_decomp_diffs_mle_vec,
                )
            })
            .multiunzip();

        let circuit = PathCheckCircuitBatchedNoMulMultiTree::new(
            decision_node_paths_vecs,
            leaf_node_paths_mle_vec,
            binary_decomp_diffs_mle_vec,
        );

        test_circuit(circuit, None);
    }
}
