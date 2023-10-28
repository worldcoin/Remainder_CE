#[cfg(test)]
mod tests {
    use std::{time::Instant, path::Path};

    use remainder_shared_types::Fr;
    use itertools::Itertools;
    
    
    

    use crate::{zkdt::{data_pipeline::dummy_data_generator::{DummyMles, generate_dummy_mles}, binary_recomp_circuit::{circuits::{PartialBitsCheckerCircuit, BinaryRecompCircuit}, dataparallel_circuits::BinaryRecompCircuitBatched, multitree_circuits::BinaryRecompCircuitMultiTree}, cache_upshot_catboost_inputs_for_testing::generate_mles_batch_catboost_single_tree, input_data_to_circuit_adapter::{BatchedZKDTCircuitMles, MinibatchData, load_upshot_data_single_tree_batch, convert_zkdt_circuit_data_into_mles}}, prover::{GKRCircuit, tests::test_circuit}};
    use remainder_shared_types::transcript::{Transcript, poseidon_transcript::PoseidonTranscript};
    

    #[test]
    fn test_partial_bits_checker() {
        let DummyMles { 
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle,
            // dummy_binary_decomp_diffs_mle,
            ..
        } = generate_dummy_mles();

        // let mut rng = test_rng();

        // let dummy_permuted_input_data = vec![
        //     InputAttribute { attr_id: Fr::from(rng.gen::<u64>()), attr_val: Fr::from(rng.gen::<u64>()) },
        //     InputAttribute { attr_id: Fr::from(rng.gen::<u64>()), attr_val: Fr::from(rng.gen::<u64>()) },
        //     InputAttribute { attr_id: Fr::from(rng.gen::<u64>()), attr_val: Fr::from(rng.gen::<u64>()) },
        //     InputAttribute { attr_id: Fr::from(rng.gen::<u64>()), attr_val: Fr::from(rng.gen::<u64>()) },
        //     InputAttribute { attr_id: Fr::from(rng.gen::<u64>()), attr_val: Fr::from(rng.gen::<u64>()) },
        //     InputAttribute { attr_id: Fr::from(rng.gen::<u64>()), attr_val: Fr::from(rng.gen::<u64>()) },
        //     InputAttribute { attr_id: Fr::from(rng.gen::<u64>()), attr_val: Fr::from(rng.gen::<u64>()) },
        //     InputAttribute { attr_id: Fr::from(rng.gen::<u64>()), attr_val: Fr::from(rng.gen::<u64>()) }
        // ];

        // let dummy_decision_node_paths = vec![
        //     DecisionNode{ node_id: Fr::from(rng.gen::<u64>()), attr_id: Fr::from(rng.gen::<u64>()), threshold: Fr::from(rng.gen::<u64>()) },
        //     DecisionNode{ node_id: Fr::from(rng.gen::<u64>()), attr_id: Fr::from(rng.gen::<u64>()), threshold: Fr::from(rng.gen::<u64>()) },
        //     // DecisionNode{ node_id: Fr::one(), attr_id: Fr::one(), threshold: Fr::one() },
        //     // DecisionNode{ node_id: Fr::from(4), attr_id: Fr::from(4), threshold: Fr::from(4) }
        // ];

    //     let dummy_permuted_input_data = (0..4).map(|idx| {
    //          InputAttribute { attr_id: Fr::from(idx + 17), attr_val: Fr::from(idx + 18) }
    //     }).collect_vec();
    //     let dummy_decision_node_paths = (0..2).map(|idx| {
    //         DecisionNode{ node_id: Fr::from(idx + 1), attr_id: Fr::from(idx + 2), threshold: Fr::from(idx + 3) }
    //    }).collect_vec();

        // let dummy_permuted_input_data_mle = DenseMle::new_from_iter(
        //     dummy_permuted_input_data
        //         .clone()
        //         .into_iter()
        //         .map(InputAttribute::from),
        //     LayerId::Input(0),
        //     None,
        // );
        
        // let dummy_decision_node_paths_mle = DenseMle::new_from_iter(
        //     dummy_decision_node_paths
        //         .clone()
        //         .into_iter()
        //         .map(DecisionNode::from),
        //     LayerId::Input(0),
        //     None,
        // );

        let mut circuit = PartialBitsCheckerCircuit::<Fr>::new(
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle,
            1,
        );

        let mut transcript = PoseidonTranscript::new("Bin Recomp Circuit Transcript");
        let now = Instant::now();
        let proof = circuit.prove(&mut transcript);
        println!("Proof generated!: Took {} seconds", now.elapsed().as_secs_f32());

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Bin Recomp Circuit Transcript");
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
    fn test_bin_recomp_circuit_non_batched() {

        // --- NOTE that this won't work unless we flip the binary decomp endian-ness!!! ---
        // let DummyMles { 
        //     dummy_permuted_input_data_mle,
        //     dummy_decision_node_paths_mle,
        //     dummy_binary_decomp_diffs_mle,
        //     ..
        // } = generate_dummy_mles();

        let (BatchedZKDTCircuitMles {
            binary_decomp_diffs_mle_vec,
            decision_node_paths_mle_vec,
            permuted_input_samples_mle_vec: permuted_input_data_mle_vec, ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        let mut circuit = BinaryRecompCircuit::<Fr>::new(
            decision_node_paths_mle_vec[0].clone(),
            permuted_input_data_mle_vec[0].clone(),
            binary_decomp_diffs_mle_vec[0].clone(),
        );

        let mut transcript = PoseidonTranscript::new("Bin Recomp Circuit Transcript");
        let now = Instant::now();
        let proof = circuit.prove(&mut transcript);
        println!("Proof generated!: Took {} seconds", now.elapsed().as_secs_f32());

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Bin Recomp Circuit Transcript");
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
    fn test_bin_recomp_circuit_batched() {

        // --- NOTE that this won't work unless we flip the binary decomp endian-ness!!! ---
        // let DummyMles { 
        //     dummy_permuted_input_data_mle,
        //     dummy_decision_node_paths_mle,
        //     dummy_binary_decomp_diffs_mle,
        //     ..
        // } = generate_dummy_mles();

        let (BatchedZKDTCircuitMles {
            binary_decomp_diffs_mle_vec,
            decision_node_paths_mle_vec,
            permuted_input_samples_mle_vec: permuted_input_data_mle_vec, ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>(2, Path::new("upshot_data/"));

        let mut circuit = BinaryRecompCircuitBatched::new(
            decision_node_paths_mle_vec,
            permuted_input_data_mle_vec,
            binary_decomp_diffs_mle_vec,
        );

        let mut transcript = PoseidonTranscript::new("Bin Recomp Circuit Transcript");
        let now = Instant::now();
        let proof = circuit.prove(&mut transcript);
        println!("Proof generated!: Took {} seconds", now.elapsed().as_secs_f32());

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Bin Recomp Circuit Transcript");
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
    fn test_bin_recomp_circuit_catboost_multitree() {

        let minibatch_data = MinibatchData {log_sample_minibatch_size: 1, sample_minibatch_number: 2};
        let trees_batched_data: Vec<BatchedZKDTCircuitMles<Fr>> = (0..2).map(
            |tree_num| {
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
        }).collect_vec();


        let (
            mut binary_decomp_diffs_mle_vec, 
            mut decision_node_paths_mle_vec, 
            mut permuted_input_samples_mle_vec
            ): (
               Vec<_>, Vec<_>, Vec<_>
            ) = trees_batched_data.into_iter()
               .map(|batched_mle| (
                   batched_mle.binary_decomp_diffs_mle_vec,
                   batched_mle.decision_node_paths_mle_vec,
                   batched_mle.permuted_input_samples_mle_vec,
               ))
               .multiunzip();
        
        let circuit = BinaryRecompCircuitMultiTree::new(
            decision_node_paths_mle_vec,
            permuted_input_samples_mle_vec,
            binary_decomp_diffs_mle_vec,
        );

        test_circuit(circuit, None);

    }

}