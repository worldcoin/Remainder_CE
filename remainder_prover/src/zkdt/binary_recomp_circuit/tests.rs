use crate::zkdt::binary_recomp_circuit::circuits::BinaryRecompCircuitBatched;

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use ark_std::{test_rng, UniformRand};
    use itertools::Itertools;
    use rand::Rng;

    use crate::{zkdt::{dummy_data_generator::{DummyMles, generate_dummy_mles, NUM_DUMMY_INPUTS, DUMMY_INPUT_LEN, TREE_HEIGHT, generate_dummy_mles_batch, BatchedDummyMles, BatchedCatboostMles, generate_mles_batch_catboost_single_tree}, zkdt_circuit_parts::NonBatchedPermutationCircuit, structs::{InputAttribute, DecisionNode}, binary_recomp_circuit::circuits::{PartialBitsCheckerCircuit, BinaryRecompCircuit, BinaryRecompCircuitBatched}}, prover::GKRCircuit, mle::{dense::DenseMle, MleRef}, layer::LayerId};
    use remainder_shared_types::transcript::{Transcript, poseidon_transcript::PoseidonTranscript};
    use crate::prover::tests::test_circuit;

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

        let (BatchedCatboostMles {
            binary_decomp_diffs_mle_vec,
            decision_node_paths_mle_vec,
            permuted_input_data_mle_vec, ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>();

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

        let (BatchedCatboostMles {
            binary_decomp_diffs_mle_vec,
            decision_node_paths_mle_vec,
            permuted_input_data_mle_vec, ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>();

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

}