use std::time::Instant;

use ark_bn254::Fr;
use ark_std::{test_rng, UniformRand};
use remainder::{zkdt::{zkdt_circuit::{BatchedDummyMles, generate_dummy_mles_batch}, zkdt_circuit_parts::PermutationCircuit}, transcript::{poseidon_transcript::PoseidonTranscript, Transcript}, prover::GKRCircuit};

fn main() {
    let mut rng = test_rng();

    let BatchedDummyMles {
        dummy_input_data_mle,
        dummy_permuted_input_data_mle, ..
    } = generate_dummy_mles_batch();

    let mut circuit = PermutationCircuit {
        dummy_input_data_mle_vec: dummy_input_data_mle,
        dummy_permuted_input_data_mle_vec: dummy_permuted_input_data_mle,
        r: Fr::rand(&mut rng),
        r_packing: Fr::rand(&mut rng),
        input_len: 700,
        num_inputs: 1,
    };

    let mut transcript = PoseidonTranscript::new("Permutation Circuit Prover Transcript");

    let now = Instant::now();

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