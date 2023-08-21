use std::time::Instant;

use ark_bn254::Fr;
use ark_std::{test_rng, UniformRand};
use remainder::{zkdt::{zkdt_helpers::{BatchedDummyMles, generate_dummy_mles_batch}, zkdt_circuit_parts::PermutationCircuit}, prover::GKRCircuit};

use lcpc_2d::fs_transcript::halo2_remainder_transcript::Transcript;
use lcpc_2d::fs_transcript::halo2_poseidon_transcript::PoseidonTranscript;


fn main() {
    let mut rng = test_rng();
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr as H2Fr;

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

    let proof = circuit.prove::<H2Fr>(&mut transcript);

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