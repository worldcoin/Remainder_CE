use ark_std::{test_rng};
use rand::Rng;
use remainder::{
    prover::GKRCircuit,
    zkdt::{
        permutation_circuit::circuits::PermutationCircuit,
        data_pipeline::dummy_data_generator::{generate_dummy_mles_batch, BatchedDummyMles},
    },
};
use remainder_shared_types::transcript::{poseidon_transcript::PoseidonTranscript, Transcript};
use std::time::Instant;

fn main() {
    let mut rng = test_rng();
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;

    let BatchedDummyMles {
        dummy_input_data_mle,
        dummy_permuted_input_data_mle,
        ..
    } = generate_dummy_mles_batch();

    let mut circuit = PermutationCircuit::new(
        dummy_input_data_mle,
        dummy_permuted_input_data_mle,
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
    );

    let mut transcript = PoseidonTranscript::new("Permutation Circuit Prover Transcript");

    let now = Instant::now();

    let proof = circuit.prove(&mut transcript);

    println!(
        "Proof generated!: Took {} seconds",
        now.elapsed().as_secs_f32()
    );

    match proof {
        Ok(proof) => {
            let mut transcript = PoseidonTranscript::new("Permutation Circuit Verifier Transcript");
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
