use std::{fs, path::Path, time::Instant};

use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use remainder::{
    prover::GKRCircuit,
    zkdt::{
        cache_upshot_catboost_inputs_for_testing::generate_mles_batch_catboost_single_tree,
        zkdt_circuit::ZKDTCircuit,
    },
};
use remainder_shared_types::{transcript::Transcript, FieldExt};
use serde_json::{from_reader, to_writer};

pub fn test_circuit<F: FieldExt, C: GKRCircuit<F>>(mut circuit: C, path: Option<&Path>)
where
    <C as GKRCircuit<F>>::Transcript: Sync,
{
    let mut transcript = C::Transcript::new("GKR Prover Transcript");
    let now = Instant::now();

    match circuit.prove(&mut transcript) {
        Ok(proof) => {
            println!(
                "proof generated successfully in {}!",
                now.elapsed().as_secs_f32()
            );
            if let Some(path) = path {
                let mut f = fs::File::create(path).unwrap();
                to_writer(&mut f, &proof).unwrap();
            }
            let mut transcript = C::Transcript::new("GKR Verifier Transcript");
            let now = Instant::now();

            let proof = if let Some(path) = path {
                let file = std::fs::File::open(path).unwrap();

                from_reader(&file).unwrap()
            } else {
                proof
            };
            match circuit.verify(&mut transcript, proof) {
                Ok(_) => {
                    println!(
                        "Verification succeeded: takes {}!",
                        now.elapsed().as_secs_f32()
                    );
                }
                Err(err) => {
                    println!("Verify failed! Error: {err}");
                    panic!();
                }
            }
        }
        Err(err) => {
            println!("Proof failed! Error: {err}");
            panic!();
        }
    }
}

fn main() {
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    let batch_size = 10;

    let (batched_catboost_mles, (_, _)) =
        generate_mles_batch_catboost_single_tree::<Fr>(batch_size, Path::new("upshot_data"));

    let combined_circuit = ZKDTCircuit {
        batched_zkdt_circuit_mles: batched_catboost_mles,
        tree_precommit_filepath: "upshot_data/tree_ligero_commitments/tree_commitment_0.json"
            .to_string(),
        sample_minibatch_precommit_filepath:
            "upshot_data/sample_minibatch_commitments/sample_minibatch_logsize_10_commitment_0.json"
                .to_string(),
    };

    test_circuit(combined_circuit, Some(Path::new("./zkdt_proof.json")));
}
