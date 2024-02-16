use crate::prover::GKRCircuit;
use ark_std::{end_timer, start_timer};
use remainder_shared_types::transcript::Transcript;
use remainder_shared_types::FieldExt;
use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

// Note: we removed this from the tests module, since that module is not visible to project
// binaries, where this function is often needed.
pub fn test_circuit<F: FieldExt, C: GKRCircuit<F>>(mut circuit: C, path: Option<&Path>)
where
    <C as GKRCircuit<F>>::Sponge: Sync,
{
    let mut transcript = C::Sponge::new("GKR Prover Transcript");
    let prover_timer = start_timer!(|| "Proof generation");

    match circuit.prove(&mut transcript) {
        Ok(proof) => {
            end_timer!(prover_timer);
            if let Some(path) = path {
                let write_out_timer = start_timer!(|| "Writing out proof");
                let f = File::create(path).unwrap();
                let writer = BufWriter::new(f);
                serde_json::to_writer(writer, &proof).unwrap();
                end_timer!(write_out_timer);
            }
            let mut transcript = C::Sponge::new("GKR Verifier Transcript");
            let verifier_timer = start_timer!(|| "Proof verification");

            let proof = if let Some(path) = path {
                let read_in_timer = start_timer!(|| "Reading in proof");
                let file = std::fs::File::open(path).unwrap();
                let reader = BufReader::new(file);
                let result = serde_json::from_reader(reader).unwrap();
                end_timer!(read_in_timer);
                result
            } else {
                proof
            };

            // Makis: Ignore verify for now.
            match circuit.verify(&mut transcript, proof) {
                Ok(_) => {
                    end_timer!(verifier_timer);
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
