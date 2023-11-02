use ark_std::{end_timer, start_timer};
use crate::mle::{dense::DenseMle, Mle, MleRef};
use crate::prover::GKRCircuit;
use remainder_shared_types::transcript::Transcript;
use remainder_shared_types::{FieldExt, Fr};
use serde_json::{from_reader, to_writer};
use std::fs::File;
use std::path::Path;

/// Note: we removed this from the tests module, since that module is not visible to project binaries.
pub fn test_circuit<F: FieldExt, C: GKRCircuit<F>>(mut circuit: C, path: Option<&Path>) 
    where <C as GKRCircuit<F>>::Transcript: Sync, 
{
    let mut transcript = C::Transcript::new("GKR Prover Transcript");
    let prover_timer = start_timer!(|| "proof generation");

    match circuit.prove(&mut transcript) {
        Ok(proof) => {
            end_timer!(prover_timer);
            if let Some(path) = path {
                let mut f = File::create(path).unwrap();
                to_writer(&mut f, &proof).unwrap();
            }
            let mut transcript = C::Transcript::new("GKR Verifier Transcript");
            let verifier_timer = start_timer!(|| "proof verification");

            let proof = if let Some(path) = path {
                let file = std::fs::File::open(path).unwrap();

                from_reader(&file).unwrap()
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
