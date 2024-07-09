use crate::prover::GKRCircuit;
use ark_std::{end_timer, start_timer};

use remainder_shared_types::transcript::{TranscriptReader, TranscriptWriter};
use remainder_shared_types::FieldExt;
use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use super::CircuitTranscript;

/// Boilerplate code for testing a circuit
pub fn test_circuit<F: FieldExt, C: GKRCircuit<F>>(mut circuit: C, path: Option<&Path>)
where
    CircuitTranscript<F, C>: Sync,
{
    let mut transcript_writer =
        TranscriptWriter::<_, CircuitTranscript<F, C>>::new("GKR Prover Transcript");
    let prover_timer = start_timer!(|| "Proof generation");

    match circuit.prove(transcript_writer) {
        Ok((transcript, gkr_circuit)) => {
            end_timer!(prover_timer);
            if let Some(path) = path {
                let write_out_timer = start_timer!(|| "Writing out proof");
                let f = File::create(path).unwrap();
                let writer = BufWriter::new(f);
                serde_json::to_writer(writer, &transcript).unwrap();
                end_timer!(write_out_timer);
            }

            let transcript = if let Some(path) = path {
                let read_in_timer = start_timer!(|| "Reading in proof");
                let file = std::fs::File::open(path).unwrap();
                let reader = BufReader::new(file);
                let result = serde_json::from_reader(reader).unwrap();
                end_timer!(read_in_timer);
                result
            } else {
                transcript
            };

            let mut transcript_reader =
                TranscriptReader::<_, CircuitTranscript<F, C>>::new(transcript);
            let verifier_timer = start_timer!(|| "Proof verification");

            match circuit.verify(&mut transcript_reader, gkr_circuit) {
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
