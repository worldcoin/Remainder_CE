use crate::prover::GKRCircuit;
use ark_std::{end_timer, start_timer};

use remainder_shared_types::transcript::{
    TranscriptReader, TranscriptSponge, TranscriptWriter,
};
use remainder_shared_types::FieldExt;
use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use super::proof_system::ProofSystem;
use super::CircuitProverTranscript;

/// Boilerplate code for testing a circuit
pub fn test_circuit<F: FieldExt, C: GKRCircuit<F, ProofSystem = PR>, PR, Sp>(
    mut circuit: C,
    path: Option<&Path>,
) where
    PR: ProofSystem<
        F,
        ProverTranscript = TranscriptWriter<F, Sp>,
        VerifierTranscript = TranscriptReader<F, Sp>,
    >,
    Sp: TranscriptSponge<F>,
{
    let mut transcript_writer = CircuitProverTranscript::<F, C>::new("GKR Prover Transcript");
    let prover_timer = start_timer!(|| "Proof generation");

    match circuit.prove(&mut transcript_writer) {
        Ok(proof) => {
            end_timer!(prover_timer);
            if let Some(path) = path {
                let write_out_timer = start_timer!(|| "Writing out proof");
                let f = File::create(path).unwrap();
                let writer = BufWriter::new(f);
                serde_json::to_writer(writer, &proof).unwrap();
                end_timer!(write_out_timer);
            }
            let transcript = transcript_writer.get_transcript();
            let mut transcript_reader = TranscriptReader::<_, Sp>::new(transcript);
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

            match circuit.verify(&mut transcript_reader, proof) {
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
