use crate::layouter::compiling::LayouterCircuit;
use crate::layouter::component::Component;
use crate::layouter::nodes::circuit_inputs::InputLayerData;
use crate::layouter::nodes::node_enum::NodeEnum;
use crate::layouter::nodes::Context;
use ark_std::{end_timer, start_timer};

use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use remainder_shared_types::transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter};
use remainder_shared_types::Field;
use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Boilerplate code for testing a circuit
pub fn test_circuit<
    F: Field,
    C: Component<NodeEnum<F>>,
    Fn: FnMut(&Context) -> (C, Vec<InputLayerData<F>>),
>(
    mut circuit: LayouterCircuit<F, C, Fn>,
    path: Option<&Path>,
) {
    let transcript_writer = TranscriptWriter::<F, PoseidonSponge<F>>::new("GKR Prover Transcript");
    let prover_timer = start_timer!(|| "Proof generation");

    match circuit.prove(transcript_writer) {
        Ok((transcript, mut gkr_circuit_description)) => {
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

            let mut transcript_reader = TranscriptReader::<F, PoseidonSponge<F>>::new(transcript);
            let verifier_timer = start_timer!(|| "Proof verification");

            match gkr_circuit_description.verify(&mut transcript_reader) {
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
