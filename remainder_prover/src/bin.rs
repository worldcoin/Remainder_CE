//! Executable for Remainder ZKDT prover!

use std::{path::Path, time::Instant, fs};

use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use remainder::{prover::{GKRError, GKRCircuit}, zkdt::{data_pipeline::dummy_data_generator::generate_mles_batch_catboost_single_tree, zkdt_circuit::ZKDTCircuit}};
use clap::Parser;
use remainder_shared_types::FieldExt;
use remainder_shared_types::transcript::Transcript;
use serde_json::{to_writer, from_reader};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the tree commitment file that we care about
    #[arg(short, long)]
    tree_commit_filepath: String,

    /// Filepath to the sample intermediates that we care about
    /// (i.e. the outputs of the Python processing)
    #[arg(short, long)]
    python_script_output_filepath: String,

    /// Number of times to greet
    #[arg(short, long, default_value_t = 1)]
    count: u8,
}

/// Runs the actual circuit on the witness data
pub fn test_circuit<F: FieldExt, C: GKRCircuit<F>>(mut circuit: C, path: Option<&Path>) {
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

/// Idea here is as follows:
/// * Take in as input the files which Ben's Rust code takes in
/// * Output as files the actual proof which gets generated from that code
fn main() -> Result<(), GKRError> {
    let args = Args::parse();

    // --- Read in the Upshot 
    let (batched_catboost_mles, (_, _)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

    let combined_circuit = ZKDTCircuit {
        batched_catboost_mles
    };

    test_circuit(combined_circuit, None);

    Ok(())
}