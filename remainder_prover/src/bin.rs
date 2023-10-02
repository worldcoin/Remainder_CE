//! Executable for Remainder ZKDT prover!

use std::{path::{Path, PathBuf}, time::Instant, fs};
use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use remainder::{prover::{GKRError, GKRCircuit}, zkdt::{data_pipeline::{dummy_data_generator::convert_zkdt_circuit_data_into_mles, dt2zkdt::load_upshot_data_single_tree_batch}, zkdt_circuit::ZKDTCircuit}};
use clap::Parser;
use remainder_shared_types::FieldExt;
use remainder_shared_types::transcript::Transcript;
use serde_json::{to_writer, from_reader};
use thiserror::Error;

#[derive(Error, Debug, Clone)]
/// Errors for running the binary over inputs and proving/verification
/// TODO!(ryancao): File-handling errors!
pub enum ZKDTBinaryError {
    #[error("GKR Verification failed! Error: {0}")]
    /// Verification failed
    GKRVerificationFailed(GKRError),
    #[error("GKR Proving failed! Error: {0}")]
    /// Proving failed
    GKRProvingFailed(GKRError),
}

/// TODOs for Ryan:
/// - Test the thing with non-powers-of-two
/// - Change the thing so that `None` gives everything instead of just 2

/// Executable for running Remainder's GKR prover over the ZKDT circuit,
/// taking in as input the outputs from the Python processing script and
/// a pre-processed Ligero decision tree commitment.
/// 
/// Should be able to:
/// * Generate a proof over any tree and any set of data
/// * Verify that proof or not
/// * Write the proof or not to a filepath of choice
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the tree commitment file that we care about
    #[arg(short, long)]
    tree_commit_filepath: String,

    /// Filepath to the sample intermediates that we care about
    /// (i.e. the outputs of the Python processing)
    #[arg(short, long)]
    quantized_samples_filepath: String,

    /// Filepath to the actual tree model that we care about
    /// (i.e. the tree JSON outputs of the Python processing)
    /// 
    /// TODO!(ryancao): Make this only need to take in the 
    /// singular tree that we are supposed to be proving, not
    /// the contents of the entire forest.
    #[arg(short, long)]
    decision_forest_model_filepath: String,

    /// log_2 of the number of samples to be read in.
    /// (Note that not passing in anything here will result in *all*
    /// samples being read.)
    #[arg(short, long)]
    log_sample_batch_size: Option<usize>,

    /// Filepath to where the final GKR proof should be written to.
    /// (Note that not passing in anything here will result in no proof
    /// being written at all.)
    #[arg(short, long)]
    gkr_proof_to_be_written_filepath: Option<String>,

    /// Whether we want the proof to be verified or not.
    #[arg(short, long, default_value_t = false)]
    verify_proof: bool,

    // /// Whether to turn on claim aggregation optimization which
    // /// reduces the number of V_i(l(x)) evaluations sent over to
    // /// the verifier, rather than the upper bound of 
    // /// `num_claims * num_challenge_points + 1`
    // #[arg(short, long, default_value_t = true)]
    // claim_agg_reduced_number_vi_l_x_evaluations_optimization: bool,

    // /// Whether to turn on claim aggregation optimization which
    // /// attempts to a) group claims by source input layers, b) aggregate
    // /// those claims first into a resulting claim, and c) aggregate
    // /// all the resulting claims.
    // /// (This includes claim de-duplicating)
    // #[arg(short, long, default_value_t = true)]
    // claim_agg_group_claims_by_input_layer_optimization: bool,

    // /// Whether to turn on flattened layer optimization for V_i(l(x)).
    // #[arg(short, long, default_value_t = true)]
    // compute_vi_l_x_flattened_optimization: bool,

    // /// Whether to turn on common-variable V_i(l(x)) computation optimization.
    // #[arg(short, long, default_value_t = true)]
    // compute_vi_l_x_common_var_optimization: bool,
}

/// Runs the actual circuit on the witness data
pub fn run_zkdt_circuit<F: FieldExt, C: GKRCircuit<F>>(
    mut circuit: C, 
    maybe_filepath_to_proof: Option<PathBuf>, 
    verify_proof: bool
) -> Result<(), ZKDTBinaryError> {
    let mut transcript = C::Transcript::new("GKR Prover Transcript");
    let now = Instant::now();

    match circuit.prove(&mut transcript) {
        Ok(proof) => {
            println!(
                "proof generated successfully in {}!",
                now.elapsed().as_secs_f32()
            );

            // --- Write proof to file, if filepath exists ---
            if let Some(filepath_to_proof) = maybe_filepath_to_proof.clone() {
                println!("Writing the serialized ZKDT GKR proof to {:?}", filepath_to_proof);
                let mut f = fs::File::create(filepath_to_proof).unwrap();
                to_writer(&mut f, &proof).unwrap();
            }
            let mut transcript = C::Transcript::new("GKR Verifier Transcript");
            let now = Instant::now();

            // --- Verify proof if asked for ---
            if verify_proof {
                // --- Grab proof from filepath, if exists; else grab from memory ---
                let proof = if let Some(filepath_to_proof) = maybe_filepath_to_proof {
                    let file = std::fs::File::open(filepath_to_proof).unwrap();
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
                        Ok(())
                    }
                    Err(err) => {
                        println!("Verify failed! Error: {err}");
                        Err(ZKDTBinaryError::GKRVerificationFailed(err))
                    }
                }
            } else {
                Ok(())
            }
        }
        Err(err) => {
            println!("Proof failed! Error: {err}");
            Err(ZKDTBinaryError::GKRProvingFailed(err))
        }
    }
}

/// Idea here is as follows:
/// * Take in as input the files which Ben's Rust code takes in
/// * Output as files the actual proof which gets generated from that code
fn main() -> Result<(), ZKDTBinaryError> {
    let args = Args::parse();

    // --- Read in the Upshot data from file ---
    let (zkdt_circuit_data, (tree_height, input_len)) = load_upshot_data_single_tree_batch::<Fr>(
        args.log_sample_batch_size,
        None,
        Path::new(&args.decision_forest_model_filepath),
        Path::new(&args.quantized_samples_filepath),
    );
    let (batched_catboost_mles, (_, _)) = convert_zkdt_circuit_data_into_mles(zkdt_circuit_data, tree_height, input_len);

    // --- Create the full ZKDT circuit ---
    let full_zkdt_circuit = ZKDTCircuit {
        batched_catboost_mles,
        tree_precommit_filepath: args.tree_commit_filepath
    };

    // --- Grab the proof filepath to write to and compute the circuit + prove ---
    let maybe_proof_filepath = args.gkr_proof_to_be_written_filepath.map(|maybe_path| {
        Path::new(&maybe_path).to_owned()
    });
    run_zkdt_circuit(full_zkdt_circuit, maybe_proof_filepath, args.verify_proof)
}