//! Executable which generates the Ligero samples commitment

use std::{path::Path, fs, io::BufWriter};
use ark_std::{log2, start_timer, end_timer};
use remainder_shared_types::Fr;
use itertools::{Itertools, repeat_n};
use remainder::{zkdt::{data_pipeline::dt2zkdt::{RawSamples, load_raw_samples, Samples}, structs::InputAttribute, constants::{get_sample_minibatch_commitment_filepath_for_batch_size, get_sample_minibatch_commitment_filepath_for_batch_size_tree_batch}}, mle::{dense::DenseMle, Mle}, layer::LayerId, prover::input_layer::{combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer}};
use clap::Parser;
use remainder_ligero::ligero_commit::remainder_ligero_commit_prove;
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};
use thiserror::Error;
use tracing::debug;
use tracing_subscriber::{FmtSubscriber, fmt::format::FmtSpan};

#[derive(Error, Debug, Clone)]
/// Errors for running the samples commitment binary.
/// TODO!(ryancao): File-handling errors!
pub enum GenerateSampleMinibatchCommitmentsError {

}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File path to the (Python) processed `samples.json` file.
    #[arg(long)]
    raw_samples_path: String,

    /// Path to the directory in which to write all the sample minibatch commitments.
    #[arg(long)]
    sample_minibatch_commitments_dir: String,

    /// Base 2 logarithm of the sample batch size. This will
    /// perform truncation of the sample batch size if specified, 
    /// or will grab all the samples and pad to the nearest power 
    /// of two (TODO!(benjaminwilson)) if `None`.
    #[arg(long)]
    log_sample_minibatch_commitment_size: Option<usize>,

    /// Whether we want debug tracing subscriber logs or not.
    /// By default, we use `DEBUG` as the subscriber level.
    /// 
    /// TODO!(ryancao): Figure out `structopt` so we can pass in
    /// different trace levels
    #[arg(long, default_value_t = false)]
    debug_tracing_subscriber: bool,

    #[arg(long, default_value_t = 1)]
    tree_batch_size: usize,
}

/// Generates Ligero commitments for the raw samples (i.e. x_1, ..., x_n).
/// Note that these commitments are re-used across all trees (although the
/// sample auxiliaries are dependent on the actual tree which is running 
/// over the samples).
/// 
/// ## Arguments
/// * `raw_samples_path` - File path to the (Python) processed `samples.json` file.
/// * `sample_minibatch_commitments_dir` - Path to the directory in which to write all the sample minibatch commitments.
/// * `log_sample_minibatch_commitment_size` - Base 2 logarithm of the sample batch size. This will
///     perform chunking for the sample batch size if specified, or will grab all the
///     samples and pad to the nearest power of two (TODO!(benjaminwilson)) if `None`.
pub fn generate_ligero_sample_minibatch_commitments<F: FieldExt>(
    raw_samples_path: &Path,
    sample_minibatch_commitments_dir: &Path,
    maybe_log_sample_minibatch_commitment_size: Option<usize>,
) {

    let raw_samples: RawSamples = load_raw_samples(raw_samples_path);

    // --- TODO!(ryancao): Fix this as soon as Ben gets the padding code to us ---
    let sample_minibatch_commitment_size = match maybe_log_sample_minibatch_commitment_size {
        Some(log_sample_minibatch_commitment_size) => 2_usize.pow(log_sample_minibatch_commitment_size as u32),
        None => raw_samples.values.len(),
    };
    let log_sample_minibatch_commitment_size = maybe_log_sample_minibatch_commitment_size.unwrap_or(log2(sample_minibatch_commitment_size) as usize);

    // --- For each minibatch of samples, convert into `Vec<Vec<InputAttribute<F>>` and compute Ligero commitment ---
    raw_samples.values.chunks(sample_minibatch_commitment_size).enumerate().for_each(
        |(minibatch_idx, sample_minibatch)| {

            // --- Grab the file save path ---
            let sample_minibatch_commitment_filepath = get_sample_minibatch_commitment_filepath_for_batch_size(
                log_sample_minibatch_commitment_size,
                minibatch_idx,
                sample_minibatch_commitments_dir
            );

            // --- Check if the cached file already exists ---
            match fs::metadata(sample_minibatch_commitment_filepath.clone()) {

                // --- File already exists; no need to commit ---
                Ok(_) => {
                    debug!(sample_minibatch_commitment_filepath, "File already exists! Skipping.");
                },

                // --- No commitment exists yet; create commitment ---
                Err(_) => {
                    debug!(sample_minibatch_commitment_filepath, "File doesn't exist yet! Generating commitment...");

                    // --- Create dummy `RawSamples` and convert to `Samples` ---
                    let minibatch_raw_samples = RawSamples {
                        values: sample_minibatch.to_vec(),
                        sample_length: sample_minibatch[0].len(), // TODO!(ryancao): Is this actually correct?
                    };
                    let minibatch_samples: Samples = (&minibatch_raw_samples).into();

                    // --- Convert into `Vec<Vec<InputAttribute<F>>` ---
                    let minibatch_converted_samples: Vec<Vec<InputAttribute<F>>> = minibatch_samples.values.iter().map(
                        |minibatch_samples_sample| {
                            minibatch_samples_sample.iter().enumerate().map(
                                |(idx, minibatch_samples_sample_val)| {
                                    InputAttribute {
                                        attr_id: F::from(idx as u64),
                                        attr_val: F::from(*minibatch_samples_sample_val as u64),
                                    }
                                }
                            ).collect_vec()
                        }
                    ).collect_vec();

                    // ------ Compute Ligero commitment ------

                    // --- First, compute the combined MLE to commit to ---
                    let minibatch_converted_samples_mle_vec = minibatch_converted_samples
                        .into_iter()
                        .map(|single_minibatch_sample| 
                            DenseMle::new_from_iter(single_minibatch_sample
                            .into_iter()
                            .map(InputAttribute::from), LayerId::Input(0), None)
                        ).collect_vec();
                    let mut minibatch_converted_samples_mle_combined: DenseMle<F, F> = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(minibatch_converted_samples_mle_vec);
                    let minibatch_converted_samples_mle_combined_dummy_input_layer_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
                        Box::new(&mut minibatch_converted_samples_mle_combined),
                    ];
                    let minibatch_converted_samples_mle_combined_dummy_input_layer_mles_input_layer_builder = InputLayerBuilder::new(minibatch_converted_samples_mle_combined_dummy_input_layer_mles, None, LayerId::Input(0));
                    let minibatch_converted_samples_mle_combined_dummy_input_layer_mles_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> = minibatch_converted_samples_mle_combined_dummy_input_layer_mles_input_layer_builder.to_input_layer();

                    // --- Create commitment to the combined MLEs via the input layer ---
                    let rho_inv = 4;
                    let ratio = 1_f64;
                    let ligero_commitment = remainder_ligero_commit_prove(
                        &minibatch_converted_samples_mle_combined_dummy_input_layer_mles_input_layer.mle.mle_ref().bookkeeping_table, rho_inv, ratio);

                    // --- Write to file ---
                    let file = fs::File::create(sample_minibatch_commitment_filepath).unwrap();
                    let bw = BufWriter::new(file);
                    serde_json::to_writer(bw, &ligero_commitment).unwrap();
                }
            }
        }
    );

    // --- Do the actual commitment to `converted_samples` ---
}



pub fn generate_ligero_sample_minibatch_commitments_batched<F: FieldExt>(
    raw_samples_path: &Path,
    sample_minibatch_commitments_dir: &Path,
    maybe_log_sample_minibatch_commitment_size: Option<usize>,
    tree_batch_size: usize,
) {

    let raw_samples: RawSamples = load_raw_samples(raw_samples_path);

    // --- TODO!(ryancao): Fix this as soon as Ben gets the padding code to us ---
    let sample_minibatch_commitment_size = match maybe_log_sample_minibatch_commitment_size {
        Some(log_sample_minibatch_commitment_size) => 2_usize.pow(log_sample_minibatch_commitment_size as u32),
        None => raw_samples.values.len(),
    };
    let log_sample_minibatch_commitment_size = maybe_log_sample_minibatch_commitment_size.unwrap_or(log2(sample_minibatch_commitment_size) as usize);

    // --- For each minibatch of samples, convert into `Vec<Vec<InputAttribute<F>>` and compute Ligero commitment ---
    raw_samples.values.chunks(sample_minibatch_commitment_size).enumerate().for_each(
        |(minibatch_idx, sample_minibatch)| {

            // --- Grab the file save path ---
            let sample_minibatch_commitment_filepath = get_sample_minibatch_commitment_filepath_for_batch_size_tree_batch(
                log_sample_minibatch_commitment_size,
                minibatch_idx,
                sample_minibatch_commitments_dir,
                tree_batch_size
            );

            // --- Check if the cached file already exists ---
            match fs::metadata(sample_minibatch_commitment_filepath.clone()) {

                // --- File already exists; no need to commit ---
                Ok(_) => {
                    debug!(sample_minibatch_commitment_filepath, "File already exists! Skipping.");
                },

                // --- No commitment exists yet; create commitment ---
                Err(_) => {
                    debug!(sample_minibatch_commitment_filepath, "File doesn't exist yet! Generating commitment...");

                    // --- Create dummy `RawSamples` and convert to `Samples` ---
                    let minibatch_raw_samples = RawSamples {
                        values: sample_minibatch.to_vec(),
                        sample_length: sample_minibatch[0].len(), // TODO!(ryancao): Is this actually correct?
                    };
                    let minibatch_samples: Samples = (&minibatch_raw_samples).into();

                    // --- Convert into `Vec<Vec<InputAttribute<F>>` ---
                    let minibatch_converted_samples: Vec<Vec<InputAttribute<F>>> = minibatch_samples.values.iter().map(
                        |minibatch_samples_sample| {
                            minibatch_samples_sample.iter().enumerate().map(
                                |(idx, minibatch_samples_sample_val)| {
                                    InputAttribute {
                                        attr_id: F::from(idx as u64),
                                        attr_val: F::from(*minibatch_samples_sample_val as u64),
                                    }
                                }
                            ).collect_vec()
                        }
                    ).collect_vec();

                    // ------ Compute Ligero commitment ------

                    // --- First, compute the combined MLE to commit to ---
                    let minibatch_converted_samples_mle_vec = minibatch_converted_samples
                        .into_iter()
                        .map(|single_minibatch_sample| 
                            DenseMle::new_from_iter(single_minibatch_sample
                            .into_iter()
                            .map(InputAttribute::from), LayerId::Input(0), None)
                        ).collect_vec();
                    let minibatch_converted_samples_mle_vec_vec = repeat_n(minibatch_converted_samples_mle_vec, tree_batch_size).collect_vec();
                    let minibatch_converted_samples_mle_vec = minibatch_converted_samples_mle_vec_vec.into_iter().map(
                        |minibatch_converted_samples_mle_vec| {
                            DenseMle::<F, InputAttribute<F>>::combine_mle_batch(minibatch_converted_samples_mle_vec)
                        }
                    ).collect_vec();
                    let mut minibatch_converted_samples_mle_combined: DenseMle<F, F> = DenseMle::<F, F>::combine_mle_batch(minibatch_converted_samples_mle_vec);
                    let minibatch_converted_samples_mle_combined_dummy_input_layer_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
                        Box::new(&mut minibatch_converted_samples_mle_combined),
                    ];
                    let minibatch_converted_samples_mle_combined_dummy_input_layer_mles_input_layer_builder = InputLayerBuilder::new(minibatch_converted_samples_mle_combined_dummy_input_layer_mles, None, LayerId::Input(0));
                    let minibatch_converted_samples_mle_combined_dummy_input_layer_mles_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> = minibatch_converted_samples_mle_combined_dummy_input_layer_mles_input_layer_builder.to_input_layer();

                    // --- Create commitment to the combined MLEs via the input layer ---
                    let rho_inv = 4;
                    let ratio = 1_f64;
                    let ligero_commitment = remainder_ligero_commit_prove(
                        &minibatch_converted_samples_mle_combined_dummy_input_layer_mles_input_layer.mle.mle_ref().bookkeeping_table, rho_inv, ratio);

                    // --- Write to file ---
                    let file = fs::File::create(sample_minibatch_commitment_filepath).unwrap();
                    let bw = BufWriter::new(file);
                    serde_json::to_writer(bw, &ligero_commitment).unwrap();
                }
            }
        }
    );

    // --- Do the actual commitment to `converted_samples` ---
}

/// This binary performs the following:
/// * Take in as input the quantized samples which the Python code generates.
/// * Output as files Ligero commitments to each minibatch within the batch
///     of samples. 
fn main() -> Result<(), GenerateSampleMinibatchCommitmentsError> {
    let args = Args::parse();

    // --- Tracing subscriber (i.e. outputs trace messages in stdout) if asked for ---
    if args.debug_tracing_subscriber {
        let subscriber = FmtSubscriber::builder()
            .with_line_number(true)
            .with_max_level(tracing::Level::DEBUG)
            .with_level(true)
            .with_span_events(FmtSpan::ACTIVE)
            .finish();
        let _default_guard = tracing::subscriber::set_global_default(subscriber);
    }

    // --- Log the args ---
    let args_as_string = format!("{:?}", args);
    debug!(args_as_string);

    // --- Compute the commitment ---
    generate_ligero_sample_minibatch_commitments_batched::<Fr>(
        Path::new(&args.raw_samples_path),
        Path::new(&args.sample_minibatch_commitments_dir),
        args.log_sample_minibatch_commitment_size, 
        args.tree_batch_size
    );

    Ok(())
}
