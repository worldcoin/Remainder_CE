use std::path::Path;

use clap::Parser;
use remainder::{prover::helpers::test_circuit, worldcoin::{circuits::build_circuit, data::{load_worldcoin_data, CircuitData}, parameters_v2::{CONSTANT_DATA_FOLDER, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS}}};
use remainder_shared_types::Fr;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// This is the filepath which contains the normalized iris image (as bytes).
    #[arg(long)]
    image_filepath: String,

    // Whether we are processing the iris image or the mask image.
    #[arg(long)]
    is_mask: bool,
}

/// Usage: `target/release/run_iriscode_circuit --image-filepath worldcoin/v2/mask/test_image.npy --is-mask`
fn main() {
    let args = Args::parse();
    let path = Path::new("worldcoin/v2/").to_path_buf();
    let image_path = Path::new(&args.image_filepath).to_path_buf();
    let data = load_worldcoin_data::<Fr, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS>(path.clone(), image_path, args.is_mask);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}