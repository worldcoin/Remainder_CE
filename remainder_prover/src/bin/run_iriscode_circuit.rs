use std::path::Path;

use clap::Parser;
use remainder::{
    prover::helpers::test_circuit,
    worldcoin::{circuits::build_circuit, data::load_worldcoin_data},
};
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

    /// The version of the iriscode circuit to run
    #[arg(long, value_parser = clap::value_parser!(u8))]
    version: u8,
}

/// Usage: `target/release/run_iriscode_circuit --image-filepath worldcoin/v2/mask/test_image.npy --version 2 --is-mask`
fn main() {
    let args = Args::parse();
    let image_path = Path::new(&args.image_filepath).to_path_buf();
    if args.version == 2 {
        use remainder::worldcoin::parameters_v2::{
            BASE, MATMULT_INTERNAL_DIM_VARS, MATMULT_NUM_COLS_VARS, MATMULT_NUM_ROWS_VARS, NUM_DIGITS,
        };
        let path = Path::new("worldcoin/v2/").to_path_buf();
        let data = load_worldcoin_data::<
            Fr,
            MATMULT_NUM_ROWS_VARS,
            MATMULT_NUM_COLS_VARS,
            MATMULT_INTERNAL_DIM_VARS,
            BASE,
            NUM_DIGITS,
        >(path, image_path, args.is_mask);
        let circuit = build_circuit(data);
        test_circuit(circuit, None);
    } else if args.version == 3 {
        use remainder::worldcoin::parameters_v3::{
            BASE, MATMULT_INTERNAL_DIM, MATMULT_NUM_COLS, MATMULT_NUM_ROWS, NUM_DIGITS,
        };
        let path = Path::new("worldcoin/v3/").to_path_buf();
        let data = load_worldcoin_data::<
            Fr,
            MATMULT_NUM_ROWS,
            MATMULT_NUM_COLS,
            MATMULT_INTERNAL_DIM,
            BASE,
            NUM_DIGITS,
        >(path, image_path, args.is_mask);
        let circuit = build_circuit(data);
        test_circuit(circuit, None);
    } else {
        panic!();
    }
}
