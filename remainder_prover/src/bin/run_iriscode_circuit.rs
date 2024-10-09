use std::path::Path;

use clap::Parser;
use remainder::{
    prover::helpers::test_circuit,
    worldcoin::{
        circuits::build_circuit,
        data::{load_worldcoin_data_v2, load_worldcoin_data_v3, wirings_to_reroutings},
        parameters::decode_wirings,
    },
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
            BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
            MATMULT_ROWS_NUM_VARS, NUM_DIGITS, WIRINGS, IM_NUM_ROWS, IM_NUM_COLS,
        };
        let data = load_worldcoin_data_v2::<
            Fr,
            MATMULT_ROWS_NUM_VARS,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            BASE,
            NUM_DIGITS,
            IM_NUM_ROWS,
            IM_NUM_COLS,
        >(image_path, args.is_mask);
        let reroutings = wirings_to_reroutings(
            &decode_wirings(WIRINGS),
            IM_NUM_COLS,
            1 << MATMULT_INTERNAL_DIM_NUM_VARS,
        );
        let circuit = build_circuit::<
            Fr,
            MATMULT_ROWS_NUM_VARS,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            BASE,
            NUM_DIGITS,
        >(data, reroutings);
        test_circuit(circuit, None);
    } else if args.version == 3 {
        use remainder::worldcoin::parameters_v3::{
            BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
            MATMULT_ROWS_NUM_VARS, NUM_DIGITS, WIRINGS, IM_NUM_ROWS, IM_NUM_COLS,
        };
        let data = load_worldcoin_data_v3::<
            Fr,
            MATMULT_ROWS_NUM_VARS,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            BASE,
            NUM_DIGITS,
            IM_NUM_ROWS,
            IM_NUM_COLS,
        >(image_path, args.is_mask);
        let reroutings = wirings_to_reroutings(
            &decode_wirings(WIRINGS),
            IM_NUM_COLS,
            1 << MATMULT_INTERNAL_DIM_NUM_VARS,
        );
        let circuit = build_circuit::<
            Fr,
            MATMULT_ROWS_NUM_VARS,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            BASE,
            NUM_DIGITS,
        >(data, reroutings);
        test_circuit(circuit, None);
    } else {
        panic!();
    }
}
