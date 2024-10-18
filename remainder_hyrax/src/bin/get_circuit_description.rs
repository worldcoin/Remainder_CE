use std::path::Path;

use clap::Parser;
use remainder::{
    prover::helpers::write_circuit_description_to_file,
    worldcoin::{
        circuits::build_iriscode_circuit_description, data::wirings_to_reroutings,
        parameters::decode_wirings,
    },
};
use remainder_shared_types::Fr;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The version of the iriscode circuit to run
    #[arg(long, value_parser = clap::value_parser!(u8))]
    version: u8,

    /// The file to output the circuit description to.
    #[arg(long)]
    circuit_description_filepath: String,
}

/// Usage: `target/release/get_circuit_description --circuit-description-filepath output.json --version 2`
fn main() {
    let args = Args::parse();
    if args.version == 2 {
        use remainder::worldcoin::parameters_v2::{
            BASE, IM_NUM_COLS, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
            MATMULT_ROWS_NUM_VARS, NUM_DIGITS, TO_REROUTE_NUM_VARS, WIRINGS,
        };
        let wirings = &decode_wirings(WIRINGS);
        let reroutings =
            wirings_to_reroutings(wirings, IM_NUM_COLS, 1 << MATMULT_INTERNAL_DIM_NUM_VARS);
        let (circuit_description, _, _) = build_iriscode_circuit_description::<
            Fr,
            TO_REROUTE_NUM_VARS,
            MATMULT_ROWS_NUM_VARS,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            BASE,
            NUM_DIGITS,
        >(reroutings);
        write_circuit_description_to_file(
            &circuit_description,
            Path::new(&args.circuit_description_filepath),
        );
    } else if args.version == 3 {
        use remainder::worldcoin::parameters_v3::{
            BASE, IM_NUM_COLS, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
            MATMULT_ROWS_NUM_VARS, NUM_DIGITS, TO_REROUTE_NUM_VARS, WIRINGS,
        };
        let wirings = &decode_wirings(WIRINGS);
        let reroutings =
            wirings_to_reroutings(wirings, IM_NUM_COLS, 1 << MATMULT_INTERNAL_DIM_NUM_VARS);
        let (circuit_description, _, _) = build_iriscode_circuit_description::<
            Fr,
            TO_REROUTE_NUM_VARS,
            MATMULT_ROWS_NUM_VARS,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            BASE,
            NUM_DIGITS,
        >(reroutings);
        write_circuit_description_to_file(
            &circuit_description,
            Path::new(&args.circuit_description_filepath),
        );
    } else {
        panic!();
    }
}
