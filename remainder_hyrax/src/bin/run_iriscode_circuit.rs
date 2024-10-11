use std::path::Path;

use clap::Parser;
use remainder::{
    prover::helpers::{test_circuit, write_circuit_description_to_file},
    worldcoin::{
        circuits::build_circuit,
        data::{load_worldcoin_data_v2, load_worldcoin_data_v3, wirings_to_reroutings},
        parameters::decode_wirings,
        test_helpers::{v2_circuit_description_and_inputs, v3_circuit_description_and_inputs},
    },
};
use remainder_hyrax::hyrax_worldcoin::test_worldcoin::test_iriscode_circuit_with_hyrax_helper;
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

    /// The file to output the circuit description to, if any.
    /// NOTE THAT PASSING SOMETHING HERE WILL CAUSE THE BINARY TO **ONLY**
    /// PRINT THE CIRCUIT DESCRIPTION AND TO **NOT** PROVE!!!
    #[arg(long)]
    circuit_description_filepath: Option<String>,
}

/// Usage: `target/release/run_iriscode_circuit --image-filepath worldcoin/v2/mask/test_image.npy --version 2 --is-mask`
fn main() {
    let args = Args::parse();
    let image_path = Path::new(&args.image_filepath).to_path_buf();
    if args.version == 2 {
        if let Some(circuit_description_path) = args.circuit_description_filepath {
            let (circuit_description, _, _) = v2_circuit_description_and_inputs(false);
            write_circuit_description_to_file(
                &circuit_description,
                Path::new(&circuit_description_path),
            );
        } else {
            let (desc, priv_layer_desc, inputs) = v2_circuit_description_and_inputs(false);
            test_iriscode_circuit_with_hyrax_helper(desc, priv_layer_desc, inputs);
        }
    } else if args.version == 3 {
        if let Some(circuit_description_path) = args.circuit_description_filepath {
            let (circuit_description, _, _) = v3_circuit_description_and_inputs(false);
            write_circuit_description_to_file(
                &circuit_description,
                Path::new(&circuit_description_path),
            );
        } else {
            let (desc, priv_layer_desc, inputs) = v3_circuit_description_and_inputs(false);
            test_iriscode_circuit_with_hyrax_helper(desc, priv_layer_desc, inputs);
        }
    } else {
        panic!();
    }
}
