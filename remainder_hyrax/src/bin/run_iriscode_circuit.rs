use clap::Parser;
use remainder::worldcoin::{
    io::read_bytes_from_file, test_helpers::circuit_description_and_inputs,
};
use remainder_hyrax::hyrax_worldcoin::test_worldcoin::test_iriscode_circuit_with_hyrax_helper;

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

/// Usage: `run_iriscode_circuit --image-filepath remainder_prover/src/worldcoin/constants/v2/mask/test_image.bin --version 2 --is-mask`
fn main() {
    let args = Args::parse();
    let image_bytes = read_bytes_from_file(&args.image_filepath);
    let (desc, inputs) = circuit_description_and_inputs(args.version, false, Some(image_bytes));
    test_iriscode_circuit_with_hyrax_helper(desc, inputs);
}
