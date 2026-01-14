use std::path::{Path, PathBuf};

use clap::Parser;
use frontend::{
    hyrax_worldcoin_mpc::mpc_prover::print_features_status,
    zk_iriscode_ss::{self, io::read_bytes_from_file, parameters::IRISCODE_LEN},
};
use shared_types::perform_function_under_expected_configs;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct CliArguments {
    /// Path to the file where secret share bytes should be written to.
    #[arg(long)]
    secret_share_bytes_path: PathBuf,
}

fn main() {
    // Sanitycheck by logging the current settings.
    perform_function_under_expected_configs!(
        print_features_status,
        &zk_iriscode_ss::EXPECTED_PROVER_CONFIG,
        &zk_iriscode_ss::EXPECTED_VERIFIER_CONFIG,
    );

    // Parse arguments and verify secret share generation proofs.
    let cli = CliArguments::parse();
    perform_function_under_expected_configs!(
        sanitycheck_secret_shares,
        &zk_iriscode_ss::EXPECTED_PROVER_CONFIG,
        &zk_iriscode_ss::EXPECTED_VERIFIER_CONFIG,
        &cli.secret_share_bytes_path
    );
}

/// Ascertains that the length of the secret shares written to
/// `secret_shares_bytes_path` is as expected.
///
/// Note that the format of the file located at `secret_shares_bytes_path`
/// is expected to be a serialized `Vec<u16>` of length `IRISCODE_LEN`.
fn sanitycheck_secret_shares(secret_shares_bytes_path: &Path) {
    // Read from file...
    let serialized_secret_share_bytes =
        read_bytes_from_file(secret_shares_bytes_path.as_os_str().to_str().unwrap());
    // ...And deserialize!
    let secret_shares: Vec<u16> = bincode::deserialize(&serialized_secret_share_bytes).unwrap();

    // Check that lengths are correct.
    assert_eq!(secret_shares.len(), IRISCODE_LEN);

    println!("Secret shares sanitycheck passed!");
}
