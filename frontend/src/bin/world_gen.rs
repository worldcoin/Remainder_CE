use std::{fs::File, io::Write, path::PathBuf};

use clap::Parser;
use frontend::{
    hyrax_worldcoin_mpc::mpc_prover::{
        generate_mpc_circuit_and_aux_mles_all_3_parties, print_features_status,
        V3MPCCircuitAndAuxMles,
    },
    zk_iriscode_ss::{self, v3::generate_iriscode_circuit_and_aux_data},
};
use shared_types::{perform_function_under_expected_configs, Fr};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct CliArguments {
    /// File path where the circuit description is written
    #[arg(long)]
    circuit: PathBuf,
}

fn main() {
    // Sanitycheck by logging the current settings.
    perform_function_under_expected_configs!(
        print_features_status,
        &zk_iriscode_ss::EXPECTED_PROVER_CONFIG,
        &zk_iriscode_ss::EXPECTED_VERIFIER_CONFIG,
    );

    // Parse args and perform circuit generation + serialization to disk.
    let cli = CliArguments::parse();
    perform_function_under_expected_configs!(
        generate_circuit_description_helper,
        &zk_iriscode_ss::EXPECTED_PROVER_CONFIG,
        &zk_iriscode_ss::EXPECTED_VERIFIER_CONFIG,
        cli.circuit
    );
}

/// Helper function which actually does the circuit description generation
/// and writes the circuit description + auxiliary MLEs for the iriscode V3
/// and secret sharing circuit to the `circuit_output_filepath` given.
fn generate_circuit_description_helper(circuit_output_filepath: PathBuf) {
    let v3_circuit_and_aux_data = generate_iriscode_circuit_and_aux_data();

    let mpc_circuit_and_aux_mles_all_3_parties =
        generate_mpc_circuit_and_aux_mles_all_3_parties::<Fr>();

    let all_circuits = V3MPCCircuitAndAuxMles {
        v3_circuit_and_aux_data,
        mpc_circuit_and_aux_mles_all_3_parties,
    };

    let serialized_circuit_description =
        bincode::serialize(&all_circuits).expect("Failed to serialize circuits");

    let mut f = File::create(circuit_output_filepath).expect("Failed to create/open output file.");
    f.write_all(&serialized_circuit_description)
        .expect("Failed to write serialized circuit description to file.");
}
