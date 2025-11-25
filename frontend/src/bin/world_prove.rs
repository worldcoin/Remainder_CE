use std::{
    fs::{create_dir_all, File},
    io::Write,
    path::{Path, PathBuf},
};

use clap::{command, Parser};
use frontend::{
    hyrax_worldcoin::orb::load_image_commitment,
    hyrax_worldcoin_mpc::mpc_prover::{print_features_status, V3MPCCircuitAndAuxMles, V3MPCProver},
    worldcoin_mpc::parameters::NUM_PARTIES,
    zk_iriscode_ss::{self, io::read_bytes_from_file},
};
use hyrax::gkr::input_layer::HyraxProverInputCommitment;
use rand::{rngs::OsRng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use shared_types::{perform_function_under_expected_configs, Bn256Point, Fr};
use zeroize::Zeroize;

#[cfg(feature = "print-trace")]
use tracing::Level;
#[cfg(feature = "print-trace")]
use tracing_subscriber::fmt;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct CliArguments {
    /// File path where the circuit description is written
    #[arg(long)]
    circuit: PathBuf,

    /// Input directory containing image and mask binaries
    #[arg(long)]
    input: PathBuf,

    /// Output directory where the generated proof is to be stored
    #[arg(long)]
    output_dir: PathBuf,
}

fn main() {
    #[cfg(feature = "print-trace")]
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    // Sanitycheck by logging the current settings.
    perform_function_under_expected_configs!(
        print_features_status,
        &zk_iriscode_ss::EXPECTED_PROVER_CONFIG,
        &zk_iriscode_ss::EXPECTED_VERIFIER_CONFIG,
    );

    // Compute all proofs and serialize.
    let cli = CliArguments::parse();
    perform_function_under_expected_configs!(
        prove_all_proofs,
        &zk_iriscode_ss::EXPECTED_PROVER_CONFIG,
        &zk_iriscode_ss::EXPECTED_VERIFIER_CONFIG,
        &cli.circuit,
        &cli.input,
        &cli.output_dir
    );
}

/// Computes ten Hyrax proofs for the following:
/// * Proof of {left eye, mask} --> left mask code
/// * Proof of {left eye, iris} --> left iris code
/// * Proof of {right eye, mask} --> right mask code
/// * Proof of {right eye, iris} --> right iris code
/// * Proof of {left mask code, left iris code, party 0} --> left secret share 0
/// * Proof of {left mask code, left iris code, party 1} --> left secret share 1
/// * Proof of {left mask code, left iris code, party 2} --> left secret share 2
/// * Proof of {right mask code, right iris code, party 0} --> right secret share 0
/// * Proof of {right mask code, right iris code, party 1} --> right secret share 1
/// * Proof of {right mask code, right iris code, party 2} --> right secret share 2
///
/// And serializes them to the filepath given by "{output_dir_for_proof}/world_v3.zkp".
fn prove_all_proofs(
    path_to_serialized_circuit: &Path,
    path_to_iris_image_commitment_inputs: &Path,
    output_dir_for_proof: &Path,
) {
    // Sample randomness for the generation of the Shamir polynomial slopes
    // (note that `OsRng` calls `/dev/urandom` under the hood)
    let mut seed = [0u8; 32];
    OsRng.fill_bytes(&mut seed);
    let mut prng = ChaCha20Rng::from_seed(seed);

    create_dir_all(output_dir_for_proof).expect("Failed to create output directory.");
    println!("Proving...");

    let serialized_circuit =
        read_bytes_from_file(path_to_serialized_circuit.as_os_str().to_str().unwrap());
    let v3_mpc_circuit_and_aux_data =
        V3MPCCircuitAndAuxMles::<Fr>::deserialize(&serialized_circuit);

    let mut v3_mpc_prover = V3MPCProver::new(
        zk_iriscode_ss::EXPECTED_PROVER_CONFIG,
        v3_mpc_circuit_and_aux_data.v3_circuit_and_aux_data,
        v3_mpc_circuit_and_aux_data.mpc_circuit_and_aux_mles_all_3_parties,
        &mut prng,
    );

    for is_mask in [false, true] {
        for is_left_eye in [false, true] {
            println!("Proving combination (is_mask, is_left_eye) = ({is_mask}, {is_left_eye}).");

            // Get the pre-existing commitment to the image.
            let mut serialized_image_commitment = load_image_commitment(
                path_to_iris_image_commitment_inputs,
                3,
                is_mask,
                is_left_eye,
            );
            let image_commitment: HyraxProverInputCommitment<Bn256Point> =
                serialized_image_commitment.clone().into();

            v3_mpc_prover.prove_v3(
                is_mask,
                is_left_eye,
                serialized_image_commitment.image,
                image_commitment,
                &mut prng,
            );

            serialized_image_commitment.blinding_factors_bytes.zeroize();
            serialized_image_commitment.commitment_bytes.zeroize();
        }
    }

    for is_left_eye in [false, true] {
        println!("Proving 3 mpc secret shares for (is_left_eye) = ({is_left_eye}).");
        v3_mpc_prover.prove_mpc(is_left_eye, &mut prng);
    }

    let v3_mpc_proof = v3_mpc_prover
        .finalize()
        .expect("Proof is missing while trying to finalize");

    #[cfg(feature = "print-trace")]
    {
        v3_mpc_proof
            .get_v3_proof_ref()
            .get_left_iris_proof()
            .print_size();
        v3_mpc_proof
            .get_v3_proof_ref()
            .get_right_iris_proof()
            .print_size();
        v3_mpc_proof
            .get_v3_proof_ref()
            .get_left_mask_proof()
            .print_size();
        v3_mpc_proof
            .get_v3_proof_ref()
            .get_left_mask_proof()
            .print_size();

        (0..NUM_PARTIES)
            .for_each(|party_idx| v3_mpc_proof.get_party_proof_ref(party_idx).print_size());
    }

    // Write the V3 proof to file
    {
        let serialized_v3_proof = v3_mpc_proof.get_v3_proof_ref().serialize();

        let mut f_v3 = File::create(output_dir_for_proof.join("world_v3.zkp"))
            .expect("Failed to create/open proof file.");

        f_v3.write_all(&serialized_v3_proof)
            .expect("Failed to write serialized v3 proof to file.");
    }

    // Write the MPC proof to file
    {
        (0..NUM_PARTIES).for_each(|party_idx| {
            let serialized_mpc_proof = v3_mpc_proof.get_party_proof_ref(party_idx).serialize();

            let mut f_mpc =
                File::create(output_dir_for_proof.join(format!("world_mpc_party_{party_idx}.zkp")))
                    .expect("Failed to create/open proof file.");

            f_mpc
                .write_all(&serialized_mpc_proof)
                .expect("Failed to write serialized mpc proof to file.");
        });
    }

    // Write the commitments (left/right iris/mask code & slope)
    {
        let commitments = v3_mpc_proof.get_commitments_ref();
        let serialized_commitments = commitments.serialize();

        let mut f = File::create(output_dir_for_proof.join("commitments.zkp"))
            .expect("Failed to create/open commitments file.");

        f.write_all(&serialized_commitments)
            .expect("Failed to write serialized commitments to file.");
    }
}
