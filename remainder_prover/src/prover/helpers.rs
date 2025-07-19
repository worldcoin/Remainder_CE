#![allow(clippy::type_complexity)]

use crate::layouter::builder::ProvableCircuit;
use crate::prover::verify;
use ark_std::{end_timer, start_timer};

use itertools::Itertools;

use remainder_shared_types::circuit_hash::CircuitHashType;
use remainder_shared_types::config::global_config::{
    global_prover_circuit_description_hash_type, global_verifier_circuit_description_hash_type,
};
use remainder_shared_types::config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig};
use remainder_shared_types::transcript::poseidon_sponge::PoseidonSponge;
use remainder_shared_types::transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter};
use remainder_shared_types::{perform_function_under_expected_configs, Field};
use serde_json;
use sha3::Digest;
use sha3::Sha3_256;
use std::fs::File;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::BufWriter;
use std::path::Path;

use super::{prove, GKRCircuitDescription};

/// Writes the circuit description for the provided [GKRCircuitDescription]
/// to the `circuit_description_path` specified, in JSON format.
///
/// ## Arguments
/// * `circuit_description` - The [GKRCircuitDescription] to be written to file.
/// * `circuit_description_path` - The filepath to which the JSON description
///   will be saved.
pub fn write_circuit_description_to_file<F: Field>(
    circuit_description: &GKRCircuitDescription<F>,
    circuit_description_path: &Path,
) {
    let f = File::create(circuit_description_path).unwrap();
    let writer = BufWriter::new(f);
    serde_json::to_writer(writer, circuit_description).unwrap();
}

/// Returns an equivalent set of field elements to be absorbed into a
/// transcript, given a circuit description and the type of hash function
/// to be used.
///
/// ## Arguments
/// * `circuit_description` - The circuit description to be hashed and
///   added to transcript.
/// * `circuit_description_hash_type` - The type of hash function to be
///   used.
pub fn get_circuit_description_hash_as_field_elems<F: Field>(
    circuit_description: &GKRCircuitDescription<F>,
    circuit_description_hash_type: CircuitHashType,
) -> Vec<F> {
    match circuit_description_hash_type {
        CircuitHashType::DefaultRustHash => {
            let mut hasher = DefaultHasher::new();
            circuit_description.hash(&mut hasher);
            let hash_value = hasher.finish();
            vec![F::from(hash_value)]
        }
        CircuitHashType::Sha3_256 => {
            // First, serialize the circuit description to be hashed
            let serialized = serde_json::to_vec(&circuit_description).expect("Failed to serialize");
            let mut hasher = Sha3_256::new();
            hasher.update(&serialized);
            let circuit_description_hash_bytes = hasher.finalize();

            // Since the output is 32 bytes and can be out of range,
            // we split instead into two chunks of 16 bytes each and
            // absorb two field elements.
            // TODO(ryancao): Update this by using `REPR_NUM_BYTES` after merging with the testing branch
            let mut circuit_description_hash_bytes_first_half = [0; 32];
            let mut circuit_description_hash_bytes_second_half = [0; 32];

            circuit_description_hash_bytes_first_half[..16]
                .copy_from_slice(&circuit_description_hash_bytes.to_vec()[..16]);
            circuit_description_hash_bytes_second_half[..16]
                .copy_from_slice(&circuit_description_hash_bytes.to_vec()[16..]);

            vec![
                F::from_bytes_le(circuit_description_hash_bytes_first_half.as_ref()),
                F::from_bytes_le(circuit_description_hash_bytes_second_half.as_ref()),
            ]
        }
        CircuitHashType::Poseidon => {
            // First, serialize the circuit description to be hashed
            let serialized = serde_json::to_vec(&circuit_description).expect("Failed to serialize");
            // Run through the bytes of `serialized` in chunks of 16 so we don't overflow
            // TODO(ryancao): Update this by using `REPR_NUM_BYTES` after merging with the testing branch
            let circuit_field_elem_desc = serialized
                .chunks(16)
                .map(|byte_chunk| F::from_bytes_le(byte_chunk))
                .collect_vec();
            let mut poseidon_sponge: PoseidonSponge<F> = PoseidonSponge::default();
            poseidon_sponge.absorb_elements(&circuit_field_elem_desc);
            vec![poseidon_sponge.squeeze()]
        }
    }
}

/// Function which calls [test_circuit_internal] with the appropriate expected
/// prover/verifier config.
pub fn test_circuit_with_config<F: Field>(
    provable_circuit: &ProvableCircuit<F>,
    expected_prover_config: &GKRCircuitProverConfig,
    expected_verifier_config: &GKRCircuitVerifierConfig,
) {
    perform_function_under_expected_configs!(
        test_circuit_internal,
        expected_prover_config,
        expected_verifier_config,
        provable_circuit
    )
}

/// Function which calls [test_circuit_internal] with the appropriate expected
/// prover/verifier config.
pub fn test_circuit_with_runtime_optimized_config<F: Field>(provable_circuit: &ProvableCircuit<F>) {
    let expected_prover_config = GKRCircuitProverConfig::runtime_optimized_default();
    let expected_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&expected_prover_config, false);
    perform_function_under_expected_configs!(
        test_circuit_internal,
        &expected_prover_config,
        &expected_verifier_config,
        provable_circuit
    )
}

/// Function which calls [test_circuit_internal] with a memory-optimized default.
pub fn test_circuit_with_memory_optimized_config<F: Field>(provable_circuit: &ProvableCircuit<F>) {
    let expected_prover_config = GKRCircuitProverConfig::memory_optimized_default();
    let expected_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&expected_prover_config, true);
    perform_function_under_expected_configs!(
        test_circuit_internal,
        &expected_prover_config,
        &expected_verifier_config,
        provable_circuit
    )
}

/// Function which instantiates a circuit description with the given inputs
/// and precommits and both attempts to both prove and verify said circuit.
fn test_circuit_internal<F: Field>(provable_circuit: &ProvableCircuit<F>) {
    let mut transcript_writer =
        TranscriptWriter::<F, PoseidonSponge<F>>::new("GKR Prover Transcript");
    let prover_timer = start_timer!(|| "Proof generation");

    match prove(
        provable_circuit,
        global_prover_circuit_description_hash_type(),
        &mut transcript_writer,
    ) {
        Ok(proof_config) => {
            end_timer!(prover_timer);
            let transcript = transcript_writer.get_transcript();
            let mut transcript_reader = TranscriptReader::<F, PoseidonSponge<F>>::new(transcript);
            let verifier_timer = start_timer!(|| "Proof verification");

            let verifiable_circuit = provable_circuit._gen_verifiable_circuit();

            match verify(
                &verifiable_circuit,
                global_verifier_circuit_description_hash_type(),
                &mut transcript_reader,
                &proof_config,
            ) {
                Ok(_) => {
                    end_timer!(verifier_timer);
                }
                Err(err) => {
                    println!("Verify failed! Error: {err}");
                    panic!();
                }
            }
        }
        Err(err) => {
            println!("Proof failed! Error: {err}");
            panic!();
        }
    }
}
