use crate::layouter::compiling::{CircuitHashType, LayouterCircuit};
use crate::layouter::component::Component;
use crate::layouter::nodes::circuit_inputs::InputLayerData;
use crate::layouter::nodes::node_enum::NodeEnum;
use crate::layouter::nodes::Context;
use ark_std::{end_timer, start_timer};

use itertools::Itertools;
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use remainder_shared_types::transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter};
use remainder_shared_types::Field;
use serde_json;
use sha3::Digest;
use sha3::Sha3_256;
use std::fs::File;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufReader, BufWriter};
use std::path::Path;

use super::GKRCircuitDescription;

/// Writes the circuit description for the provided [LayouterCircuit]
/// to the `circuit_description_path` specified, in JSON format.
///
/// WARNING: THIS FUNCTION CALLS THE `self.witness_builder` FUNCTION
/// ON THE CIRCUIT AND THUS CANNOT BE USED IN TANDEM WITH [test_circuit()]!!!
/// INSTEAD, CREATE A NEW INSTANCE OF THE CIRCUIT IF YOU ARE TO USE
/// [test_circuit()] AS WELL!!!
///
/// ## Arguments
/// * `circuit_to_be_built` - The [LayouterCircuit] whose circuit description
///     will be generated.
/// * `circuit_description_path` - The filepath to which the JSON description
///     will be saved.
pub fn write_circuit_description_to_file<
    F: Field,
    C: Component<NodeEnum<F>>,
    Fn: FnMut(&Context) -> (C, Vec<InputLayerData<F>>),
>(
    mut circuit_to_be_built: LayouterCircuit<F, C, Fn>,
    circuit_description_path: &Path,
) {
    let circuit_description = circuit_to_be_built.get_circuit_description();
    let f = File::create(circuit_description_path).unwrap();
    let writer = BufWriter::new(f);
    serde_json::to_writer(writer, &circuit_description).unwrap();
}

/// Returns an equivalent set of field elements to be absorbed into a
/// transcript, given a circuit description and the type of hash function
/// to be used.
///
/// ## Arguments
/// * `circuit_description` - The circuit description to be hashed and
///     added to transcript.
/// * `circuit_description_hash_type` - The type of hash function to be
///     used.
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
            let circuit_description_hash_bytes_first_half =
                &circuit_description_hash_bytes.to_vec()[..16];
            let circuit_description_hash_bytes_second_half =
                &circuit_description_hash_bytes.to_vec()[16..];
            vec![
                F::from_bytes_le(circuit_description_hash_bytes_first_half.to_vec()),
                F::from_bytes_le(circuit_description_hash_bytes_second_half.to_vec()),
            ]
        }
        CircuitHashType::Poseidon => {
            // First, serialize the circuit description to be hashed
            let serialized = serde_json::to_vec(&circuit_description).expect("Failed to serialize");
            // Run through the bytes of `serialized` in chunks of 16 so we don't overflow
            // TODO(ryancao): Update this by using `REPR_NUM_BYTES` after merging with the testing branch
            let circuit_field_elem_desc = serialized
                .chunks(16)
                .map(|byte_chunk| F::from_bytes_le(byte_chunk.to_vec()))
                .collect_vec();
            let mut poseidon_sponge: PoseidonSponge<F> = PoseidonSponge::default();
            poseidon_sponge.absorb_elements(&circuit_field_elem_desc);
            vec![poseidon_sponge.squeeze()]
        }
    }
}

/// TODO(ryancao): Move this into the prover/verifier settings!!! (This is already a TDH ticket)
const CIRCUIT_DESCRIPTION_HASH_TYPE: CircuitHashType = CircuitHashType::DefaultRustHash;

/// Boilerplate code for testing a circuit
pub fn test_circuit<
    F: Field,
    C: Component<NodeEnum<F>>,
    Fn: FnMut(&Context) -> (C, Vec<InputLayerData<F>>),
>(
    mut circuit: LayouterCircuit<F, C, Fn>,
    path: Option<&Path>,
) {
    let transcript_writer = TranscriptWriter::<F, PoseidonSponge<F>>::new("GKR Prover Transcript");
    let prover_timer = start_timer!(|| "Proof generation");

    match circuit.prove(transcript_writer, CIRCUIT_DESCRIPTION_HASH_TYPE) {
        Ok((transcript, mut gkr_circuit_description)) => {
            end_timer!(prover_timer);
            if let Some(path) = path {
                let write_out_timer = start_timer!(|| "Writing out proof");
                let f = File::create(path).unwrap();
                let writer = BufWriter::new(f);
                serde_json::to_writer(writer, &transcript).unwrap();
                end_timer!(write_out_timer);
            }

            let transcript = if let Some(path) = path {
                let read_in_timer = start_timer!(|| "Reading in proof");
                let file = std::fs::File::open(path).unwrap();
                let reader = BufReader::new(file);
                let result = serde_json::from_reader(reader).unwrap();
                end_timer!(read_in_timer);
                result
            } else {
                transcript
            };

            let mut transcript_reader = TranscriptReader::<F, PoseidonSponge<F>>::new(transcript);
            let verifier_timer = start_timer!(|| "Proof verification");

            match gkr_circuit_description
                .verify(&mut transcript_reader, CIRCUIT_DESCRIPTION_HASH_TYPE)
            {
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
