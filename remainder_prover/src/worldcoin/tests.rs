use crate::input_layer::ligero_input_layer::LigeroInputLayerDescription;
use crate::prover::{prove, verify};
use remainder_ligero::ligero_structs::LigeroAuxInfo;
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use remainder_shared_types::transcript::{TranscriptReader, TranscriptWriter};
use remainder_shared_types::Fr;
use std::collections::HashMap;

use super::test_helpers::{small_circuit_description_and_inputs, v2_circuit_description_and_inputs};

#[test]
fn test_small_circuit_both_layers_public() {
    let (circuit_desc, _, inputs) = small_circuit_description_and_inputs();
    let mut transcript_writer = TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("GKR Prover Transcript");
    prove(&inputs, &HashMap::new(), &circuit_desc, &mut transcript_writer).unwrap();
    let mut transcript_reader = TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(transcript_writer.get_transcript());
    verify(&inputs, &[], &circuit_desc, &mut transcript_reader).unwrap();
}

#[test]
fn test_small_circuit_with_a_ligero_layer() {
    let (circuit_desc, private_input_layer_desc, mut inputs) = small_circuit_description_and_inputs();

    // Create the Ligero input layer specifcation.
    let aux = LigeroAuxInfo::<Fr>::new(
        1 << private_input_layer_desc.num_vars,
        4, // rho_inv
        1.0, // ratio
        Some(1), // maybe_num_col_opens
    );
    let ligero_layer_desc = LigeroInputLayerDescription::<Fr> {
        layer_id: private_input_layer_desc.layer_id,
        num_vars: private_input_layer_desc.num_vars,
        aux,
    };
    let mut ligero_layer_spec_map = HashMap::new();
    ligero_layer_spec_map.insert(private_input_layer_desc.layer_id, (ligero_layer_desc.clone(), None));

    // Prove.
    let mut transcript_writer = TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("GKR Prover Transcript");
    prove(&inputs, &ligero_layer_spec_map, &circuit_desc, &mut transcript_writer).unwrap();

    // Verify (remembering to remove the private input layer from the inputs).
    let mut transcript_reader = TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(transcript_writer.get_transcript());
    inputs.remove(&private_input_layer_desc.layer_id);
    verify(&inputs, &[ligero_layer_desc], &circuit_desc, &mut transcript_reader).unwrap();
}

#[ignore]
#[test]
fn test_worldcoin_circuit_iris_v2_public_inputs() {
    let (circuit_desc, _, inputs) = v2_circuit_description_and_inputs(false);
    let mut transcript_writer = TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("GKR Prover Transcript");
    prove(&inputs, &HashMap::new(), &circuit_desc, &mut transcript_writer).unwrap();
    let mut transcript_reader = TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(transcript_writer.get_transcript());
    verify(&inputs, &[], &circuit_desc, &mut transcript_reader).unwrap();
}