use crate::input_layer::ligero_input_layer::LigeroInputLayerDescription;
use crate::input_layer::InputLayerDescription;
use crate::prover::{prove, verify};
use remainder_ligero::ligero_structs::LigeroAuxInfo;
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use remainder_shared_types::transcript::{TranscriptReader, TranscriptWriter};
use remainder_shared_types::Fr;
use std::collections::HashMap;

use super::test_helpers::{
    small_circuit_description_and_inputs, circuit_description_and_inputs,
};

#[test]
fn test_small_circuit_both_layers_public() {
    let (desc, inputs) = small_circuit_description_and_inputs();
    let mut transcript_writer =
        TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("GKR Prover Transcript");
    prove(
        &inputs,
        &HashMap::new(),
        &desc.circuit_description,
        &mut transcript_writer,
    )
    .unwrap();
    let mut transcript_reader =
        TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(transcript_writer.get_transcript());
    verify(
        &inputs,
        &[],
        &desc.circuit_description,
        crate::layouter::compiling::CircuitHashType::Sha3_256,
        &mut transcript_reader,
    )
    .unwrap();
}

// Helper function for [test_small_circuit_with_hyrax_layer].
fn build_ligero_layer_spec(input_layer_desc: &InputLayerDescription) -> LigeroInputLayerDescription<Fr> {
    let aux = LigeroAuxInfo::<Fr>::new(
        1 << input_layer_desc.num_vars,
        4,       // rho_inv
        1.0,     // ratio
        Some(1), // maybe_num_col_opens
    );
    LigeroInputLayerDescription {
        layer_id: input_layer_desc.layer_id,
        num_vars: input_layer_desc.num_vars,
        aux,
    }
}

#[test]
fn test_small_circuit_with_ligero_layers() {
    let (desc, mut inputs) = small_circuit_description_and_inputs();

    // Create the Ligero input layer specifcations.
    let mut ligero_layer_spec_map = HashMap::new();
    let image_ligero_spec = build_ligero_layer_spec(&desc.image_input_layer); 
    ligero_layer_spec_map.insert(
        desc.image_input_layer.layer_id,
        (image_ligero_spec.clone(), None),
    );
    let digits_ligero_spec = build_ligero_layer_spec(&desc.digits_input_layer);
    ligero_layer_spec_map.insert(
        desc.digits_input_layer.layer_id,
        (digits_ligero_spec.clone(), None),
    );

    // Prove.
    let mut transcript_writer =
        TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("GKR Prover Transcript");
    prove(
        &inputs,
        &ligero_layer_spec_map,
        &desc.circuit_description,
        &mut transcript_writer,
    )
    .unwrap();

    // Verify (remembering to remove the private input layers from the inputs).
    let mut transcript_reader =
        TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(transcript_writer.get_transcript());
    inputs.remove(&desc.image_input_layer.layer_id);
    inputs.remove(&desc.digits_input_layer.layer_id);
    verify(
        &inputs,
        &[image_ligero_spec, digits_ligero_spec],
        &desc.circuit_description,
        crate::layouter::compiling::CircuitHashType::Sha3_256,
        &mut transcript_reader,
    )
    .unwrap();
}

#[ignore]
#[test]
fn test_worldcoin_circuit_iris_v2_public_inputs() {
    let (desc, inputs) = circuit_description_and_inputs(2, false, None);
    let mut transcript_writer =
        TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("GKR Prover Transcript");
    prove(
        &inputs,
        &HashMap::new(),
        &desc.circuit_description,
        &mut transcript_writer,
    )
    .unwrap();
    let mut transcript_reader =
        TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(transcript_writer.get_transcript());
    verify(
        &inputs,
        &[],
        &desc.circuit_description,
        crate::layouter::compiling::CircuitHashType::Sha3_256,
        &mut transcript_reader,
    )
    .unwrap();
}
