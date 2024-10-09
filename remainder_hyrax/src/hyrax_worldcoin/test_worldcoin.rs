use std::{collections::HashMap, path::Path};

use remainder::{input_layer::hyrax_input_layer, worldcoin::{data::{load_worldcoin_data_v2, load_worldcoin_data_v3, wirings_to_reroutings, IriscodeCircuitData}, parameters::decode_wirings}};
use remainder_shared_types::{
    halo2curves::{bn256::G1 as Bn256Point, group::Group, CurveExt},
    transcript::{
        ec_transcript::ECTranscript,
        poseidon_transcript::PoseidonSponge,
    },
};

use crate::{
    hyrax_gkr::{hyrax_input_layer::{HyraxInputLayer, HyraxInputLayerDescription}, HyraxProof}, hyrax_worldcoin::build_hyrax_circuit_hyrax_input_layer,
    pedersen::PedersenCommitter, utils::vandermonde::VandermondeInverse,
};

// use super::build_hyrax_circuit_public_input_layer;
type Scalar = <Bn256Point as Group>::Scalar;
type Base = <Bn256Point as CurveExt>::Base;

use remainder::worldcoin::tests::small_circuit_description_and_inputs;

#[test]
fn test_small_circuit_both_layers_public() {
    let (circuit_desc, _, inputs) = small_circuit_description_and_inputs();
    dbg!(&circuit_desc);
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let blinding_rng = &mut rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    let num_generators = 100;
    let committer = PedersenCommitter::<Bn256Point>::new(
        num_generators + 1,
        "modulus modulus modulus modulus modulus",
        None,
    );
    let proof = HyraxProof::prove(
        &inputs,
        &HashMap::new(),
        &circuit_desc,
        &committer,
        blinding_rng,
        converter,
        &mut transcript,
    );
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    proof.verify(
        &HashMap::new(),
        &circuit_desc,
        &committer,
        &mut transcript,
    );
}

#[test]
fn test_small_circuit_with_hyrax_layer() {
    let (circuit_desc, private_layer_desc, inputs) = small_circuit_description_and_inputs();
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let blinding_rng = &mut rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    let num_generators = 100;
    let committer = PedersenCommitter::<Bn256Point>::new(
        num_generators + 1,
        "modulus modulus modulus modulus modulus",
        None,
    );
    let mut prover_hyrax_input_layers = HashMap::new();
    let hyrax_input_layer_desc: HyraxInputLayerDescription = private_layer_desc.into();
    prover_hyrax_input_layers.insert(
        hyrax_input_layer_desc.layer_id,
        (hyrax_input_layer_desc.clone(), None),
    );

    let proof = HyraxProof::prove(
        &inputs,
        &prover_hyrax_input_layers,
        &circuit_desc,
        &committer,
        blinding_rng,
        converter,
        &mut transcript,
    );
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let mut verifier_hyrax_input_layers = HashMap::new();
    verifier_hyrax_input_layers.insert(
        hyrax_input_layer_desc.layer_id,
        hyrax_input_layer_desc.clone(),
    );
    proof.verify(
        &verifier_hyrax_input_layers,
        &circuit_desc,
        &committer,
        &mut transcript,
    );
}

// /// Helper function that runs the Hyrax Worldcoin test against a given data set
// /// with public input layers.
// pub fn test_hyrax_worldcoin_public_input_layer<
//     const MATMULT_ROWS_NUM_VARS: usize,
//     const MATMULT_COLS_NUM_VARS: usize,
//     const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
//     const BASE: u64,
//     const NUM_DIGITS: usize,
// >(
//     data: CircuitData<
//         Scalar,
//     >,
//     reroutings: Vec<(usize, usize)>,
//     num_generators: usize,
// ) {
//     let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
//         ECTranscriptWriter::new("");
//     let blinding_rng = &mut rand::thread_rng();
//     let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
//     let committer = PedersenCommitter::<Bn256Point>::new(
//         num_generators + 1,
//         "hi why is this not working, please help me",
//         None,
//     );
//     let mut hyrax_prover = HyraxProver::new(&committer, blinding_rng, converter);

//     let witness_function = build_hyrax_circuit_public_input_layer(data, reroutings);

//     let (mut circuit_description, hyrax_proof) =
//         hyrax_prover.prove_gkr_circuit(witness_function, &mut prover_transcript);

//     let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
//         ECTranscriptReader::new(prover_transcript.get_transcript());

//     hyrax_prover.verify_gkr_circuit(
//         &hyrax_proof,
//         &mut circuit_description,
//         &mut verifier_transcript,
//     );
// }

// /// Helper function that runs the Hyrax Worldcoin test against a given data set
// /// with hyrax input layers when data needs to be blinded.
// pub fn test_hyrax_worldcoin_hyrax_input_layer<
//     const MATMULT_ROWS_NUM_VARS: usize,
//     const MATMULT_COLS_NUM_VARS: usize,
//     const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
//     const BASE: u64,
//     const NUM_DIGITS: usize,
// >(
//     data: CircuitData<
//         Scalar,
//     >,
//     reroutings: Vec<(usize, usize)>,
//     num_generators: usize,
// ) {
//     let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
//         ECTranscriptWriter::new("");
//     let blinding_rng = &mut rand::thread_rng();
//     let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
//     let committer = PedersenCommitter::<Bn256Point>::new(
//         num_generators + 1,
//         "hi why is this not working, please help me",
//         None,
//     );
//     let mut hyrax_prover = HyraxProver::new(&committer, blinding_rng, converter);

//     let witness_function = build_hyrax_circuit_hyrax_input_layer(data, reroutings, None);

//     let (mut circuit_description, hyrax_proof) =
//         hyrax_prover.prove_gkr_circuit(witness_function, &mut prover_transcript);

//     let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
//         ECTranscriptReader::new(prover_transcript.get_transcript());

//     hyrax_prover.verify_gkr_circuit(
//         &hyrax_proof,
//         &mut circuit_description,
//         &mut verifier_transcript,
//     );
// }

// #[ignore] // Takes a long time to run
// #[test]
// fn test_hyrax_worldcoin_v2_iris_public_input_layer() {
//     use remainder::worldcoin::parameters_v2::{
//         IM_NUM_COLS,
//         BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
//         MATMULT_ROWS_NUM_VARS, NUM_DIGITS, WIRINGS
//     };
//     let image_path = Path::new("../remainder_prover/src/worldcoin/constants/v2/iris/test_image.npy").to_path_buf();
//     let data = load_worldcoin_data_v2::<
//         Scalar,
//         MATMULT_ROWS_NUM_VARS,
//         MATMULT_COLS_NUM_VARS,
//         MATMULT_INTERNAL_DIM_NUM_VARS,
//         BASE,
//         NUM_DIGITS,
//     >(image_path, false);
//     let reroutings = wirings_to_reroutings(&decode_wirings(WIRINGS), IM_NUM_COLS, 1 << MATMULT_INTERNAL_DIM_NUM_VARS);
//     test_hyrax_worldcoin_public_input_layer(data, reroutings, 100);
// }

// #[ignore] // Takes a long time to run
// #[test]
// fn test_hyrax_worldcoin_v2_mask_public_input_layer() {
//     use remainder::worldcoin::parameters_v2::{
//         IM_NUM_COLS,
//         BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
//         MATMULT_ROWS_NUM_VARS, NUM_DIGITS, WIRINGS
//     };
//     let image_path = Path::new("../remainder_prover/src/worldcoin/constants/v2/mask/test_image.npy").to_path_buf();
//     let data = load_worldcoin_data_v2::<
//         Scalar,
//         MATMULT_ROWS_NUM_VARS,
//         MATMULT_COLS_NUM_VARS,
//         MATMULT_INTERNAL_DIM_NUM_VARS,
//         BASE,
//         NUM_DIGITS,
//     >(image_path, true);
//     let reroutings = wirings_to_reroutings(&decode_wirings(WIRINGS), IM_NUM_COLS, 1 << MATMULT_INTERNAL_DIM_NUM_VARS);
//     test_hyrax_worldcoin_public_input_layer(data, reroutings, 100);
// }

// #[ignore] // Takes a long time to run
// #[test]
// fn test_hyrax_worldcoin_v3_iris_public_input_layer() {
//     use remainder::worldcoin::parameters_v3::{
//         IM_NUM_COLS,
//         BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
//         MATMULT_ROWS_NUM_VARS, NUM_DIGITS, WIRINGS
//     };
//     let image_path = Path::new("../remainder_prover/src/worldcoin/constants/v3/iris/test_image.npy").to_path_buf();
//     let data = load_worldcoin_data_v3::<
//         Scalar,
//         MATMULT_ROWS_NUM_VARS,
//         MATMULT_COLS_NUM_VARS,
//         MATMULT_INTERNAL_DIM_NUM_VARS,
//         BASE,
//         NUM_DIGITS,
//     >(image_path, false);
//     let reroutings = wirings_to_reroutings(&decode_wirings(WIRINGS), IM_NUM_COLS, 1 << MATMULT_INTERNAL_DIM_NUM_VARS);
//     test_hyrax_worldcoin_public_input_layer(data, reroutings, 100);
// }

// #[ignore] // Takes a long time to run
// #[test]
// fn test_hyrax_worldcoin_v3_mask_public_input_layer() {
//     use remainder::worldcoin::parameters_v3::{
//         IM_NUM_COLS,
//         BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
//         MATMULT_ROWS_NUM_VARS, NUM_DIGITS, WIRINGS
//     };
//     let image_path = Path::new("../remainder_prover/src/worldcoin/constants/v3/mask/test_image.npy").to_path_buf();
//     let data = load_worldcoin_data_v3::<
//         Scalar,
//         MATMULT_ROWS_NUM_VARS,
//         MATMULT_COLS_NUM_VARS,
//         MATMULT_INTERNAL_DIM_NUM_VARS,
//         BASE,
//         NUM_DIGITS,
//     >(image_path, true);
//     let reroutings = wirings_to_reroutings(&decode_wirings(WIRINGS), IM_NUM_COLS, 1 << MATMULT_INTERNAL_DIM_NUM_VARS);
//     test_hyrax_worldcoin_public_input_layer(data, reroutings, 100);
// }

// #[ignore] // Takes a long time to run
// #[test]
// fn test_hyrax_worldcoin_v2_iris_hyrax_input_layer() {
//     use remainder::worldcoin::parameters_v3::{
//         IM_NUM_COLS,
//         BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
//         MATMULT_ROWS_NUM_VARS, NUM_DIGITS, WIRINGS
//     };
//     let image_path = Path::new("../remainder_prover/src/worldcoin/constants/v2/iris/test_image.npy").to_path_buf();
//     let data = load_worldcoin_data_v2::<
//         Scalar,
//         MATMULT_ROWS_NUM_VARS,
//         MATMULT_COLS_NUM_VARS,
//         MATMULT_INTERNAL_DIM_NUM_VARS,
//         BASE,
//         NUM_DIGITS,
//     >(image_path, false);
//     let reroutings = wirings_to_reroutings(&decode_wirings(WIRINGS), IM_NUM_COLS, 1 << MATMULT_INTERNAL_DIM_NUM_VARS);
//     test_hyrax_worldcoin_hyrax_input_layer(data, reroutings, 512);
// }
