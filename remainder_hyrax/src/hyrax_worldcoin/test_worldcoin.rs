use std::{collections::HashMap, env};

use crate::{
    hyrax_gkr::HyraxProof,
    utils::vandermonde::VandermondeInverse,
};
use remainder::{
    layer::LayerId,
    mle::evals::MultilinearExtension,
    worldcoin::{circuits::IriscodeProofDescription, io::read_bytes_from_file, test_helpers::circuit_description_and_inputs},
};
use remainder_shared_types::{
    halo2curves::bn256::G1 as Bn256Point,
    pedersen::PedersenCommitter,
    transcript::{ec_transcript::ECTranscript, poseidon_transcript::PoseidonSponge},
    Base, Scalar,
};

use super::{orb::{load_image_commitment, IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING}, upgrade::{prove_single, verify_single}};

#[test]
fn test_small_circuit_both_layers_public() {
    use remainder::worldcoin::test_helpers::small_circuit_description_and_inputs;
    let (proof_desc, inputs) = small_circuit_description_and_inputs();
    test_iriscode_circuit_with_public_layers_helper(proof_desc, inputs);
}

#[test]
/// Test a small version of the iriscode circuit with a Hyrax input layer.
fn test_small_circuit_with_hyrax_layer() {
    use remainder::worldcoin::test_helpers::small_circuit_description_and_inputs;
    let (proof_desc, inputs) = small_circuit_description_and_inputs();
    test_iriscode_circuit_with_hyrax_helper(proof_desc, inputs);
}

#[ignore] // Takes a long time to run
#[test]
fn test_v2_iris_with_hyrax_precommit() {
    let version = 2;
    let left_eye = true;
    let mask = false;
    // Create the Pedersen committer using the same reference string and parameters as on the Orb
    let committer: PedersenCommitter<Bn256Point> = PedersenCommitter::new(1 << IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING, None);
    // Create a single RNG and Vandermonde inverse converter for all proofs.
    let blinding_rng = &mut rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    let (image, commitment, blinding_factors) = load_image_commitment(version, mask, left_eye);
    let proof = prove_single(
        version,
        mask,
        image.to_vec(),
        commitment.to_vec(),
        blinding_factors.to_vec(),
        &committer,
        blinding_rng,
        converter,
    );
    let code = verify_single(version, mask, &proof, &committer);
    //FIXME assert length of code
}

#[ignore] // Takes a long time to run
#[test]
fn test_upgrade_v2_v3() {
    let mut data: HashMap<(u8, bool, bool), (Vec<u8>, Vec<u8>, Vec<u8>)> = HashMap::new();
    for version in 2..=3 {
        for mask in [false, true] {
            for left_eye in [false, true] {
                data.insert((version, mask, left_eye), load_image_commitment(version, mask, left_eye));
            }
        }
    }
    let proofs = super::upgrade::prove_upgrade_v2_to_v3(&data);
    let _codes = super::upgrade::verify_upgrade_v2_to_v3(&proofs).unwrap();
}

#[ignore] // Takes a long time to run
#[test]
fn test_v2_iris_with_hyrax_layer() {
    test_iriscode_v2_with_hyrax_helper(false);
}

#[ignore] // Takes a long time to run
#[test]
fn test_v2_mask_with_hyrax_layer() {
    test_iriscode_v2_with_hyrax_helper(false);
}

#[ignore] // Takes a long time to run
#[test]
fn test_v3_iris_with_hyrax_layer() {
    test_iriscode_v3_with_hyrax_helper(false);
}

#[ignore] // Takes a long time to run
#[test]
fn test_v3_mask_with_hyrax_layer() {
    test_iriscode_v3_with_hyrax_helper(false);
}

/// Test the iriscode circuit v2 with a Hyrax input layer in either the mask (true) or iris (false)
/// case.
pub fn test_iriscode_v2_with_hyrax_helper(mask: bool) {
    let (desc, inputs) = circuit_description_and_inputs(2, mask, None);
    test_iriscode_circuit_with_hyrax_helper(desc, inputs);
}

/// Test the iriscode circuit v3 with a Hyrax input layer in either the mask (true) or iris (false)
/// case.
pub fn test_iriscode_v3_with_hyrax_helper(mask: bool) {
    let (desc, inputs) = circuit_description_and_inputs(3, mask, None);
    test_iriscode_circuit_with_hyrax_helper(desc, inputs);
}

/// Helper function for testing an iriscode circuit (of any version, with any data) with a Hyrax input layer.
pub fn test_iriscode_circuit_with_public_layers_helper(
    proof_desc: IriscodeProofDescription<Scalar>,
    inputs: HashMap<LayerId, MultilinearExtension<Scalar>>,
) {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let blinding_rng = &mut rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    let num_generators = 512;
    let committer = PedersenCommitter::<Bn256Point>::new(
        num_generators + 1,
        "modulus modulus modulus modulus modulus",
        None,
    );
    let proof = HyraxProof::prove(
        &inputs,
        &HashMap::new(),
        &proof_desc.circuit_description,
        &committer,
        blinding_rng,
        converter,
        &mut transcript,
    );
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    proof.verify(&HashMap::new(), &proof_desc.circuit_description, &committer, &mut transcript);
}

/// Helper function for testing an iriscode circuit (of any version, with any data) with Hyrax input
/// layers for the private data.
pub fn test_iriscode_circuit_with_hyrax_helper(
    proof_desc: IriscodeProofDescription<Scalar>,
    inputs: HashMap<LayerId, MultilinearExtension<Scalar>>,
) {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let blinding_rng = &mut rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    let num_generators = 512;
    let committer = PedersenCommitter::<Bn256Point>::new(
        num_generators + 1,
        "modulus modulus modulus modulus modulus",
        None,
    );
    // Set up Hyrax input layer specification.
    let mut prover_hyrax_input_layers = HashMap::new();
    prover_hyrax_input_layers.insert(
        proof_desc.image_input_layer.layer_id,
        (proof_desc.image_input_layer.clone().into(), None),
    );
    prover_hyrax_input_layers.insert(
        proof_desc.digits_input_layer.layer_id,
        (proof_desc.digits_input_layer.clone().into(), None),
    );
    // Prove.
    let proof = HyraxProof::prove(
        &inputs,
        &prover_hyrax_input_layers,
        &proof_desc.circuit_description,
        &committer,
        blinding_rng,
        converter,
        &mut transcript,
    );
    // Verify.
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let mut verifier_hyrax_input_layers = HashMap::new();
    verifier_hyrax_input_layers.insert(
        proof_desc.image_input_layer.layer_id,
        proof_desc.image_input_layer.clone().into(),
    );
    verifier_hyrax_input_layers.insert(
        proof_desc.digits_input_layer.layer_id,
        proof_desc.digits_input_layer.clone().into(),
    );
    proof.verify(
        &verifier_hyrax_input_layers,
        &proof_desc.circuit_description,
        &committer,
        &mut transcript,
    );
}
