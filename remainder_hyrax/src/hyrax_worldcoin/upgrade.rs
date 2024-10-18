#![allow(warnings)]
use std::collections::HashMap;
use std::hash::Hash;

use ff::Field;
use remainder::prover::GKRCircuitDescription;
use remainder::utils::mle::pad_with;
use remainder::worldcoin::test_helpers::{circuit_description_and_inputs, v2_circuit_description_and_inputs, v3_circuit_description_and_inputs};
use remainder_shared_types::halo2curves::{bn256::G1 as Bn256Point, group::Group};
use remainder_shared_types::pedersen::PedersenCommitter;
use remainder_shared_types::transcript::ec_transcript::{ECTranscript, ECTranscriptTrait};
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use remainder_shared_types::transcript::Transcript;
use remainder_shared_types::{Fq, Fr};

use crate::hyrax_gkr::hyrax_input_layer::HyraxInputLayerDescription;
use crate::hyrax_gkr::{self, HyraxProof};
use crate::hyrax_pcs::MleCoefficientsVector;
use crate::hyrax_worldcoin::orb::{IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING};
use crate::utils::vandermonde::VandermondeInverse;

use super::orb::SerializedImageCommitment;

type Scalar = Fr;
type Base = Fq;

#[derive(Debug)]
pub enum UpgradeError {
}

/// Verify the upgrade from v2 to v3 using the Hyrax proof system. Receives the [HyraxProof] for each combination of
/// version, eye, and type (iris or mask), and returns, for each such combination, the corresponding mask or iris code.
pub fn verify_upgrade_v2_to_v3(
    proofs: &HashMap<(u8, bool, bool), HyraxProof<Bn256Point>>,
) -> Result<HashMap<(u8, bool, bool), (Vec<bool>, Vec<Bn256Point>)>, UpgradeError> {
    // Create the Pedersen committer using the same reference string and parameters as on the Orb
    let committer: PedersenCommitter<Bn256Point> = PedersenCommitter::new(1 << IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING, None);

    let mut results = HashMap::new();
    for version in [2u8, 3u8] {
        for mask in [false, true] {
            for left_eye in [false, true] {
                dbg!(version, mask, left_eye);
                let proof = proofs.get(&(version, mask, left_eye)).unwrap();
                let code = verify_single(version, mask, proof, &committer);
                results.insert((version, mask, left_eye), code);
            }
        }
    }
    Ok(results)
}

pub fn verify_single(
    version: u8,
    mask: bool,
    proof: &HyraxProof<Bn256Point>,
    committer: &PedersenCommitter<Bn256Point>,
) -> (Vec<bool>, Vec<Bn256Point>) {
    // Get the circuit description, as well as the inputs if we were using a test image;
    // we'll use these to ensure that the correct kernel values and thresholds are being
    // used in the supplied proof.
    let (proof_desc, test_inputs) = circuit_description_and_inputs(version, mask, None);

    let mut hyrax_input_layers = HashMap::new();
    // The image, with the precommit (must use the same number of columns as were used at the time of committing!)
    let image_hyrax_input_layer_desc = HyraxInputLayerDescription {
        layer_id: proof_desc.image_input_layer.layer_id,
        num_bits: proof_desc.image_input_layer.num_vars,
        log_num_cols: IMAGE_COMMIT_LOG_NUM_COLS,
    };
    hyrax_input_layers.insert(
        proof_desc.image_input_layer.layer_id,
        image_hyrax_input_layer_desc,
    );
    hyrax_input_layers.insert(
        proof_desc.digits_input_layer.layer_id,
        proof_desc.digits_input_layer.clone().into(),
    );

    // Create a fresh transcript.
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    // Verify the relationship between iris/mask code and image.
    proof.verify(
        &hyrax_input_layers,
        &proof_desc.circuit_description,
        &committer,
        &mut transcript,
    );
    // Check that the correct kernel values and thresholds are being used.
    let expected_aux_mle = test_inputs.get(&proof_desc.auxiliary_input_layer.layer_id).unwrap();
    let aux_mle_in_proof = proof.public_inputs.iter().find(|(id, _)| id == &proof_desc.auxiliary_input_layer.layer_id).unwrap().1.clone();
    assert_eq!(*expected_aux_mle, aux_mle_in_proof); // FIXME raise error

    // Extract the iris/mask code from the proof, and convert to Vec<bool>.
    let code_mle = proof.public_inputs
        .iter()
        .find(|(id, _)| id == &proof_desc.code_input_layer.layer_id)
        .unwrap().1.get_evals_vector();
    let code = code_mle
        .iter()
        .map(|b|
            match b {
                &Fr::ONE => true,
                &Fr::ZERO => false,
                _ => panic!() // FIXME raise error
            }
        ).collect();
    let image_commitment = proof.hyrax_input_proofs
        .iter()
        .find(|proof| proof.layer_id == proof_desc.image_input_layer.layer_id)
        .unwrap().input_commitment.clone();
    (code, image_commitment)
}

// FIXME(Ben) document
/// * `image` is unpadded
pub fn prove_single(
    version: u8,
    mask: bool,
    image_commitment: SerializedImageCommitment,
    committer: &PedersenCommitter<Bn256Point>,
    blinding_rng: &mut rand::rngs::ThreadRng,
    converter: &mut VandermondeInverse<Scalar>,
) -> HyraxProof<Bn256Point> {
    // Build the circuit description and calculate inputs (including the iris/mask code).
    let (proof_desc, inputs) = circuit_description_and_inputs(version, mask, Some(image_commitment.image.clone()));
    use crate::hyrax_gkr::hyrax_input_layer::HyraxProverInputCommitment;

    // Set up Hyrax input layer specification.
    let mut hyrax_input_layers = HashMap::new();
    // The image, with the precommit (must use the same number of columns as were used at the time of committing!)
    let image_hyrax_input_layer_desc = HyraxInputLayerDescription {
        layer_id: proof_desc.image_input_layer.layer_id,
        num_bits: proof_desc.image_input_layer.num_vars,
        log_num_cols: IMAGE_COMMIT_LOG_NUM_COLS,
    };
    hyrax_input_layers.insert(
        proof_desc.image_input_layer.layer_id,
        (image_hyrax_input_layer_desc, Some(image_commitment.into())),
    );
    // The digit multiplicities, without the precommit
    hyrax_input_layers.insert(
        proof_desc.digits_input_layer.layer_id,
        (proof_desc.digits_input_layer.clone().into(), None),
    );

    // Create a fresh transcript.
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    // Prove the relationship between iris/mask code and image.
    HyraxProof::prove(
        &inputs,
        &hyrax_input_layers,
        &proof_desc.circuit_description,
        &committer,
        blinding_rng,
        converter,
        &mut transcript,
    )
}

/// Prove the upgrade from v2 to v3 using the Hyrax proof system. Receives the image, commitment and
/// blinding for each combination of version, eye, and type (iris or mask), and returns, for each
/// such combination, the corresponding [HyraxProof].
pub fn prove_upgrade_v2_to_v3(
    data: &HashMap<(u8, bool, bool), SerializedImageCommitment>,
) -> HashMap<(u8, bool, bool), HyraxProof<Bn256Point>> {
    // Create the Pedersen committer using the same reference string and parameters as on the Orb
    let committer: PedersenCommitter<Bn256Point> = PedersenCommitter::new(1 << IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING, None);
    // Create a single RNG and Vandermonde inverse converter for all proofs.
    let blinding_rng = &mut rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();

    let mut proofs = HashMap::new();
    for version in [2u8, 3u8] {
        for mask in [false, true] {
            for left_eye in [false, true] {
                dbg!(version, mask, left_eye);
                let commitment = data.get(&(version, left_eye, mask)).unwrap();
                let proof = prove_single(
                    version,
                    mask,
                    commitment.clone(),
                    &committer,
                    blinding_rng,
                    converter,
                );
                proofs.insert((version, mask, left_eye), proof);
            }
        }
    }
    proofs
}