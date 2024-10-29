#![allow(warnings)]
use std::collections::HashMap;
use std::hash::Hash;

use clap::error;
use ff::Field;
use remainder::prover::GKRCircuitDescription;
use remainder::utils::mle::pad_with;
use remainder::worldcoin::parameters_v2::IRISCODE_LEN as V2_IRISCODE_LEN;
use remainder::worldcoin::parameters_v3::IRISCODE_LEN as V3_IRISCODE_LEN;
use remainder::worldcoin::test_helpers::{
    circuit_description_and_inputs, v2_circuit_description_and_inputs,
    v3_circuit_description_and_inputs,
};
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::halo2curves::{bn256::G1 as Bn256Point, group::Group};
use remainder_shared_types::pedersen::PedersenCommitter;
use remainder_shared_types::transcript::ec_transcript::{ECTranscript, ECTranscriptTrait};
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use remainder_shared_types::transcript::Transcript;
use remainder_shared_types::{Fq, Fr};

use super::orb::SerializedImageCommitment;
use crate::hyrax_gkr::hyrax_input_layer::HyraxInputLayerDescription;
use crate::hyrax_gkr::{self, HyraxProof};
use crate::hyrax_pcs::MleCoefficientsVector;
use crate::hyrax_worldcoin::orb::{IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING};
use crate::utils::vandermonde::VandermondeInverse;
use sha256::digest as sha256_digest;
use thiserror::Error;

type Scalar = Fr;
type Base = Fq;

#[derive(Debug, Error)]
pub enum UpgradeError {
    #[error("Non-zero padding bits in iris/mask code in version {0} circuit, is_mask={1}, left_eye={2}.")]
    NonZeroPaddingBits(u8, bool, bool),
    #[error("Non-binary iris/mask code in version {0} circuit, is_mask={1}, left_eye={2}.")]
    NonBinaryIrisMaskCode(u8, bool, bool),
    #[error(
        "Incorrect kernel values or thresholds in version {0} circuit, is_mask={1}, left_eye={2}."
    )]
    IncorrectKernelValuesOrThresholds(u8, bool, bool),
    #[error("Image commitment does not match expected hash in version {0} circuit, is_mask={1}, left_eye={2}.")]
    WrongHash(u8, bool, bool),
}

/// Verify the upgrade from v2 to v3 using the Hyrax proof system. Receives the [HyraxProof] for each combination of
/// version, type (iris or mask) and eye, and returns, for each such combination, the corresponding mask or iris code.
/// Checks that the image commitment matches the expected hash.
pub fn verify_upgrade_v2_to_v3(
    proofs_and_hashes: &HashMap<(u8, bool, bool), (HyraxProof<Bn256Point>, String)>,
) -> Result<HashMap<(u8, bool, bool), Vec<bool>>, UpgradeError> {
    // Create the Pedersen committer using the same reference string and parameters as on the Orb
    let committer: PedersenCommitter<Bn256Point> =
        PedersenCommitter::new(1 << IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING, None);

    let mut results = HashMap::new();
    for version in [2u8, 3u8] {
        for is_mask in [false, true] {
            for is_left_eye in [false, true] {
                let (proof, expected_hash) = proofs_and_hashes
                    .get(&(version, is_mask, is_left_eye))
                    .unwrap();
                match verify_iriscode(version, is_mask, is_left_eye, proof, &committer) {
                    Ok(result) => {
                        let (code, commitment) = result;
                        // Check that the image commitment matches the expected hash.
                        let commitment_hash = sha256_digest(
                            &commitment
                                .iter()
                                .flat_map(|p| p.to_bytes_compressed())
                                .collect::<Vec<u8>>(),
                        );
                        if commitment_hash != *expected_hash {
                            return Err(UpgradeError::WrongHash(version, is_left_eye, is_mask));
                        }
                        results.insert((version, is_mask, is_left_eye), code);
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
        }
    }
    Ok(results)
}

/// Verify that the iris/mask code in the supplied proof is correct, and return the unpadded iris/mask code, along with the
/// commitment to the image.
/// Checks, in particular:
/// * That the [HyraxProof] verifies.
/// * That the correct kernel values and thresholds are being used in the supplied proof.
/// * That the MLE encoding the iris/mask code has only 0s in the padding region.
/// * That the unpadded iris/mask code consists only of 0s and 1s.
/// This is a helper function for the upgrade from v2 to v3.
pub(crate) fn verify_iriscode(
    version: u8,
    is_mask: bool,
    is_left_eye: bool,
    proof: &HyraxProof<Bn256Point>,
    committer: &PedersenCommitter<Bn256Point>,
) -> Result<(Vec<bool>, Vec<Bn256Point>), UpgradeError> {
    assert!(version == 2 || version == 3);
    // Get the circuit description, as well as the inputs if we were using a test image;
    // we'll use these to ensure that the correct kernel values and thresholds are being
    // used in the supplied proof.
    let (proof_desc, test_inputs) = circuit_description_and_inputs(version, is_mask, None);

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
    let expected_aux_mle = test_inputs
        .get(&proof_desc.auxiliary_input_layer.layer_id)
        .unwrap();
    let aux_mle_in_proof = proof
        .public_inputs
        .iter()
        .find(|(id, _)| id == &proof_desc.auxiliary_input_layer.layer_id)
        .unwrap()
        .1
        .clone();
    if *expected_aux_mle != aux_mle_in_proof {
        return Err(UpgradeError::IncorrectKernelValuesOrThresholds(
            version,
            is_left_eye,
            is_mask,
        ));
    }

    // Extract the iris/mask code from the proof, and convert to Vec<bool>.
    let code_mle = proof
        .public_inputs
        .iter()
        .find(|(id, _)| id == &proof_desc.code_input_layer.layer_id)
        .unwrap()
        .1
        .to_vec();
    let code: Vec<bool> = code_mle
        .iter()
        .map(|b| match b {
            &Fr::ONE => Ok(true),
            &Fr::ZERO => Ok(false),
            _ => Err(UpgradeError::NonBinaryIrisMaskCode(
                version,
                is_left_eye,
                is_mask,
            )),
        })
        .collect::<Result<Vec<bool>, UpgradeError>>()?;
    // Check that the MLE encoding the code used only 0s in the padding region
    let code_len = if version == 2 {
        V2_IRISCODE_LEN
    } else {
        V3_IRISCODE_LEN
    };
    for &b in &code[code_len..] {
        if b {
            return Err(UpgradeError::NonZeroPaddingBits(
                version,
                is_left_eye,
                is_mask,
            ));
        }
    }

    // Extract the commitment to the image from the proof.
    let image_commitment = proof
        .hyrax_input_proofs
        .iter()
        .find(|proof| proof.layer_id == proof_desc.image_input_layer.layer_id)
        .unwrap()
        .input_commitment
        .clone();

    // Return the unpadded code and the commitment to the image.
    Ok((code[..code_len].to_vec(), image_commitment))
}

/// Prove a single instance of the iriscode circuit using the Hyrax proof system, using the provided
/// precommitment for the image.
/// This is a helper function for the upgrade from v2 to v3.
pub(crate) fn prove_with_image_precommit(
    version: u8,
    is_mask: bool,
    image_commitment: SerializedImageCommitment,
    committer: &PedersenCommitter<Bn256Point>,
    blinding_rng: &mut rand::rngs::ThreadRng,
    converter: &mut VandermondeInverse<Scalar>,
) -> HyraxProof<Bn256Point> {
    // Build the circuit description and calculate inputs (including the iris/mask code).
    let (proof_desc, inputs) =
        circuit_description_and_inputs(version, is_mask, Some(image_commitment.image.clone()));
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

/// Prove the upgrade from v2 to v3 using the Hyrax proof system. Receives a [SerializedImageCommitment]
/// for each combination of version, type (iris or mask) and eye, and returns, for each
/// such combination, the corresponding [HyraxProof].
/// (This is a convenience function for the mobile app).
pub fn prove_upgrade_v2_to_v3(
    data: &HashMap<(u8, bool, bool), SerializedImageCommitment>,
) -> HashMap<(u8, bool, bool), HyraxProof<Bn256Point>> {
    // Create the Pedersen committer using the same reference string and parameters as on the Orb
    let committer: PedersenCommitter<Bn256Point> =
        PedersenCommitter::new(1 << IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING, None);
    // Create a single RNG and Vandermonde inverse converter for all proofs.
    let blinding_rng = &mut rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();

    let mut proofs = HashMap::new();
    // We iterate explicitly over the expected keys, to ensure that the HashMap is complete.
    for version in [2u8, 3u8] {
        for is_mask in [false, true] {
            for is_left_eye in [false, true] {
                let commitment = data.get(&(version, is_mask, is_left_eye)).unwrap();
                let proof = prove_with_image_precommit(
                    version,
                    is_mask,
                    commitment.clone(),
                    &committer,
                    blinding_rng,
                    converter,
                );
                proofs.insert((version, is_mask, is_left_eye), proof);
            }
        }
    }
    proofs
}
