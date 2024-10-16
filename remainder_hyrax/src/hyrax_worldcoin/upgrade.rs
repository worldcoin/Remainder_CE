#![allow(warnings)]
use std::collections::HashMap;
use std::hash::Hash;

use ff::Field;
use remainder::prover::GKRCircuitDescription;
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
use crate::hyrax_worldcoin::orb::{deserialize_blinding_factors_from_bytes_compressed, deserialize_commitment_from_bytes_compressed, LOG_NUM_COLS, PUBLIC_STRING};
use crate::utils::vandermonde::VandermondeInverse;

type Scalar = Fr;
type Base = Fq;

#[derive(Debug)]
pub enum UpgradeError {
}

/// Verify the upgrade from v2 to v3 using the Hyrax proof system. Receives the [HyraxProof] for each combination of
/// version, eye, and type (iris or mask), and returns, for each such combination, the corresponding mask or iris code.
pub fn verify_upgrade_v2_to_v3(
    proofs: &HashMap<(u8, bool, bool), HyraxProof<Bn256Point>>,
) -> Result<HashMap<(u8, bool, bool), Vec<bool>>, UpgradeError> {
    // Create the Pedersen committer using the same reference string and parameters as on the Orb
    let committer: PedersenCommitter<Bn256Point> = PedersenCommitter::new(1 << LOG_NUM_COLS, PUBLIC_STRING, None);

    let mut codes = HashMap::new();
    for version in [2u8, 3u8] {
        for mask in [false, true] {
            for left_eye in [false, true] {
                dbg!(version, mask, left_eye);
                // Get the circuit description, as well as the inputs if we were using a test image;
                // we'll use these to ensure that the correct kernel values and thresholds are being
                // used in the supplied proof.
                let (proof_desc, test_inputs) = circuit_description_and_inputs(version, mask, None);

                let mut hyrax_input_layers = HashMap::new();
                hyrax_input_layers.insert(
                    proof_desc.image_input_layer.layer_id,
                    proof_desc.image_input_layer.clone().into(),
                );
                hyrax_input_layers.insert(
                    proof_desc.digits_input_layer.layer_id,
                    proof_desc.digits_input_layer.clone().into(),
                );

                // Create a fresh transcript.
                let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
                    ECTranscript::new("modulus modulus modulus modulus modulus");
                // Verify the relationship between iris/mask code and image.
                let proof = proofs.get(&(version, mask, left_eye)).unwrap();
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
                let code: Vec<bool> = code_mle
                    .iter()
                    .map(|b|
                        match b {
                            &Fr::ONE => true,
                            &Fr::ZERO => false,
                            _ => panic!() // FIXME raise error
                        }
                    ).collect();
                codes.insert((version, mask, left_eye), code);
            }
        }
    }
    Ok(codes)
}

/// Prove the upgrade from v2 to v3 using the Hyrax proof system. Receives the image, commitment and
/// blinding for each combination of version, eye, and type (iris or mask), and returns, for each
/// such combination, the corresponding [HyraxProof].
pub fn prove_upgrade_v2_to_v3(
    data: &HashMap<(u8, bool, bool), (Vec<u8>, Vec<u8>, Vec<u8>)>,
) -> HashMap<(u8, bool, bool), HyraxProof<Bn256Point>> {
    // Create the Pedersen committer using the same reference string and parameters as on the Orb
    let committer: PedersenCommitter<Bn256Point> = PedersenCommitter::new(1 << LOG_NUM_COLS, PUBLIC_STRING, None);
    // Create a single RNG and Vandermonde inverse converter for all proofs.
    let blinding_rng = &mut rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();

    let mut proofs = HashMap::new();
    for version in [2u8, 3u8] {
        for mask in [false, true] {
            for left_eye in [false, true] {
                dbg!(version, mask, left_eye);
                let (image, commitment_bytes, blinding_factors_bytes) = data.get(&(version, left_eye, mask)).unwrap();
                // Deserialize the commitment and blinding factors.
                let commitment = deserialize_commitment_from_bytes_compressed(&commitment_bytes);
                let blinding_factors_matrix = deserialize_blinding_factors_from_bytes_compressed::<Bn256Point>(&blinding_factors_bytes);
                // Build the circuit description and calculate inputs (including the iris/mask code).
                let (proof_desc, inputs) = circuit_description_and_inputs(version, mask, Some(image.to_vec()));
                use crate::hyrax_gkr::hyrax_input_layer::HyraxProverInputCommitment;
                // Rebuild the image precommitment.
                let image_precommit = HyraxProverInputCommitment {
                    mle: MleCoefficientsVector::<Bn256Point>::U8Vector(image.to_vec()),
                    commitment: commitment,
                    blinding_factors_matrix: blinding_factors_matrix,
                };

                // Set up Hyrax input layer specification.
                let mut hyrax_input_layers = HashMap::new();
                // The image, with the precommit
                hyrax_input_layers.insert(
                    proof_desc.image_input_layer.layer_id,
                    (proof_desc.image_input_layer.clone().into(), Some(image_precommit)),
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
                let proof = HyraxProof::prove(
                    &inputs,
                    &hyrax_input_layers,
                    &proof_desc.circuit_description,
                    &committer,
                    blinding_rng,
                    converter,
                    &mut transcript,
                );
                proofs.insert((version, mask, left_eye), proof);
            }
        }
    }
    proofs
}