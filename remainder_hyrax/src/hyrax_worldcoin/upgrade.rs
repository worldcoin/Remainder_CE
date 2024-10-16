#![allow(warnings)]
use std::collections::HashMap;
use std::hash::Hash;

use remainder::prover::GKRCircuitDescription;
use remainder::worldcoin::test_helpers::{v2_circuit_description_and_inputs, v3_circuit_description_and_inputs};
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

pub enum UpgradeError {}

type Scalar = Fr;
type Base = Fq;

/// Prove the upgrade from v2 to v3 using the Hyrax proof system. Receives the image, commitment and
/// blinding for each combination of version, eye, and type (iris or mask), and returns, for each
/// such combination, the transcript and the corresponding iris/mask code (in the clear, as bytes).
pub fn prove_upgrade_v2_to_v3(
    data: &HashMap<(u8, bool, bool), (&[u8], &[u8], &[u8])>,
) -> Result<HashMap<(u8, bool, bool), HyraxProof<Bn256Point>>, UpgradeError> {
    // Create the Pedersen committer using the same reference string and parameters as on the Orb
    let committer: PedersenCommitter<Bn256Point> = PedersenCommitter::new(1 << LOG_NUM_COLS, PUBLIC_STRING, None);

    let blinding_rng = &mut rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    
    let mut results = HashMap::new();

    for version in [2u8, 3u8] {
        let circuit_builder = match version {
            2 => v2_circuit_description_and_inputs,
            3 => v3_circuit_description_and_inputs,
            _ => unreachable!(),
        };
        for mask in [false, true] {
            for left_eye in [false, true] {
                let (image, commitment_bytes, blinding_factors_bytes) = data.get(&(version, left_eye, mask)).unwrap();
                // Build the circuit description and calculate inputs (including the iris/mask code)
                let (desc, priv_layer_desc, inputs) = circuit_builder(mask, image.to_vec());
                use crate::hyrax_gkr::hyrax_input_layer::HyraxProverInputCommitment;
                // Rebuild the image precommitment
                let image_precommit = HyraxProverInputCommitment {
                    mle: MleCoefficientsVector::<Bn256Point>::U8Vector(image.to_vec()),
                    commitment: deserialize_commitment_from_bytes_compressed(&commitment_bytes),
                    blinding_factors_matrix: deserialize_blinding_factors_from_bytes_compressed::<Bn256Point>(&blinding_factors_bytes)
                };
                // Set up Hyrax input layer specification
                let mut prover_hyrax_input_layers = HashMap::new();
                let hyrax_input_layer_desc: HyraxInputLayerDescription = priv_layer_desc.into();
                prover_hyrax_input_layers.insert(
                    hyrax_input_layer_desc.layer_id,
                    (hyrax_input_layer_desc.clone(), Some(image_precommit)),
                );
                // Create a fresh transcript
                let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
                    ECTranscript::new("modulus modulus modulus modulus modulus");
                // Prove the relationship between iris/mask code and image
                let proof = HyraxProof::prove(
                    &inputs,
                    &prover_hyrax_input_layers,
                    &desc,
                    &committer,
                    blinding_rng,
                    converter,
                    &mut transcript,
                );
                results.insert((version, mask, left_eye), proof);
            }
        }
    }
    Ok(results)
}