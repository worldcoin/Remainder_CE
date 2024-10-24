use std::collections::HashMap;

use remainder::{
    layer::LayerId,
    mle::evals::MultilinearExtension,
    worldcoin::{circuits::IriscodeProofDescription, test_helpers::circuit_description_and_inputs},
};

use remainder_shared_types::{
    halo2curves::bn256::G1 as Bn256Point,
    pedersen::PedersenCommitter,
    transcript::{ec_transcript::ECTranscript, poseidon_transcript::PoseidonSponge},
    Base, Scalar,
};

use crate::{hyrax_gkr::HyraxProof, utils::vandermonde::VandermondeInverse};

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{hyrax_gkr::HyraxProof, hyrax_worldcoin::{test_worldcoin::{test_iriscode_circuit_with_hyrax_helper, test_iriscode_circuit_with_public_layers_helper}, upgrade::{prove_upgrade_v2_to_v3, verify_upgrade_v2_to_v3}}, utils::vandermonde::VandermondeInverse};
    use remainder::
        worldcoin::{parameters_v2::IRISCODE_LEN as V2_IRISCODE_LEN, parameters_v3::IRISCODE_LEN as V3_IRISCODE_LEN}
    ;
    use remainder_shared_types::{
        halo2curves::bn256::G1 as Bn256Point,
        pedersen::PedersenCommitter,
        Scalar,
    };

    use super::{super::{orb::{load_image_commitment, SerializedImageCommitment, IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING}, upgrade::{prove_with_image_precommit, verify_iriscode}}, test_iriscode_v2_with_hyrax_helper, test_iriscode_v3_with_hyrax_helper};

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
        let is_left_eye = true;
        let is_mask = false;
        // Create the Pedersen committer using the same reference string and parameters as on the Orb
        let committer: PedersenCommitter<Bn256Point> = PedersenCommitter::new(1 << IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING, None);
        // Create a single RNG and Vandermonde inverse converter for all proofs.
        let blinding_rng = &mut rand::thread_rng();
        let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
        let proof = prove_with_image_precommit(
            version,
            is_mask,
            load_image_commitment(version, is_mask, is_left_eye),
            &committer,
            blinding_rng,
            converter,
        );
        let (code, _commitment) = verify_iriscode(version, is_mask, is_left_eye, &proof, &committer).unwrap();
        assert_eq!(code.len(), V2_IRISCODE_LEN);
    }

    #[ignore] // Takes a long time to run
    #[test]
    fn test_upgrade_v2_v3() {
        use sha256::digest as sha256_digest;

        let mut commitments: HashMap<(u8, bool, bool), SerializedImageCommitment> = HashMap::new();
        for version in 2..=3 {
            for mask in [false, true] {
                for left_eye in [false, true] {
                    let serialized_commitment = load_image_commitment(version, mask, left_eye);
                    commitments.insert((version, mask, left_eye), serialized_commitment);
                }
            }
        }
        let proofs = prove_upgrade_v2_to_v3(&commitments.clone());

        // Get expected hashes for the commitments.
        // In production, the verifier should be obtaining the hashes from the signed hashes.json file.
        let mut proofs_and_hashes: HashMap<(u8, bool, bool), (HyraxProof<Bn256Point>, String)> = HashMap::new();
        for ((version, mask, left_eye), proof) in proofs {
            let serialized_commitment = commitments.get(&(version, mask, left_eye)).unwrap();
            let hash = sha256_digest(&serialized_commitment.commitment_bytes.clone());
            proofs_and_hashes.insert((version, mask, left_eye),
                ( proof, hash)
            );
        }

        let results = verify_upgrade_v2_to_v3(&proofs_and_hashes).unwrap();
        for version in 2..=3 {
            for mask in [false, true] {
                for left_eye in [false, true] {
                    let code = results.get(&(version, mask, left_eye)).unwrap();
                    assert_eq!(code.len(), if version == 2 { V2_IRISCODE_LEN } else { V3_IRISCODE_LEN });
                }
            }
        }
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
