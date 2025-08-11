use std::collections::HashMap;

use crate::zk_iriscode_ss::{
    circuits::IriscodeCircuitDescription, v3::circuit_description_and_inputs,
};
use ark_std::{end_timer, start_timer};
use remainder::{
    circuit_layout::ProvableCircuit, layer::LayerId, mle::evals::MultilinearExtension,
};

use remainder_shared_types::{
    config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig},
    halo2curves::bn256::G1 as Bn256Point,
    pedersen::PedersenCommitter,
    perform_function_under_prover_config, perform_function_under_verifier_config,
    transcript::{ec_transcript::ECTranscript, poseidon_sponge::PoseidonSponge},
    Base, Scalar,
};

use remainder_hyrax::{
    hyrax_gkr::{verify_hyrax_proof, HyraxProof},
    utils::vandermonde::VandermondeInverse,
};

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::{
        hyrax_worldcoin::{
            test_worldcoin::{
                test_iriscode_circuit_with_hyrax_helper,
                test_iriscode_circuit_with_public_layers_helper,
            },
            v3::verify_v3_iriscode_proof_and_hash,
        },
        zk_iriscode_ss::{
            circuits::iriscode_ss_attach_data,
            v3::{circuit_description_and_input_builder, load_worldcoin_data},
        },
    };
    use remainder_hyrax::{
        hyrax_gkr::hyrax_input_layer::HyraxProverInputCommitment,
        utils::vandermonde::VandermondeInverse,
    };
    use remainder_shared_types::{
        config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig},
        curves::PrimeOrderCurve,
        halo2curves::bn256::G1 as Bn256Point,
        pedersen::PedersenCommitter,
        perform_function_under_prover_config, perform_function_under_verifier_config, Fr, Scalar,
    };

    use super::{
        super::{
            orb::{load_image_commitment, IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING},
            v3::prove_with_image_precommit,
        },
        test_iriscode_v3_with_hyrax_helper,
    };

    #[test]
    fn test_small_circuit_both_layers_public() {
        use crate::zk_iriscode_ss::test_helpers::small_circuit_description_and_inputs;
        let provable_circuit = small_circuit_description_and_inputs(false).unwrap();
        test_iriscode_circuit_with_public_layers_helper(provable_circuit);
    }

    #[test]
    /// Test a small version of the iriscode circuit with a Hyrax input layer.
    fn test_small_circuit_with_hyrax_layer() {
        use crate::zk_iriscode_ss::test_helpers::small_circuit_description_and_inputs;
        let (ic_circuit_desc, inputs) = small_circuit_description_and_inputs();
        test_iriscode_circuit_with_hyrax_helper(ic_circuit_desc, inputs);
    }

    #[ignore] // Takes a long time to run
    #[test]
    // Test the proving and verifying of the v3 iriscode circuit with the image precommit;
    // verification includes checking the hash. This is testing the fundamental prove and verify
    // functions that will be used by the user/smartphone and the server.
    fn test_v3_masked_iriscode_proof_and_verification() {
        use sha256::digest as sha256_digest;
        // Get the proof description and input builder.
        // This is shared by the prover and the verifier.
        let (ic_circuit_desc, input_builder_metadata) = circuit_description_and_input_builder();

        // Create the Pedersen committer using the same reference string and parameters as on the Orb
        let committer: PedersenCommitter<Bn256Point> =
            PedersenCommitter::new(1 << IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING, None);
        // Create a single RNG and Vandermonde inverse converter for all proofs.
        let blinding_rng = &mut rand::thread_rng();
        let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();

        // Prover config is Hyrax-compatible, memory-efficient config
        let gkr_circuit_prover_config =
            GKRCircuitProverConfig::hyrax_compatible_memory_optimized_default();
        let gkr_circuit_verifier_config =
            GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

        for is_mask in [false, true] {
            for is_left_eye in [false, true] {
                // Get the pre-existing commitment to the image.
                let serialized_image_commitment = load_image_commitment(
                    &Path::new("iriscode_pcp_example").to_path_buf(),
                    3,
                    is_mask,
                    is_left_eye,
                );
                let image_commitment: HyraxProverInputCommitment<Bn256Point> =
                    serialized_image_commitment.clone().into();
                // Derive the hash of the image commitment.  In production, this has been calculated by the Orb and sent over in a signed file.
                let expected_commitment_hash = sha256_digest(
                    &image_commitment
                        .commitment
                        .iter()
                        .flat_map(|p| p.to_bytes_compressed())
                        .collect::<Vec<u8>>(),
                );

                // Load the inputs to the circuit (these are all MLEs, i.e. in the clear).
                let data = load_worldcoin_data::<Fr>(serialized_image_commitment.image, is_mask);
                let inputs = iriscode_ss_attach_data(&input_builder_metadata, data);

                // Extract the auxiliary public inputs for later use by the verifier.
                // In production, the verifier will have these already.
                let auxiliary_mle = inputs
                    .get(&ic_circuit_desc.auxiliary_input_layer.layer_id)
                    .unwrap()
                    .clone();

                // Prove the iriscode circuit with the image precommit.
                // In production, the prover needs to retain `prover_code_commit`, since it will need this
                // to prove the MPC circuit later on.
                let (proof, proof_config, _image_commit) = perform_function_under_prover_config!(
                    prove_with_image_precommit,
                    &gkr_circuit_prover_config,
                    &ic_circuit_desc,
                    inputs,
                    image_commitment.into(),
                    &committer,
                    blinding_rng,
                    converter
                );

                let (_code_commitment, _image_commitment) =
                    perform_function_under_verifier_config!(
                        verify_v3_iriscode_proof_and_hash,
                        &gkr_circuit_verifier_config,
                        &proof,
                        &ic_circuit_desc,
                        &auxiliary_mle,
                        &expected_commitment_hash,
                        &committer,
                        &proof_config
                    )
                    .unwrap();
            }
        }
    }

    #[ignore] // Takes a long time to run
    #[test]
    fn test_v3_iris_with_hyrax_layer() {
        test_iriscode_v3_with_hyrax_helper(false);
    }

    #[ignore] // Takes a long time to run
    #[test]
    fn test_v3_mask_with_hyrax_layer() {
        test_iriscode_v3_with_hyrax_helper(true);
    }
}

/// Test the iriscode circuit v3 with a Hyrax input layer in either the mask (true) or iris (false)
/// case.
pub fn test_iriscode_v3_with_hyrax_helper(mask: bool) {
    let (desc, inputs) = circuit_description_and_inputs(mask, None);
    test_iriscode_circuit_with_hyrax_helper(desc, inputs);
}

/// Helper function for testing an iriscode circuit (with any data) with a Hyrax input layer.
pub fn test_iriscode_circuit_with_public_layers_helper(provable_circuit: ProvableCircuit<Scalar>) {
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

    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    // --- Compute actual Hyrax proof ---
    let (proof, proof_config) = perform_function_under_prover_config!(
        HyraxProof::prove,
        &gkr_circuit_prover_config,
        provable_circuit.get_inputs_ref(),
        &HashMap::new(),
        provable_circuit.get_gkr_circuit_description_ref(),
        &committer,
        blinding_rng,
        converter,
        &mut transcript
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    // Verify.
    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &gkr_circuit_verifier_config,
        &proof,
        &HashMap::new(),
        provable_circuit.get_gkr_circuit_description_ref(),
        &committer,
        &mut transcript,
        &proof_config
    );
}

/// Helper function for testing an iriscode circuit (with any data) with Hyrax input
/// layers for the private data.
pub fn test_iriscode_circuit_with_hyrax_helper(
    ic_circuit_desc: IriscodeCircuitDescription<Scalar>,
    inputs: HashMap<LayerId, MultilinearExtension<Scalar>>,
) {
    // Print the layer IDs for debugging.
    println!("Input layer ids:");
    println!("Image: {:?}", ic_circuit_desc.image_input_layer.layer_id);
    println!("Digits: {:?}", ic_circuit_desc.digits_input_layer.layer_id);
    println!(
        "Auxiliary: {:?}",
        ic_circuit_desc.auxiliary_input_layer.layer_id
    );
    println!("Code: {:?}", ic_circuit_desc.code_input_layer.layer_id);

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
        ic_circuit_desc.image_input_layer.layer_id,
        (ic_circuit_desc.image_input_layer.clone().into(), None),
    );
    prover_hyrax_input_layers.insert(
        ic_circuit_desc.code_input_layer.layer_id,
        (ic_circuit_desc.code_input_layer.clone().into(), None),
    );
    prover_hyrax_input_layers.insert(
        ic_circuit_desc.digits_input_layer.layer_id,
        (ic_circuit_desc.digits_input_layer.clone().into(), None),
    );
    // Prove.
    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_memory_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    // --- Compute actual Hyrax proof ---
    let prove_timer = start_timer!(|| "Proving");
    let (proof, proof_config) = perform_function_under_prover_config!(
        HyraxProof::prove,
        &gkr_circuit_prover_config,
        &inputs,
        &prover_hyrax_input_layers,
        &ic_circuit_desc.circuit_description,
        &committer,
        blinding_rng,
        converter,
        &mut transcript
    );
    end_timer!(prove_timer);

    // Verify.
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let mut verifier_hyrax_input_layers = HashMap::new();
    verifier_hyrax_input_layers.insert(
        ic_circuit_desc.image_input_layer.layer_id,
        ic_circuit_desc.image_input_layer.clone().into(),
    );
    verifier_hyrax_input_layers.insert(
        ic_circuit_desc.code_input_layer.layer_id,
        ic_circuit_desc.code_input_layer.clone().into(),
    );
    verifier_hyrax_input_layers.insert(
        ic_circuit_desc.digits_input_layer.layer_id,
        ic_circuit_desc.digits_input_layer.clone().into(),
    );
    let verification_timer = start_timer!(|| "verification timer");
    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &gkr_circuit_verifier_config,
        &proof,
        &verifier_hyrax_input_layers,
        &ic_circuit_desc.circuit_description,
        &committer,
        &mut transcript,
        &proof_config
    );
    end_timer!(verification_timer);
}
