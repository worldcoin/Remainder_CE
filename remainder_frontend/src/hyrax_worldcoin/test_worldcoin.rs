use crate::zk_iriscode_ss::v3::circuit_description_and_inputs;
use ark_std::{end_timer, start_timer};

use remainder_shared_types::{
    config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig},
    halo2curves::bn256::G1 as Bn256Point,
    pedersen::PedersenCommitter,
    perform_function_under_prover_config, perform_function_under_verifier_config,
    transcript::{ec_transcript::ECTranscript, poseidon_sponge::PoseidonSponge},
    Base, Scalar,
};

use remainder_hyrax::{
    circuit_layout::HyraxProvableCircuit, hyrax_gkr::verify_hyrax_proof,
    utils::vandermonde::VandermondeInverse,
};

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        super::orb::{load_image_commitment, IMAGE_COMMIT_LOG_NUM_COLS, PUBLIC_STRING},
        test_iriscode_v3_with_hyrax_helper,
    };
    use crate::{
        hyrax_worldcoin::{
            test_worldcoin::{
                test_iriscode_circuit_with_hyrax_helper,
                test_iriscode_circuit_with_public_layers_helper,
            },
            v3::verify_v3_iriscode_proof_and_hash,
        },
        layouter::builder::Circuit,
        zk_iriscode_ss::{
            circuits::{iriscode_ss_attach_input_data, V3_INPUT_IMAGE_LAYER, V3_SIGN_BITS_LAYER},
            test_helpers::{
                small_hyrax_circuit_with_private_inputs, small_hyrax_circuit_with_public_inputs,
            },
            v3::{build_worldcoin_aux_data, circuit_description, load_worldcoin_data},
        },
    };
    use rand::rngs::ThreadRng;
    use remainder_hyrax::{
        circuit_layout::HyraxVerifiableCircuit,
        hyrax_gkr::{hyrax_input_layer::HyraxProverInputCommitment, HyraxProof},
        utils::vandermonde::VandermondeInverse,
    };
    use remainder_shared_types::{
        config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig, ProofConfig},
        curves::PrimeOrderCurve,
        halo2curves::bn256::G1 as Bn256Point,
        pedersen::PedersenCommitter,
        perform_function_under_prover_config, perform_function_under_verifier_config,
        transcript::{ec_transcript::ECTranscript, poseidon_sponge::PoseidonSponge},
        Base, Fr, Scalar,
    };

    #[test]
    fn test_small_circuit_both_layers_public() {
        let provable_circuit = small_hyrax_circuit_with_public_inputs().unwrap();
        test_iriscode_circuit_with_public_layers_helper(provable_circuit);
    }

    #[test]
    /// Test a small version of the iriscode circuit with a Hyrax input layer.
    fn test_small_circuit_with_hyrax_layer() {
        let provable_circuit = small_hyrax_circuit_with_private_inputs().unwrap();
        test_iriscode_circuit_with_hyrax_helper(provable_circuit);
    }

    fn v3_masked_iriscode_prove(
        is_mask: bool,
        is_left_eye: bool,
        committer: &PedersenCommitter<Bn256Point>,
        blinding_rng: &mut ThreadRng,
        converter: &mut VandermondeInverse<Scalar>,
        ic_circuit: Circuit<Fr>,
    ) -> (
        HyraxVerifiableCircuit<Bn256Point>,
        HyraxProof<Bn256Point>,
        ProofConfig,
        String,
        HyraxProverInputCommitment<Bn256Point>,
        HyraxProverInputCommitment<Bn256Point>,
    ) {
        use sha256::digest;

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
        let expected_commitment_hash = digest(
            &image_commitment
                .commitment
                .iter()
                .flat_map(|p| p.to_bytes_compressed())
                .collect::<Vec<u8>>(),
        );

        // Load the inputs to the circuit (these are all MLEs, i.e. in the clear).
        let input_data = load_worldcoin_data::<Fr>(serialized_image_commitment.image, is_mask);
        let aux_data = build_worldcoin_aux_data::<Fr>(is_mask);
        let circuit =
            iriscode_ss_attach_input_data::<_, { crate::zk_iriscode_ss::parameters::BASE }>(
                ic_circuit.clone(),
                input_data,
                aux_data,
            )
            .unwrap();

        let mut provable_circuit = circuit.gen_hyrax_provable_circuit().unwrap();

        provable_circuit
            .set_pre_commitment(
                V3_INPUT_IMAGE_LAYER,
                image_commitment.into(),
                Some(IMAGE_COMMIT_LOG_NUM_COLS),
            )
            .unwrap();

        let verifiable_circuit = provable_circuit._gen_hyrax_verifiable_circuit();

        // Create a fresh transcript.
        let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
            ECTranscript::new("V3 Iriscode Circuit Pipeline");

        // Prove the relationship between iris/mask code and image.
        let (proof, proof_config) =
            provable_circuit.prove(&committer, blinding_rng, converter, &mut transcript);

        let code_commit = provable_circuit
            .get_commitment_ref_by_label(V3_SIGN_BITS_LAYER)
            .unwrap()
            .clone();
        let image_commit = provable_circuit
            .get_commitment_ref_by_label(V3_INPUT_IMAGE_LAYER)
            .unwrap()
            .clone();

        (
            verifiable_circuit,
            proof,
            proof_config,
            expected_commitment_hash,
            image_commit,
            code_commit,
        )
    }

    #[ignore] // Takes a long time to run
    #[test]
    // Test the proving and verifying of the v3 iriscode circuit with the image precommit;
    // verification includes checking the hash. This is testing the fundamental prove and verify
    // functions that will be used by the user/smartphone and the server.
    fn test_v3_masked_iriscode_proof_and_verification() {
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

        // Get the proof description and input builder.
        // This is shared by the prover and the verifier.
        let ic_circuit =
            perform_function_under_prover_config!(circuit_description, &gkr_circuit_prover_config,)
                .unwrap();

        for is_mask in [false, true] {
            for is_left_eye in [false, true] {
                let (
                    verifiable_circuit,
                    proof,
                    proof_config,
                    expected_commitment_hash,
                    _image_commit,
                    _code_commit,
                ) = perform_function_under_prover_config!(
                    v3_masked_iriscode_prove,
                    &gkr_circuit_prover_config,
                    is_mask,
                    is_left_eye,
                    &committer,
                    blinding_rng,
                    converter,
                    ic_circuit.clone()
                );

                // let (_code_commitment, _image_commitment) =
                perform_function_under_verifier_config!(
                    verify_v3_iriscode_proof_and_hash,
                    &gkr_circuit_verifier_config,
                    &proof,
                    &verifiable_circuit,
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
    let circuit = circuit_description_and_inputs(mask, None).unwrap();
    let provable_circuit = circuit.gen_hyrax_provable_circuit().unwrap();
    test_iriscode_circuit_with_hyrax_helper(provable_circuit);
}

/// Helper function for testing an iriscode circuit (with any data) with a Hyrax input layer.
pub fn test_iriscode_circuit_with_public_layers_helper(
    mut provable_circuit: HyraxProvableCircuit<Bn256Point>,
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

    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    let verifiable_circuit = provable_circuit._gen_hyrax_verifiable_circuit();

    // --- Compute actual Hyrax proof ---
    let (proof, proof_config) = perform_function_under_prover_config!(
        HyraxProvableCircuit::prove,
        &gkr_circuit_prover_config,
        &mut provable_circuit,
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
        &verifiable_circuit,
        &committer,
        &mut transcript,
        &proof_config
    );
}

/// Helper function for testing an iriscode circuit (with any data) with Hyrax input
/// layers for the private data.
pub fn test_iriscode_circuit_with_hyrax_helper(
    mut provable_circuit: HyraxProvableCircuit<Bn256Point>,
    /*
    ic_circuit_desc: IriscodeCircuitDescription<Scalar>,
    inputs: HashMap<LayerId, MultilinearExtension<Scalar>>,
    */
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
    /*
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
    */
    // Prove.
    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_memory_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    let verifiable_circuit = provable_circuit._gen_hyrax_verifiable_circuit();

    // --- Compute actual Hyrax proof ---
    let prove_timer = start_timer!(|| "Proving");
    let (proof, proof_config) = perform_function_under_prover_config!(
        HyraxProvableCircuit::prove,
        &gkr_circuit_prover_config,
        &mut provable_circuit,
        &committer,
        blinding_rng,
        converter,
        &mut transcript
    );
    end_timer!(prove_timer);

    // Verify.
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    /*
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
    */
    let verification_timer = start_timer!(|| "verification timer");
    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &gkr_circuit_verifier_config,
        &proof,
        &verifiable_circuit,
        &committer,
        &mut transcript,
        &proof_config
    );
    end_timer!(verification_timer);
}
