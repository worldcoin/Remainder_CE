use remainder_shared_types::{
    config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig},
    curves::PrimeOrderCurve,
    pedersen::PedersenCommitter,
    perform_function_under_expected_configs,
    transcript::{ec_transcript::ECTranscript, poseidon_sponge::PoseidonSponge},
};

use crate::{
    hyrax_gkr::verify_hyrax_proof,
    provable_circuit::HyraxProvableCircuit,
    utils::{get_crypto_chacha20_prng, vandermonde::VandermondeInverse},
};

/// Helper function for testing an iriscode circuit (of any version, with any
/// data) with a Hyrax input layer.
pub fn test_iriscode_circuit_with_hyrax_helper<C: PrimeOrderCurve>(
    mut provable_circuit: HyraxProvableCircuit<C>,
    /*
    circuit_desc: GKRCircuitDescription<C::Scalar>,
    private_layer_descriptions: HyraxInputLayerDescriptionWithPrecommit<C>,
    inputs: HashMap<LayerId, MultilinearExtension<C::Scalar>>,
    */
) {
    let mut transcript: ECTranscript<C, PoseidonSponge<C::Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let mut blinding_rng = get_crypto_chacha20_prng();
    let converter: &mut VandermondeInverse<C::Scalar> = &mut VandermondeInverse::new();
    let num_generators = 512;
    let committer = PedersenCommitter::<C>::new(
        num_generators + 1,
        "modulus modulus modulus modulus modulus",
        None,
    );

    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_memory_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    let verifiable_circuit = provable_circuit._gen_hyrax_verifiable_circuit();

    // --- Compute actual Hyrax proof ---
    let (proof, proof_config) = perform_function_under_expected_configs!(
        HyraxProvableCircuit::prove,
        &gkr_circuit_prover_config,
        &gkr_circuit_verifier_config,
        &mut provable_circuit,
        &committer,
        &mut blinding_rng,
        converter,
        &mut transcript
    );

    let mut transcript: ECTranscript<C, PoseidonSponge<C::Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    /*
    let verifier_hyrax_input_layers = private_layer_descriptions
        .into_iter()
        .map(|(k, v)| (k, v.0))
        .collect();
    */

    perform_function_under_expected_configs!(
        verify_hyrax_proof,
        &gkr_circuit_prover_config,
        &gkr_circuit_verifier_config,
        &proof,
        &verifiable_circuit,
        &committer,
        &mut transcript,
        &proof_config
    );
}
