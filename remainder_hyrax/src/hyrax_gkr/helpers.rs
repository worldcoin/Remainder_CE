use std::collections::HashMap;

use remainder::{
    layer::LayerId,
    mle::evals::MultilinearExtension,
    prover::{
        config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig},
        global_config::perform_function_under_expected_configs,
        GKRCircuitDescription,
    },
};
use remainder_shared_types::{
    curves::PrimeOrderCurve,
    pedersen::PedersenCommitter,
    transcript::{ec_transcript::ECTranscript, poseidon_transcript::PoseidonSponge},
};

use crate::utils::vandermonde::VandermondeInverse;

use super::{
    hyrax_input_layer::HyraxInputLayerDescriptionWithPrecommit, verify_hyrax_proof, HyraxProof,
};

/// Helper function for testing an iriscode circuit (of any version, with any
/// data) with a Hyrax input layer.
pub fn test_iriscode_circuit_with_hyrax_helper<C: PrimeOrderCurve>(
    circuit_desc: GKRCircuitDescription<C::Scalar>,
    private_layer_descriptions: HyraxInputLayerDescriptionWithPrecommit<C>,
    inputs: HashMap<LayerId, MultilinearExtension<C::Scalar>>,
) {
    let mut transcript: ECTranscript<C, PoseidonSponge<C::Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let blinding_rng = &mut rand::thread_rng();
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

    // --- Compute actual Hyrax proof ---
    let (proof, proof_config) = perform_function_under_expected_configs(
        HyraxProof::prove,
        (
            &inputs,
            &private_layer_descriptions,
            &circuit_desc,
            &committer,
            blinding_rng,
            converter,
            &mut transcript,
        ),
        &gkr_circuit_prover_config,
        &gkr_circuit_verifier_config,
    );

    let mut transcript: ECTranscript<C, PoseidonSponge<C::Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let verifier_hyrax_input_layers = private_layer_descriptions
        .into_iter()
        .map(|(k, v)| (k, v.0))
        .collect();

    perform_function_under_expected_configs(
        verify_hyrax_proof,
        (
            &proof,
            &verifier_hyrax_input_layers,
            &circuit_desc,
            &committer,
            &mut transcript,
            &proof_config,
        ),
        &gkr_circuit_prover_config,
        &gkr_circuit_verifier_config,
    );
}
