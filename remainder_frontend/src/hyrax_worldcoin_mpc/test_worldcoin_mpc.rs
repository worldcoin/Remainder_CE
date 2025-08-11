use std::collections::HashMap;

use crate::worldcoin_mpc::circuits::MPCCircuitDescription;
use ark_std::{end_timer, start_timer};
use remainder::{layer::LayerId, mle::evals::MultilinearExtension};
use remainder_shared_types::{
    config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig},
    pedersen::PedersenCommitter,
    perform_function_under_prover_config, perform_function_under_verifier_config,
    transcript::{ec_transcript::ECTranscript, poseidon_sponge::PoseidonSponge},
    Base, Bn256Point, Scalar,
};

use remainder_hyrax::{
    hyrax_gkr::{hyrax_input_layer::commit_to_input_values, verify_hyrax_proof, HyraxProof},
    utils::vandermonde::VandermondeInverse,
};

#[cfg(test)]
mod tests {

    use crate::worldcoin_mpc::test_helpers::{
        inversed_circuit_description_and_inputs, small_circuit_description_and_inputs,
    };
    use remainder_shared_types::{
        config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig},
        perform_function_under_expected_configs,
    };

    use crate::hyrax_worldcoin_mpc::test_worldcoin_mpc::{
        test_mpc_circuit_with_precommits_hyrax_helper, test_mpc_circuit_with_public_layers_helper,
    };

    use super::test_mpc_circuit_with_hyrax_helper;

    #[ignore] // takes a long time to run!
    #[test]
    fn test_small_mpc_circuit_with_hyrax_layers() {
        const NUM_IRIS_4_CHUNKS: usize = 1;
        const PARTY_IDX: usize = 0;
        let (desc, inputs) = small_circuit_description_and_inputs::<NUM_IRIS_4_CHUNKS, PARTY_IDX>();
        test_mpc_circuit_with_hyrax_helper(desc, inputs);
    }

    #[ignore] // takes a long time to run!
    #[test]
    fn test_small_mpc_circuit_with_hyrax_layers_batched() {
        const NUM_IRIS_4_CHUNKS: usize = 4;
        const PARTY_IDX: usize = 0;
        let (desc, inputs) = small_circuit_description_and_inputs::<NUM_IRIS_4_CHUNKS, PARTY_IDX>();
        test_mpc_circuit_with_hyrax_helper(desc, inputs);
    }

    #[ignore] // takes a long time to run!
    #[test]
    fn test_small_mpc_circuit_with_hyrax_layers_batched_non_power_of_2() {
        const NUM_IRIS_4_CHUNKS: usize = 3;
        const PARTY_IDX: usize = 0;
        let (desc, inputs) = small_circuit_description_and_inputs::<NUM_IRIS_4_CHUNKS, PARTY_IDX>();
        test_mpc_circuit_with_hyrax_helper(desc, inputs);
    }

    #[ignore] // takes a long time to run!
    #[test]
    fn test_small_mpc_circuit_with_precommits_hyrax_layers_batched_non_power_of_2() {
        const NUM_IRIS_4_CHUNKS: usize = 3;
        const PARTY_IDX: usize = 0;
        let (desc, inputs) = small_circuit_description_and_inputs::<NUM_IRIS_4_CHUNKS, PARTY_IDX>();
        test_mpc_circuit_with_precommits_hyrax_helper(desc, inputs);
    }

    #[ignore] // takes a long time to run!
    #[test]
    fn test_small_mpc_circuit_with_precommits_hyrax_layers_batched_non_power_of_2_all_3_parties() {
        const NUM_IRIS_4_CHUNKS: usize = 3;
        // party 0
        let (desc, inputs) = small_circuit_description_and_inputs::<NUM_IRIS_4_CHUNKS, 0>();
        test_mpc_circuit_with_precommits_hyrax_helper(desc, inputs);
        // party 1
        let (desc, inputs) = small_circuit_description_and_inputs::<NUM_IRIS_4_CHUNKS, 1>();
        test_mpc_circuit_with_precommits_hyrax_helper(desc, inputs);
        // party 2
        let (desc, inputs) = small_circuit_description_and_inputs::<NUM_IRIS_4_CHUNKS, 2>();
        test_mpc_circuit_with_precommits_hyrax_helper(desc, inputs);
    }

    #[test]
    fn test_small_mpc_circuit_both_layers_public() {
        const NUM_IRIS_4_CHUNKS: usize = 1;
        const PARTY_IDX: usize = 0;
        let (desc, inputs) = small_circuit_description_and_inputs::<NUM_IRIS_4_CHUNKS, PARTY_IDX>();
        // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
        let gkr_circuit_prover_config =
            GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
        let gkr_circuit_verifier_config =
            GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);
        perform_function_under_expected_configs!(
            test_mpc_circuit_with_public_layers_helper,
            &gkr_circuit_prover_config,
            &gkr_circuit_verifier_config,
            desc,
            inputs
        );
    }

    #[ignore] // takes a long time to run!
    #[test]
    fn test_inversed_mpc_circuit_with_hyrax_layers() {
        const TEST_IDX: usize = 2;
        const NUM_IRIS_4_CHUNKS: usize = 1;
        const PARTY_IDX: usize = 0;
        let (desc, inputs) =
            inversed_circuit_description_and_inputs::<NUM_IRIS_4_CHUNKS, PARTY_IDX>(TEST_IDX);
        test_mpc_circuit_with_hyrax_helper(desc, inputs);
    }

    #[ignore] // takes a long time to run!
    #[test]
    fn test_inversed_mpc_circuit_with_hyrax_layers_batched() {
        const TEST_IDX: usize = 2;
        const NUM_IRIS_4_CHUNKS: usize = 4;
        const PARTY_IDX: usize = 0;
        let (desc, inputs) =
            inversed_circuit_description_and_inputs::<NUM_IRIS_4_CHUNKS, PARTY_IDX>(TEST_IDX);
        test_mpc_circuit_with_hyrax_helper(desc, inputs);
    }

    #[ignore] // takes a long time to run!
    #[test]
    fn test_inversed_mpc_circuit_with_hyrax_layers_batched_non_power_of_2() {
        const TEST_IDX: usize = 2;
        const NUM_IRIS_4_CHUNKS: usize = 3;
        const PARTY_IDX: usize = 0;
        let (desc, inputs) =
            inversed_circuit_description_and_inputs::<NUM_IRIS_4_CHUNKS, PARTY_IDX>(TEST_IDX);
        test_mpc_circuit_with_hyrax_helper(desc, inputs);
    }

    #[ignore] // takes a long time to run!
    #[test]
    fn test_inversed_mpc_circuit_both_layers_public() {
        const TEST_IDX: usize = 2;
        const NUM_IRIS_4_CHUNKS: usize = 1;
        const PARTY_IDX: usize = 0;
        let (desc, inputs) =
            inversed_circuit_description_and_inputs::<NUM_IRIS_4_CHUNKS, PARTY_IDX>(TEST_IDX);
        test_mpc_circuit_with_public_layers_helper(desc, inputs);
    }
}

/// Helper function for testing an Shamir's secret sharing circuit with a public input layer.
pub fn test_mpc_circuit_with_public_layers_helper(
    mpc_circuit_desc: MPCCircuitDescription<Scalar>,
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

    // --- Compute actual Hyrax proof ---
    let (proof, proof_config) = HyraxProof::prove(
        &inputs,
        &HashMap::new(),
        &mpc_circuit_desc.circuit_description,
        &committer,
        blinding_rng,
        converter,
        &mut transcript,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    // Verify.
    verify_hyrax_proof(
        &proof,
        &HashMap::new(),
        &mpc_circuit_desc.circuit_description,
        &committer,
        &mut transcript,
        &proof_config,
    );
}

/// Helper function for testing an Shamir's secret sharing circuit with Hyrax input
/// layers for the private data.
pub fn test_mpc_circuit_with_hyrax_helper(
    mpc_circuit_desc: MPCCircuitDescription<Scalar>,
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
        mpc_circuit_desc.slope_input_layer.layer_id,
        (mpc_circuit_desc.slope_input_layer.clone().into(), None),
    );
    prover_hyrax_input_layers.insert(
        mpc_circuit_desc.iris_code_input_layer.layer_id,
        (mpc_circuit_desc.iris_code_input_layer.clone().into(), None),
    );
    prover_hyrax_input_layers.insert(
        mpc_circuit_desc.mask_code_input_layer.layer_id,
        (mpc_circuit_desc.mask_code_input_layer.clone().into(), None),
    );
    prover_hyrax_input_layers.insert(
        mpc_circuit_desc.auxilary_input_layer.layer_id,
        (mpc_circuit_desc.auxilary_input_layer.clone().into(), None),
    );
    // Prove.
    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_memory_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    // --- Compute actual Hyrax proof ---
    let (proof, proof_config) = perform_function_under_prover_config!(
        HyraxProof::prove,
        &gkr_circuit_prover_config,
        &inputs,
        &prover_hyrax_input_layers,
        &mpc_circuit_desc.circuit_description,
        &committer,
        blinding_rng,
        converter,
        &mut transcript
    );
    // Verify.
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let mut verifier_hyrax_input_layers = HashMap::new();
    verifier_hyrax_input_layers.insert(
        mpc_circuit_desc.slope_input_layer.layer_id,
        mpc_circuit_desc.slope_input_layer.clone().into(),
    );
    verifier_hyrax_input_layers.insert(
        mpc_circuit_desc.iris_code_input_layer.layer_id,
        mpc_circuit_desc.iris_code_input_layer.clone().into(),
    );
    verifier_hyrax_input_layers.insert(
        mpc_circuit_desc.mask_code_input_layer.layer_id,
        mpc_circuit_desc.mask_code_input_layer.clone().into(),
    );
    verifier_hyrax_input_layers.insert(
        mpc_circuit_desc.auxilary_input_layer.layer_id,
        mpc_circuit_desc.auxilary_input_layer.clone().into(),
    );
    let verification_timer = start_timer!(|| "verification timer");
    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &gkr_circuit_verifier_config,
        &proof,
        &verifier_hyrax_input_layers,
        &mpc_circuit_desc.circuit_description,
        &committer,
        &mut transcript,
        &proof_config
    );
    end_timer!(verification_timer);
}

/// Helper function for testing an Shamir's secret sharing circuit with Hyrax input
/// layers for the private data.
pub fn test_mpc_circuit_with_precommits_hyrax_helper(
    mpc_circuit_desc: MPCCircuitDescription<Scalar>,
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

    let slope_precommit = commit_to_input_values(
        &mpc_circuit_desc.slope_input_layer.clone().into(),
        inputs
            .get(&mpc_circuit_desc.slope_input_layer.layer_id)
            .unwrap(),
        &committer,
        blinding_rng,
    );

    let iris_precommit = commit_to_input_values(
        &mpc_circuit_desc.iris_code_input_layer.clone().into(),
        inputs
            .get(&mpc_circuit_desc.iris_code_input_layer.layer_id)
            .unwrap(),
        &committer,
        blinding_rng,
    );

    let mask_code_precommit = commit_to_input_values(
        &mpc_circuit_desc.mask_code_input_layer.clone().into(),
        inputs
            .get(&mpc_circuit_desc.mask_code_input_layer.layer_id)
            .unwrap(),
        &committer,
        blinding_rng,
    );

    // Set up Hyrax input layer specification.
    let mut prover_hyrax_input_layers = HashMap::new();
    prover_hyrax_input_layers.insert(
        mpc_circuit_desc.slope_input_layer.layer_id,
        (
            mpc_circuit_desc.slope_input_layer.clone().into(),
            Some(slope_precommit),
        ),
    );
    prover_hyrax_input_layers.insert(
        mpc_circuit_desc.iris_code_input_layer.layer_id,
        (
            mpc_circuit_desc.iris_code_input_layer.clone().into(),
            Some(iris_precommit),
        ),
    );
    prover_hyrax_input_layers.insert(
        mpc_circuit_desc.mask_code_input_layer.layer_id,
        (
            mpc_circuit_desc.mask_code_input_layer.clone().into(),
            Some(mask_code_precommit),
        ),
    );
    prover_hyrax_input_layers.insert(
        mpc_circuit_desc.auxilary_input_layer.layer_id,
        (mpc_circuit_desc.auxilary_input_layer.clone().into(), None),
    );
    // Prove.
    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_memory_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    // --- Compute actual Hyrax proof ---
    let (proof, proof_config) = perform_function_under_prover_config!(
        HyraxProof::prove,
        &gkr_circuit_prover_config,
        &inputs,
        &prover_hyrax_input_layers,
        &mpc_circuit_desc.circuit_description,
        &committer,
        blinding_rng,
        converter,
        &mut transcript
    );
    // Verify.
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let mut verifier_hyrax_input_layers = HashMap::new();
    verifier_hyrax_input_layers.insert(
        mpc_circuit_desc.slope_input_layer.layer_id,
        mpc_circuit_desc.slope_input_layer.clone().into(),
    );
    verifier_hyrax_input_layers.insert(
        mpc_circuit_desc.iris_code_input_layer.layer_id,
        mpc_circuit_desc.iris_code_input_layer.clone().into(),
    );
    verifier_hyrax_input_layers.insert(
        mpc_circuit_desc.mask_code_input_layer.layer_id,
        mpc_circuit_desc.mask_code_input_layer.clone().into(),
    );
    verifier_hyrax_input_layers.insert(
        mpc_circuit_desc.auxilary_input_layer.layer_id,
        mpc_circuit_desc.auxilary_input_layer.clone().into(),
    );
    let verification_timer = start_timer!(|| "verification timer");
    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &gkr_circuit_verifier_config,
        &proof,
        &verifier_hyrax_input_layers,
        &mpc_circuit_desc.circuit_description,
        &committer,
        &mut transcript,
        &proof_config
    );
    end_timer!(verification_timer);
}
