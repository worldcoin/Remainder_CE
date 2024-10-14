use std::collections::HashMap;

use crate::{
    hyrax_gkr::{hyrax_input_layer::HyraxInputLayerDescription, HyraxProof},
    utils::vandermonde::VandermondeInverse,
};
use remainder::{
    input_layer::InputLayerDescription,
    layer::LayerId,
    mle::evals::MultilinearExtension,
    prover::GKRCircuitDescription,
    worldcoin::test_helpers::{
        v2_circuit_description_and_inputs, v3_circuit_description_and_inputs,
    },
};
use remainder_shared_types::{
    halo2curves::{bn256::G1 as Bn256Point, group::Group, CurveExt},
    pedersen::PedersenCommitter,
    transcript::{ec_transcript::ECTranscript, poseidon_transcript::PoseidonSponge},
};

type Scalar = <Bn256Point as Group>::Scalar;
type Base = <Bn256Point as CurveExt>::Base;

#[test]
fn test_small_circuit_both_layers_public() {
    use remainder::worldcoin::test_helpers::small_circuit_description_and_inputs;
    let (circuit_desc, _, inputs) = small_circuit_description_and_inputs();
    test_iriscode_circuit_with_public_layers_helper(circuit_desc, inputs);
}

#[test]
/// Test a small version of the iriscode circuit with a Hyrax input layer.
fn test_small_circuit_with_hyrax_layer() {
    use remainder::worldcoin::test_helpers::small_circuit_description_and_inputs;
    let (desc, priv_layer_desc, inputs) = small_circuit_description_and_inputs();
    test_iriscode_circuit_with_hyrax_helper(desc, priv_layer_desc, inputs);
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

/// Test the iriscode circuit v2 with a Hyrax input layer in either the mask (true) or iris (false)
/// case.
pub fn test_iriscode_v2_with_hyrax_helper(mask: bool) {
    let (desc, priv_layer_desc, inputs) = v2_circuit_description_and_inputs(mask, None);
    test_iriscode_circuit_with_hyrax_helper(desc, priv_layer_desc, inputs);
}

/// Test the iriscode circuit v3 with a Hyrax input layer in either the mask (true) or iris (false)
/// case.
pub fn test_iriscode_v3_with_hyrax_helper(mask: bool) {
    let (desc, priv_layer_desc, inputs) = v3_circuit_description_and_inputs(mask, None);
    test_iriscode_circuit_with_hyrax_helper(desc, priv_layer_desc, inputs);
}

/// Helper function for testing an iriscode circuit (of any version, with any data) with a Hyrax input layer.
pub fn test_iriscode_circuit_with_public_layers_helper(
    circuit_desc: GKRCircuitDescription<Scalar>,
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
        &circuit_desc,
        &committer,
        blinding_rng,
        converter,
        &mut transcript,
    );
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    proof.verify(&HashMap::new(), &circuit_desc, &committer, &mut transcript);
}

/// Helper function for testing an iriscode circuit (of any version, with any data) with a Hyrax input layer.
pub fn test_iriscode_circuit_with_hyrax_helper(
    circuit_desc: GKRCircuitDescription<Scalar>,
    private_layer_desc: InputLayerDescription,
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
    let mut prover_hyrax_input_layers = HashMap::new();
    let hyrax_input_layer_desc: HyraxInputLayerDescription = private_layer_desc.into();
    prover_hyrax_input_layers.insert(
        hyrax_input_layer_desc.layer_id,
        (hyrax_input_layer_desc.clone(), None),
    );

    let proof = HyraxProof::prove(
        &inputs,
        &prover_hyrax_input_layers,
        &circuit_desc,
        &committer,
        blinding_rng,
        converter,
        &mut transcript,
    );
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let mut verifier_hyrax_input_layers = HashMap::new();
    verifier_hyrax_input_layers.insert(
        hyrax_input_layer_desc.layer_id,
        hyrax_input_layer_desc.clone(),
    );
    proof.verify(
        &verifier_hyrax_input_layers,
        &circuit_desc,
        &committer,
        &mut transcript,
    );
}
