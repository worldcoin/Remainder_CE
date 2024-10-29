use std::collections::HashMap;

use remainder::{
    layer::LayerId, layouter::compiling::CircuitHashType, mle::evals::MultilinearExtension,
    prover::GKRCircuitDescription,
};
use remainder_shared_types::{
    curves::PrimeOrderCurve,
    pedersen::PedersenCommitter,
    transcript::{ec_transcript::ECTranscript, poseidon_transcript::PoseidonSponge},
};

use crate::utils::vandermonde::VandermondeInverse;

use super::{hyrax_input_layer::HyraxInputLayerDescriptionWithPrecommit, HyraxProof};

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

    let proof = HyraxProof::prove(
        &inputs,
        &private_layer_descriptions,
        &circuit_desc,
        &committer,
        blinding_rng,
        converter,
        &mut transcript,
        CircuitHashType::Sha3_256,
    );
    let mut transcript: ECTranscript<C, PoseidonSponge<C::Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let verifier_hyrax_input_layers = private_layer_descriptions
        .into_iter()
        .map(|(k, v)| (k, v.0))
        .collect();
    proof.verify(
        &verifier_hyrax_input_layers,
        &circuit_desc,
        &committer,
        &mut transcript,
        CircuitHashType::Sha3_256,
    );
}
