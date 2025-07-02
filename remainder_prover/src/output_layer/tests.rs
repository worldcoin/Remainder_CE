//! Unit tests for [crate::output_layer].

use itertools::{repeat_n, Itertools};
use pretty_assertions::assert_eq;

use remainder_shared_types::ff_field;
use remainder_shared_types::transcript::ProverTranscript;
use remainder_shared_types::{
    transcript::{test_transcript::TestSponge, TranscriptReader, TranscriptWriter},
    Fr,
};

use crate::claims::Claim;
use crate::mle::Mle;
use crate::{
    layer::LayerId,
    mle::{zero::ZeroMle, MleIndex},
    output_layer::OutputLayerDescription,
};

use super::OutputLayer;

#[test]
fn test_fix_layer() {
    let layer_id = LayerId::Layer(0);
    let num_vars = 2;

    let mle = ZeroMle::new(num_vars, None, layer_id);

    let mut output_layer = OutputLayer::new_zero(mle);

    let challenges = vec![Fr::ONE, Fr::ONE];
    // Fix `x_1 = 1` and `x_2 = 1`.
    output_layer.fix_layer(&challenges).unwrap();

    // Expect the output layer to be fully bound and evaluating to
    // `f(1, 1)` which is equal to `0`.
    assert_eq!(output_layer.value().unwrap(), Fr::ZERO);
}

#[test]
fn test_output_layer_get_claims() {
    // ---- Part 1: Generate Output layer.
    let layer_id = LayerId::Layer(0);
    let num_vars = 2;

    let mle = ZeroMle::new(num_vars, None, layer_id);

    let mut output_layer = OutputLayer::new_zero(mle.clone());
    let mut circuit_output_layer = OutputLayerDescription::new_zero(
        layer_id,
        &repeat_n(MleIndex::Free, num_vars).collect_vec(),
    );
    circuit_output_layer.index_mle_indices(0);

    // ---- Part 2: Fix output layer and generate claims.

    // Use a `TestSponge` which always returns `1`.
    let mut transcript_writer: TranscriptWriter<Fr, TestSponge<Fr>> =
        TranscriptWriter::new("Test Transcript Writer");

    transcript_writer.append_elements(
        "Output layer MLE evals",
        &output_layer.get_mle().iter().collect_vec(),
    );
    let challenges = transcript_writer.get_challenges("Challenge on the output layer", 2);
    // Fix `x_1 = 1` and `x_2 = 1`.
    output_layer.fix_layer(&challenges).unwrap();
    let claim = output_layer.get_claim().unwrap();

    let expected_point = vec![Fr::ONE, Fr::ONE];
    let expected_result = Fr::from(0);
    let expected_claim = Claim::new(expected_point.clone(), expected_result, layer_id, layer_id);

    assert_eq!(claim, expected_claim);

    // ---- Part 3: Generate claims from the verifier's side.
    let transcript = transcript_writer.get_transcript();
    let mut transcript_reader = TranscriptReader::<Fr, TestSponge<Fr>>::new(transcript);

    let verifier_output_layer = circuit_output_layer
        .retrieve_mle_from_transcript_and_fix_layer(&mut transcript_reader)
        .unwrap();

    let claim = verifier_output_layer.get_claim().unwrap();

    assert_eq!(claim, expected_claim);
}

#[test]
fn test_output_layer_get_claims_with_prefix_bits() {
    // ---- Part 1: Generate Output layer.
    let layer_id = LayerId::Layer(0);
    let num_free_vars = 2;
    let prefix_bits = vec![MleIndex::Fixed(true), MleIndex::Fixed(false)];

    let mle = ZeroMle::new(num_free_vars, Some(prefix_bits.clone()), layer_id);

    let mut output_layer = OutputLayer::new_zero(mle.clone());
    let mut circuit_output_layer = OutputLayerDescription::new_zero(
        layer_id,
        &prefix_bits
            .into_iter()
            .chain(repeat_n(MleIndex::Free, num_free_vars))
            .collect_vec(),
    );
    circuit_output_layer.index_mle_indices(0);

    // ---- Part 2: Fix output layer and generate claims.

    // Use a `TestSponge` which always returns `1`.
    let mut transcript_writer: TranscriptWriter<Fr, TestSponge<Fr>> =
        TranscriptWriter::new("Test Transcript Writer");

    transcript_writer.append_elements(
        "Output layer MLE evals",
        &output_layer.get_mle().iter().collect_vec(),
    );
    let challenges = transcript_writer.get_challenges(
        "Challenge on the output layer",
        output_layer.num_free_vars(),
    );
    output_layer.fix_layer(&challenges).unwrap();
    let claim = output_layer.get_claim().unwrap();

    let expected_point = vec![Fr::ONE, Fr::ZERO, Fr::ONE, Fr::ONE];
    let expected_result = Fr::from(0);

    let expected_claim = Claim::new(expected_point.clone(), expected_result, layer_id, layer_id);

    assert_eq!(claim, expected_claim);

    // ---- Part 3: Generate claims from the verifier's side.
    let transcript = transcript_writer.get_transcript();
    let mut transcript_reader = TranscriptReader::<Fr, TestSponge<Fr>>::new(transcript);

    let verifier_output_layer = circuit_output_layer
        .retrieve_mle_from_transcript_and_fix_layer(&mut transcript_reader)
        .unwrap();

    let claim = verifier_output_layer.get_claim().unwrap();

    // TODO(Makis): This still fails because the verifier doesn't know about
    // the prefix bits. I think we should pass this information to the verifier
    // key.
    assert_eq!(claim, expected_claim)
}
