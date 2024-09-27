//! Unit tests for [crate::output_layer::mle_output_layer].

use itertools::{repeat_n, Itertools};
use pretty_assertions::assert_eq;

use remainder_shared_types::{ff_field, Field};
use remainder_shared_types::{
    transcript::{test_transcript::TestSponge, TranscriptReader, TranscriptWriter},
    Fr,
};

use crate::{
    claims::{wlx_eval::ClaimMle, YieldClaim},
    layer::LayerId,
    mle::{mle_enum::MleEnum, zero::ZeroMle, Mle, MleIndex},
    output_layer::{mle_output_layer::CircuitMleOutputLayer, CircuitOutputLayer},
};

use super::{mle_output_layer::MleOutputLayer, OutputLayer};

impl<F: Field> ZeroMle<F> {
    /// To be used internally for testing.
    pub(crate) fn new_raw(
        mle_indices: Vec<MleIndex<F>>,
        original_mle_indices: Vec<MleIndex<F>>,
        num_vars: usize,
        layer_id: LayerId,
        zero: [F; 1],
        indexed: bool,
    ) -> Self {
        Self {
            mle_indices,
            original_mle_indices,
            num_vars,
            layer_id,
            zero,
            indexed,
        }
    }
}

#[test]
fn test_fix_layer() {
    let layer_id = LayerId::Layer(0);
    let num_vars = 2;

    let mle = ZeroMle::new(num_vars, None, layer_id);

    let mut output_layer = MleOutputLayer::new_zero(mle);

    // Use a `TestSponge` which always returns `1`.
    let mut transcript_writer: TranscriptWriter<Fr, TestSponge<Fr>> =
        TranscriptWriter::new("Test Transcript Writer");

    // Fix `x_1 = 1` and `x_2 = 1`.
    output_layer.fix_layer(&mut transcript_writer).unwrap();

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

    let mut output_layer = MleOutputLayer::new_zero(mle.clone());
    let mut circuit_output_layer = CircuitMleOutputLayer::new_zero(
        layer_id,
        &repeat_n(MleIndex::Free, num_vars).collect_vec(),
    );
    circuit_output_layer.index_mle_indices(0);

    // ---- Part 2: Fix output layer and generate claims.

    // Use a `TestSponge` which always returns `1`.
    let mut transcript_writer: TranscriptWriter<Fr, TestSponge<Fr>> =
        TranscriptWriter::new("Test Transcript Writer");

    output_layer.append_mle_to_transcript(&mut transcript_writer);
    output_layer.fix_layer(&mut transcript_writer).unwrap();
    let claims = output_layer.get_claims().unwrap();

    let expected_point = vec![Fr::ONE, Fr::ONE];
    let expected_result = Fr::from(0);
    let expected_indices = vec![MleIndex::Bound(Fr::ONE, 0), MleIndex::Bound(Fr::ONE, 1)];
    let expected_zero_mle = ZeroMle::new_raw(
        expected_indices,
        mle.mle_indices().to_vec(),
        0,
        layer_id,
        [Fr::ZERO],
        false,
    );
    let expected_claims = vec![ClaimMle::new(
        expected_point.clone(),
        expected_result,
        None,
        Some(layer_id),
        Some(MleEnum::Zero(expected_zero_mle)),
    )];

    assert_eq!(claims, expected_claims);

    // ---- Part 3: Generate claims from the verifier's side.
    let transcript = transcript_writer.get_transcript();
    let mut transcript_reader = TranscriptReader::<Fr, TestSponge<Fr>>::new(transcript);

    let verifier_output_layer = circuit_output_layer
        .retrieve_mle_from_transcript_and_fix_layer(&mut transcript_reader)
        .unwrap();

    let claims = verifier_output_layer.get_claims().unwrap();

    assert_eq!(claims, expected_claims);
}

#[test]
fn test_output_layer_get_claims_with_prefix_bits() {
    // ---- Part 1: Generate Output layer.
    let layer_id = LayerId::Layer(0);
    let num_free_vars = 2;
    let prefix_bits = vec![MleIndex::Fixed(true), MleIndex::Fixed(false)];

    let mle = ZeroMle::new(num_free_vars, Some(prefix_bits.clone()), layer_id);

    let mut output_layer = MleOutputLayer::new_zero(mle.clone());
    let mut circuit_output_layer = CircuitMleOutputLayer::new_zero(
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

    output_layer.append_mle_to_transcript(&mut transcript_writer);
    output_layer.fix_layer(&mut transcript_writer).unwrap();
    let claims = output_layer.get_claims().unwrap();

    let expected_point = vec![Fr::ONE, Fr::ZERO, Fr::ONE, Fr::ONE];
    let expected_result = Fr::from(0);
    let expected_indices = vec![
        MleIndex::Fixed(true),
        MleIndex::Fixed(false),
        MleIndex::Bound(Fr::ONE, 0),
        MleIndex::Bound(Fr::ONE, 1),
    ];
    let expected_zero_mle = ZeroMle::new_raw(
        expected_indices,
        mle.mle_indices().to_vec(),
        0,
        layer_id,
        [Fr::ZERO],
        false,
    );
    let expected_claims = vec![ClaimMle::new(
        expected_point.clone(),
        expected_result,
        None,
        Some(layer_id),
        Some(MleEnum::Zero(expected_zero_mle)),
    )];

    assert_eq!(claims, expected_claims);

    // ---- Part 3: Generate claims from the verifier's side.
    let transcript = transcript_writer.get_transcript();
    let mut transcript_reader = TranscriptReader::<Fr, TestSponge<Fr>>::new(transcript);

    let verifier_output_layer = circuit_output_layer
        .retrieve_mle_from_transcript_and_fix_layer(&mut transcript_reader)
        .unwrap();

    let claims = verifier_output_layer.get_claims().unwrap();

    // TODO(Makis): This still fails because the verifier doesn't know about
    // the prefix bits. I think we should pass this information to the verifier
    // key.
    assert_eq!(claims, expected_claims)
}
