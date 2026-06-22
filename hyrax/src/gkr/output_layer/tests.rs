use super::*;
use remainder::layer::LayerId;
use remainder::mle::zero::ZeroMle;
use shared_types::halo2curves::bn256::{Fr, G1 as Bn256Point};
use shared_types::halo2curves::CurveExt;
use shared_types::transcript::ec_transcript::ECTranscript;
use shared_types::transcript::poseidon_sponge::PoseidonSponge;

type Base = <Bn256Point as CurveExt>::Base;

const INIT_STR: &str = "modulus modulus modulus modulus modulus modulus";

/// An honest zero-test output layer commits its claim to zero, and the verifier
/// accepts it by recomputing the commitment from the revealed blinding factor.
#[test]
fn zero_output_layer_commits_to_zero_and_verifies() {
    let committer = PedersenCommitter::<Bn256Point>::new(2, INIT_STR, None);
    let layer_id = LayerId::Input(0);

    let mut output_layer: OutputLayer<Fr> = OutputLayer::new_zero(ZeroMle::new(2, None, layer_id));

    let mut prover_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new(INIT_STR);
    let (proof, committed_claim) = HyraxOutputLayerProof::prove(
        &mut output_layer,
        &mut prover_transcript,
        &mut rand::thread_rng(),
        &committer,
    );

    // The claim value is zero, and the commitment opens to zero under the
    // revealed blinding factor.
    assert_eq!(committed_claim.evaluation.value, Fr::from(0_u64));
    assert_eq!(
        proof.claim_commitment,
        committer.scalar_commit(&Fr::from(0_u64), &proof.claim_blinding)
    );

    let mut layer_desc =
        OutputLayerDescription::new_zero(layer_id, &[MleIndex::Free, MleIndex::Free]);
    layer_desc.index_mle_indices(0);

    let mut verifier_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new(INIT_STR);
    let claim =
        HyraxOutputLayerProof::verify(&proof, &layer_desc, &committer, &mut verifier_transcript);
    assert_eq!(claim.evaluation, proof.claim_commitment);
}

/// A malicious prover that commits the true non-zero output (instead of zero)
/// must be rejected by the verifier of a zero-test output layer, regardless of
/// the blinding factor it reveals.
#[test]
#[should_panic(expected = "Zero-test output layer claim does not open to zero")]
fn zero_output_layer_rejects_nonzero_commitment() {
    let committer = PedersenCommitter::<Bn256Point>::new(2, INIT_STR, None);
    let layer_id = LayerId::Input(0);

    let malicious = HyraxOutputLayerProof {
        claim_commitment: committer.scalar_commit(&Fr::from(7_u64), &Fr::from(3_u64)),
        claim_blinding: Fr::from(3_u64),
    };

    let mut layer_desc =
        OutputLayerDescription::new_zero(layer_id, &[MleIndex::Free, MleIndex::Free]);
    layer_desc.index_mle_indices(0);

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new(INIT_STR);
    HyraxOutputLayerProof::verify(&malicious, &layer_desc, &committer, &mut transcript);
}
