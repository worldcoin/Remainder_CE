use crate::hyrax_gkr::hyrax_layer::HyraxClaim;
use crate::hyrax_primitives::proof_of_claim_agg::barycentric_weights;
use crate::hyrax_primitives::proof_of_claim_agg::ProofOfClaimAggregation;
use crate::pedersen::PedersenCommitter;

use remainder::layer::LayerId;
use remainder_shared_types::halo2curves::bn256::Fr;
use remainder_shared_types::halo2curves::bn256::G1 as Bn256Point;
use remainder_shared_types::halo2curves::CurveExt;
use remainder_shared_types::transcript::ec_transcript::ECTranscriptReader;
use remainder_shared_types::transcript::ec_transcript::ECTranscriptWriter;
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;

type Base = <Bn256Point as CurveExt>::Base;

// a static string constant
const INIT_STR: &str = "modulus modulus modulus modulus modulus modulus";

#[test]
pub fn test_barycentric_weights() {
    let weights = barycentric_weights(Fr::zero(), 3);
    assert_eq!(weights, vec![Fr::one(), Fr::zero(), Fr::zero()]);
    let weights = barycentric_weights(Fr::one(), 3);
    assert_eq!(weights, vec![Fr::zero(), Fr::one(), Fr::zero()]);
}

#[test]
pub fn test_completeness() {
    let committer = PedersenCommitter::<Bn256Point>::new(1, INIT_STR, None);
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test completeness transcript");
    let claims = vec![
        HyraxClaim {
            to_layer_id: LayerId::Layer(0),
            point: vec![Fr::from(23), Fr::from(6)],
            mle_enum: None,
            evaluation: committer.committed_scalar(&Fr::from(6), &Fr::from(2)),
        },
        HyraxClaim {
            to_layer_id: LayerId::Layer(0),
            point: vec![Fr::from(3), Fr::from(7)],
            mle_enum: None,
            evaluation: committer.committed_scalar(&Fr::from(5), &Fr::from(3)),
        },
    ];
    // An MLE with two coordinates is (generically) determined by any four evaluations, and the two
    // claims only specify two!  So we can choose any value for the last coefficient of the
    // interpolating polynomial so long as we are careful to adjust the linear coeff accordingly.
    let arbitrary_value = Fr::one();
    let coeffs = vec![
        claims[0].evaluation.value,
        claims[1].evaluation.value - claims[0].evaluation.value - arbitrary_value,
        arbitrary_value,
    ];
    let (proof, agg_claim_prover) = ProofOfClaimAggregation::prove(
        &claims,
        &coeffs,
        &committer,
        &mut rand::thread_rng(),
        &mut prover_transcript,
    );

    let transcript = prover_transcript.get_transcript();
    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(transcript);

    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();
    let agg_claim_verifier = proof.verify(&claim_commitments, &committer, &mut verifier_transcript);
    assert_eq!(agg_claim_prover.point, agg_claim_verifier.point);
    assert_eq!(
        agg_claim_prover.evaluation.commitment,
        agg_claim_verifier.evaluation
    );
}

#[test]
pub fn test_agg_one_claim() {
    let committer = PedersenCommitter::<Bn256Point>::new(1, INIT_STR, None);
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test claim agg transcript");
    let claims = vec![HyraxClaim {
        to_layer_id: LayerId::Layer(0),
        point: vec![Fr::from(23), Fr::from(6)],
        mle_enum: None,
        evaluation: committer.committed_scalar(&Fr::from(6), &Fr::from(2)),
    }];
    let coeffs = vec![claims[0].evaluation.value];
    let (proof, agg_claim_prover) = ProofOfClaimAggregation::prove(
        &claims,
        &coeffs,
        &committer,
        &mut rand::thread_rng(),
        &mut prover_transcript,
    );

    let transcript = prover_transcript.get_transcript();
    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(transcript);

    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();
    let agg_claim_verifier = proof.verify(&claim_commitments, &committer, &mut verifier_transcript);
    assert_eq!(agg_claim_prover.point, agg_claim_verifier.point);
    assert_eq!(
        agg_claim_prover.evaluation.commitment,
        agg_claim_verifier.evaluation
    );
}

#[test]
#[should_panic]
pub fn test_soundness() {
    // use a false interpolating polynomial (but still of the correct degree) - should fail.
    let committer = PedersenCommitter::<Bn256Point>::new(1, INIT_STR, None);
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test soundness transcript");
    let claims = vec![
        HyraxClaim {
            to_layer_id: LayerId::Layer(0),
            point: vec![Fr::from(23), Fr::from(6)],
            mle_enum: None,
            evaluation: committer.committed_scalar(&Fr::from(6), &Fr::from(2)),
        },
        HyraxClaim {
            to_layer_id: LayerId::Layer(0),
            point: vec![Fr::from(3), Fr::from(7)],
            mle_enum: None,
            evaluation: committer.committed_scalar(&Fr::from(5), &Fr::from(3)),
        },
    ];
    let coeffs = vec![Fr::from(123123), Fr::from(789789)];
    let (proof, _) = ProofOfClaimAggregation::prove(
        &claims,
        &coeffs,
        &committer,
        &mut rand::thread_rng(),
        &mut prover_transcript,
    );

    let transcript = prover_transcript.get_transcript();
    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(transcript);

    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();
    proof.verify(&claim_commitments, &committer, &mut verifier_transcript);
}
