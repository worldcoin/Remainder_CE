use super::*;
use remainder_shared_types::halo2curves::bn256::Fr;
use remainder_shared_types::halo2curves::bn256::G1 as Bn256Point;
/// Tests for the Pedersen commitment scheme using the BN254 (aka BN256) curve and its scalar field (Fr).
use remainder_shared_types::halo2curves::CurveExt;
use remainder_shared_types::transcript::ec_transcript::ECTranscriptReader;
use remainder_shared_types::transcript::ec_transcript::ECTranscriptWriter;
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;

type Base = <Bn256Point as CurveExt>::Base;

// a static string constant
const INIT_STR: &str = "modulus modulus modulus modulus modulus modulus";

#[test]
fn test_completeness() {
    let committer = PedersenCommitter::<Bn256Point>::new(1, INIT_STR, None);

    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("testing proof of product (completeness) - prover");

    let x = committer.committed_scalar(&Fr::from(7_u64), &Fr::from(4211_u64));
    let y = committer.committed_scalar(&Fr::from(4_u64), &Fr::from(112_u64));
    let z = committer.committed_scalar(&Fr::from(28_u64), &Fr::from(741_u64));
    let proof = ProofOfProduct::prove(
        &x,
        &y,
        &z,
        &committer,
        &mut rand::thread_rng(),
        &mut prover_transcript,
    );

    let transcript = prover_transcript.get_transcript();
    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(transcript);

    proof.verify(
        x.commitment,
        y.commitment,
        z.commitment,
        &committer,
        &mut verifier_transcript,
    );
}

#[test]
#[should_panic]
fn test_soundness() {
    // test soundness - try to verify when the product relation doesn't hold
    let committer = PedersenCommitter::<Bn256Point>::new(1, INIT_STR, None);

    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("testing proof of product (soundness) - prover");

    let x = committer.committed_scalar(&Fr::from(7_u64), &Fr::from(20_u64));
    let y = committer.committed_scalar(&Fr::from(4_u64), &Fr::from(31_u64));
    let z = committer.committed_scalar(&Fr::from(28_u64).neg(), &Fr::from(414_u64));
    let proof = ProofOfProduct::prove(
        &x,
        &y,
        &z,
        &committer,
        &mut rand::thread_rng(),
        &mut prover_transcript,
    );

    let transcript = prover_transcript.get_transcript();
    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(transcript);

    proof.verify(
        x.commitment,
        y.commitment,
        z.commitment,
        &committer,
        &mut verifier_transcript,
    );
}
