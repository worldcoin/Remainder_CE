use super::*;
use remainder_shared_types::halo2curves::bn256::Fr;
use remainder_shared_types::halo2curves::bn256::G1 as Bn256Point;
/// Tests for the Pedersen commitment scheme using the BN254 (aka BN256) curve and its scalar field (Fr).
use remainder_shared_types::halo2curves::CurveExt;
use remainder_shared_types::transcript::ec_transcript::ECTranscript;
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;

type Base = <Bn256Point as CurveExt>::Base;

// a static string constant
const INIT_STR: &str = "modulus modulus modulus modulus modulus modulus";

#[test]
fn test_completeness() {
    let committer = PedersenCommitter::<Bn256Point>::new(1, INIT_STR, None);

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let commit0 = committer.committed_scalar(&Fr::from(23_u64), &Fr::from(2_u64));
    let commit1 = committer.committed_scalar(&Fr::from(23_u64), &Fr::from(3_u64));
    let proof = ProofOfEquality::prove(
        &commit0,
        &commit1,
        &committer,
        &mut rand::thread_rng(),
        &mut transcript,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    proof.verify(
        commit0.commitment,
        commit1.commitment,
        &committer,
        &mut transcript,
    );
}

#[test]
#[should_panic]
fn test_soundness() {
    // test soundness - try to verify for two commitments that are not to the same value
    let committer = PedersenCommitter::<Bn256Point>::new(1, INIT_STR, None);

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let commit0 = committer.committed_scalar(&Fr::from(23_u64), &Fr::from(2_u64));
    let commit1 = committer.committed_scalar(&Fr::from(41_u64), &Fr::from(3_u64));
    let proof = ProofOfEquality::prove(
        &commit0,
        &commit1,
        &committer,
        &mut rand::thread_rng(),
        &mut transcript,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    proof.verify(
        commit0.commitment,
        commit1.commitment,
        &committer,
        &mut transcript,
    );
}
