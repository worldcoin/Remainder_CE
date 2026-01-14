use super::*;
use shared_types::halo2curves::bn256::Fr;
use shared_types::halo2curves::bn256::G1 as Bn256Point;
/// Tests for the Pedersen commitment scheme using the BN254 (aka BN256) curve and its scalar field (Fr).
use shared_types::halo2curves::CurveExt;
use shared_types::transcript::ec_transcript::ECTranscript;
use shared_types::transcript::poseidon_sponge::PoseidonSponge;

type Base = <Bn256Point as CurveExt>::Base;

// a static string constant
const INIT_STR: &str = "modulus modulus modulus modulus modulus modulus";

#[test]
fn test_completeness() {
    let committer = PedersenCommitter::<Bn256Point>::new(1, INIT_STR, None);

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let x = committer.committed_scalar(&Fr::from(7_u64), &Fr::from(4211_u64));
    let y = committer.committed_scalar(&Fr::from(4_u64), &Fr::from(112_u64));
    let z = committer.committed_scalar(&Fr::from(28_u64), &Fr::from(741_u64));
    let proof = ProofOfProduct::prove(
        &x,
        &y,
        &z,
        &committer,
        &mut rand::thread_rng(),
        &mut transcript,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    proof.verify(
        x.commitment,
        y.commitment,
        z.commitment,
        &committer,
        &mut transcript,
    );
}

#[test]
#[should_panic]
fn test_soundness() {
    // test soundness - try to verify when the product relation doesn't hold
    let committer = PedersenCommitter::<Bn256Point>::new(1, INIT_STR, None);

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let x = committer.committed_scalar(&Fr::from(7_u64), &Fr::from(20_u64));
    let y = committer.committed_scalar(&Fr::from(4_u64), &Fr::from(31_u64));
    let z = committer.committed_scalar(&Fr::from(28_u64).neg(), &Fr::from(414_u64));
    let proof = ProofOfProduct::prove(
        &x,
        &y,
        &z,
        &committer,
        &mut rand::thread_rng(),
        &mut transcript,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    proof.verify(
        x.commitment,
        y.commitment,
        z.commitment,
        &committer,
        &mut transcript,
    );
}
