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

    let x = committer.committed_scalar(&Fr::from(23_u64), &Fr::from(2_u64));
    let proof = ProofOfOpening::prove(&x, &committer, &mut rand::thread_rng(), &mut transcript);

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    proof.verify(x.commitment, &committer, &mut transcript);
}

#[test]
#[should_panic]
fn test_soundness() {
    // test soundness - try to verify for a different commitment value than the one in the proof
    let committer = PedersenCommitter::<Bn256Point>::new(1, INIT_STR, None);

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let x = committer.committed_scalar(&Fr::from(23_u64), &Fr::from(2_u64));
    let proof = ProofOfOpening::prove(&x, &committer, &mut rand::thread_rng(), &mut transcript);

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    proof.verify(x.commitment + x.commitment, &committer, &mut transcript);
}
