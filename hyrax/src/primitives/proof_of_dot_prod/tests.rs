use super::*;
use shared_types::halo2curves::bn256::Fr;
use shared_types::halo2curves::bn256::G1 as Bn256Point;
/// Tests for the Pedersen commitment scheme using the BN254 (aka BN256) curve and its scalar field (Fr).
use shared_types::halo2curves::CurveExt;
use shared_types::transcript::ec_transcript::ECTranscript;
use shared_types::transcript::poseidon_sponge::PoseidonSponge;

type Base = <Bn256Point as CurveExt>::Base;

#[test]
// if the dot product is computed correctly, the evaluation proof should pass the verifier's check.
fn sanity_check_test_honest_prover() {
    let committer = PedersenCommitter::<Bn256Point>::new(
        3,
        "sanity checksanity checksanity checksanity check",
        None,
    );
    let x = committer.committed_vector(&[Fr::one(), Fr::one()], &Fr::one());
    let y = committer.committed_scalar(&(Fr::one() + Fr::one()), &Fr::one());
    let a = (0..2).map(|_| Fr::one()).collect_vec();
    let prover_random_generator = &mut rand::thread_rng();

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let eval_proof = ProofOfDotProduct::prove(
        &x,
        &y,
        &a,
        &committer,
        prover_random_generator,
        &mut transcript,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    eval_proof.verify(
        &x.commitment,
        &y.commitment,
        &a,
        &committer,
        &mut transcript,
    );
}

#[test]
// if the dot product is computed correctly, the evaluation proof should pass the verifier's check.
// with slightly larger vectors and actual randomness.
fn sanity_check_honest_prover_2() {
    let committer = PedersenCommitter::<Bn256Point>::new(
        (1 << 5) + 1,
        "another sanity checkanother sanity checkanother sanity check",
        None,
    );
    let x = committer.committed_vector(
        &(0..(1 << 5))
            .map(|_| Fr::from(rand::random::<u8>() as u64))
            .collect_vec(),
        &Fr::from(rand::random::<u8>() as u64),
    );
    let a = (0..(1 << 5))
        .map(|_| Fr::from(rand::random::<u8>() as u64))
        .collect_vec();
    let y = committer.committed_scalar(
        &ProofOfDotProduct::<Bn256Point>::compute_dot_product(&x.value, &a),
        &Fr::from(rand::random::<u8>() as u64),
    );
    let prover_random_generator = &mut rand::thread_rng();

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let eval_proof = ProofOfDotProduct::prove(
        &x,
        &y,
        &a,
        &committer,
        prover_random_generator,
        &mut transcript,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    eval_proof.verify(
        &x.commitment,
        &y.commitment,
        &a,
        &committer,
        &mut transcript,
    );
}
