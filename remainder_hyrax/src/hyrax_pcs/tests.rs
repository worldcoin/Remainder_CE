use super::*;
use rand::rngs::OsRng;
use rand::RngCore;
use remainder_shared_types::halo2curves::bn256::G1 as Bn256Point;
/// Tests for the Pedersen commitment scheme using the BN254 (aka BN256) curve and its scalar field (Fr).
use remainder_shared_types::halo2curves::group::Group;
use remainder_shared_types::halo2curves::CurveExt;
use remainder_shared_types::transcript::ec_transcript::ECTranscriptReader;
use remainder_shared_types::transcript::ec_transcript::ECTranscriptWriter;
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;

type Scalar = <Bn256Point as Group>::Scalar;
type Base = <Bn256Point as CurveExt>::Base;

#[test]
/// test on a 2 x 2 matrix of all identity elements
fn sanity_check_test_honest_prover_small_identity() {
    let committer = PedersenCommitter::<Bn256Point>::new(
        3,
        "small prover small proversmall proversmall proversmall prover",
        None,
    );
    let committer_copy = PedersenCommitter::new_with_generators(
        committer.generators.clone(),
        committer.blinding_generator.clone(),
        Some(committer.int_abs_val_bitwidth),
    );
    let input_layer_mle_coeff =
        MleCoefficientsVector::ScalarFieldVector((0..4).map(|_| Scalar::one()).collect_vec());
    let challenge_coordinates = (0..2).map(|_| Scalar::one()).collect_vec();
    let mle_evaluation_at_challenge = Scalar::one();
    let log_split_point = 1;
    let prover_random_generator = &mut rand::thread_rng();
    let mut prover_transcript = ECTranscriptWriter::<Bn256Point, PoseidonSponge<Base>>::new(
        "testing proof of dot product - prover",
    );

    let blinding_factors_matrix_rows = (0..2).map(|_| Scalar::one()).collect_vec();
    let blinding_factor_evaluation = Scalar::one();

    let mut seed_matrix = [0u8; 32];
    OsRng.fill_bytes(&mut seed_matrix);
    let mut seed_eval = [0u8; 32];
    OsRng.fill_bytes(&mut seed_eval);

    let comm_to_matrix = HyraxPCSProof::compute_matrix_commitments(
        log_split_point,
        &input_layer_mle_coeff,
        &committer,
        &blinding_factors_matrix_rows,
    );

    let hyrax_eval_proof = HyraxPCSProof::prove(
        log_split_point,
        &input_layer_mle_coeff,
        &challenge_coordinates,
        &mle_evaluation_at_challenge,
        &committer_copy,
        blinding_factor_evaluation,
        prover_random_generator,
        &mut prover_transcript,
        &blinding_factors_matrix_rows,
    );

    let mut verifier_transcript = ECTranscriptReader::<Bn256Point, PoseidonSponge<Base>>::new(
        prover_transcript.get_transcript(),
    );
    hyrax_eval_proof.verify_hyrax_evaluation_proof(
        log_split_point,
        &committer,
        &comm_to_matrix,
        &challenge_coordinates,
        &mut verifier_transcript,
    );
}

#[test]
/// test on a 4 x 2 matrix of all identity elements
fn sanity_check_test_honest_prover_small_asymmetric_one() {
    let committer = PedersenCommitter::<Bn256Point>::new(
        9,
        "all onesall onesall onesall onesall onesall onesall ones",
        None,
    );
    let committer_copy = PedersenCommitter::new_with_generators(
        committer.generators.clone(),
        committer.blinding_generator.clone(),
        Some(committer.int_abs_val_bitwidth),
    );
    let input_layer_mle_coeff =
        MleCoefficientsVector::ScalarFieldVector((0..8).map(|_| Scalar::one()).collect_vec());
    let challenge_coordinates = (0..3).map(|_| Scalar::one()).collect_vec();
    let mle_evaluation_at_challenge = Scalar::one();

    let log_split_point = 1;
    let prover_random_generator = &mut rand::thread_rng();
    let mut prover_transcript = ECTranscriptWriter::<Bn256Point, PoseidonSponge<Base>>::new(
        "testing proof of dot product - prover",
    );

    let blinding_factors_matrix_rows = (0..4).map(|_| Scalar::one()).collect_vec();
    let blinding_factor_evaluation = Scalar::one();

    let comm_to_matrix = HyraxPCSProof::compute_matrix_commitments(
        log_split_point,
        &input_layer_mle_coeff,
        &committer,
        &blinding_factors_matrix_rows,
    );

    let hyrax_eval_proof = HyraxPCSProof::prove(
        log_split_point,
        &input_layer_mle_coeff,
        &challenge_coordinates,
        &mle_evaluation_at_challenge,
        &committer_copy,
        blinding_factor_evaluation,
        prover_random_generator,
        &mut prover_transcript,
        &blinding_factors_matrix_rows,
    );

    let mut verifier_transcript = ECTranscriptReader::<Bn256Point, PoseidonSponge<Base>>::new(
        prover_transcript.get_transcript(),
    );
    hyrax_eval_proof.verify_hyrax_evaluation_proof(
        log_split_point,
        &committer,
        &comm_to_matrix,
        &challenge_coordinates,
        &mut verifier_transcript,
    );
}

#[test]
/// test on a 4 x 2 of all random elements
fn sanity_check_test_honest_prover_small_asymmetric_random() {
    let committer = PedersenCommitter::<Bn256Point>::new(3, "asymmetric proverasymmetric proverasymmetric proverasymmetric proverasymmetric proverasymmetric prover", None);
    let committer_copy = PedersenCommitter::new_with_generators(
        committer.generators.clone(),
        committer.blinding_generator.clone(),
        Some(committer.int_abs_val_bitwidth),
    );
    let input_layer_mle_coeff_raw_vec = (0..8)
        .map(|_| Scalar::from(rand::random::<u64>()))
        .collect_vec();
    let input_layer_mle_coeff =
        MleCoefficientsVector::ScalarFieldVector(input_layer_mle_coeff_raw_vec.clone());
    let challenge_coordinates = (0..3)
        .map(|_| Scalar::from(rand::random::<u64>()))
        .collect_vec();

    let (r_1, r_2, r_3) = (
        challenge_coordinates[0].clone(),
        challenge_coordinates[1].clone(),
        challenge_coordinates[2].clone(),
    );

    let challenge_vec = vec![
        (Scalar::one() - r_1) * (Scalar::one() - r_2) * (Scalar::one() - r_3),
        (r_1) * (Scalar::one() - r_2) * (Scalar::one() - r_3),
        (Scalar::one() - r_1) * (r_2) * (Scalar::one() - r_3),
        (r_1) * (r_2) * (Scalar::one() - r_3),
        (Scalar::one() - r_1) * (Scalar::one() - r_2) * (r_3),
        (r_1) * (Scalar::one() - r_2) * (r_3),
        (Scalar::one() - r_1) * (r_2) * (r_3),
        (r_1) * (r_2) * (r_3),
    ];
    let mle_evaluation_at_challenge = input_layer_mle_coeff_raw_vec
        .iter()
        .zip(challenge_vec.iter())
        .fold(Scalar::zero(), |acc, (mle_coeff, challenge_eval)| {
            acc + (*mle_coeff * challenge_eval)
        });
    let log_split_point = 1;
    let prover_random_generator = &mut rand::thread_rng();
    let mut prover_transcript = ECTranscriptWriter::<Bn256Point, PoseidonSponge<Base>>::new(
        "testing proof of dot product - prover",
    );

    let blinding_factors_matrix_rows = (0..4)
        .map(|_| Scalar::from(rand::random::<u64>()))
        .collect_vec();
    let blinding_factor_evaluation = Scalar::from(rand::random::<u64>());

    let comm_to_matrix = HyraxPCSProof::compute_matrix_commitments(
        log_split_point,
        &input_layer_mle_coeff,
        &committer,
        &blinding_factors_matrix_rows,
    );

    let hyrax_eval_proof = HyraxPCSProof::prove(
        log_split_point,
        &input_layer_mle_coeff,
        &challenge_coordinates,
        &mle_evaluation_at_challenge,
        &committer_copy,
        blinding_factor_evaluation,
        prover_random_generator,
        &mut prover_transcript,
        &blinding_factors_matrix_rows,
    );

    let mut verifier_transcript = ECTranscriptReader::<Bn256Point, PoseidonSponge<Base>>::new(
        prover_transcript.get_transcript(),
    );
    hyrax_eval_proof.verify_hyrax_evaluation_proof(
        log_split_point,
        &committer,
        &comm_to_matrix,
        &challenge_coordinates,
        &mut verifier_transcript,
    );
}

#[test]
/// test on a 2^9 x 2^9 matrix with all random elements
fn sanity_check_test_honest_prover_iris_size_symmetric_random() {
    let committer = PedersenCommitter::<Bn256Point>::new(
        (1 << 9) + 1,
        "random symmetricrandom symmetricrandom symmetric",
        None,
    );
    let committer_copy = PedersenCommitter::new_with_generators(
        committer.generators.clone(),
        committer.blinding_generator.clone(),
        Some(committer.int_abs_val_bitwidth),
    );
    let input_layer_mle_coeff_raw_vec = (0..(1 << 18))
        .map(|_| Scalar::from(rand::random::<u64>()))
        .collect_vec();
    let input_layer_mle_coeff =
        MleCoefficientsVector::ScalarFieldVector(input_layer_mle_coeff_raw_vec.clone());
    let challenge_coordinates = (0..18)
        .map(|_| Scalar::from(rand::random::<u64>()))
        .collect_vec();

    let (challenge_vec, _) =
        HyraxPCSProof::<Bn256Point>::compute_l_r_from_log_n_cols(0, &challenge_coordinates);

    let mle_evaluation_at_challenge = input_layer_mle_coeff_raw_vec
        .iter()
        .zip(challenge_vec.iter())
        .fold(Scalar::zero(), |acc, (mle_coeff, challenge_eval)| {
            acc + (*mle_coeff * challenge_eval)
        });
    let log_split_point = 9;
    let prover_random_generator = &mut rand::thread_rng();
    let mut prover_transcript = ECTranscriptWriter::<Bn256Point, PoseidonSponge<Base>>::new(
        "testing proof of dot product - prover",
    );

    let blinding_factors_matrix_rows = (0..(1 << 9))
        .map(|_| Scalar::from(rand::random::<u64>()))
        .collect_vec();
    let blinding_factor_evaluation = Scalar::from(rand::random::<u64>());

    let comm_to_matrix = HyraxPCSProof::compute_matrix_commitments(
        log_split_point,
        &input_layer_mle_coeff,
        &committer,
        &blinding_factors_matrix_rows,
    );

    let hyrax_eval_proof = HyraxPCSProof::prove(
        log_split_point,
        &input_layer_mle_coeff,
        &challenge_coordinates,
        &mle_evaluation_at_challenge,
        &committer_copy,
        blinding_factor_evaluation,
        prover_random_generator,
        &mut prover_transcript,
        &blinding_factors_matrix_rows,
    );

    let mut verifier_transcript = ECTranscriptReader::<Bn256Point, PoseidonSponge<Base>>::new(
        prover_transcript.get_transcript(),
    );
    hyrax_eval_proof.verify_hyrax_evaluation_proof(
        log_split_point,
        &committer,
        &comm_to_matrix,
        &challenge_coordinates,
        &mut verifier_transcript,
    );
}
