use ark_std::end_timer;
use ark_std::start_timer;
use itertools::Itertools;
use rand::rngs::OsRng;
use rand::RngCore;
use remainder_shared_types::halo2curves::bn256::G1 as Bn256Point;
/// Tests for the Pedersen commitment scheme using the BN254 (aka BN256) curve and its scalar field (Fr).
use remainder_shared_types::halo2curves::group::Group;
use remainder_shared_types::halo2curves::CurveExt;
use remainder_shared_types::pedersen::PedersenCommitter;
use remainder_shared_types::transcript::ec_transcript::ECTranscript;
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;

use crate::hyrax_pcs::HyraxPCSEvaluationProof;
use crate::hyrax_pcs::MleCoefficientsVector;

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
        committer.blinding_generator,
        Some(committer.int_abs_val_bitwidth),
    );
    let input_layer_mle_coeff =
        MleCoefficientsVector::ScalarFieldVector((0..4).map(|_| Scalar::one()).collect_vec());
    let challenge_coordinates = (0..2).map(|_| Scalar::one()).collect_vec();
    let mle_evaluation_at_challenge = Scalar::one();
    let log_split_point = 1;
    let prover_random_generator = &mut rand::thread_rng();
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let blinding_factors_matrix_rows = (0..2).map(|_| Scalar::one()).collect_vec();

    let mut seed_matrix = [0u8; 32];
    OsRng.fill_bytes(&mut seed_matrix);
    let mut seed_eval = [0u8; 32];
    OsRng.fill_bytes(&mut seed_eval);

    let comm_to_matrix = HyraxPCSEvaluationProof::compute_matrix_commitments(
        log_split_point,
        &input_layer_mle_coeff,
        &committer,
        &blinding_factors_matrix_rows,
    );

    let hyrax_eval_proof = HyraxPCSEvaluationProof::prove(
        log_split_point,
        &input_layer_mle_coeff,
        &challenge_coordinates,
        &mle_evaluation_at_challenge,
        &committer_copy,
        prover_random_generator,
        &mut transcript,
        &blinding_factors_matrix_rows,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    hyrax_eval_proof.verify(
        log_split_point,
        &committer,
        &comm_to_matrix,
        &challenge_coordinates,
        &mut transcript,
    );
}

#[test]
/// test on a 2 x 4 matrix of all identity elements
fn sanity_check_test_honest_prover_small_asymmetric_one() {
    let committer = PedersenCommitter::<Bn256Point>::new(
        9,
        "all onesall onesall onesall onesall onesall onesall ones",
        None,
    );
    let committer_copy = PedersenCommitter::new_with_generators(
        committer.generators.clone(),
        committer.blinding_generator,
        Some(committer.int_abs_val_bitwidth),
    );
    let input_layer_mle_coeff =
        MleCoefficientsVector::ScalarFieldVector((0..8).map(|_| Scalar::one()).collect_vec());
    let challenge_coordinates = (0..3).map(|_| Scalar::one()).collect_vec();
    let mle_evaluation_at_challenge = Scalar::one();

    let log_split_point = 1;
    let prover_random_generator = &mut rand::thread_rng();
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let blinding_factors_matrix_rows = (0..4).map(|_| Scalar::zero()).collect_vec();

    let comm_to_matrix = HyraxPCSEvaluationProof::compute_matrix_commitments(
        log_split_point,
        &input_layer_mle_coeff,
        &committer,
        &blinding_factors_matrix_rows,
    );

    let hyrax_eval_proof = HyraxPCSEvaluationProof::prove(
        log_split_point,
        &input_layer_mle_coeff,
        &challenge_coordinates,
        &mle_evaluation_at_challenge,
        &committer_copy,
        prover_random_generator,
        &mut transcript,
        &blinding_factors_matrix_rows,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    hyrax_eval_proof.verify(
        log_split_point,
        &committer,
        &comm_to_matrix,
        &challenge_coordinates,
        &mut transcript,
    );
}

#[test]
/// test on a 4 x 2 of all random elements
fn sanity_check_test_honest_prover_small_asymmetric_random() {
    let committer = PedersenCommitter::<Bn256Point>::new(3, "asymmetric proverasymmetric proverasymmetric proverasymmetric proverasymmetric proverasymmetric prover", None);
    let committer_copy = PedersenCommitter::new_with_generators(
        committer.generators.clone(),
        committer.blinding_generator,
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
        challenge_coordinates[0],
        challenge_coordinates[1],
        challenge_coordinates[2],
    );

    let challenge_vec = vec![
        (Scalar::one() - r_1) * (Scalar::one() - r_2) * (Scalar::one() - r_3),
        (Scalar::one() - r_1) * (Scalar::one() - r_2) * (r_3),
        (Scalar::one() - r_1) * (r_2) * (Scalar::one() - r_3),
        (Scalar::one() - r_1) * (r_2) * (r_3),
        (r_1) * (Scalar::one() - r_2) * (Scalar::one() - r_3),
        (r_1) * (Scalar::one() - r_2) * (r_3),
        (r_1) * (r_2) * (Scalar::one() - r_3),
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
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let blinding_factors_matrix_rows = (0..4)
        .map(|_| Scalar::from(rand::random::<u64>()))
        .collect_vec();

    let comm_to_matrix = HyraxPCSEvaluationProof::compute_matrix_commitments(
        log_split_point,
        &input_layer_mle_coeff,
        &committer,
        &blinding_factors_matrix_rows,
    );

    let hyrax_eval_proof = HyraxPCSEvaluationProof::prove(
        log_split_point,
        &input_layer_mle_coeff,
        &challenge_coordinates,
        &mle_evaluation_at_challenge,
        &committer_copy,
        prover_random_generator,
        &mut transcript,
        &blinding_factors_matrix_rows,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    hyrax_eval_proof.verify(
        log_split_point,
        &committer,
        &comm_to_matrix,
        &challenge_coordinates,
        &mut transcript,
    );
}

#[ignore] // takes a long time to run!
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
        committer.blinding_generator,
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

    let (challenge_vec, _) = HyraxPCSEvaluationProof::<Bn256Point>::compute_l_r_from_log_n_cols(
        0,
        &challenge_coordinates,
    );

    let mle_evaluation_at_challenge = input_layer_mle_coeff_raw_vec
        .iter()
        .zip(challenge_vec.iter())
        .fold(Scalar::zero(), |acc, (mle_coeff, challenge_eval)| {
            acc + (*mle_coeff * challenge_eval)
        });
    let log_split_point = 9;
    let prover_random_generator = &mut rand::thread_rng();
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let blinding_factors_matrix_rows = (0..(1 << 9))
        .map(|_| Scalar::from(rand::random::<u64>()))
        .collect_vec();

    let comm_to_matrix = HyraxPCSEvaluationProof::compute_matrix_commitments(
        log_split_point,
        &input_layer_mle_coeff,
        &committer,
        &blinding_factors_matrix_rows,
    );

    let hyrax_eval_proof = HyraxPCSEvaluationProof::prove(
        log_split_point,
        &input_layer_mle_coeff,
        &challenge_coordinates,
        &mle_evaluation_at_challenge,
        &committer_copy,
        prover_random_generator,
        &mut transcript,
        &blinding_factors_matrix_rows,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    hyrax_eval_proof.verify(
        log_split_point,
        &committer,
        &comm_to_matrix,
        &challenge_coordinates,
        &mut transcript,
    );
}

#[test]
/// Test on a 2^9 x 2^9 matrix with all zeroes to test internal scalar mult optimization,
/// to see if only doing double-and-add for the significant bits makes a difference.
fn sanity_check_test_honest_prover_iris_size_symmetric_all_zero() {
    let committer = PedersenCommitter::<Bn256Point>::new(
        (1 << 9) + 1,
        "zerozerozerozerozerozerozerozero",
        None,
    );
    let input_layer_mle_coeff_raw_vec = (0..(1 << 18)).map(|_| Scalar::zero()).collect_vec();
    let input_layer_mle_coeff =
        MleCoefficientsVector::ScalarFieldVector(input_layer_mle_coeff_raw_vec.clone());

    let blinding_factors_matrix_rows = (0..(1 << 9))
        .map(|_| Scalar::from(rand::random::<u64>()))
        .collect_vec();

    let commit_timer = start_timer!(|| "commit time");
    HyraxPCSEvaluationProof::compute_matrix_commitments(
        9,
        &input_layer_mle_coeff,
        &committer,
        &blinding_factors_matrix_rows,
    );
    end_timer!(commit_timer);
}
