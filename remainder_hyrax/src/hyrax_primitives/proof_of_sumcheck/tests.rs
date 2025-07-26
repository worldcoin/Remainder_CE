use crate::hyrax_gkr::hyrax_layer::commit_to_post_sumcheck_layer;
use crate::hyrax_gkr::hyrax_layer::committed_scalar_psl_as_commitments;
use crate::utils::vandermonde::VandermondeInverse;

use super::*;
use remainder::layer::product::PostSumcheckLayerTree;
use remainder::layer::LayerId;
use remainder::mle::betavalues::BetaValues;
use remainder::mle::dense::DenseMle;
use remainder::mle::Mle;
use remainder_shared_types::curves::ConstantRng;
use remainder_shared_types::halo2curves::bn256::Fr;
use remainder_shared_types::halo2curves::bn256::G1 as Bn256Point;
use remainder_shared_types::halo2curves::group::Group;
use remainder_shared_types::halo2curves::CurveExt;
use remainder_shared_types::transcript::ec_transcript::ECTranscript;
use remainder_shared_types::transcript::poseidon_sponge::PoseidonSponge;

type Base = <Bn256Point as CurveExt>::Base;

// a static string constant
const INIT_STR: &str = "modulus modulus modulus modulus modulus modulus";

#[test]
fn test_calculate_j_star() {
    let bindings = vec![Fr::from(7), Fr::from(4), Fr::from(28), Fr::from(23)];
    let rhos = vec![
        Fr::from(1),
        Fr::from(2),
        Fr::from(3),
        Fr::from(4),
        Fr::from(5),
    ];
    let gammas = vec![Fr::from(11), Fr::from(12), Fr::from(13), Fr::from(14)];
    let degree = 2;
    let j_star = ProofOfSumcheck::<Bn256Point>::calculate_j_star(&bindings, &rhos, &gammas, degree);
    assert_eq!(j_star.len(), 12);
    assert_eq!(
        j_star[0],
        gammas[0].invert().unwrap() * (rhos[0] * Fr::from(2) - rhos[1] * Fr::one())
    );
    assert_eq!(
        j_star[1],
        gammas[0].invert().unwrap() * (rhos[0] * Fr::from(1) - rhos[1] * bindings[0])
    );
    assert_eq!(
        j_star[2],
        gammas[0].invert().unwrap() * (rhos[0] * Fr::from(1) - rhos[1] * bindings[0] * bindings[0])
    );
    assert_eq!(
        j_star[3],
        gammas[1].invert().unwrap() * (rhos[1] * Fr::from(2) - rhos[2] * Fr::one())
    );
    assert_eq!(
        j_star[11],
        gammas[3].invert().unwrap() * (rhos[3] * Fr::from(1) - rhos[4] * bindings[3] * bindings[3])
    );
}

#[test]
fn test_completeness() {
    let committer = PedersenCommitter::<Bn256Point>::new(5, INIT_STR, None);

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    // the mle
    let v00 = Fr::from(7);
    let v01 = Fr::from(28);
    let v10 = Fr::from(4);
    let v11 = Fr::from(23);
    let mut mle = DenseMle::new_from_raw(vec![v00, v01, v10, v11], LayerId::Input(0));
    // the sum
    let sum = v00 + v01 + v10 + v11;
    let sum_commit = committer.committed_scalar(&sum, &Fr::from(2));
    // the bindings
    let r1 = Fr::from(3).neg();
    let r2 = Fr::from(2);
    let bindings = vec![r1, r2];
    // bind the variables
    mle.index_mle_indices(0);
    mle.fix_variable(0, r1);
    mle.fix_variable(1, r2);
    let mle_eval = mle.value();
    // first sumcheck message f1
    let f10 = v00 + v01;
    let f11 = v10 + v11 - v00 - v01;
    assert_eq!(f10 + f10 + f11, sum); // f1(0) + f1(1) = sum
    let f1_padded = vec![f10, f11, Fr::zero(), Fr::zero()];
    let message1 = committer.committed_vector(&f1_padded, &Fr::from(6));
    // second sumcheck message f2
    let f20 = (Fr::one() - r1) * v00 + r1 * v10;
    let f21 = (Fr::one() - r1) * (v01 - v00) + r1 * (v11 - v10);
    assert_eq!(f10 + r1 * f11, f20 + f20 + f21); // f1(r1) = f2(0) + f2(1)
    let f2_padded = vec![Fr::zero(), Fr::zero(), f20, f21];
    let message2 = committer.committed_vector(&f2_padded, &Fr::from(7));
    assert_eq!(f20 + r2 * f21, mle_eval); // f2(r2) = mle_eval
    let post_sumcheck_layer = commit_to_post_sumcheck_layer(
        &PostSumcheckLayerTree::<Fr, Fr>::mle(&mle),
        &committer,
        &mut rand::thread_rng(),
    );
    let proof = ProofOfSumcheck::prove(
        &sum_commit,
        &vec![message1, message2],
        1, // the degree of the messages
        &post_sumcheck_layer,
        &bindings,
        &committer,
        &mut rand::thread_rng(),
        &mut transcript,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let psl_as_commits = committed_scalar_psl_as_commitments(&post_sumcheck_layer);

    proof.verify(
        &sum_commit.commitment,
        1, // the degree of the messages
        &psl_as_commits,
        &bindings,
        &committer,
        &mut transcript,
    );
}

#[test]
fn test_example_with_regular_layer() {
    let committer = PedersenCommitter::<Bn256Point>::new_with_generators(
        vec![
            Bn256Point::generator(),
            Bn256Point::generator(),
            Bn256Point::generator(),
            Bn256Point::generator(),
            Bn256Point::generator(),
            Bn256Point::generator(),
            Bn256Point::generator(),
        ],
        Bn256Point::identity(),
        None,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    // the mle
    let v00 = Fr::from(1);
    let v10 = Fr::from(1);
    let v01 = Fr::from(1);
    let v11 = Fr::from(1);
    let mut mle = DenseMle::new_from_raw(vec![v00, v10, v01, v11], LayerId::Input(0));
    let layer_claim_for_beta = vec![Fr::one(), Fr::one()];
    let mut equality_mle = BetaValues::new_beta_equality_mle(layer_claim_for_beta);
    let mut constant_rng = ConstantRng::new(1);

    // the sum
    let sum = Fr::one();
    let sum_commit = committer.committed_scalar(&sum, &Fr::one());

    // the bindings
    let r1 = Fr::one();
    let r2 = Fr::one();
    let bindings = vec![r1, r2];

    // bind the variables
    mle.index_mle_indices(0);

    // first sumcheck message f1
    let f1_padded = vec![
        Fr::zero(),
        Fr::one(),
        Fr::zero(),
        Fr::zero(),
        Fr::zero(),
        Fr::zero(),
    ];
    let message1 = committer.committed_vector(&f1_padded, &Fr::one());

    // second sumcheck message f2
    equality_mle.fix_variable(r1);
    mle.fix_variable(0, r1);

    let evaluations = [Fr::from(0), Fr::from(1), Fr::from(2)];
    let mut converter = VandermondeInverse::<Fr>::new();
    let coefficients = converter.convert_to_coefficients(evaluations.to_vec());
    assert_eq!(coefficients, vec![Fr::from(0), Fr::from(1), Fr::from(0)]);

    let f2_padded = vec![
        Fr::zero(),
        Fr::zero(),
        Fr::zero(),
        Fr::from(0),
        Fr::from(1),
        Fr::from(0),
    ];
    let message2 = committer.committed_vector(&f2_padded, &Fr::from(7));

    equality_mle.fix_variable(r2);
    mle.fix_variable(1, r2);

    let _mle_eval = mle.value() * equality_mle.value();
    let post_sumcheck_layer = PostSumcheckLayerTree::<Fr, Fr>::mult(
        PostSumcheckLayerTree::<Fr, Fr>::mle(&mle),
        PostSumcheckLayerTree::constant(equality_mle.value())
    );
    // deliberately avoiding the cleanup to test for prover - verifier coherence
    // post_sumcheck_layer.remove_add_values(false);
    let post_sumcheck_layer = commit_to_post_sumcheck_layer(
        &post_sumcheck_layer,
        &committer,
        &mut rand::thread_rng(),
    );
    let proof = ProofOfSumcheck::prove(
        &sum_commit,
        &vec![message1, message2],
        2, // the degree of the messages
        &post_sumcheck_layer,
        &bindings,
        &committer,
        &mut constant_rng,
        &mut transcript,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let committed_psl = committed_scalar_psl_as_commitments(&post_sumcheck_layer);

    proof.verify(
        &sum_commit.commitment,
        2, // the degree of the messages
        &committed_psl,
        &bindings,
        &committer,
        &mut transcript,
    );
}

#[test]
#[should_panic]
fn test_soundness() {
    // use the same setup as in the completeness test, but change the sumcheck messages to
    // something random.  Also, cheat in the choice of padding.
    let committer = PedersenCommitter::<Bn256Point>::new(5, INIT_STR, None);

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let sum_commit = committer.committed_scalar(&Fr::from(2), &Fr::from(2));
    let f1_padded = vec![Fr::from(123), Fr::from(123), Fr::from(123), Fr::from(333)];
    let message1 = committer.committed_vector(&f1_padded, &Fr::from(6));
    // second sumcheck message f2
    let f2_padded = vec![Fr::from(555), Fr::from(555), Fr::from(111), Fr::from(1)];
    let message2 = committer.committed_vector(&f2_padded, &Fr::from(7));
    // the mle
    let v00 = Fr::from(7);
    let v01 = Fr::from(4);
    let v10 = Fr::from(28);
    let v11 = Fr::from(23);
    let mut mle = DenseMle::new_from_raw(vec![v00, v10, v01, v11], LayerId::Input(0));
    let bindings = vec![Fr::from(3), Fr::from(2)];
    mle.fix_variable(1, bindings[0]);
    mle.fix_variable(2, bindings[1]);
    let post_sumcheck_layer = commit_to_post_sumcheck_layer(
        &PostSumcheckLayerTree::<Fr, Fr>::mle(&mle),
        &committer,
        &mut rand::thread_rng(),
    );
    let proof = ProofOfSumcheck::prove(
        &sum_commit,
        &vec![message1, message2],
        1, // the degree of the messages
        &post_sumcheck_layer,
        &bindings,
        &committer,
        &mut rand::thread_rng(),
        &mut transcript,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let committed_psl = committed_scalar_psl_as_commitments(&post_sumcheck_layer);
    proof.verify(
        &sum_commit.commitment,
        1, // the degree of the messages
        &committed_psl,
        &bindings,
        &committer,
        &mut transcript,
    );
}
