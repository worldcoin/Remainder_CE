use super::*;
use crate::{
    expression::generic_expr::ExpressionNode,
    layer::{
        // claims::tests::claim_aggregation_testing_wrapper,
        claims::Claim,
        claims::ClaimGroup,
        from_mle,
        GKRLayer,
        Layer,
        LayerBuilder,
        LayerId,
    },
    mle::{
        dense::{DenseMle, Tuple2},
        Mle,
    },
};
use ark_std::test_rng;

use ark_std::{One, Zero};
use rand::Rng;
use remainder_shared_types::transcript::{poseidon_transcript::PoseidonSponge, Transcript};
use remainder_shared_types::Fr;

/// Does a dummy version of sumcheck with a testing RNG
pub fn dummy_sumcheck<F: FieldExt>(
    expr: &mut Expression<F, ProverExpr>,
    rng: &mut impl Rng,
    layer_claim: Claim<F>,
) -> Vec<(Vec<F>, Option<F>)> {
    let claim_point = layer_claim.get_point();
    let expression_nonlinear_indices = expr.get_all_nonlinear_rounds();
    let expression_linear_indices = expr.get_all_linear_rounds();
    expression_linear_indices
        .iter()
        .sorted()
        .for_each(|round_idx| {
            expr.fix_variable_at_index(*round_idx, claim_point[*round_idx]);
        });
    let betavec = expression_nonlinear_indices
        .iter()
        .map(|idx| (*idx, claim_point[*idx]))
        .collect_vec();
    let mut newbeta = BetaValues::new(betavec);

    // --- Does the bit indexing ---

    // --- The prover messages to the verifier...? ---
    let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
    let mut challenge: Option<F> = None;

    for round_index in expression_nonlinear_indices {
        // --- First fix the variable representing the challenge from the last round ---
        // (This doesn't happen for the first round)
        if let Some(challenge) = challenge {
            expr.fix_variable(round_index - 1, challenge);
            newbeta.beta_update(round_index - 1, challenge);
        }

        // --- Grabs the degree of univariate polynomial we are sending over ---
        let degree = get_round_degree(expr, round_index);

        // --- Gives back the evaluations g(0), g(1), ..., g(d - 1) ---
        let eval = compute_sumcheck_message_beta_cascade(expr, round_index, degree, &newbeta);

        if let Ok(Evals(evaluations)) = eval {
            // dbg!(&evaluations);
            messages.push((evaluations, challenge))
        } else {
            // dbg!(&eval);
            panic!();
        };

        challenge = Some(F::from(rng.gen::<u64>()));
        // challenge = Some(F::one());
    }

    // expr.fix_variable(max_round - 1, challenge.unwrap());
    // beta_table.beta_update(max_round - 1, challenge.unwrap()).unwrap();

    messages
}

/// Returns the curr random challenge if verified correctly, otherwise verify error
/// can change this to take prev round random challenge, and then compute the new random challenge
/// TODO!(ryancao): Change this to take in the expression as well and do the final sumcheck check
pub fn verify_sumcheck_messages<F: FieldExt>(
    messages: Vec<(Vec<F>, Option<F>)>,
    mut expression: Expression<F, ProverExpr>,
    layer_claim: Claim<F>,
    rng: &mut impl Rng,
) -> Result<F, VerifyError> {
    if messages.len() == 0 {
        return Ok(F::zero());
    }
    let mut prev_evals = &messages[0].0;
    let mut chal = F::zero();

    // Thaler book page 34
    // First round:
    // For the first two evals, i.e. g_1(0), g_1(1), add them together.
    // This is the claimed sumcheck result from the prover.
    // TODO!(ende): degree check?
    let claimed_val = messages[0].0[0] + messages[0].0[1];
    if claimed_val != layer_claim.get_result() {
        return Err(VerifyError::SumcheckBad);
    }
    let mut challenges = vec![];

    // --- Go through sumcheck messages + (FS-generated) challenges ---
    // Round j, 1 < j < v
    for (evals, challenge) in messages.iter().skip(1) {
        let curr_evals = evals;
        chal = (*challenge).unwrap();
        // --- Evaluate the previous round's polynomial at the random challenge point, i.e. g_{i - 1}(r_i) ---
        let prev_at_r = evaluate_at_a_point(prev_evals, challenge.unwrap())
            .expect("could not evaluate at challenge point");

        // dbg!(&prev_evals);
        // dbg!(-prev_evals[0]);
        // dbg!(prev_at_r);
        // dbg!(curr_evals[0], curr_evals[1], curr_evals[0] + curr_evals[1]);
        // --- g_{i - 1}(r) should equal g_i(0) + g_i(1) ---
        if prev_at_r != curr_evals[0] + curr_evals[1] {
            return Err(VerifyError::SumcheckBad);
        };
        prev_evals = curr_evals;
        challenges.push(chal);
    }

    // Round v, again Thaler book page 34
    let final_chal = F::from(rng.gen::<u64>());
    // let final_chal = F::one();
    challenges.push(final_chal);

    // uses the expression to make one single oracle query
    let mle_bound = expression.evaluate_expr(challenges.clone()).unwrap();
    let beta_bound =
        BetaValues::compute_beta_over_two_challenges(layer_claim.get_point(), &challenges);
    let oracle_query = mle_bound * beta_bound;

    let prev_at_r =
        evaluate_at_a_point(prev_evals, final_chal).expect("could not evaluate at challenge point");
    if oracle_query != prev_at_r {
        return Err(VerifyError::SumcheckBad);
    }

    Ok(chal)
}

pub fn get_dummy_claim<F: FieldExt>(
    mle_ref: DenseMleRef<F>,
    rng: &mut impl Rng,
    challenges: Option<Vec<F>>,
) -> Claim<F> {
    let mut expression = mle_ref.expression();
    let num_vars = expression.index_mle_indices(0);
    let challenges = if let Some(challenges) = challenges {
        assert_eq!(challenges.len(), num_vars);
        challenges
    } else {
        (0..num_vars)
            .map(|_| F::from(rng.gen::<u64>()))
            .collect_vec()
    };
    let eval = expression.evaluate_expr(challenges).unwrap();

    let (expression_node, mle_vec) = expression.deconstruct_mut();

    let claim = match expression_node {
        ExpressionNode::Mle(mle_vec_index) => mle_vec_index
            .get_mle(mle_vec)
            .mle_indices
            .iter()
            .map(|index: &MleIndex<F>| index.val().unwrap())
            .collect_vec(),
        _ => panic!(),
    };
    Claim::new_raw(claim, eval)
}

pub(crate) fn get_dummy_expression_eval<F: FieldExt>(
    expression: &Expression<F, ProverExpr>,
    rng: &mut impl Rng,
) -> Claim<F> {
    let mut expression = expression.clone();
    let num_vars = expression.index_mle_indices(0);

    let expression_nonlinear_indices = expression.get_all_nonlinear_rounds();
    let expression_linear_indices = expression.get_all_linear_rounds();

    let challenges = (0..num_vars)
        .map(|_| (F::from(rng.gen::<u64>())))
        .collect_vec();
    let challenges_enumerate = expression_nonlinear_indices
        .iter()
        .map(|index| (*index, challenges[*index]))
        .collect_vec();
    expression_linear_indices
        .iter()
        .for_each(|round| expression.fix_variable_at_index(*round, challenges[*round]));

    let beta = BetaValues::new(challenges_enumerate.clone());
    let eval = compute_sumcheck_message_beta_cascade(&expression, 0, 2, &beta).unwrap();
    let Evals(evals) = eval;
    let result = if evals.len() > 1 {
        evals[0] + evals[1]
    } else {
        evals[0]
    };

    Claim::new_raw(challenges, result)
}

/// Test regular numerical evaluation, last round type beat
#[test]
fn eval_expr_nums() {
    let new_beta = BetaValues::new(vec![]);
    let mut expression1: Expression<Fr, ProverExpr> = Expression::constant(Fr::from(6));
    let res = compute_sumcheck_message_beta_cascade(&mut expression1, 0, 0, &new_beta);
    let exp = Evals(vec![Fr::from(6)]);
    assert_eq!(res.unwrap(), exp);
}

/// Test the evaluation at an arbitrary point, all positives
#[test]
fn eval_at_point_pos() {
    //poly = 3x^2 + 5x + 9
    let evals = vec![Fr::from(9), Fr::from(17), Fr::from(31)];
    let point = Fr::from(3);
    let evald = evaluate_at_a_point(&evals, point);
    assert_eq!(
        evald.unwrap(),
        Fr::from(3) * point * point + Fr::from(5) * point + Fr::from(9)
    );
}

/// Test the evaluation at an arbitrary point, neg numbers
#[test]
fn eval_at_point_neg() {
    // poly = 2x^2 - 6x + 3
    let evals = vec![Fr::from(3), Fr::from(1).neg(), Fr::from(1).neg()];
    let _degree = 2;
    let point = Fr::from(3);
    let evald = evaluate_at_a_point(&evals, point);
    assert_eq!(
        evald.unwrap(),
        Fr::from(2) * point * point - Fr::from(6) * point + Fr::from(3)
    );
}

/// Test the evaluation at an arbitrary point, more evals than degree
#[test]
fn eval_at_point_more_than_degree() {
    // poly = 3 + 10x
    let evals = vec![Fr::from(3), Fr::from(13), Fr::from(23)];
    let point = Fr::from(3);
    let evald = evaluate_at_a_point(&evals, point);
    assert_eq!(evald.unwrap(), Fr::from(3) + Fr::from(10) * point);
}

/// Test whether evaluate_mle_ref correctly computes the evaluations for a single MLE
#[test]
fn test_linear_sum() {
    let newbeta = BetaValues::new(vec![]);

    let mle_v1 = vec![Fr::from(3), Fr::from(2), Fr::from(2), Fr::from(5)];
    let mle1: DenseMleRef<Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None).mle_ref();
    let mut mleexpr = Expression::mle(mle1);
    mleexpr.index_mle_indices(0);
    mleexpr.fix_variable_at_index(0, Fr::from(2));
    mleexpr.fix_variable_at_index(1, Fr::from(4));

    let res = compute_sumcheck_message_beta_cascade(&mut mleexpr, 1, 0, &newbeta);
    let exp = Evals(vec![Fr::from(29)]);
    assert_eq!(res.unwrap(), exp);
}

/// Test whether evaluate_mle_ref correctly computes the evaluations for a product of MLEs
#[test]
fn test_quadratic_sum() {
    let new_beta = BetaValues::new(vec![(0, Fr::from(2)), (1, Fr::from(4))]);

    let mle_v1 = vec![Fr::from(1), Fr::from(0), Fr::from(2), Fr::from(3)];
    let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);

    let mle_v2 = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(5)];
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let mut expression = Expression::products(vec![mle1.mle_ref(), mle2.mle_ref()]);
    expression.index_mle_indices(0);

    let res = compute_sumcheck_message_beta_cascade(&mut expression, 1, 3, &new_beta);
    let exp = Evals(vec![
        Fr::from(2).neg(),
        Fr::from(120),
        Fr::from(780),
        Fr::from(2320),
    ]);
    assert_eq!(res.unwrap(), exp);
}

/// test whether evaluate_mle_ref correctly computes the evalutaions for a product of MLEs
/// where one of the MLEs is a log size step smaller than the other (e.g. V(b_1, b_2)*V(b_1))
#[test]
fn test_quadratic_sum_differently_sized_mles2() {
    let new_beta = BetaValues::new(vec![(0, Fr::from(2)), (1, Fr::from(4))]);

    let mle_v1 = vec![
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(3),
        Fr::from(1),
        Fr::from(4),
    ];
    let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);

    let mle_v2 = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(5)];
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let mut expression = Expression::products(vec![mle1.mle_ref(), mle2.mle_ref()]);
    expression.index_mle_indices(0);
    expression.fix_variable_at_index(2, Fr::from(3));

    let res = compute_sumcheck_message_beta_cascade(&mut expression, 1, 3, &new_beta);
    let exp = Evals(vec![
        Fr::from(12).neg(),
        Fr::from(230),
        Fr::from(1740),
        Fr::from(5688),
    ]);
    assert_eq!(res.unwrap(), exp);
}

/// test dummy sumcheck against verifier for product of the same mle
#[test]
fn test_dummy_sumcheck_1() {
    // let layer_claims = (vec![Fr::from(1), Fr::from(-1)], Fr::from(2));
    let mut rng = test_rng();
    let mle_vec = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(2)];

    let mle_new: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mle_v2 = vec![Fr::from(1), Fr::from(5), Fr::from(1), Fr::from(5)];
    let mle_2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let mle_output = DenseMle::new_from_iter(
        mle_new
            .into_iter()
            .zip(mle_2.into_iter())
            .map(|(first, second)| first * second),
        LayerId::Input(0),
        None,
    );
    let layer_claims = get_dummy_claim(mle_output.mle_ref(), &mut rng, None);

    // dbg!(claimed_claim);

    let mle_ref_1 = mle_new.mle_ref();
    let mle_ref_2 = mle_2.mle_ref();

    let mut expression = Expression::products(vec![mle_ref_1, mle_ref_2]);
    let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
    let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
    assert!(verifyres.is_ok());
}

/// test dummy sumcheck against product of two diff mles
#[test]
fn test_dummy_sumcheck_2() {
    // let layer_claims = (vec![Fr::from(3), Fr::from(4), Fr::from(2)], Fr::one());
    let mut rng = test_rng();
    let mle_v1 = vec![Fr::from(1), Fr::from(0), Fr::from(2), Fr::from(3)];
    let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);

    let mle_v2 = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(5)];
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let mle_output = DenseMle::new_from_iter(
        mle1.into_iter()
            .zip(mle2.into_iter())
            .map(|(first, second)| first * second),
        LayerId::Input(0),
        None,
    );
    let layer_claims = get_dummy_claim(mle_output.mle_ref(), &mut rng, None);

    let mle_ref_1 = mle1.mle_ref();
    let mle_ref_2 = mle2.mle_ref();

    let mut expression = Expression::products(vec![mle_ref_1, mle_ref_2]);
    let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
    let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
    assert!(verifyres.is_ok());
}

/// test dummy sumcheck against product of two mles diff sizes
#[test]
fn test_dummy_sumcheck_3() {
    let mut rng = test_rng();
    let mle_v1 = vec![
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(3),
        Fr::from(1),
        Fr::from(4),
    ];
    let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);

    let mle_v2 = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(5)];
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let mle_output = DenseMle::new_from_iter(
        mle1.into_iter()
            .zip(mle2.into_iter().chain(mle2.into_iter()))
            .map(|(first, second)| first * second),
        LayerId::Input(0),
        None,
    );
    let layer_claims = get_dummy_claim(mle_output.mle_ref(), &mut rng, None);

    let mle_ref_1 = mle1.mle_ref();
    let mle_ref_2 = mle2.mle_ref();

    let mut expression = Expression::products(vec![mle_ref_1, mle_ref_2]);
    let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
    let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
    assert!(verifyres.is_ok());
}

/// test dummy sumcheck against sum of two mles
#[test]
fn test_dummy_sumcheck_sum_small() {
    let mut rng = test_rng();
    let mle_v1 = vec![Fr::from(1), Fr::from(0), Fr::from(1), Fr::from(2)];
    let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);

    let mle_v2 = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(5)];
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let mle_output = DenseMle::new_from_iter(
        mle1.into_iter()
            .zip(mle2.into_iter())
            .map(|(first, second)| first + second),
        LayerId::Input(0),
        None,
    );
    let layer_claims = get_dummy_claim(mle_output.mle_ref(), &mut rng, None);

    let mle_ref_1 = mle1.mle_ref();
    let mle_ref_2 = mle2.mle_ref();

    let mut expression = mle_ref_1.expression() + mle_ref_2.expression();

    let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
    let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
    assert!(verifyres.is_ok());
}

/// test dummy sumcheck for concatenated expr SMALL
#[test]
fn test_dummy_sumcheck_concat() {
    // let layer_claims = (vec![Fr::from(3), Fr::from(1), Fr::from(2)], Fr::one());
    let mut rng = test_rng();
    let mle_v1 = vec![Fr::from(5), Fr::from(2)];
    let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);

    let mle_v2 = vec![Fr::from(2), Fr::from(3)];
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let output: DenseMle<Fr, Fr> = DenseMle::new_from_raw(
        vec![Fr::from(5), Fr::from(2), Fr::from(2), Fr::from(3)],
        LayerId::Input(0),
        None,
    );

    let layer_claims = get_dummy_claim(output.mle_ref(), &mut rng, None);

    let mle_ref_1 = mle1.mle_ref();
    let mle_ref_2 = mle2.mle_ref();

    let expression = Expression::mle(mle_ref_1);
    let expr2 = Expression::mle(mle_ref_2);

    let mut expression = expr2.concat_expr(expression);
    let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
    let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
    assert!(verifyres.is_ok());
}

/// test dummy sumcheck for concatenated expr SMALL BUT LESS SMALL
#[test]
fn test_dummy_sumcheck_concat_2() {
    // let layer_claims = (
    //     vec![Fr::from(2), Fr::from(4), Fr::from(2), Fr::from(3)],
    //     Fr::one(),
    // );
    let mut rng = test_rng();
    let mle_v1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
    let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);

    let mle_v2 = vec![Fr::from(1), Fr::from(3), Fr::from(1), Fr::from(6)];
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let mle_out = DenseMle::<Fr, Fr>::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
            Fr::from(6),
        ],
        LayerId::Input(0),
        None,
    );
    let layer_claims = get_dummy_claim(mle_out.mle_ref(), &mut rng, None);

    let mle_ref_1 = mle1.mle_ref();
    let mle_ref_2 = mle2.mle_ref();

    let expression = Expression::mle(mle_ref_1);
    let expr2 = Expression::mle(mle_ref_2);

    let mut expression = expr2.concat_expr(expression);
    let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());
    let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
    assert!(verifyres.is_ok());
}

/// test dummy sumcheck for concatenated expr
#[test]
fn test_dummy_sumcheck_concat_aggro() {
    // let layer_claims = (
    //     vec![Fr::from(2), Fr::from(4), Fr::from(2), Fr::from(3)],
    //     Fr::one(),
    // );
    let mut rng = test_rng();
    let mle_v1 = vec![
        Fr::from(1),
        Fr::from(2),
        Fr::from(23).neg(),
        Fr::from(47).neg(),
        Fr::from(5),
        Fr::from(22),
        Fr::from(31),
        Fr::from(4).neg(),
    ];
    let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);

    let mle_v2 = vec![Fr::from(1), Fr::from(3), Fr::from(1), Fr::from(6)];
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let mle_ref_1 = mle1.mle_ref();
    let mle_ref_2 = mle2.mle_ref();

    let mle_output = DenseMle::new_from_iter(
        mle1.into_iter()
            .zip(mle2.clone().into_iter().chain(mle2.into_iter()))
            .map(|(first, second)| first * second)
            .flat_map(|x| vec![x, x]),
        LayerId::Input(0),
        None,
    );
    let layer_claims = get_dummy_claim(mle_output.mle_ref(), &mut rng, None);

    let expression = Expression::products(vec![mle_ref_1, mle_ref_2]);
    let expr2 = expression.clone();

    let mut expression = expr2.concat_expr(expression);
    let res_messages = dummy_sumcheck(&mut expression, &mut rng, layer_claims.clone());

    let verifyres = verify_sumcheck_messages(res_messages, expression, layer_claims, &mut rng);
    assert!(verifyres.is_ok());
}

#[test]
fn test_dummy_sumcheck_concat_aggro_aggro() {
    // let layer_claims = (
    //     vec![Fr::from(12190), Fr::from(28912), Fr::from(1)],
    //     Fr::one(),
    // );
    let mut rng = test_rng();
    let mle_v1 = vec![Fr::from(1), Fr::from(2)];
    let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);

    let mle_v2 = vec![Fr::from(5), Fr::from(1291)];
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let mle_ref_1 = mle1.mle_ref();
    let mle_ref_2 = mle2.mle_ref();

    let expression = Expression::mle(mle_ref_1);
    let expr2 = Expression::mle(mle_ref_2);

    let expression = expr2.clone().concat_expr(expression);
    let mut expression_aggro = expression.concat_expr(expr2);
    let layer_claims = get_dummy_expression_eval(&expression_aggro, &mut rng);
    let res_messages = dummy_sumcheck(&mut expression_aggro, &mut rng, layer_claims.clone());
    let verifyres =
        verify_sumcheck_messages(res_messages, expression_aggro, layer_claims, &mut rng);
    assert!(verifyres.is_ok());
}

/// test dummy sumcheck for concatenated expr
#[test]
fn test_dummy_sumcheck_concat_aggro_aggro_aggro() {
    let mut rng = test_rng();
    let mle_v1 = vec![Fr::from(1390), Fr::from(222104)];
    let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);

    let mle_v2 = vec![Fr::from(5), Fr::from(1291)];
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let mle_ref_1 = mle1.mle_ref();
    let mle_ref_2 = mle2.mle_ref();

    let expression = Expression::mle(mle_ref_1);
    let expr2 = Expression::mle(mle_ref_2);

    let expression = expr2.clone().concat_expr(expression);
    let expression_aggro = expression.concat_expr(expr2.clone());
    let mut expression_aggro_aggro = expression_aggro.concat_expr(expr2);
    let layer_claims = get_dummy_expression_eval(&expression_aggro_aggro, &mut rng);
    let res_messages = dummy_sumcheck(&mut expression_aggro_aggro, &mut rng, layer_claims.clone());
    let verifyres =
        verify_sumcheck_messages(res_messages, expression_aggro_aggro, layer_claims, &mut rng);
    assert!(verifyres.is_ok());
}

#[test]
fn test_dummy_sumcheck_sum() {
    // let layer_claims = (vec![Fr::from(2), Fr::from(1), Fr::from(10)], Fr::one());
    let mut rng = test_rng();
    let mle_v1 = vec![Fr::from(0), Fr::from(2)];
    let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);

    let mle_v2 = vec![Fr::from(5), Fr::from(1291)];
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let mle_ref_1 = mle1.mle_ref();
    let mle_ref_2 = mle2.mle_ref();

    let expression = Expression::mle(mle_ref_1);
    let expr2 = Expression::mle(mle_ref_2);

    let expression = expr2.clone().concat_expr(expression);
    let mut expression_aggro = expression.concat_expr(expr2);
    let layer_claims = get_dummy_expression_eval(&expression_aggro, &mut rng);
    let res_messages = dummy_sumcheck(&mut expression_aggro, &mut rng, layer_claims.clone());
    let verifyres =
        verify_sumcheck_messages(res_messages, expression_aggro, layer_claims, &mut rng);
    assert!(verifyres.is_ok());
}

#[test]
fn test_beta_cascade_1() {
    let mle_1_vec = vec![Fr::one(), Fr::from(2_u64), Fr::from(2_u64), Fr::from(3_u64)];
    let mut mle_ref_1: DenseMleRef<Fr> =
        DenseMle::new_from_raw(mle_1_vec, LayerId::Input(0), None).mle_ref();
    mle_ref_1.index_mle_indices(0);
    let beta_vals = vec![Fr::from(2_u64), Fr::from(3_u64)];
    let betacascade_evals = beta_cascade(&[&mle_ref_1], 2, 0, &beta_vals, &vec![]);
    let expected_evals = Evals(vec![Fr::from(4).neg(), Fr::from(10), Fr::from(30)]);
    assert_eq!(betacascade_evals, expected_evals);
}

#[test]
fn test_beta_cascade_2() {
    let mle_1_vec = vec![Fr::one(), Fr::from(2_u64), Fr::from(2_u64), Fr::from(3_u64)];
    let mle_2_vec = vec![Fr::from(4_u64), Fr::from(2_u64), Fr::from(3_u64), Fr::one()];
    let mut mle_ref_1: DenseMleRef<Fr> =
        DenseMle::new_from_raw(mle_1_vec, LayerId::Input(0), None).mle_ref();
    let mut mle_ref_2: DenseMleRef<Fr> =
        DenseMle::new_from_raw(mle_2_vec, LayerId::Input(0), None).mle_ref();
    mle_ref_1.index_mle_indices(0);
    mle_ref_2.index_mle_indices(0);
    let beta_vals = vec![Fr::from(2_u64), Fr::from(3_u64)];
    let betacascade_evals = beta_cascade(&[&mle_ref_1, &mle_ref_2], 3, 0, &beta_vals, &vec![]);
    let expected_evals = Evals(vec![
        Fr::from(10).neg(),
        Fr::from(2),
        Fr::from(60).neg(),
        Fr::from(232).neg(),
    ]);
    assert_eq!(betacascade_evals, expected_evals);
}

#[test]
fn test_beta_cascade_3() {
    let mle_1_vec = vec![Fr::one(), Fr::from(2_u64), Fr::from(2_u64), Fr::from(3_u64)];
    let mle_2_vec = vec![Fr::from(4_u64), Fr::from(2_u64), Fr::from(3_u64), Fr::one()];
    let mle_3_vec = vec![Fr::one(), Fr::from(2_u64), Fr::from(2_u64), Fr::from(3_u64)];
    let mut mle_ref_1: DenseMleRef<Fr> =
        DenseMle::new_from_raw(mle_1_vec, LayerId::Input(0), None).mle_ref();
    let mut mle_ref_2: DenseMleRef<Fr> =
        DenseMle::new_from_raw(mle_2_vec, LayerId::Input(0), None).mle_ref();
    let mut mle_ref_3: DenseMleRef<Fr> =
        DenseMle::new_from_raw(mle_3_vec, LayerId::Input(0), None).mle_ref();
    mle_ref_1.index_mle_indices(0);
    mle_ref_2.index_mle_indices(0);
    mle_ref_3.index_mle_indices(0);

    let beta_vals = vec![Fr::from(2_u64), Fr::from(3_u64)];
    let betacascade_evals = beta_cascade(
        &[&mle_ref_1, &mle_ref_2, &mle_ref_3],
        4,
        0,
        &beta_vals,
        &vec![],
    );
    let expected_evals = Evals(vec![
        Fr::from(28).neg(),
        Fr::from(22),
        Fr::from(240).neg(),
        Fr::from(1288).neg(),
        Fr::from(3740).neg(),
    ]);
    assert_eq!(betacascade_evals, expected_evals);
}

#[test]
fn test_successors_from_mle_ref_product() {
    let mle_1_vec = vec![Fr::one(), Fr::from(2_u64), Fr::from(2_u64), Fr::from(3_u64)];
    let mle_2_vec = vec![Fr::from(4_u64), Fr::from(2_u64), Fr::from(3_u64), Fr::one()];
    let mle_3_vec = vec![Fr::one(), Fr::from(2_u64), Fr::from(2_u64), Fr::from(3_u64)];
    let mut mle_ref_1: DenseMleRef<Fr> =
        DenseMle::new_from_raw(mle_1_vec, LayerId::Input(0), None).mle_ref();
    let mut mle_ref_2: DenseMleRef<Fr> =
        DenseMle::new_from_raw(mle_2_vec, LayerId::Input(0), None).mle_ref();
    let mut mle_ref_3: DenseMleRef<Fr> =
        DenseMle::new_from_raw(mle_3_vec, LayerId::Input(0), None).mle_ref();
    mle_ref_1.index_mle_indices(0);
    mle_ref_2.index_mle_indices(0);
    mle_ref_3.index_mle_indices(0);

    let successors_vec =
        successors_from_mle_ref_product(&[&mle_ref_1, &mle_ref_2, &mle_ref_3], 4).unwrap();
    let expected_vec = vec![
        Fr::from(4),
        Fr::from(8),
        Fr::zero(),
        Fr::from(32).neg(),
        Fr::from(100).neg(),
        Fr::from(12),
        Fr::from(9),
        Fr::from(16).neg(),
        Fr::from(75).neg(),
        Fr::from(180).neg(),
    ];
    assert_eq!(successors_vec, expected_vec);
}

#[test]
fn test_beta_cascade_step() {
    let mle_1_vec = vec![Fr::one(), Fr::from(2_u64), Fr::from(2_u64), Fr::from(3_u64)];
    let mle_2_vec = vec![Fr::from(4_u64), Fr::from(2_u64), Fr::from(3_u64), Fr::one()];
    let mle_3_vec = vec![Fr::one(), Fr::from(2_u64), Fr::from(2_u64), Fr::from(3_u64)];
    let mut mle_ref_1: DenseMleRef<Fr> =
        DenseMle::new_from_raw(mle_1_vec, LayerId::Input(0), None).mle_ref();
    let mut mle_ref_2: DenseMleRef<Fr> =
        DenseMle::new_from_raw(mle_2_vec, LayerId::Input(0), None).mle_ref();
    let mut mle_ref_3: DenseMleRef<Fr> =
        DenseMle::new_from_raw(mle_3_vec, LayerId::Input(0), None).mle_ref();
    mle_ref_1.index_mle_indices(0);
    mle_ref_2.index_mle_indices(0);
    mle_ref_3.index_mle_indices(0);

    let mut successors_vec =
        successors_from_mle_ref_product(&[&mle_ref_1, &mle_ref_2, &mle_ref_3], 4).unwrap();

    let one_step_with_beta_val_3 = beta_cascade_step(&mut successors_vec, Fr::from(3));
    let expected_vec = vec![
        Fr::from(28),
        Fr::from(11),
        Fr::from(48).neg(),
        Fr::from(161).neg(),
        Fr::from(340).neg(),
    ];
    assert_eq!(one_step_with_beta_val_3, expected_vec);
}
