use crate::builders::layer_builder::from_mle;
use crate::expression::generic_expr::Expression;
use crate::expression::prover_expr::ProverExpr;
use crate::layer::regular_layer::RegularLayer;
use crate::layer::Layer;
use crate::mle::dense::DenseMle;
use crate::mle::{Mle, MleIndex};
use crate::sumcheck::evaluate_at_a_point;
use crate::utils::test_utils::DummySponge;
use claim_aggregation::prover_aggregate_claims;
use rand::Rng;
use remainder_shared_types::transcript::{TranscriptSponge, TranscriptWriter};

use super::*;
use ark_std::test_rng;
use remainder_shared_types::Fr;

#[test]
fn test_get_claim() {
    // [1, 1, 1, 1] \oplus (1 - (1 * (1 + V[1, 1, 1, 1]))) * 2
    let expression1: Expression<Fr, ProverExpr> = Expression::<Fr, ProverExpr>::constant(Fr::one());
    let mle = DenseMle::<Fr>::new_from_raw(
        vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()],
        LayerId::Input(0),
    );
    let expression3 = Expression::<Fr, ProverExpr>::mle(mle);
    let expression = expression1.clone() + expression3.clone();
    let expression = expression1 - expression;
    let expression = expression * Fr::from(2);
    let _expression = expression.select(expression3);
}

// ------- Helper functions for claim aggregation tests -------

/// Builds a vector of [Claim]s by evaluation an expression `expr` on
/// each point in `points`.
fn claims_from_expr_and_points(
    expr: &Expression<Fr, ProverExpr>,
    points: &Vec<Vec<Fr>>,
) -> Vec<Claim<Fr>> {
    points
        .iter()
        .flat_map(|point| {
            let mut exp = expr.clone();
            exp.index_mle_indices(0);
            exp.evaluate_expr(point.clone()).unwrap();
            // ClaimMle::new_raw(point.clone(), result)
            RegularLayer::new_raw(LayerId::Layer(0), exp)
                .get_claims()
                .unwrap()
        })
        .collect()
}

/// Builds GKR layer whose MLE is the function whose evaluations
/// on the boolean hypercube are given by `mle_evals`.
fn layer_from_evals(mle_evals: Vec<Fr>) -> RegularLayer<Fr> {
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_evals, LayerId::Input(0));

    let layer = from_mle(
        mle,
        |mle| mle.clone().expression(),
        |_, _, _| unimplemented!(),
    );

    RegularLayer::new(layer, LayerId::Input(0))
}

/// Returns a random MLE expression with an associated GKR layer, along with the
/// MLE used to create the layer.
fn build_random_mle_layer(num_vars: usize) -> (RegularLayer<Fr>, DenseMle<Fr>) {
    let mut rng = test_rng();
    let mle_evals: Vec<Fr> = (0..num_vars).map(|_| Fr::from(rng.gen::<u64>())).collect();
    (
        layer_from_evals(mle_evals.clone()),
        DenseMle::new_from_raw(mle_evals, LayerId::Input(0)),
    )
}

/*
fn compute_claim_wlx<F: Field, Sp: TranscriptSponge<F>>(
    claims: &ClaimGroup<F>,
    layer: &impl YieldWLXEvals<F>,
) -> (Claim<F>, Vec<Vec<F>>) {
    let num_claims = claims.get_num_claims();
    let num_vars = claims.get_num_vars();

    let points_matrix = claims.get_claim_points_matrix();

    debug_assert_eq!(points_matrix.len(), num_claims);
    debug_assert_eq!(points_matrix[0].len(), num_vars);

    let mut transcript: TranscriptWriter<F, Sp> = TranscriptWriter::new("Claims Test Transcript");

    let claim_proof = prover_aggregate_claims_helper(claims, layer, &mut transcript).unwrap();
    (claim_proof.claim, claim_proof.proof)
}
*/

fn claim_aggregation_wrapper<Sp: TranscriptSponge<Fr>>(
    output_mles_from_layer: Vec<DenseMle<Fr>>,
    claims: &[Claim<Fr>],
) -> RawClaim<Fr> {
    let mut transcript: TranscriptWriter<_, Sp> = TranscriptWriter::new("Claims Test Transcript");
    prover_aggregate_claims(claims, &output_mles_from_layer, &mut transcript).unwrap()
}

/// Compute l* = l(r*).
fn compute_l_star(claims: &[Claim<Fr>], r_star: Fr) -> Vec<Fr> {
    assert!(claims.len() > 0);
    let num_vars = claims[0].get_num_vars();

    (0..num_vars)
        .map(|i| {
            let evals: Vec<Fr> = claims.iter().map(|claim| claim.get_point()[i]).collect();
            evaluate_at_a_point(&evals, r_star).unwrap()
        })
        .collect()
}

// Returns expected aggregated claim of `expr` on l(r_star) = `l_star`.
fn compute_expected_claim(layer: &RegularLayer<Fr>, l_star: &Vec<Fr>) -> RawClaim<Fr> {
    let mut expr = layer.get_expression().clone();
    expr.index_mle_indices(0);
    let result = expr.evaluate_expr(l_star.clone()).unwrap();
    RawClaim::new(l_star.clone(), result)
}

// ----------------------------------------------------------

/// Test claim aggregation small MLE on 2 variables
/// with 2 claims.
#[test]
fn test_aggro_claim_1() {
    let mle_evals = vec![Fr::from(1), Fr::from(0), Fr::from(2), Fr::from(3)];
    let points = vec![
        vec![Fr::from(3), Fr::from(3)],
        vec![Fr::from(2), Fr::from(7)],
    ];
    let r_star = Fr::from(10);

    let output_mles_for_layer = vec![DenseMle::new_from_raw(mle_evals.clone(), LayerId::Input(0))];
    let layer = layer_from_evals(mle_evals);
    let claims = claims_from_expr_and_points(layer.get_expression(), &points);

    let l_star = compute_l_star(&claims, r_star);

    // Compare to l(10) computed by hand.
    assert_eq!(l_star, vec![Fr::from(7).neg(), Fr::from(43)]);

    let aggregated_claim =
        claim_aggregation_wrapper::<DummySponge<Fr, 10>>(output_mles_for_layer, &claims);
    let expected_claim = compute_expected_claim(&layer, &l_star);

    assert_eq!(aggregated_claim.get_eval(), expected_claim.get_eval());
}

/// Test claim aggregation on another small MLE on 2 variables
/// with 3 claims.
#[test]

fn test_aggro_claim_2() {
    let mle_evals = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
    let points = vec![
        vec![Fr::from(1), Fr::from(2)],
        vec![Fr::from(2), Fr::from(3)],
        vec![Fr::from(3), Fr::from(1)],
    ];

    let r_star = Fr::from(2).neg();

    let output_mles_for_layer = vec![DenseMle::new_from_raw(mle_evals.clone(), LayerId::Input(0))];
    let layer = layer_from_evals(mle_evals);
    let claims = claims_from_expr_and_points(layer.get_expression(), &points);

    let l_star = compute_l_star(&claims, r_star);

    let aggregated_claim =
        claim_aggregation_wrapper::<DummySponge<Fr, -2>>(output_mles_for_layer, &claims);
    let expected_claim = compute_expected_claim(&layer, &l_star);

    assert_eq!(aggregated_claim.get_eval(), expected_claim.get_eval());
}

/// Test claim aggregation for 3 claims on a random MLE on 3
/// variables with random challenge.
#[test]
fn test_aggro_claim_3() {
    let points = vec![
        vec![Fr::from(2).neg(), Fr::from(192013).neg(), Fr::from(2148)],
        vec![Fr::from(123), Fr::from(482), Fr::from(241)],
        vec![Fr::from(92108), Fr::from(29014), Fr::from(524)],
    ];
    let r_star = Fr::from(25);

    // ---------------
    let (layer, layerwise_bt) = build_random_mle_layer(3);
    let output_mles_for_layer = vec![layerwise_bt];
    let claims = claims_from_expr_and_points(layer.get_expression(), &points);

    let l_star = compute_l_star(&claims, r_star);

    let aggregated_claim =
        claim_aggregation_wrapper::<DummySponge<Fr, 25>>(output_mles_for_layer, &claims);
    let expected_claim = compute_expected_claim(&layer, &l_star);

    assert_eq!(aggregated_claim.get_eval(), expected_claim.get_eval());
}

/// Test claim aggregation on a random, product MLE on 3 variables
/// (1 + 2) with 3 claims.
#[test]
fn test_aggro_claim_4() {
    let mut rng = test_rng();
    let mle1_evals = vec![Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())];
    let mle2_evals = vec![
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
    ];

    let mut mle1: DenseMle<Fr> = DenseMle::new_from_raw(mle1_evals, LayerId::Input(0));
    mle1.add_prefix_bits(vec![MleIndex::Fixed(false), MleIndex::Fixed(false)]);

    let mut mle2: DenseMle<Fr> = DenseMle::new_from_raw(mle2_evals, LayerId::Input(0));
    mle2.add_prefix_bits(vec![MleIndex::Fixed(true)]);

    let mle_ref = mle1.clone();
    let mle_ref2 = mle2.clone();

    let output_mles_from_layer = vec![mle_ref.clone(), mle_ref2.clone()];
    let expr = Expression::<Fr, ProverExpr>::products(vec![mle_ref, mle_ref2]);
    let mut expr_copy = expr.clone();

    let layer = from_mle(
        (mle1, mle2),
        |mle| Expression::<Fr, ProverExpr>::products(vec![mle.clone().0, mle.clone().1]),
        |_, _, _| unimplemented!(),
    );
    let layer: RegularLayer<_> = RegularLayer::new(layer, LayerId::Input(0));

    let chals1 = vec![Fr::from(2).neg(), Fr::from(192013).neg(), Fr::from(2148)];
    let chals2 = vec![Fr::from(123), Fr::from(482), Fr::from(241)];
    let chals3 = vec![Fr::from(92108), Fr::from(29014), Fr::from(524)];
    let chals = vec![chals1, chals2, chals3];

    let claim_group = claims_from_expr_and_points(&layer.expression, &chals);

    let rchal = Fr::from(40).neg();

    let res =
        claim_aggregation_wrapper::<DummySponge<_, -40>>(output_mles_from_layer, &claim_group);

    let transpose1 = vec![Fr::from(2).neg(), Fr::from(123), Fr::from(92108)];
    let transpose2 = vec![Fr::from(192013).neg(), Fr::from(482), Fr::from(29014)];
    let transpose3 = vec![Fr::from(2148), Fr::from(241), Fr::from(524)];

    let fix_vars: Vec<Fr> = vec![transpose1, transpose2, transpose3]
        .into_iter()
        .map(|evals| evaluate_at_a_point(&evals, rchal).unwrap())
        .collect();

    expr_copy.index_mle_indices(0);

    let eval_fixed_vars = expr_copy.evaluate_expr(fix_vars.clone()).unwrap();
    let claim_fixed_vars: RawClaim<Fr> = RawClaim::new(fix_vars, eval_fixed_vars);
    assert_ne!(res.get_eval(), claim_fixed_vars.get_eval());
}

/// Make sure claim aggregation FAILS for a WRONG CLAIM!
#[test]
fn test_aggro_claim_negative_1() {
    let mut rng = test_rng();
    let mle_v1 = vec![
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
    ];
    let mle1: DenseMle<Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0));
    let output_mles_from_layer = vec![mle1.clone()];

    let mle_ref = mle1.clone();
    let mut expr = Expression::<Fr, ProverExpr>::mle(mle_ref);

    let layer = from_mle(
        mle1,
        |mle| mle.clone().expression(),
        |_, _, _| unimplemented!(),
    );
    let layer: RegularLayer<_> = RegularLayer::new(layer, LayerId::Input(0));

    let chals1 = vec![Fr::from(2).neg(), Fr::from(192013).neg(), Fr::from(2148)];
    let chals2 = vec![Fr::from(123), Fr::from(482), Fr::from(241)];
    let chals3 = vec![Fr::from(92108), Fr::from(29014), Fr::from(524)];
    let chals = vec![chals1, chals2, chals3];

    let rchal = Fr::from(76);

    let mut claims = claims_from_expr_and_points(&layer.expression, &chals);
    let test_claim = &claims[0];
    claims[0] = Claim::new(
        test_claim.get_point().to_vec(),
        test_claim.get_eval() - Fr::one(),
        test_claim.from_layer_id,
        test_claim.to_layer_id,
    );
    let res = claim_aggregation_wrapper::<DummySponge<_, 76>>(output_mles_from_layer, &claims);

    let transpose1 = vec![Fr::from(2).neg(), Fr::from(123), Fr::from(92108)];
    let transpose2 = vec![Fr::from(192013).neg(), Fr::from(482), Fr::from(29014)];
    let transpose3 = vec![Fr::from(2148), Fr::from(241), Fr::from(524)];

    let fix_vars: Vec<Fr> = vec![transpose1, transpose2, transpose3]
        .into_iter()
        .map(|evals| evaluate_at_a_point(&evals, rchal).unwrap())
        .collect();

    expr.index_mle_indices(0);

    let eval_fixed_vars = expr.evaluate_expr(fix_vars.clone()).unwrap();
    let claim_fixed_vars: RawClaim<Fr> = RawClaim::new(fix_vars, eval_fixed_vars);
    assert_ne!(res.get_eval(), claim_fixed_vars.get_eval());
}

/// Make sure claim aggregation fails for ANOTHER WRONG CLAIM!
#[test]
fn test_aggro_claim_negative_2() {
    let _dummy_claim = (vec![Fr::from(1); 3], Fr::from(0));
    let mut rng = test_rng();
    let mle_v1 = vec![
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
        Fr::from(rng.gen::<u64>()),
    ];
    let mle1: DenseMle<Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0));
    let mle_ref = mle1.clone();
    let expr = Expression::<Fr, ProverExpr>::mle(mle_ref);
    let mut expr_copy = expr.clone();
    let output_mle_from_layer = vec![mle1.clone()];

    let layer = from_mle(
        mle1,
        |mle| mle.clone().expression(),
        |_, _, _| unimplemented!(),
    );
    let layer: RegularLayer<_> = RegularLayer::new(layer, LayerId::Input(0));

    let chals1 = vec![Fr::from(2).neg(), Fr::from(192013).neg(), Fr::from(2148)];
    let chals2 = vec![Fr::from(123), Fr::from(482), Fr::from(241)];
    let chals3 = vec![Fr::from(92108), Fr::from(29014), Fr::from(524)];
    let chals = vec![chals1, chals2, chals3];

    let rchal = Fr::from(76);

    let mut claims = claims_from_expr_and_points(&layer.expression, &chals);
    let test_claim = claims[2].clone();
    claims[2] = Claim::new(
        test_claim.get_point().to_vec(),
        test_claim.get_eval() + Fr::one(),
        test_claim.from_layer_id,
        test_claim.to_layer_id,
    );
    let res = claim_aggregation_wrapper::<DummySponge<_, 40>>(output_mle_from_layer, &claims);

    let transpose1 = vec![Fr::from(2).neg(), Fr::from(123), Fr::from(92108)];
    let transpose2 = vec![Fr::from(192013).neg(), Fr::from(482), Fr::from(29014)];
    let transpose3 = vec![Fr::from(2148), Fr::from(241), Fr::from(524)];

    let fix_vars: Vec<Fr> = vec![transpose1, transpose2, transpose3]
        .into_iter()
        .map(|evals| evaluate_at_a_point(&evals, rchal).unwrap())
        .collect();

    expr_copy.index_mle_indices(0);

    let eval_fixed_vars = expr_copy.evaluate_expr(fix_vars.clone()).unwrap();

    let claim_fixed_vars: RawClaim<Fr> = RawClaim::new(fix_vars, eval_fixed_vars);
    assert_ne!(res.get_eval(), claim_fixed_vars.get_eval());
}

#[test]
fn test_aggro_claim_common_suffix1() {
    // MLE on 3 variables (2^3 = 8 evals)
    let mle_evals: Vec<Fr> = vec![1, 2, 42, 4, 5, 6, 7, 17]
        .into_iter()
        .map(Fr::from)
        .collect();
    let points = vec![
        vec![Fr::from(1), Fr::from(3), Fr::from(5)],
        vec![Fr::from(2), Fr::from(4), Fr::from(5)],
    ];
    let r_star = Fr::from(10);

    // ---------------

    let output_mle_from_layer = vec![DenseMle::new_from_raw(mle_evals.clone(), LayerId::Input(0))];
    let layer = layer_from_evals(mle_evals);
    let claims = claims_from_expr_and_points(layer.get_expression(), &points);

    // W(l(0)), W(l(1)) computed by hand.
    assert_eq!(claims[0].get_eval(), Fr::from(163));
    assert_eq!(claims[1].get_eval(), Fr::from(1015));

    let l_star = compute_l_star(&claims, r_star);

    // Compare to l(10) computed by hand.
    assert_eq!(l_star, vec![Fr::from(11), Fr::from(13), Fr::from(5)]);

    /*
    let wlx = compute_claim_wlx::<_, DummySponge<Fr, 10>>(&claims, &layer);
    assert_eq!(wlx.1.first().unwrap().clone(), vec![Fr::from(2269)]);
    */

    let aggregated_claim =
        claim_aggregation_wrapper::<DummySponge<Fr, 10>>(output_mle_from_layer, &claims);
    let expected_claim = compute_expected_claim(&layer, &l_star);

    // Compare to W(l_star) computed by hand.
    assert_eq!(expected_claim.get_eval(), Fr::from(26773));

    assert_eq!(aggregated_claim.get_eval(), expected_claim.get_eval());
}

#[test]
fn test_aggro_claim_common_suffix2() {
    // MLE on 3 variables (2^3 = 8 evals)
    let mle_evals: Vec<Fr> = vec![1, 2, 42, 4, 5, 6, 7, 17]
        .into_iter()
        .map(Fr::from)
        .collect();
    let points = vec![
        vec![Fr::from(1), Fr::from(3), Fr::from(5)],
        vec![Fr::from(2), Fr::from(3), Fr::from(5)],
    ];
    let r_star = Fr::from(10);

    // ---------------
    let output_mle_from_layer = vec![DenseMle::new_from_raw(mle_evals.clone(), LayerId::Input(0))];
    let layer = layer_from_evals(mle_evals);
    let claims = claims_from_expr_and_points(layer.get_expression(), &points);

    // W(l(0)), W(l(1)) computed by hand.
    assert_eq!(claims[0].get_eval(), Fr::from(163));
    assert_eq!(claims[1].get_eval(), Fr::from(767));

    let l_star = compute_l_star(&claims, r_star);

    // Compare to l(10) computed by hand.
    assert_eq!(l_star, vec![Fr::from(11), Fr::from(3), Fr::from(5)]);

    let aggregated_claim =
        claim_aggregation_wrapper::<DummySponge<Fr, 10>>(output_mle_from_layer, &claims);
    let expected_claim = compute_expected_claim(&layer, &l_star);

    // Compare to W(l_star) computed by hand.
    assert_eq!(expected_claim.get_eval(), Fr::from(6203));

    assert_eq!(aggregated_claim.get_eval(), expected_claim.get_eval());
}
