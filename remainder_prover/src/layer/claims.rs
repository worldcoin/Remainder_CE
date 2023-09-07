//!Utilities involving the claims a layer makes

use remainder_shared_types::FieldExt;

use crate::sumcheck::*;

use ark_std::{cfg_into_iter, cfg_iter};

use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use thiserror::Error;

use super::Layer;
use crate::layer::LayerId;

use serde::{Deserialize, Serialize};

#[derive(Error, Debug, Clone)]
///Errors to do with aggregating and collecting claims
pub enum ClaimError {
    #[error("The Layer has not finished the sumcheck protocol")]
    ///The Layer has not finished the sumcheck protocol
    SumCheckNotComplete,
    #[error("MLE indices must all be fixed")]
    ///MLE indices must all be fixed
    ClaimMleIndexError,
    #[error("Layer ID not assigned")]
    ///Layer ID not assigned
    LayerMleError,
    #[error("MLE within MleRef has multiple values within it")]
    ///MLE within MleRef has multiple values within it
    MleRefMleError,
    #[error("Error aggregating claims")]
    ///Error aggregating claims
    ClaimAggroError,
    #[error("Should be evaluating to a sum")]
    ///Should be evaluating to a sum
    ExpressionEvalError,
    #[error("Number of variables mismatch.")]
    /// Number of variables mismatch when building `ClaimGroup<F>`.
    NumVarsMismatch,
    #[error("Point coordinate index out of bounds.")]
    IndexOutOfBounds,
}

/// A claim contains a `point` \in F^n along with the `result` \in F
/// that an associated layer MLE is expected to evaluate to. In other
/// words, if `W : F^n -> F` is the MLE, then the claim asserts:
/// `W(point) == result`.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub(crate) struct Claim<F> {
    /// The point in F^n where the layer MLE is to be evaluated on.
    point: Vec<F>,
    /// The expected result of evaluating this layer's MLE on `point`.
    result: F,
}

impl<F: FieldExt> Claim<F> {
    pub fn new(point: Vec<F>, result: F) -> Self {
        Self { point, result }
    }

    pub fn get_point(&self) -> &Vec<F> {
        &self.point
    }

    /// The length of the `point` vector.
    pub fn get_num_vars(&self) -> usize {
        self.point.len()
    }

    pub fn get_result(&self) -> F {
        self.result
    }
}

/// A list of claims in F^n
/// all originating from the same layer with ID `from_layer_id`.
/// Invariant: All claims are on the same number of variables, i.e.
/// their points are elements of F^n for some common n among all
/// claims.
#[derive(Clone)]
pub(crate) struct ClaimGroup<F> {
    /// A vector of claims in F^n.
    claims: Vec<Claim<F>>,
    /// The layer ID of the source layer.
    from_layer_id: LayerId,
    /// The points in `claims` is effectively a matrix of elements in
    /// F. We also store the transpose of this matrix for convenient
    /// access.
    /// TODO(Makis): Consider only providing iterator access instead
    /// of storing separate, redundant copies.
    claim_points_transpose: Vec<Vec<F>>,
    /// A vector of `self.get_num_claims()` elements.
    /// For each claim i, stores the expected result of the i-th
    /// claim.
    result_vector: Vec<F>,
}

impl<F: FieldExt> ClaimGroup<F> {
    /// Build a ClaimGroup<F> struct through a vector of claims.
    pub fn new(claims: Vec<Claim<F>>, from_layer_id: LayerId) -> Result<Self, ClaimError> {
        let num_claims = claims.len();
        if num_claims == 0 {
            return Ok(Self {
                claims: vec![],
                from_layer_id,
                claim_points_transpose: vec![],
                result_vector: vec![],
            });
        }
        let num_vars = claims[0].get_num_vars();

        // Check all claims match on the number of variables.
        if !claims
            .into_iter()
            .all(|claim| claim.get_num_vars() == num_vars)
        {
            return Err(ClaimError::NumVarsMismatch);
        }

        // Compute the claim points transpose.
        let claim_points_transpose: Vec<Vec<F>> = (0..num_vars)
            .map(|j| (0..num_claims).map(|i| claims[i].get_point()[j]).collect())
            .collect();

        // Compute the result vector.
        let result_vector: Vec<F> = (0..num_claims).map(|i| claims[i].get_result()).collect();

        Ok(Self {
            claims,
            from_layer_id,
            claim_points_transpose,
            result_vector,
        })
    }

    pub fn get_num_claims(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    /// Returns the number of indices of the claims stored.
    /// Panics if no claims present.
    pub fn get_num_vars(&self) -> usize {
        self.claims[0].get_num_vars()
    }

    pub fn get_from_layer_id(&self) -> LayerId {
        self.from_layer_id
    }

    /// Returns the i-th result.
    /// Panics if i is not in the range
    /// 0 <= i < `self.get_num_claims()`.
    pub fn get_result(&self, i: usize) -> F {
        self.claims[i].get_result()
    }

    /// Returns the i-th point.
    /// Panics if i is not in the range
    /// 0 <= i < `self.get_num_claims()`.
    pub fn get_challenge(&self, i: usize) -> &Vec<F> {
        &self.claims[i].get_point()
    }

    /// Returns a vector of `self.get_num_claims()` elements, the j-th
    /// entry of which is the i-th coordinate of the j-th claim's
    /// point.
    /// Panics if i is not in the range: 0 <= i < `self.get_num_vars()`.
    pub fn get_points_column(&self, i: usize) -> &Vec<F> {
        &self.claim_points_transpose[i]
    }

    /// Returns an "n x m" matrix where n = `self.get_num_vars()` and
    /// m = `self.get_num_claims()`, containing the claim points as
    /// its columns.
    pub fn get_claim_points_transpose(&self) -> &Vec<Vec<F>> {
        &self.claim_points_transpose
    }

    pub fn get_result_vector(&self) -> &Vec<F> {
        &self.result_vector
    }
}

/// Type for storing all the claims to/from a layer.
type LayersClaim<F: FieldExt> = Vec<ClaimGroup<F>>;

pub(crate) fn compute_aggregated_challenges<F: FieldExt>(
    claims: &ClaimGroup<F>,
    r_star: F,
) -> Result<Vec<F>, ClaimError> {
    let num_claims = claims.get_num_claims();
    let num_vars = claims.get_num_vars();

    let challenge_matrix_transpose = claims.get_claim_points_transpose();
    debug_assert_eq!(challenge_matrix_transpose.len(), num_vars);

    if claims.is_empty() {
        return Err(ClaimError::ClaimAggroError);
    }
    debug_assert_eq!(challenge_matrix_transpose[0].len(), num_claims);

    // Compute r = l(r*).
    let r: Vec<F> = cfg_into_iter!(0..num_vars)
        .map(|idx| {
            let evals = challenge_matrix_transpose[idx];
            evaluate_at_a_point(&evals, r_star).unwrap()
        })
        .collect();

    Ok(r)
}

pub(crate) fn compute_claim_wlx<F: FieldExt>(
    claims: &ClaimGroup<F>,
    layer: &impl Layer<F>,
) -> Result<Vec<F>, ClaimError> {
    let num_claims = claims.get_num_claims();
    let num_vars = claims.get_num_vars();

    let results = claims.get_result_vector();
    let challenge_matrix_transpose = claims.get_claim_points_transpose();
    debug_assert_eq!(challenge_matrix_transpose.len(), num_vars);

    if claims.is_empty() {
        return Err(ClaimError::ClaimAggroError);
    }
    debug_assert_eq!(challenge_matrix_transpose[0].len(), num_claims);

    // Get the evals [W(l(0)), W(l(1)), ... ]
    let wlx =
        layer.get_wlx_evaluations(challenge_matrix_transpose, results, num_claims, num_vars)?;

    Ok(wlx)
}

/// verifies the claim aggregation
pub(crate) fn verify_aggregate_claim<F: FieldExt>(
    wlx: &Vec<F>, // synonym for qx
    claims: &ClaimGroup<F>,
    r_star: F,
) -> Result<Claim<F>, ClaimError> {
    let num_claims = claims.get_num_claims();
    let num_vars = claims.get_num_vars();

    // Check q(0), q(1) equals the claimed value -- i.e. W(l(0)),
    // W(l(1)), etc.
    for idx in 0..num_claims {
        if claims.get_result(idx) != wlx[idx] {
            return Err(ClaimError::ClaimAggroError);
        }
    }

    let r = compute_aggregated_challenges(claims, r_star)?;
    let q_rstar = evaluate_at_a_point(wlx, r_star).unwrap();

    let aggregated_claim: Claim<F> = Claim::new(r, q_rstar);
    Ok(aggregated_claim)
}

#[cfg(test)]
mod tests {

    use crate::expression::{Expression, ExpressionStandard};
    use crate::layer::{from_mle, GKRLayer, LayerId};
    use crate::mle::dense::DenseMle;
    use rand::Rng;
    use remainder_shared_types::transcript::poseidon_transcript::PoseidonTranscript;

    use super::*;
    use ark_std::test_rng;
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;

    #[test]
    fn test_get_claim() {
        // [1, 1, 1, 1] \oplus (1 - (1 * (1 + V[1, 1, 1, 1]))) * 2
        let expression1: ExpressionStandard<Fr> = ExpressionStandard::Constant(Fr::one());
        let mle = DenseMle::<_, Fr>::new_from_raw(
            vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()],
            LayerId::Input(0),
            None,
        );
        let expression3 = ExpressionStandard::Mle(mle.mle_ref());
        let expression = expression1.clone() + expression3.clone();
        // let expression = expression1.clone() * expression;
        let expression = expression1 - expression;
        let expression = expression * Fr::from(2);
        let _expression = expression3.concat_expr(expression);

        // TODO(ryancao): Need to create a layer and fix all the MLE variables...
    }

    // ------- Helper functions for claim aggregation tests -------

    /// Builds `ClaimGroup<Fr>` by evaluation an expression `expr` on each
    /// point in `points`.
    fn claims_from_expr_and_points(
        expr: &ExpressionStandard<Fr>,
        points: &Vec<Vec<Fr>>,
    ) -> ClaimGroup<Fr> {
        let claims_vector: Vec<Claim<Fr>> = cfg_into_iter!(points)
            .map(|point| {
                let mut exp = expr.clone();
                exp.index_mle_indices(0);
                let result = exp.evaluate_expr(point.clone()).unwrap();
                Claim::new(point.clone(), result)
            })
            .collect();

        ClaimGroup::new(claims_vector, LayerId::Output(0)).unwrap()
    }

    /// Builds an expression corresponding to the MLE of a function
    /// whose evaluations on the boolean hypercube are given by
    /// `mle_evals`, along with a dummy GKR layer.
    fn expr_and_layer_from_evals(
        mle_evals: Vec<Fr>,
    ) -> (ExpressionStandard<Fr>, GKRLayer<Fr, PoseidonTranscript<Fr>>) {
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_evals, LayerId::Input(0), None);
        let mle_ref = mle.mle_ref();

        let mut expr = ExpressionStandard::Mle(mle_ref);

        let layer = from_mle(
            mle,
            |mle| mle.mle_ref().expression(),
            |_, _, _| unimplemented!(),
        );

        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input(0));

        (expr, layer)
    }

    fn claim_aggregation_wrapper(
        expr: &ExpressionStandard<Fr>,
        layer: &impl Layer<Fr>,
        claims: &ClaimGroup<Fr>,
        r_star: Fr,
    ) -> Claim<Fr> {
        let r = compute_aggregated_challenges(claims, r_star).unwrap();
        let wlx = compute_claim_wlx(claims, layer).unwrap();
        let claimed_val = evaluate_at_a_point(&wlx, r_star).unwrap();
        Claim::new(r, claimed_val)
    }

    /// Compute l* = l(r*).
    fn compute_l_star(claims: &ClaimGroup<Fr>, r_star: &Fr) -> Vec<Fr> {
        let num_claims = claims.get_num_claims();
        let num_vars = claims.get_num_vars();

        (0..num_vars)
            .map(|i| {
                let evals: &Vec<Fr> = claims.get_points_column(i);
                evaluate_at_a_point(evals, r_star.clone()).unwrap()
            })
            .collect()
    }

    // Returns expected aggregated claim of `expr` on l(r_star) = `l_star`.
    fn compute_expected_claim(expr: &ExpressionStandard<Fr>, l_star: &Vec<Fr>) -> Claim<Fr> {
        let mut expr = expr.clone();
        expr.index_mle_indices(0);
        let result = expr.evaluate_expr(l_star.clone()).unwrap();
        Claim::new(l_star.clone(), result)
    }

    // ----------------------------------------------------------

    /// Test claim aggregation small mle
    #[test]
    fn test_aggro_claim_1() {
        let mle_evals = vec![Fr::from(1), Fr::from(0), Fr::from(2), Fr::from(3)];
        let points = vec![
            vec![Fr::from(3), Fr::from(3)],
            vec![Fr::from(2), Fr::from(7)],
        ];
        let r_star = Fr::from(10);

        // ---------------

        let (expr, layer) = expr_and_layer_from_evals(mle_evals);
        let claims = claims_from_expr_and_points(&expr, &points);

        let l_star = compute_l_star(&claims, &r_star);

        // Compare to l(10) computed by hand.
        assert_eq!(l_star, vec![Fr::from(7).neg(), Fr::from(43)]);

        let aggregated_claim = claim_aggregation_wrapper(&expr, &layer, &claims, r_star.clone());
        let expected_claim = compute_expected_claim(&expr, &l_star);

        assert_eq!(aggregated_claim, expected_claim);
    }

    /// Test claim aggregation on another small mle
    #[test]
    fn test_aggro_claim_2() {
        let mle_evals = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
        let points = vec![
            vec![Fr::from(1), Fr::from(2)],
            vec![Fr::from(2), Fr::from(3)],
            vec![Fr::from(3), Fr::from(1)],
        ];
        let r_star = Fr::from(2).neg();

        // ---------------

        let (expr, layer) = expr_and_layer_from_evals(mle_evals);
        let claims = claims_from_expr_and_points(&expr, &points);

        let l_star = compute_l_star(&claims, &r_star);

        // TODO: Assert l_star was computed correctly.

        let aggregated_claim = claim_aggregation_wrapper(&expr, &layer, &claims, r_star.clone());
        let expected_claim = compute_expected_claim(&expr, &l_star);

        assert_eq!(aggregated_claim, expected_claim);
    }

    /// Test claim aggregation for 3 claims on a random MLE on 8
    /// variables with random challenge.
    #[test]
    fn test_aggro_claim_3() {
        let mut rng = test_rng();
        let mle_evals = vec![
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
        ];

        let points = vec![
            vec![Fr::from(2).neg(), Fr::from(192013).neg(), Fr::from(2148)],
            vec![Fr::from(123), Fr::from(482), Fr::from(241)],
            vec![Fr::from(92108), Fr::from(29014), Fr::from(524)],
        ];

        let r_star = Fr::from(rng.gen::<u64>());

        // ---------------

        let (expr, layer) = expr_and_layer_from_evals(mle_evals);
        let claims = claims_from_expr_and_points(&expr, &points);

        let l_star = compute_l_star(&claims, &r_star);

        let aggregated_claim = claim_aggregation_wrapper(&expr, &layer, &claims, r_star.clone());
        let expected_claim = compute_expected_claim(&expr, &l_star);

        assert_eq!(aggregated_claim, expected_claim);
    }

    /*
    /// Test claim aggregation on a RANDOM mle
    #[test]
    fn test_aggro_claim_4() {
        let mut rng = test_rng();
        let mle_v1 = vec![Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())];
        let mle_v2 = vec![
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
            Fr::from(rng.gen::<u64>()),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);
        let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);
        let mle_ref = mle1.mle_ref();
        let mle_ref2 = mle2.mle_ref();

        let mut expr = ExpressionStandard::Product(vec![mle_ref, mle_ref2]);
        let mut expr_copy = expr.clone();

        let layer = from_mle(
            (mle1, mle2),
            |mle| ExpressionStandard::products(vec![mle.0.mle_ref(), mle.1.mle_ref()]),
            |_, _, _| unimplemented!(),
        );
        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input(0));

        let chals1 = vec![Fr::from(2).neg(), Fr::from(192013).neg(), Fr::from(2148)];
        let chals2 = vec![Fr::from(123), Fr::from(482), Fr::from(241)];
        let chals3 = vec![Fr::from(92108), Fr::from(29014), Fr::from(524)];
        let chals = vec![&chals1, &chals2, &chals3];
        let mut valchal: Vec<Fr> = Vec::new();
        for i in 0..3 {
            let mut exp = expr.clone();
            exp.index_mle_indices(0);
            let eval = exp.evaluate_expr((*chals[i]).clone());
            valchal.push(eval.unwrap());
        }

        let claim1: Claim<Fr> = (chals1, valchal[0]);
        let claim2: Claim<Fr> = (chals2, valchal[1]);
        let claim3: Claim<Fr> = (chals3, valchal[2] + Fr::one());

        let rchal = Fr::from(rng.gen::<u64>());

        let res: Claim<Fr> = aggregate_claims(&[claim1, claim2, claim3], &layer, rchal)
            .unwrap()
            .0;

        let transpose1 = vec![Fr::from(2).neg(), Fr::from(123), Fr::from(92108)];
        let transpose2 = vec![Fr::from(192013).neg(), Fr::from(482), Fr::from(29014)];
        let transpose3 = vec![Fr::from(2148), Fr::from(241), Fr::from(524)];

        let fix_vars: Vec<Fr> = vec![transpose1, transpose2, transpose3]
            .into_iter()
            .map(|evals| evaluate_at_a_point(&evals, rchal).unwrap())
            .collect();

        expr_copy.index_mle_indices(0);

        let eval_fixed_vars = expr_copy.evaluate_expr(fix_vars.clone()).unwrap();
        let claim_fixed_vars: Claim<Fr> = (fix_vars, eval_fixed_vars);
        assert_ne!(res, claim_fixed_vars);
    }

    /// Make sure claim aggregation FAILS for a WRONG CLAIM!
    #[test]
    fn test_aggro_claim_negative_1() {
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
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);
        let mle_ref = mle1.mle_ref();
        let mut expr = ExpressionStandard::Mle(mle_ref);
        let mut expr_copy = expr.clone();

        let layer = from_mle(
            mle1,
            |mle| mle.mle_ref().expression(),
            |_, _, _| unimplemented!(),
        );
        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input(0));

        let chals1 = vec![Fr::from(2).neg(), Fr::from(192013).neg(), Fr::from(2148)];
        let chals2 = vec![Fr::from(123), Fr::from(482), Fr::from(241)];
        let chals3 = vec![Fr::from(92108), Fr::from(29014), Fr::from(524)];
        let chals = vec![&chals1, &chals2, &chals3];
        let mut valchal: Vec<Fr> = Vec::new();
        for i in 0..3 {
            let mut exp = expr.clone();
            exp.index_mle_indices(0);
            let eval = exp.evaluate_expr((*chals[i]).clone());
            valchal.push(eval.unwrap());
        }

        let claim1: Claim<Fr> = (chals1, valchal[0] - Fr::one());
        let claim2: Claim<Fr> = (chals2, valchal[1]);
        let claim3: Claim<Fr> = (chals3, valchal[2]);

        let rchal = Fr::from(rng.gen::<u64>());

        let res: Claim<Fr> = aggregate_claims(&[claim1, claim2, claim3], &layer, rchal)
            .unwrap()
            .0;

        let transpose1 = vec![Fr::from(2).neg(), Fr::from(123), Fr::from(92108)];
        let transpose2 = vec![Fr::from(192013).neg(), Fr::from(482), Fr::from(29014)];
        let transpose3 = vec![Fr::from(2148), Fr::from(241), Fr::from(524)];

        let fix_vars: Vec<Fr> = vec![transpose1, transpose2, transpose3]
            .into_iter()
            .map(|evals| evaluate_at_a_point(&evals, rchal).unwrap())
            .collect();

        expr_copy.index_mle_indices(0);

        let eval_fixed_vars = expr_copy.evaluate_expr(fix_vars.clone()).unwrap();
        let claim_fixed_vars: Claim<Fr> = (fix_vars, eval_fixed_vars);
        assert_ne!(res, claim_fixed_vars);
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
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);
        let mle_ref = mle1.mle_ref();
        let mut expr = ExpressionStandard::Mle(mle_ref);
        let mut expr_copy = expr.clone();

        let layer = from_mle(
            mle1,
            |mle| mle.mle_ref().expression(),
            |_, _, _| unimplemented!(),
        );
        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input(0));

        let chals1 = vec![Fr::from(2).neg(), Fr::from(192013).neg(), Fr::from(2148)];
        let chals2 = vec![Fr::from(123), Fr::from(482), Fr::from(241)];
        let chals3 = vec![Fr::from(92108), Fr::from(29014), Fr::from(524)];
        let chals = vec![&chals1, &chals2, &chals3];
        let mut valchal: Vec<Fr> = Vec::new();
        for i in 0..3 {
            let mut exp = expr.clone();
            exp.index_mle_indices(0);
            let eval = exp.evaluate_expr((*chals[i]).clone());
            valchal.push(eval.unwrap());
        }

        let claim1: Claim<Fr> = (chals1, valchal[0]);
        let claim2: Claim<Fr> = (chals2, valchal[1]);
        let claim3: Claim<Fr> = (chals3, valchal[2] + Fr::one());

        let rchal = Fr::from(rng.gen::<u64>());

        let res: Claim<Fr> = aggregate_claims(&[claim1, claim2, claim3], &layer, rchal)
            .unwrap()
            .0;

        let transpose1 = vec![Fr::from(2).neg(), Fr::from(123), Fr::from(92108)];
        let transpose2 = vec![Fr::from(192013).neg(), Fr::from(482), Fr::from(29014)];
        let transpose3 = vec![Fr::from(2148), Fr::from(241), Fr::from(524)];

        let fix_vars: Vec<Fr> = vec![transpose1, transpose2, transpose3]
            .into_iter()
            .map(|evals| evaluate_at_a_point(&evals, rchal).unwrap())
            .collect();

        expr_copy.index_mle_indices(0);

        let eval_fixed_vars = expr_copy.evaluate_expr(fix_vars.clone()).unwrap();
        let claim_fixed_vars: Claim<Fr> = (fix_vars, eval_fixed_vars);
        assert_ne!(res, claim_fixed_vars);
    }

    #[test]
    fn test_verify_claim_aggro() {
        let mle_v1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);
        let mle_ref = mle1.mle_ref();
        let mut expr = ExpressionStandard::Mle(mle_ref);
        let mut expr_copy = expr.clone();

        let layer = from_mle(
            mle1,
            |mle| mle.mle_ref().expression(),
            |_, _, _| unimplemented!(),
        );
        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input(0));

        let chals1 = vec![Fr::from(1), Fr::from(2)];
        let chals2 = vec![Fr::from(2), Fr::from(3)];
        let chals3 = vec![Fr::from(3), Fr::from(1)];
        let chals = vec![&chals1, &chals2, &chals3];
        let mut valchal: Vec<Fr> = Vec::new();

        for i in 0..3 {
            let mut exp = expr.clone();
            exp.index_mle_indices(0);
            let eval = exp.evaluate_expr((*chals[i]).clone());
            valchal.push(eval.unwrap());
        }

        let claim1: Claim<Fr> = (chals1, valchal[0]);
        let claim2: Claim<Fr> = (chals2, valchal[1]);
        let claim3: Claim<Fr> = (chals3, valchal[2]);
        let claims = vec![claim1, claim2, claim3];

        let rchal = Fr::from(2).neg();

        let (res, wlx) = aggregate_claims(&claims, &layer, rchal).unwrap();
        let rounds = expr.index_mle_indices(0);
        // for round in 0..rounds {
        //     expr.fix
        // }
        let verify_result = verify_aggregate_claim(&wlx, &claims, rchal).unwrap();
    }
    */
}
