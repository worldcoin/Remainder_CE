//!Utilities involving the claims a layer makes

use crate::{expression::ExpressionStandard, mle::beta::BetaTable};
use lcpc_2d::FieldExt;

use itertools::Itertools;
use crate::mle::MleRef;
use crate::sumcheck::*;

use ark_std::{cfg_into_iter, cfg_iter};

use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use thiserror::Error;

use super::{Claim, Layer};

#[derive(Error, Debug, Clone)]
///Errors to do with aggregating and collecting claims
///Errors to do with aggregating and collecting claims
pub enum ClaimError {
    #[error("The Layer has not finished the sumcheck protocol")]
    ///The Layer has not finished the sumcheck protocol
    ///The Layer has not finished the sumcheck protocol
    SumCheckNotComplete,
    #[error("MLE indices must all be fixed")]
    ///MLE indices must all be fixed
    ///MLE indices must all be fixed
    ClaimMleIndexError,
    #[error("Layer ID not assigned")]
    ///Layer ID not assigned
    ///Layer ID not assigned
    LayerMleError,
    #[error("MLE within MleRef has multiple values within it")]
    ///MLE within MleRef has multiple values within it
    ///MLE within MleRef has multiple values within it
    MleRefMleError,
    #[error("Error aggregating claims")]
    ///Error aggregating claims
    ///Error aggregating claims
    ClaimAggroError,
    #[error("Should be evaluating to a sum")]
    ///Should be evaluating to a sum
    ///Should be evaluating to a sum
    ExpressionEvalError,
}

// /// Compute evaluations of W(l(x))
// fn compute_wlx<F: FieldExt>(
//     layer: &impl Layer<F>,
//     claim_vecs: Vec<Vec<F>>,
//     claimed_vals: &mut Vec<F>,
//     num_claims: usize,
//     num_idx: usize,
//     // prev_layer_claim: Claim<F>,
// ) -> Result<Vec<F>, ClaimError> {
//     //fix variable hella times
//     //evaluate expr on the mutated expr

//     // get the number of evaluations
//     let num_vars = expr.index_mle_indices(0);
//     let degree = get_round_degree(&expr, 0);
//     // expr.init_beta_tables(prev_layer_claim);
//     let num_evals = (num_vars) * (num_claims); //* degree;

//     // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
//     let next_evals: Result<Vec<F>, ClaimError> = cfg_into_iter!(num_claims..num_evals)
//         .map(|idx| {
//             // get the challenge l(idx)
//             let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
//                 .map(|claim_idx| {
//                     let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
//                         .map(|claim| claim[claim_idx])
//                         .collect();
//                     let res = evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap();
//                     res
//                 })
//                 .collect();

//             // use fix_var to compute W(l(index))
//             // let mut fix_expr = expr.clone();
//             // let eval_w_l = fix_expr.evaluate_expr(new_chal);

//             let mut beta = BetaTable::new((new_chal, F::zero())).unwrap();
//             beta.table.index_mle_indices(0);
//             let eval = compute_sumcheck_message(expr, 0, degree, &beta).unwrap();
//             if let SumOrEvals::Evals(evals) = eval {
//                 Ok(evals[0] + evals[1])
//             } else {
//                 panic!()
//             }

//             // this has to be a sum--get the overall evaluation
//             // match eval_w_l {
//             //     Ok(evaluation) => Ok(evaluation),
//             //     Err(_) => Err(ClaimError::ExpressionEvalError)
//             // }
//         })
//         .collect();

//     // concat this with the first k evaluations from the claims to get num_evals evaluations
//     claimed_vals.extend(&next_evals.unwrap());
//     let wlx_evals = claimed_vals.clone();
//     Ok(wlx_evals)
// }

/// Aggregate several claims into one
pub(crate) fn aggregate_claims<F: FieldExt>(
    claims: &[Claim<F>],
    layer: &impl Layer<F>,
    rstar: F,
) -> Result<(Claim<F>, Vec<F>), ClaimError> {
    let (claim_vecs, mut vals): (Vec<Vec<F>>, Vec<F>) = cfg_iter!(claims).cloned().unzip();

    if claims.is_empty() {
        return Err(ClaimError::ClaimAggroError);
    }

    // --- Number of variables in the claim ---
    let num_idx = claim_vecs[0].len();

    // get the claim (r = l(r*))
    let r: Vec<F> = cfg_into_iter!(0..num_idx)
        .map(|idx| {
            let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                .map(|claim| claim[idx])
                .collect();
            // --- Interpolate u_1, v_1, w_1 --> r_1 ---
            // l(0) = u_1, ... what is l(r^*)?
            evaluate_at_a_point(&evals, rstar).unwrap()
        })
        .collect();

    // get the evals [W(l(0)), W(l(1)), ...]
    let wlx = layer.get_wlx_evaluations(claim_vecs, &mut vals, claims.len(), num_idx)?;

    // interpolate to get W(l(r)), that's the claimed value
    let claimed_val = evaluate_at_a_point(&wlx, rstar);

    Ok(((r, claimed_val.unwrap()), wlx))
}

/// verifies the claim aggregation
pub(crate) fn verify_aggregate_claim<F: FieldExt>(
    wlx: &Vec<F>, // synonym for qx
    claims: &[Claim<F>],
    r_star: F,
) -> Result<Claim<F>, ClaimError> {
    let (claim_vecs, _): (Vec<Vec<F>>, Vec<F>) = cfg_iter!(claims).cloned().unzip();
    let num_idx = claim_vecs[0].len();

    // check q(0), q(1) equals the claimed value (or wl(0), wl(1))
    for (idx, claim) in claims.iter().enumerate() {
        if claim.1 != wlx[idx] {
            return Err(ClaimError::ClaimAggroError);
        }
    }

    // compute r = l(r_star)
    let r: Vec<F> = cfg_into_iter!(0..num_idx)
        .map(|idx| {
            let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                .map(|claim| claim[idx])
                .collect();
            evaluate_at_a_point(&evals, r_star).unwrap()
        })
        .collect();

    let q_rstar = evaluate_at_a_point(wlx, r_star).unwrap();

    let aggregated_claim: Claim<F> = (r, q_rstar);

    Ok(aggregated_claim)
}

#[cfg(test)]
mod tests {

    use crate::expression::Expression;
    use crate::layer::{from_mle, GKRLayer, LayerId};
    use crate::mle::{dense::DenseMle, Mle};
    use crate::transcript::poseidon_transcript::PoseidonTranscript;

    use super::*;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;
    use ark_std::One;

    #[test]
    fn test_get_claim() {
        // [1, 1, 1, 1] \oplus (1 - (1 * (1 + V[1, 1, 1, 1]))) * 2
        let expression1: ExpressionStandard<Fr> = ExpressionStandard::Constant(Fr::one());
        let mle = DenseMle::<_, Fr>::new_from_raw(
            vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()],
            LayerId::Input,
            None,
        );
        let expression3 = ExpressionStandard::Mle(mle.mle_ref());
        let expression = expression1.clone() + expression3.clone();
        // let expression = expression1.clone() * expression;
        let expression = expression1 - expression;
        let expression = expression * Fr::from(2);
        let _expression = expression3.concat(expression);

        // TODO(ryancao): Need to create a layer and fix all the MLE variables...
    }

    /// Test claim aggregation small mle
    #[test]
    fn test_aggro_claim() {
        let _dummy_claim = (vec![Fr::one(); 2], Fr::from(0));

        let mle_v1 = vec![Fr::from(1), Fr::from(0), Fr::from(2), Fr::from(3)];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input, None);
        let mle_ref = mle1.mle_ref();

        let mut expr = ExpressionStandard::Mle(mle_ref);

        let layer = from_mle(mle1, |mle| mle.mle_ref().expression(), |_, _, _| unimplemented!());
        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input);
        let mut expr_copy = expr.clone();

        let chals1 = vec![Fr::from(3), Fr::from(3)];
        let chals2 = vec![Fr::from(2), Fr::from(7)];
        let chals = vec![&chals1, &chals2];

        let mut valchal: Vec<Fr> = Vec::new();
        for i in 0..2 {
            let mut exp = expr.clone();

            exp.index_mle_indices(0);
            let eval = exp.evaluate_expr((*chals[i]).clone());
            valchal.push(eval.unwrap());
        }

        let claim1: Claim<Fr> = (chals1, valchal[0]);
        let claim2: Claim<Fr> = (chals2, valchal[1]);

        let res = aggregate_claims(&[claim1, claim2], &layer, Fr::from(10))
            .unwrap()
            .0;

        expr_copy.index_mle_indices(0);
        let challenge_l_10 = vec![Fr::from(-7), Fr::from(43)]; // l(10), by hand
        let eval_l_10 = expr_copy.evaluate_expr(challenge_l_10.clone()).unwrap();
        let claim_l_10: Claim<Fr> = (challenge_l_10, eval_l_10);
        assert_eq!(res, claim_l_10);
    }

    /// Test claim aggregation on another small mle
    #[test]
    fn test_aggro_claim_2() {
        let _dummy_claim = (vec![Fr::one(); 2], Fr::from(0));

        let mle_v1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input, None);
        let mle_ref = mle1.mle_ref();
        let mut expr = ExpressionStandard::Mle(mle_ref);
        let mut expr_copy = expr.clone();

        let layer = from_mle(mle1, |mle| mle.mle_ref().expression(), |_, _, _| unimplemented!());
        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input);

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

        let rchal = Fr::from(-2);

        let res: Claim<Fr> = aggregate_claims(&[claim1, claim2, claim3], &layer, rchal)
            .unwrap()
            .0;

        let transpose1 = vec![Fr::from(1), Fr::from(2), Fr::from(3)];
        let transpose2 = vec![Fr::from(2), Fr::from(3), Fr::from(1)];

        let fix_vars: Vec<Fr> = vec![transpose1, transpose2]
            .into_iter()
            .map(|evals| evaluate_at_a_point(&evals, rchal).unwrap())
            .collect();

        expr_copy.index_mle_indices(0);

        let eval_fixed_vars = expr_copy.evaluate_expr(fix_vars.clone()).unwrap();
        let claim_fixed_vars: Claim<Fr> = (fix_vars, eval_fixed_vars);
        assert_eq!(res, claim_fixed_vars);
    }

    /// Test claim aggregation on random mle
    #[test]
    fn test_aggro_claim_3() {
        let _dummy_claim = (vec![Fr::one(); 3], Fr::from(0));
        let mut rng = test_rng();
        let mle_v1 = vec![
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input, None);
        let mle_ref = mle1.mle_ref();
        let mut expr = ExpressionStandard::Mle(mle_ref);
        let mut expr_copy = expr.clone();

        let layer = from_mle(mle1, |mle| mle.mle_ref().expression(), |_, _, _| unimplemented!());
        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input);

        let chals1 = vec![Fr::from(-2), Fr::from(-192013), Fr::from(2148)];
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
        let claim3: Claim<Fr> = (chals3, valchal[2]);

        let rchal = Fr::rand(&mut rng);

        let res: Claim<Fr> = aggregate_claims(&[claim1, claim2, claim3], &layer, rchal)
            .unwrap()
            .0;

        let transpose1 = vec![Fr::from(-2), Fr::from(123), Fr::from(92108)];
        let transpose2 = vec![Fr::from(-192013), Fr::from(482), Fr::from(29014)];
        let transpose3 = vec![Fr::from(2148), Fr::from(241), Fr::from(524)];

        let fix_vars: Vec<Fr> = vec![transpose1, transpose2, transpose3]
            .into_iter()
            .map(|evals| evaluate_at_a_point(&evals, rchal).unwrap())
            .collect();

        expr_copy.index_mle_indices(0);
        // expr_copy.init_beta_tables(dummy_claim);

        let eval_fixed_vars = expr_copy.evaluate_expr(fix_vars.clone()).unwrap();
        let claim_fixed_vars: Claim<Fr> = (fix_vars, eval_fixed_vars);
        assert_eq!(res, claim_fixed_vars);
    }

    /// Test claim aggregation on a RANDOM mle
    #[test]
    fn test_aggro_claim_4() {
        let _dummy_claim = (vec![Fr::from(1); 3], Fr::from(0));
        let mut rng = test_rng();
        let mle_v1 = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let mle_v2 = vec![
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input, None);
        let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input, None);
        let mle_ref = mle1.mle_ref();
        let mle_ref2 = mle2.mle_ref();

        let mut expr = ExpressionStandard::Product(vec![mle_ref, mle_ref2]);
        let mut expr_copy = expr.clone();

        let layer = from_mle((mle1, mle2), |mle| ExpressionStandard::products(vec![mle.0.mle_ref(), mle.1.mle_ref()]), |_, _, _| unimplemented!());
        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input);

        let chals1 = vec![Fr::from(-2), Fr::from(-192013), Fr::from(2148)];
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

        let rchal = Fr::rand(&mut rng);

        let res: Claim<Fr> = aggregate_claims(&[claim1, claim2, claim3], &layer, rchal)
            .unwrap()
            .0;

        let transpose1 = vec![Fr::from(-2), Fr::from(123), Fr::from(92108)];
        let transpose2 = vec![Fr::from(-192013), Fr::from(482), Fr::from(29014)];
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
        let _dummy_claim = (vec![Fr::from(1); 3], Fr::from(0));
        let mut rng = test_rng();
        let mle_v1 = vec![
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input, None);
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input, None);
        let mle_ref = mle1.mle_ref();
        let mut expr = ExpressionStandard::Mle(mle_ref);
        let mut expr_copy = expr.clone();

        let layer = from_mle(mle1, |mle| mle.mle_ref().expression(), |_, _, _| unimplemented!());
        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input);

        let chals1 = vec![Fr::from(-2), Fr::from(-192013), Fr::from(2148)];
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

        let rchal = Fr::rand(&mut rng);

        let res: Claim<Fr> = aggregate_claims(&[claim1, claim2, claim3], &layer, rchal)
            .unwrap()
            .0;

        let transpose1 = vec![Fr::from(-2), Fr::from(123), Fr::from(92108)];
        let transpose2 = vec![Fr::from(-192013), Fr::from(482), Fr::from(29014)];
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
        let _dummy_claim = (vec![Fr::from(1); 3], Fr::from(0));
        let mut rng = test_rng();
        let mle_v1 = vec![
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
        ];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input, None);
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input, None);
        let mle_ref = mle1.mle_ref();
        let mut expr = ExpressionStandard::Mle(mle_ref);
        let mut expr_copy = expr.clone();

        let layer = from_mle(mle1, |mle| mle.mle_ref().expression(), |_, _, _| unimplemented!());
        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input);

        let chals1 = vec![Fr::from(-2), Fr::from(-192013), Fr::from(2148)];
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

        let rchal = Fr::rand(&mut rng);

        let res: Claim<Fr> = aggregate_claims(&[claim1, claim2, claim3], &layer, rchal)
            .unwrap()
            .0;

        let transpose1 = vec![Fr::from(-2), Fr::from(123), Fr::from(92108)];
        let transpose2 = vec![Fr::from(-192013), Fr::from(482), Fr::from(29014)];
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
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input, None);
        let mle_ref = mle1.mle_ref();
        let mut expr = ExpressionStandard::Mle(mle_ref);
        let mut expr_copy = expr.clone();

        let layer = from_mle(mle1, |mle| mle.mle_ref().expression(), |_, _, _| unimplemented!());
        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input);

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

        let rchal = Fr::from(-2);

        let (_res, wlx) = aggregate_claims(&claims, &layer, rchal).unwrap();
        let _rounds = expr.index_mle_indices(0);
        let _verify_result = verify_aggregate_claim(&wlx, &claims, rchal).unwrap();
    }
}
