//! A set of functions providing the interface for perfoming claim aggregation.

use std::cmp::max;

use ark_std::{cfg_into_iter, end_timer, start_timer};
use remainder_shared_types::{
    config::global_config::global_prover_claim_agg_constant_column_optimization,
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};
use tracing::{debug, info};

use crate::{
    claims::{Claim, RawClaim},
    layer::combine_mles::{
        combine_mles_with_aggregate, get_indexed_layer_mles_to_combine, pre_fix_mles,
    },
    mle::dense::DenseMle,
    sumcheck::evaluate_at_a_point,
};

use super::claim_group::ClaimGroup;

use anyhow::{Ok, Result};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Performs claim aggregation on the prover side and, if successful, returns a
/// single, raw aggregated claim.
///
/// # Parameters
///
/// * `claims`: a slice of claims, all on the same layer (same `to_layer_id`),
///   to be aggregated into one.
/// * `layer`: the GKR layer that this claim group is making claims on; the
///   layer whose ID matches the `to_layer_id` field of all elements in
///   `claims`.
/// * `output_mles_from_layer`: the compiled bookkeeping tables that result from
///   this layer, in order to aggregate into the layerwise bookkeeping table.
/// * `transcript_writer`: is used to post the interpolation polynomial
///   evaluations and generate challenges.
pub fn prover_aggregate_claims<F: Field>(
    claims: &[Claim<F>],
    output_mles_from_layer: Vec<DenseMle<F>>,
    transcript_writer: &mut impl ProverTranscript<F>,
) -> Result<RawClaim<F>> {
    let num_claims = claims.len();
    debug_assert!(num_claims > 0);
    info!("High-level claim aggregation on {num_claims} claims.");

    let claim_preproc_timer = start_timer!(|| "Claim preprocessing".to_string());

    let fixed_output_mles = get_indexed_layer_mles_to_combine(output_mles_from_layer);

    let claim_groups = ClaimGroup::form_claim_groups(claims.to_vec());

    debug!("Grouped claims for aggregation: ");
    for group in &claim_groups {
        debug!("GROUP: {:#?}", group.get_raw_claims());
    }

    end_timer!(claim_preproc_timer);
    let intermediate_timer = start_timer!(|| "Intermediate claim aggregation.".to_string());

    let intermediate_claims = claim_groups
        .into_iter()
        .map(|claim_group| claim_group.prover_aggregate(&fixed_output_mles, transcript_writer))
        .collect::<Result<Vec<_>>>()?;

    end_timer!(intermediate_timer);
    let final_timer = start_timer!(|| "Final stage aggregation.".to_string());

    let intermediate_claims_group = ClaimGroup::new_from_raw_claims(intermediate_claims).unwrap();

    // Finally, aggregate all intermediate claims.
    let claim =
        intermediate_claims_group.prover_aggregate(&fixed_output_mles, transcript_writer)?;

    end_timer!(final_timer);
    Ok(claim)
}

/// Returns an upper bound on the number of evaluations needed to represent the
/// polynomial `P(x) = W(l(x))` where `W : F^n -> F` is a multilinear polynomial
/// on `n` variables and `l : F -> F^n` is such that:
///  * `l(0) = claim_vecs[0]`,
///  * `l(1) = claim_vecs[1]`,
///  * ...,
///  * `l(m-1) = claim_vecs[m-1]`.
///
/// It is guaranteed that the returned value is at least `num_claims =
/// claim_vecs.len()`.
///
/// # Panics
///  if `claim_vecs` is empty.
pub fn get_num_wlx_evaluations<F: Field>(
    claim_vecs: &[Vec<F>],
) -> (usize, Option<Vec<usize>>, Vec<usize>) {
    let num_claims = claim_vecs.len();
    let num_vars = claim_vecs[0].len();

    debug!("Smart num_evals");
    let mut num_constant_columns = num_vars as i64;
    let mut common_idx = vec![];
    let mut non_common_idx = vec![];
    #[allow(clippy::needless_range_loop)]
    for j in 0..num_vars {
        let mut degree_reduced = true;
        for i in 1..num_claims {
            if claim_vecs[i][j] != claim_vecs[i - 1][j] {
                num_constant_columns -= 1;
                degree_reduced = false;
                non_common_idx.push(j);
                break;
            }
        }
        if degree_reduced {
            common_idx.push(j);
        }
    }
    assert!(num_constant_columns >= 0);
    debug!("degree_reduction = {}", num_constant_columns);

    // Evaluate the P(x) := W(l(x)) polynomial at deg(P) + 1
    // points. W : F^n -> F is a multi-linear polynomial on
    // `num_vars` variables and l : F -> F^n is a canonical
    // polynomial passing through `num_claims` points so its degree is
    // at most `num_claims - 1`. This imposes an upper
    // bound of `num_vars * (num_claims - 1)` to the degree of P.
    // However, the actual degree of P might be lower.
    // For any coordinate `i` such that all claims agree
    // on that coordinate, we can quickly deduce that `l_i(x)` is a
    // constant polynomial of degree zero instead of `num_claims -
    // 1` which brings down the total degree by the same amount.
    let num_evals =
        (num_vars) * (num_claims - 1) + 1 - (num_constant_columns as usize) * (num_claims - 1);
    debug!("num_evals originally = {}", num_evals);
    (max(num_evals, num_claims), Some(common_idx), non_common_idx)
}

/// Returns a vector of evaluations of this layer's MLE on a sequence of
/// points computed by interpolating a polynomial that passes through the
/// points of `claims_vecs`.
pub fn get_wlx_evaluations<F: Field>(
    claim_vecs: &[Vec<F>],
    claimed_vals: &[F],
    claim_mles: Vec<DenseMle<F>>,
    num_claims: usize,
    num_idx: usize,
) -> Result<Vec<F>> {
    // get the number of evaluations

    let (num_evals, common_idx) = if global_prover_claim_agg_constant_column_optimization() {
        let (num_evals, common_idx, _) = get_num_wlx_evaluations(claim_vecs);
        (num_evals, common_idx)
    } else {
        (((num_claims - 1) * num_idx) + 1, None)
    };

    let mut claim_mles = claim_mles;

    if let Some(common_idx) = common_idx {
        pre_fix_mles(&mut claim_mles, &claim_vecs[0], common_idx);
    }

    // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
    let next_evals: Vec<F> = cfg_into_iter!(num_claims..num_evals)
        .map(|idx| {
            // get the challenge l(idx)
            let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                .map(|claim_idx| {
                    let evals: Vec<F> = cfg_into_iter!(claim_vecs)
                        .map(|claim| claim[claim_idx])
                        .collect();
                    evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap()
                })
                .collect();

            let wlx_eval_on_mle = combine_mles_with_aggregate(&claim_mles, &new_chal);
            wlx_eval_on_mle.unwrap()
        })
        .collect();

    // concat this with the first k evaluations from the claims to
    // get num_evals evaluations
    let mut wlx_evals = claimed_vals.to_vec();
    wlx_evals.extend(&next_evals);
    Ok(wlx_evals)
}

/// Performs claim aggregation on the verifier side.
/// * `claims`: a group of claims, all on the same layer (same `to_layer_id`),
///   to be aggregated into one.
/// * `transcript_reader`: is used to retrieve the wlx evaluations and generate
///   challenges.
///
/// # Returns
///
/// If successful, returns a single aggregated claim.
pub fn verifier_aggregate_claims<F: Field>(
    claims: &[Claim<F>],
    transcript_reader: &mut impl VerifierTranscript<F>,
) -> Result<RawClaim<F>> {
    let num_claims = claims.len();
    debug_assert!(num_claims > 0);
    info!("High-level claim aggregation on {num_claims} claims.");

    let claim_preproc_timer = start_timer!(|| "Claim preprocessing".to_string());

    let claim_groups = ClaimGroup::form_claim_groups(claims.to_vec());

    debug!("Grouped claims for aggregation: ");
    for group in &claim_groups {
        debug!("GROUP:");
        for claim in group.get_raw_claims() {
            debug!("{:#?}", claim);
        }
    }

    end_timer!(claim_preproc_timer);
    let intermediate_timer = start_timer!(|| "Intermediate claim aggregation.".to_string());

    let intermediate_claims = claim_groups
        .into_iter()
        .map(|claim_group| claim_group.verifier_aggregate(transcript_reader))
        .collect::<Result<Vec<_>>>()?;

    end_timer!(intermediate_timer);
    let final_timer = start_timer!(|| "Final stage aggregation.".to_string());

    // Finally, aggregate all intermediate claims.
    let intermediate_claim_group = ClaimGroup::new_from_raw_claims(intermediate_claims).unwrap();
    let claim = intermediate_claim_group.verifier_aggregate(transcript_reader)?;

    end_timer!(final_timer);
    Ok(claim)
}
