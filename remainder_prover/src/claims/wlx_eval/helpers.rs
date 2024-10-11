//! A set of helper functions for wlx style claim aggregation

use ark_std::{cfg_into_iter, end_timer, start_timer};
use itertools::Itertools;
use remainder_shared_types::{
    transcript::{ProverTranscript, TranscriptReaderError, VerifierTranscript},
    Field,
};
use tracing::{debug, info};

use crate::{
    claims::{
        wlx_eval::{claim_group::form_claim_groups, get_num_wlx_evaluations, ClaimMle},
        Claim, ClaimError,
    },
    layer::{
        combine_mle_refs::get_og_mle_refs,
        regular_layer::claims::CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION,
    },
    mle::dense::DenseMle,
    prover::GKRError,
};

use super::{claim_group::ClaimGroup, evaluate_at_a_point, YieldWLXEvals};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Performs claim aggregation on the prover side.
/// * `claims`: a group of claims, all on the same layer (same `to_layer_id`),
///   to be aggregated into one.
/// * `layer`: typically, the GKR layer this claim group is making claims on,
///    but in general could be anything that yields WLX evalutions.
/// * `output_mles_from_layer`: The compiled bookkeeping tables that result from
///    this layer, in order to aggregate into the layerwise bookkeeping table.
/// * `transcript_writer`: is used to post wlx evaluations and generate
///   challenges.
///
/// # Returns
///
/// If successful, returns a pair containing the aggregated claim without
/// from/to layer ID information and a vector of wlx evaluations. The vector
/// either contains no evaluations (in the trivial case of aggregating a single
/// claim) or contains `k` vectors, the wlx evaluations produced during each of
/// the `k` naive claim aggregations performed.
pub fn prover_aggregate_claims_helper<F: Field>(
    claims: &ClaimGroup<F>,
    layer: &impl YieldWLXEvals<F>,
    output_mles_from_layer: &[DenseMle<F>],
    transcript_writer: &mut impl ProverTranscript<F>,
) -> Result<Claim<F>, GKRError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("High-level claim aggregation on {num_claims} claims.");

    let claim_preproc_timer = start_timer!(|| "Claim preprocessing".to_string());

    let fixed_output_mles = get_og_mle_refs(output_mles_from_layer);

    let claim_groups = form_claim_groups(claims.get_claims().to_vec());

    debug!("Grouped claims for aggregation: ");
    for group in &claim_groups {
        debug!("GROUP: {:#?}", group.get_claims());
    }

    end_timer!(claim_preproc_timer);
    let intermediate_timer = start_timer!(|| "Intermediate claim aggregation.".to_string());

    // TODO(Makis): Parallelize
    let intermediate_results: Result<Vec<_>, _> = claim_groups
        .into_iter()
        .map(|claim_group| {
            prover_aggregate_claims_in_one_round(
                &claim_group,
                &fixed_output_mles,
                layer,
                transcript_writer,
            )
        })
        .collect();
    let intermediate_results = intermediate_results?;

    // TODO(Makis): Parallelize both
    let intermediate_claims = intermediate_results
        .clone()
        .into_iter()
        .map(|result| ClaimMle::new_raw(result.get_point().clone(), result.get_result()))
        .collect();

    end_timer!(intermediate_timer);
    let final_timer = start_timer!(|| "Final stage aggregation.".to_string());

    // Finally, aggregate all intermediate claims.
    let claim = prover_aggregate_claims_in_one_round(
        &ClaimGroup::new(intermediate_claims).unwrap(),
        &fixed_output_mles,
        layer,
        transcript_writer,
    )?;

    end_timer!(final_timer);
    Ok(claim)
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
pub fn verifier_aggregate_claims_helper<F: Field>(
    claims: &ClaimGroup<F>,
    transcript_reader: &mut impl VerifierTranscript<F>,
) -> Result<Claim<F>, TranscriptReaderError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("High-level claim aggregation on {num_claims} claims.");

    let claim_preproc_timer = start_timer!(|| "Claim preprocessing".to_string());

    let claim_groups = form_claim_groups(claims.get_claims().to_vec());

    let _num_claim_groups = claim_groups.len();

    debug!("Grouped claims for aggregation: ");
    for group in &claim_groups {
        debug!("GROUP:");
        for claim in group.get_claims() {
            debug!("{:#?}", claim);
        }
    }

    end_timer!(claim_preproc_timer);
    let intermediate_timer = start_timer!(|| "Intermediate claim aggregation.".to_string());

    // TODO(Makis): Parallelize
    let intermediate_results: Result<Vec<_>, _> = claim_groups
        .into_iter()
        .map(|claim_group| verifier_aggregate_claims_in_one_round(&claim_group, transcript_reader))
        .collect();
    let intermediate_results = intermediate_results?;

    let intermediate_claims = intermediate_results
        .clone()
        .into_iter()
        .map(|result| ClaimMle::new_raw(result.point, result.result))
        .collect();

    end_timer!(intermediate_timer);
    let final_timer = start_timer!(|| "Final stage aggregation.".to_string());

    // Finally, aggregate all intermediate claims.
    let claim = verifier_aggregate_claims_in_one_round(
        &ClaimGroup::new(intermediate_claims).unwrap(),
        transcript_reader,
    )?;

    end_timer!(final_timer);
    Ok(claim)
}

/// Aggregates a sequence of claim into a single point. If `claims` contains `m`
/// points `[u_0, u_1, ..., u_{m-1}]` where each `u_i \in F^n`, this function
/// computes a univariate polynomial vector `l : F -> F^n` such that `l(0) =
/// u_0, l(1) = u_1, ..., l(m-1) = u_{m-1}` using Lagrange interpolation, then
/// evaluates `l` on `r_star` and returns it.
/// # Requires
/// `claims_points` to be non-empty, otherwise a
/// `ClaimError::ClaimAggroError` is returned.
/// # TODO(Makis)
/// Using the ClaimGroup abstraction here is not ideal since we are only
/// operating on the points and not on the results. However, the ClaimGroup API
/// is convenient for accessing columns and makes the implementation more
/// readable. We should consider alternative designs.
pub fn compute_aggregated_challenges<F: Field>(
    claims: &ClaimGroup<F>,
    r_star: F,
) -> Result<Vec<F>, ClaimError> {
    if claims.is_empty() {
        return Err(ClaimError::ClaimAggroError);
    }

    let num_vars = claims.get_num_vars();

    // Compute r = l(r*) by performing Lagrange interpolation on each coordinate
    // using `evaluate_at_a_point`.
    let r: Vec<F> = cfg_into_iter!(0..num_vars)
        .map(|idx| {
            let evals = claims.get_points_column(idx);
            // Interpolate the value l(r*) from the values
            // l(0), l(1), ..., l(m-1) where m = # of claims.
            evaluate_at_a_point(evals, r_star).unwrap()
        })
        .collect();

    Ok(r)
}

/// Low-level analogue of `prover_aggregate_claims` which performs claim
/// aggregation on the claim group `claims` in a single stage without further
/// grouping.
/// * `claims`: the group of claims to be aggregated.
/// * `layer_mles`: the compiled bookkeeping tables from this layer, which
///    when aggregated appropriately with their prefix bits, make up the
///    layerwise bookkeeping table.
/// * `layer`: the layer whose output MLE is being made a claim on. Each of the
///    claims are aggregated into one claim, whose validity is reduced to the
///    validity of a claim in a future layer throught he sumcheck protocol.
/// * `transcript_writer`: is used to post wlx evaluations and generate
///   challenges.
///
/// # Returns
///
/// If successful, returns a single aggregated claim.
fn prover_aggregate_claims_in_one_round<F: Field>(
    claims: &ClaimGroup<F>,
    layer_mles: &[DenseMle<F>],
    layer: &impl YieldWLXEvals<F>,
    transcript_writer: &mut impl ProverTranscript<F>,
) -> Result<Claim<F>, GKRError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("Low-level claim aggregation on {num_claims} claims.");

    // Do nothing if there is only one claim.
    if num_claims == 1 {
        debug!("Received 1 claim. Doing nothing.");
        // Return the claim but erase any from/to layer info so as not to
        // trigger any checks from claim groups used in claim aggregation.
        let claim = claims.get_claim(0).claim.clone();

        return Ok(claim);
    }

    // Aggregate claims by performing the claim aggregation protocol.
    // First compute V_i(l(x)).

    let wlx_evaluations = layer
        .get_wlx_evaluations(
            claims.get_claim_points_matrix(),
            claims.get_results(),
            layer_mles.to_vec(),
            claims.get_num_claims(),
            claims.get_num_vars(),
        )
        .unwrap();
    let relevant_wlx_evaluations = wlx_evaluations[num_claims..].to_vec();

    // Append evaluations to the transcript before sampling a challenge.
    transcript_writer.append_elements(
        "Claim Aggregation Wlx_evaluations",
        &relevant_wlx_evaluations,
    );

    // Next, sample `r^\star` from the transcript.
    let agg_chal = transcript_writer.get_challenge("Challenge for claim aggregation");
    debug!("Aggregate challenge: {:#?}", agg_chal);

    let aggregated_challenges = compute_aggregated_challenges(claims, agg_chal).unwrap();
    let claimed_val = evaluate_at_a_point(&wlx_evaluations, agg_chal).unwrap();

    debug!("Aggregating claims: {:#?}", claims.get_claims());

    debug!(
        "Low level aggregated claim:\n{:#?}",
        ClaimMle::new_raw(aggregated_challenges.clone(), claimed_val)
    );

    let claim = Claim::new(aggregated_challenges, claimed_val);
    Ok(claim)
}

fn verifier_aggregate_claims_in_one_round<F: Field>(
    claims: &ClaimGroup<F>,
    transcript_reader: &mut impl VerifierTranscript<F>,
) -> Result<Claim<F>, TranscriptReaderError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("Low-level claim aggregation on {num_claims} claims.");

    // Do nothing if there is only one claim.
    if num_claims == 1 {
        debug!("Received 1 claim. Doing nothing.");
        // Return the claim but erase any from/to layer info so as not to
        // trigger any checks from claim groups used in claim aggregation.

        let claim = claims.get_claim(0).claim.clone();

        return Ok(claim);
    }

    // Aggregate claims by performing the claim aggregation protocol.
    // First retrieve V_i(l(x)).

    let num_wlx_evaluations = if CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION {
        let (num_wlx_evaluations, _, _) = get_num_wlx_evaluations(claims.get_claim_points_matrix());
        num_wlx_evaluations
    } else {
        ((num_claims - 1) * claims.get_num_vars()) + 1
    };

    let num_relevant_wlx_evaluations = num_wlx_evaluations - num_claims;
    let relevant_wlx_evaluations = transcript_reader.consume_elements(
        "Claim Aggregation Wlx_evaluations",
        num_relevant_wlx_evaluations,
    )?;
    let wlx_evaluations = claims
        .get_results()
        .clone()
        .into_iter()
        .chain(relevant_wlx_evaluations.clone())
        .collect_vec();

    // Next, sample `r^\star` from the transcript.
    let agg_chal = transcript_reader.get_challenge("Challenge for claim aggregation")?;
    debug!("Aggregate challenge: {:#?}", agg_chal);

    let aggregated_challenges = compute_aggregated_challenges(claims, agg_chal).unwrap();
    let claimed_val = evaluate_at_a_point(&wlx_evaluations, agg_chal).unwrap();

    debug!("Aggregating claims: {:#?}", claims.get_claims());

    debug!(
        "Low level aggregated claim:\n{:#?}",
        ClaimMle::new_raw(aggregated_challenges.clone(), claimed_val)
    );

    let claim = Claim::new(aggregated_challenges, claimed_val);
    Ok(claim)
}
