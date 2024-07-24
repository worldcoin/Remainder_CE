//! A set of helper functions for wlx style claim aggregation

use ark_std::{cfg_into_iter, end_timer, start_timer};
use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use remainder_shared_types::{
    transcript::{ProverTranscript, TranscriptReaderError, VerifierTranscript},
    FieldExt,
};
use tracing::{debug, info};

use crate::{
    claims::{
        wlx_eval::{claim_group::form_claim_groups, get_num_wlx_evaluations, ClaimMle},
        Claim, ClaimAndProof, ClaimError,
    },
    layer::combine_mle_refs::get_og_mle_refs,
    mle::mle_enum::MleEnum,
    prover::GKRError,
};

use super::{claim_group::ClaimGroup, evaluate_at_a_point, YieldWLXEvals};

/// Performs claim aggregation. Can be used by both the prover and the verifier.
/// * `claims`: a group of claims, all residing in the same layer (same
///   `to_layer_id` fields), to be aggregated into one.
/// * `compute_wlx_fn`: closure for computing the wlx evaluations. If
///   `aggregate_claims` is called by the prover, the closure should compute the
///   wlx evaluations, potentially using "smart" aggregation controlled by
///   `ENABLE_REDUCED_WLX_EVALS` which provides tighter bounds on the degree of
///   `W(l(x))`. A prover's `compute_wlx_fn` should never produce an error. If
///   called by the verifier, the closure should return the next wlx evaluations
///   received from the prover. In case claim aggregation requires more
///   evaluations than the ones provided by the prover, the closure should
///   return a `GKRError` which is propagated back to the caller of
///   `aggregate_claims`.
/// * `transcript`: is used to post wlx evaluations and generate challenges.
///
/// If successful, returns a pair containing the aggregated claim without
/// from/to layer ID information and a vector of wlx evaluations. The vector
/// either contains no evaluations (in the trivial case of aggregating a single
/// claim) or contains `k` vectors, the wlx evaluations produced during each of
/// the `k` naive claim aggregations performed.
pub fn prover_aggregate_claims_helper<F: FieldExt>(
    claims: &ClaimGroup<F>,
    layer: &impl YieldWLXEvals<F>,
    transcript_writer: &mut impl ProverTranscript<F>,
) -> Result<ClaimAndProof<F, Vec<Vec<F>>>, GKRError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("High-level claim aggregation on {num_claims} claims.");

    let claim_preproc_timer = start_timer!(|| "Claim preprocessing".to_string());

    let layer_mle_refs = get_og_mle_refs(claims.get_claim_mle_refs());

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
                &layer_mle_refs,
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
        .map(|result| {
            ClaimMle::new_raw(result.claim.get_point().clone(), result.claim.get_result())
        })
        .collect();
    let intermediate_wlx_evals: Vec<Vec<F>> = intermediate_results
        .into_iter()
        .flat_map(|result| result.proof)
        .collect();

    end_timer!(intermediate_timer);
    let final_timer = start_timer!(|| "Final stage aggregation.".to_string());

    // Finally, aggregate all intermediate claims.
    let ClaimAndProof {
        claim,
        proof: wlx_evals_option,
    } = prover_aggregate_claims_in_one_round(
        &ClaimGroup::new(intermediate_claims).unwrap(),
        &layer_mle_refs,
        layer,
        transcript_writer,
    )?;

    // Holds a sequence of relevant wlx evaluations, one for each claim
    // group that is being aggregated.
    let group_wlx_evaluations = [intermediate_wlx_evals, wlx_evals_option].concat();

    end_timer!(final_timer);
    Ok(ClaimAndProof {
        claim,
        proof: group_wlx_evaluations,
    })
}

pub fn verifier_aggregate_claims_helper<F: FieldExt>(
    claims: &ClaimGroup<F>,
    transcript_reader: &mut impl VerifierTranscript<F>,
) -> Result<ClaimAndProof<F, Vec<Vec<F>>>, TranscriptReaderError> {
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
        .map(|result| {
            ClaimMle::new_raw(result.claim.get_point().clone(), result.claim.get_result())
        })
        .collect();
    let intermediate_wlx_evals: Vec<Vec<F>> = intermediate_results
        .into_iter()
        .flat_map(|result| result.proof)
        .collect();

    end_timer!(intermediate_timer);
    let final_timer = start_timer!(|| "Final stage aggregation.".to_string());

    // Finally, aggregate all intermediate claims.
    let ClaimAndProof {
        claim,
        proof: wlx_evals_option,
    } = verifier_aggregate_claims_in_one_round(
        &ClaimGroup::new(intermediate_claims).unwrap(),
        transcript_reader,
    )?;

    // Holds a sequence of relevant wlx evaluations, one for each claim
    // group that is being aggregated.
    let group_wlx_evaluations = [intermediate_wlx_evals, wlx_evals_option].concat();

    end_timer!(final_timer);
    Ok(ClaimAndProof {
        claim,
        proof: group_wlx_evaluations,
    })
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
pub fn compute_aggregated_challenges<F: FieldExt>(
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

/// Low-level analogue of `aggregate_claims` which performs claim aggregation on
/// the claim group `claims` in a single stage without further grouping.
/// * `claims`: the group of claims to be aggregated.
/// * `compute_wlx_fn`: closure for computing the wlx evaluations. If
///   `aggregate_claims_in_one_round` is called by the prover, the closure
///   should compute the wlx evaluations, potentially using "smart" aggregation
///   controlled by `ENABLE_REDUCED_WLX_EVALS` which provides tighter bounds on
///   the degree of `W(l(x))`. A prover's `compute_wlx_fn` should never produce
///   an error. If called by the verifier, the closure should return the wlx
///   evaluations sent by the prover. In case claim aggregation requires more
///   evaluations than the ones provided by the prover, the closure should
///   return a `GKRError` which is propagated back to the caller of
///   `aggregate_claims_in_one_round`.
/// * `transcript`: is used to post wlx evaluations and generate challenges.
///
/// If successful, returns a pair containing the aggregated claim without
/// from/to layer ID information and a vector of wlx evaluations. The vector
/// either contains no evaluations (in the trivial case of aggregating a single
/// claim) or contains a single vector of the wlx evaluations produced during
/// this 1-step claim aggregation.
fn prover_aggregate_claims_in_one_round<F: FieldExt>(
    claims: &ClaimGroup<F>,
    layer_mle_refs: &[MleEnum<F>],
    layer: &impl YieldWLXEvals<F>,
    transcript_writer: &mut impl ProverTranscript<F>,
) -> Result<ClaimAndProof<F, Vec<Vec<F>>>, GKRError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("Low-level claim aggregation on {num_claims} claims.");

    // Do nothing if there is only one claim.
    if num_claims == 1 {
        debug!("Received 1 claim. Doing nothing.");
        // Return the claim but erase any from/to layer info so as not to
        // trigger any checks from claim groups used in claim aggregation.
        let claim = claims.get_claim(0).claim.clone();

        return Ok(ClaimAndProof {
            claim,
            proof: vec![vec![]],
        });
    }

    // Aggregate claims by performing the claim aggregation protocol.
    // First compute V_i(l(x)).

    let wlx_evaluations = {
        let wlx_evals = layer
            .get_wlx_evaluations(
                claims.get_claim_points_matrix(),
                claims.get_results(),
                layer_mle_refs.to_vec(),
                claims.get_num_claims(),
                claims.get_num_vars(),
            )
            .unwrap();
        Ok(wlx_evals)
    }?;
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

    Ok(ClaimAndProof {
        claim: Claim::new(aggregated_challenges, claimed_val),
        proof: vec![relevant_wlx_evaluations],
    })
}

fn verifier_aggregate_claims_in_one_round<F: FieldExt>(
    claims: &ClaimGroup<F>,
    transcript_reader: &mut impl VerifierTranscript<F>,
) -> Result<ClaimAndProof<F, Vec<Vec<F>>>, TranscriptReaderError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("Low-level claim aggregation on {num_claims} claims.");

    // Do nothing if there is only one claim.
    if num_claims == 1 {
        debug!("Received 1 claim. Doing nothing.");
        // Return the claim but erase any from/to layer info so as not to
        // trigger any checks from claim groups used in claim aggregation.

        return Ok(ClaimAndProof {
            claim: claims.get_claim(0).claim.clone(),
            proof: vec![vec![]],
        });
    }

    // Aggregate claims by performing the claim aggregation protocol.
    // First retrieve V_i(l(x)).
    let (num_wlx_evaluations, _) = get_num_wlx_evaluations(claims.get_claim_points_matrix());
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

    Ok(ClaimAndProof {
        claim: Claim::new(aggregated_challenges, claimed_val),
        proof: vec![relevant_wlx_evaluations],
    })
}
