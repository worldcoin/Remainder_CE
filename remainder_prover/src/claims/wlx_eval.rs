#[cfg(test)]
pub mod tests;

mod claim_group;

use remainder_shared_types::transcript::{
    TranscriptReader, TranscriptReaderError, TranscriptSponge, TranscriptWriter,
};
use remainder_shared_types::FieldExt;
use tracing::instrument;

use crate::claims::wlx_eval::claim_group::{form_claim_groups, ClaimGroup};
use crate::layer::combine_mle_refs::get_og_mle_refs;

use crate::mle::mle_enum::MleEnum;

use crate::prover::input_layer::InputLayer;
use crate::prover::GKRError;
use crate::sumcheck::*;

use ark_std::cfg_into_iter;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::layer::LayerId;
use crate::layer::{Layer, LayerError};

use serde::{Deserialize, Serialize};

use std::cmp::max;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

use log::{debug, info};

use ark_std::{end_timer, start_timer};

use super::{Claim, ClaimAggregator, ClaimError, YieldClaim};

///The basic ClaimAggregator
///
/// Keeps tracks of claims using a hashmap with
/// the layerid as the key
///
/// Aggregates claims using univariate polynomial interpolation
///
/// Collects additional information in the `ClaimMle` struct
/// to make computation of evaluations easier, most importantly
/// the 'original_bookkeeping_table'
pub struct WLXAggregator<F: FieldExt, L, LI> {
    claims: HashMap<LayerId, Vec<ClaimMle<F>>>,
    _marker: std::marker::PhantomData<(L, LI)>,
}

impl<
        F: FieldExt,
        L: Layer<F> + YieldWLXEvals<F> + YieldClaim<F, ClaimMle<F>>,
        LI: InputLayer<F> + YieldWLXEvals<F>,
    > ClaimAggregator<F> for WLXAggregator<F, L, LI>
{
    type Claim = ClaimMle<F>;

    type AggregationProof = Vec<Vec<F>>;

    type Layer = L;
    type InputLayer = LI;

    fn prover_aggregate_claims(
        &self,
        layer: &Self::Layer,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<(Claim<F>, Self::AggregationProof), GKRError> {
        let layer_id = layer.id();
        self.prover_aggregate_claims(layer, *layer_id, transcript_writer)
    }

    fn prover_aggregate_claims_input(
        &self,
        layer: &Self::InputLayer,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<(Claim<F>, Self::AggregationProof), GKRError> {
        let layer_id = layer.layer_id();
        self.prover_aggregate_claims(layer, *layer_id, transcript_writer)
    }

    fn verifier_aggregate_claims(
        &self,
        layer_id: LayerId,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<Claim<F>, GKRError> {
        let claims = self
            .get_claims(layer_id)
            .ok_or(GKRError::ErrorWhenVerifyingLayer(
                layer_id,
                LayerError::ClaimError(ClaimError::ClaimAggroError),
            ))?;

        let claim_group = ClaimGroup::new(claims.to_vec()).unwrap();
        debug!("Layer Claim Group for input: {:#?}", claims);

        // --- Add the claimed values to the FS transcript ---
        for claim in claims {
            let claim_point_len = claim.get_point().len();
            let transcript_claim_point = transcript_reader
                .consume_elements(
                    "Claimed challenge coordinates to be aggregated",
                    claim_point_len,
                )
                .map_err(|err| {
                    GKRError::ErrorWhenVerifyingLayer(layer_id, LayerError::TranscriptError(err))
                })?;
            debug_assert_eq!(transcript_claim_point, *claim.get_point());

            let transcript_claim_result = transcript_reader
                .consume_element("Claimed value to be aggregated")
                .map_err(|err| {
                    GKRError::ErrorWhenVerifyingLayer(layer_id, LayerError::TranscriptError(err))
                })?;
            debug_assert_eq!(transcript_claim_result, claim.get_result());
        }

        if claims.len() > 1 {
            let (prev_claim, _) = verifier_aggregate_claims_helper(&claim_group, transcript_reader)
                .map_err(|err| {
                    GKRError::ErrorWhenVerifyingLayer(layer_id, LayerError::TranscriptError(err))
                })?;

            Ok(prev_claim)
        } else {
            Ok(claims[0].get_claim().clone())
        }
    }

    fn add_claims(&mut self, layer: &impl YieldClaim<F, Self::Claim>) -> Result<(), LayerError> {
        let claims = layer.get_claims()?;

        debug!("Ingesting claims: {:#?}", claims);

        for claim in claims {
            let layer_id = claim.get_to_layer_id().unwrap();
            if let Some(claims) = self.claims.get_mut(&layer_id) {
                claims.push(claim);
            } else {
                self.claims.insert(layer_id, vec![claim]);
            }
        }

        Ok(())
    }

    fn get_claims(&self, layer_id: LayerId) -> Option<&[Self::Claim]> {
        self.claims.get(&layer_id).map(|claims| claims.as_slice())
    }

    fn new() -> Self {
        Self {
            claims: HashMap::new(),
            _marker: PhantomData,
        }
    }
}

impl<
        F: FieldExt,
        L: Layer<F> + YieldWLXEvals<F> + YieldClaim<F, ClaimMle<F>>,
        LI: InputLayer<F> + YieldWLXEvals<F>,
    > WLXAggregator<F, L, LI>
{
    fn prover_aggregate_claims(
        &self,
        layer: &impl YieldWLXEvals<F>,
        layer_id: LayerId,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<(Claim<F>, <Self as ClaimAggregator<F>>::AggregationProof), GKRError> {
        let claims = self
            .get_claims(layer_id)
            .ok_or(GKRError::ErrorWhenVerifyingLayer(
                layer_id,
                LayerError::ClaimError(ClaimError::ClaimAggroError),
            ))?;
        let claim_group = ClaimGroup::new(claims.to_vec()).unwrap();
        debug!("Found Layer claims:\n{:#?}", claims);

        // --- Add the claimed values to the FS transcript ---
        for claim in claims {
            transcript_writer.append_elements("Claimed bits to be aggregated", claim.get_point());
            transcript_writer.append("Claimed value to be aggregated", claim.get_result());
        }

        prover_aggregate_claims_helper(&claim_group, layer, transcript_writer)
    }
}

///The trait that layers must implement to be compatible with the WLXEval based claim aggregator
pub trait YieldWLXEvals<F: FieldExt> {
    ///Get W(l(x)) evaluations
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, crate::claims::ClaimError>;
}

/// A claim that can optionally maintain additional source/destination layer information through
/// `from_layer_id` and `to_layer_id`. This information can be used to speed up
/// claim aggregation.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct ClaimMle<F: FieldExt> {
    ///The underlying raw Claim
    claim: Claim<F>,
    /// The layer ID of the layer that produced this claim (if present); origin
    /// layer.
    pub from_layer_id: Option<LayerId>,
    /// The layer ID of the layer containing the MLE this claim refers to (if
    /// present); destination layer.
    pub to_layer_id: Option<LayerId>,
    /// the mle ref associated with the claim
    pub mle_ref: Option<MleEnum<F>>,
}

impl<F: FieldExt> ClaimMle<F> {
    /// Generate new raw claim without any origin/destination information.
    pub fn new_raw(point: Vec<F>, result: F) -> Self {
        Self {
            claim: Claim { point, result },
            from_layer_id: None,
            to_layer_id: None,
            mle_ref: None,
        }
    }

    /// Generate new claim, potentially with origin/destination information.
    pub fn new(
        point: Vec<F>,
        result: F,
        from_layer_id: Option<LayerId>,
        to_layer_id: Option<LayerId>,
        mle_ref: Option<MleEnum<F>>,
    ) -> Self {
        Self {
            claim: Claim { point, result },
            from_layer_id,
            to_layer_id,
            mle_ref,
        }
    }

    /// Returns the length of the `point` vector.
    pub fn get_num_vars(&self) -> usize {
        self.claim.point.len()
    }

    /// Returns the point vector in F^n.
    pub fn get_point(&self) -> &Vec<F> {
        &self.claim.point
    }

    /// Returns the expected result.
    pub fn get_result(&self) -> F {
        self.claim.result
    }

    /// Returns the source Layer ID.
    pub fn get_from_layer_id(&self) -> Option<LayerId> {
        self.from_layer_id
    }

    /// Returns the destination Layer ID.
    pub fn get_to_layer_id(&self) -> Option<LayerId> {
        self.to_layer_id
    }

    /// Returns the underlying `Claim`
    pub fn get_claim(&self) -> &Claim<F> {
        &self.claim
    }
}

impl<F: fmt::Debug + FieldExt> fmt::Debug for ClaimMle<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Claim")
            .field("point", &self.claim.point)
            .field("result", &self.claim.result)
            .field("from_layer_id", &self.from_layer_id)
            .field("to_layer_id", &self.to_layer_id)
            .finish()
    }
}

// ---- Interface: Code outside `claims.rs` should be calling ----
// ---- `prover_aggregate_claims` and `verifier_aggregate_claims` to perform
// ---- claim aggregation.

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
/// If successful, returns a pair containing the aggregated claim without
/// from/to layer ID information and a vector of wlx evaluations. The vector
/// either contains no evaluations (in the trivial case of aggregating a single
/// claim) or contains `k` vectors, the wlx evaluations produced during each of
/// the `k` naive claim aggregations performed.
/// TODO(Makis): Refactor this file to better expose the interface vs
/// implementation.
fn prover_aggregate_claims_helper<F: FieldExt, Tr: TranscriptSponge<F>>(
    claims: &ClaimGroup<F>,
    layer: &impl YieldWLXEvals<F>,
    transcript_writer: &mut TranscriptWriter<F, Tr>,
) -> Result<(Claim<F>, Vec<Vec<F>>), GKRError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("High-level claim aggregation on {num_claims} claims.");

    let claim_preproc_timer = start_timer!(|| "Claim preprocessing".to_string());

    let layer_mle_refs = get_og_mle_refs(claims.get_claim_mle_refs());

    let claim_groups = form_claim_groups(claims.get_claims().to_vec());

    let num_claim_groups = claim_groups.len();

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
    let intermediate_results: Result<Vec<(ClaimMle<F>, Vec<Vec<F>>)>, GKRError> = claim_groups
        .into_iter()
        .enumerate()
        .map(|(idx, claim_group)| {
            prover_aggregate_claims_in_one_round(
                &claim_group,
                &layer_mle_refs,
                layer,
                idx,
                transcript_writer,
            )
        })
        .collect();
    let intermediate_results = intermediate_results?;

    // TODO(Makis): Parallelize both
    let intermediate_claims = intermediate_results
        .clone()
        .into_iter()
        .map(|result| result.0)
        .collect();
    let intermediate_wlx_evals: Vec<Vec<F>> = intermediate_results
        .into_iter()
        .flat_map(|result| result.1)
        .collect();

    end_timer!(intermediate_timer);
    let final_timer = start_timer!(|| "Final stage aggregation.".to_string());

    // Finally, aggregate all intermediate claims.
    let (claim, wlx_evals_option) = prover_aggregate_claims_in_one_round(
        &ClaimGroup::new(intermediate_claims).unwrap(),
        &layer_mle_refs,
        layer,
        num_claim_groups, // Should be the final prover-supplied V_i(l(x)) evaluations
        transcript_writer,
    )?;

    // Holds a sequence of relevant wlx evaluations, one for each claim
    // group that is being aggregated.
    let group_wlx_evaluations = [intermediate_wlx_evals, wlx_evals_option].concat();

    end_timer!(final_timer);
    Ok((claim.claim, group_wlx_evaluations))
}

fn verifier_aggregate_claims_helper<F: FieldExt, Tr: TranscriptSponge<F>>(
    claims: &ClaimGroup<F>,
    transcript_reader: &mut TranscriptReader<F, Tr>,
) -> Result<(Claim<F>, Vec<Vec<F>>), TranscriptReaderError> {
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
    let intermediate_results: Result<Vec<(ClaimMle<F>, Vec<Vec<F>>)>, _> = claim_groups
        .into_iter()
        .enumerate()
        .map(|(_idx, claim_group)| {
            verifier_aggregate_claims_in_one_round(&claim_group, transcript_reader)
        })
        .collect();
    let intermediate_results = intermediate_results?;

    // TODO(Makis): Parallelize both
    let intermediate_claims = intermediate_results
        .clone()
        .into_iter()
        .map(|result| result.0)
        .collect();
    let intermediate_wlx_evals: Vec<Vec<F>> = intermediate_results
        .into_iter()
        .flat_map(|result| result.1)
        .collect();

    end_timer!(intermediate_timer);
    let final_timer = start_timer!(|| "Final stage aggregation.".to_string());

    // Finally, aggregate all intermediate claims.
    let (claim, wlx_evals_option) = verifier_aggregate_claims_in_one_round(
        &ClaimGroup::new(intermediate_claims).unwrap(),
        transcript_reader,
    )?;

    // Holds a sequence of relevant wlx evaluations, one for each claim
    // group that is being aggregated.
    let group_wlx_evaluations = [intermediate_wlx_evals, wlx_evals_option].concat();

    end_timer!(final_timer);
    Ok((claim.claim, group_wlx_evaluations))
}

// ---- Implementation: The following functions are used by ----
// ---- the interface functions and/or testing functions. ----

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
#[instrument(level = "trace", err)]
#[instrument(level = "debug", skip_all, err)]
fn compute_aggregated_challenges<F: FieldExt>(
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
/// If successful, returns a pair containing the aggregated claim without
/// from/to layer ID information and a vector of wlx evaluations. The vector
/// either contains no evaluations (in the trivial case of aggregating a single
/// claim) or contains a single vector of the wlx evaluations produced during
/// this 1-step claim aggregation.
fn prover_aggregate_claims_in_one_round<F: FieldExt, Tr: TranscriptSponge<F>>(
    claims: &ClaimGroup<F>,
    layer_mle_refs: &Vec<MleEnum<F>>,
    layer: &impl YieldWLXEvals<F>,
    prover_supplied_wlx_group_idx: usize,
    transcript_writer: &mut TranscriptWriter<F, Tr>,
) -> Result<(ClaimMle<F>, Vec<Vec<F>>), GKRError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("Low-level claim aggregation on {num_claims} claims.");

    // Do nothing if there is only one claim.
    if num_claims == 1 {
        debug!("Received 1 claim. Doing nothing.");
        // Return the claim but erase any from/to layer info so as not to
        // trigger any checks from claim groups used in claim aggregation.
        let claim = ClaimMle {
            from_layer_id: None,
            to_layer_id: None,
            ..claims.get_claim(0).clone()
        };

        return Ok((claim, vec![vec![]]));
    }

    // Aggregate claims by performing the claim aggregation protocol.
    // First compute V_i(l(x)).

    let wlx_evaluations = {
        let wlx_evals = layer
            .get_wlx_evaluations(
                claims.get_claim_points_matrix(),
                claims.get_results(),
                layer_mle_refs.clone(),
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

    debug!("Aggregating claims: ");
    for c in claims.get_claims() {
        debug!("   {:#?}", c);
    }

    debug!(
        "Low level aggregated claim:\n{:#?}",
        ClaimMle::new_raw(aggregated_challenges.clone(), claimed_val)
    );

    Ok((
        ClaimMle::new_raw(aggregated_challenges, claimed_val),
        vec![relevant_wlx_evaluations],
    ))
}

fn verifier_aggregate_claims_in_one_round<F: FieldExt, Tr: TranscriptSponge<F>>(
    claims: &ClaimGroup<F>,
    transcript_reader: &mut TranscriptReader<F, Tr>,
) -> Result<(ClaimMle<F>, Vec<Vec<F>>), TranscriptReaderError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("Low-level claim aggregation on {num_claims} claims.");

    // Do nothing if there is only one claim.
    if num_claims == 1 {
        debug!("Received 1 claim. Doing nothing.");
        // Return the claim but erase any from/to layer info so as not to
        // trigger any checks from claim groups used in claim aggregation.
        let claim = ClaimMle {
            from_layer_id: None,
            to_layer_id: None,
            ..claims.get_claim(0).clone()
        };

        return Ok((claim, vec![vec![]]));
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
        .chain(relevant_wlx_evaluations.clone().into_iter())
        .collect();

    // Next, sample `r^\star` from the transcript.
    let agg_chal = transcript_reader.get_challenge("Challenge for claim aggregation")?;
    debug!("Aggregate challenge: {:#?}", agg_chal);

    let aggregated_challenges = compute_aggregated_challenges(claims, agg_chal).unwrap();
    let claimed_val = evaluate_at_a_point(&wlx_evaluations, agg_chal).unwrap();

    debug!("Aggregating claims: ");
    for c in claims.get_claims() {
        debug!("   {:#?}", c);
    }

    debug!(
        "Low level aggregated claim:\n{:#?}",
        ClaimMle::new_raw(aggregated_challenges.clone(), claimed_val)
    );

    Ok((
        ClaimMle::new_raw(aggregated_challenges, claimed_val),
        vec![relevant_wlx_evaluations],
    ))
}

/// Returns an upper bound on the number of evaluations needed to represent the
/// polynomial `P(x) = W(l(x))` where `W : F^n -> F` is a multilinear polynomial
/// on `n` variables and `l : F -> F^n` is such that:
///  * `l(0) = claim_vecs[0]`,
///  *  `l(1) = `claim_vecs[1]`,
///  *   ...,
///  *  `l(m-1) = `claim_vecs[m-1]`.
/// It is guaranteed that the returned value is at least `num_claims =
/// claim_vecs.len()`.
/// # Panics
///  if `claim_vecs` is empty.
pub fn get_num_wlx_evaluations<F: FieldExt>(
    claim_vecs: &Vec<Vec<F>>,
) -> (usize, Option<Vec<usize>>) {
    let num_claims = claim_vecs.len();
    let num_vars = claim_vecs[0].len();

    debug!("Smart num_evals");
    let mut num_constant_columns = num_vars as i64;
    let mut common_idx = vec![];
    for j in 0..num_vars {
        let mut degree_reduced = true;
        for i in 1..num_claims {
            if claim_vecs[i][j] != claim_vecs[i - 1][j] {
                num_constant_columns -= 1;
                degree_reduced = false;
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
    (max(num_evals, num_claims), Some(common_idx))
}
