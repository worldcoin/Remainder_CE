use ark_crypto_primitives::crh::sha256::digest::typenum::Or;
use itertools::Either;
use remainder_shared_types::transcript::{TranscriptReader, TranscriptReaderError, TranscriptSponge, TranscriptWriter};
use remainder_shared_types::FieldExt;
use tracing::instrument;

use crate::layer::combine_mle_refs::get_og_mle_refs;
use crate::mle::dense::{DenseMle, DenseMleRef};
use crate::mle::mle_enum::MleEnum;
use crate::mle::zero::ZeroMleRef;
use crate::mle::{MleIndex, MleRef};
use crate::prover::input_layer::enum_input_layer::InputLayerEnum;
use crate::prover::input_layer::InputLayer;
use crate::prover::{GKRError, ENABLE_OPTIMIZATION};
use crate::sumcheck::*;

use ark_std::{cfg_into_iter, cfg_iter};

use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use thiserror::Error;

use crate::layer::combine_mle_refs::CombineMleRefError;
use crate::layer::Layer;
use crate::layer::LayerId;

use serde::{Deserialize, Serialize};

use core::cmp::Ordering;
use std::cmp::max;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

use log::{debug, info, warn};

use itertools::Itertools;

use ark_std::{end_timer, start_timer};

use super::{Claim, ClaimAggregator, ClaimError, YieldClaim};

pub const ENABLE_REDUCED_WLX_EVALS: bool = true;
pub const ENABLE_CLAIM_GROUPING: bool = true;
pub const ENABLE_CLAIM_DEDUPLICATION: bool = true;
/// Controls raw mle eval vs expression eval
pub const ENABLE_RAW_MLE: bool = true;
/// Controls pre-fixing variables for shared columns, should only be true if above is true
pub const ENABLE_PRE_FIX: bool = true;

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

impl<F: FieldExt, L: YieldWLXEvals<F> + YieldClaim<F, ClaimMle<F>>, LI: YieldWLXEvals<F>> ClaimAggregator<F> for WLXAggregator<F, L, LI> {
    type Claim = ClaimMle<F>;

    type AggregationProof = Vec<Vec<F>>;

    type Layer = L;
    type InputLayer = LI;

    fn prover_aggregate_claims(claims: &[Self::Claim], layer: &Self::Layer, transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>) -> Result<(Claim<F>, Self::AggregationProof), GKRError> {
        let claim_group = ClaimGroup::new(claims.to_vec()).unwrap();
        debug!("Found Layer claims:\n{:#?}", claims);

        // --- Add the claimed values to the FS transcript ---
        for claim in claims {
            transcript_writer
                .append_elements("Claimed bits to be aggregated", claim.get_point());
            transcript_writer.append("Claimed value to be aggregated", claim.get_result());
        }

        prover_aggregate_claims(&claim_group, &mut |claims, _, layer_mle_refs| {
            let wlx_evals = layer
                .get_wlx_evaluations(
                    claims.get_claim_points_matrix(),
                    claims.get_results(),
                    layer_mle_refs.unwrap().clone(),
                    claims.get_num_claims(),
                    claims.get_num_vars(),
                )
                .unwrap();
            Ok(wlx_evals)
        }, transcript_writer)
    }

    fn prover_aggregate_claims_input(claims: &[Self::Claim], layer: &Self::InputLayer, transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>) -> Result<(Claim<F>, Self::AggregationProof), GKRError> {
        let claim_group = ClaimGroup::new(claims.to_vec()).unwrap();
        debug!("Found Layer claims:\n{:#?}", claims);

        // --- Add the claimed values to the FS transcript ---
        for claim in claims {
            transcript_writer
                .append_elements("Claimed bits to be aggregated", claim.get_point());
            transcript_writer.append("Claimed value to be aggregated", claim.get_result());
        }

        prover_aggregate_claims(&claim_group, &mut |claims, _, layer_mle_refs| {
            let wlx_evals = layer
                .get_wlx_evaluations(
                    claims.get_claim_points_matrix(),
                    claims.get_results(),
                    layer_mle_refs.unwrap().clone(),
                    claims.get_num_claims(),
                    claims.get_num_vars(),
                )
                .unwrap();
            Ok(wlx_evals)
        }, transcript_writer)
    }

    fn verifier_aggregate_claims(claims: &[Self::Claim], transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>) -> Result<Claim<F>, TranscriptReaderError> {
        let claim_group = ClaimGroup::new(claims.to_vec()).unwrap();
            debug!("Layer Claim Group for input: {:#?}", claims);

        // --- Add the claimed values to the FS transcript ---
        for claim in claims {
            let claim_point_len = claim.get_point().len();
            let transcript_claim_point = transcript_reader
                .consume_elements(
                    "Claimed challenge coordinates to be aggregated",
                    claim_point_len,
                )?;
            debug_assert_eq!(transcript_claim_point, *claim.get_point());

            let transcript_claim_result = transcript_reader
                .consume_element("Claimed value to be aggregated")?;
            debug_assert_eq!(transcript_claim_result, claim.get_result());
        }

        if claims.len() > 1 {
            let (prev_claim, _) =
                verifier_aggregate_claims(&claim_group, transcript_reader)?;

            Ok(prev_claim)
        } else {
            Ok(claims[0].get_claim().clone())
        }
    }
    
    fn add_claims(&mut self, claims: Vec<Self::Claim>) {
        for claim in claims {
            let layer_id = claim.get_to_layer_id().unwrap();
            if let Some(claims) = self.claims.get_mut(&layer_id) {
                claims.push(claim);
            } else {
                self.claims.insert(layer_id, vec![claim]);
            }
        }
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
            claim: Claim {
                point,
                result,
            },
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
            claim: Claim {
                point,
                result,
            },
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
        self.claim.result.clone()
    }

    /// Returns the source Layer ID.
    pub fn get_from_layer_id(&self) -> Option<LayerId> {
        self.from_layer_id
    }

    /// Returns the destination Layer ID.
    pub fn get_to_layer_id(&self) -> Option<LayerId> {
        self.to_layer_id
    }

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

/// A collection of claims for the same layer with an API for accessing the
/// matrix of claim points in a multitude of ways (row-wise or column-wise).
/// This struct is useful for claim aggregation.
/// Invariant: All claims have to agree on `to_layer_id` and on the number of
/// variables.
#[derive(Clone, Debug)]
pub struct ClaimGroup<F: FieldExt> {
    /// A vector of claims in F^n.
    pub claims: Vec<ClaimMle<F>>,
    /// TODO(Makis): The following fields are all redundant. We should remove
    /// them on the next refactoring and consider using iterators instead!
    /// -------------- REFACTOR NEEDED ---------------------
    /// The common layer ID of all claims stored in this group.
    src_layer_id: Option<LayerId>,
    /// A 2D matrix with the claim's points as its rows.
    claim_points_matrix: Vec<Vec<F>>,
    /// The points in `claims` is effectively a matrix of elements in F. We also
    /// store the transpose of this matrix for convenient access.
    claim_points_transpose: Vec<Vec<F>>,
    /// A vector of `self.get_num_claims()` elements. For each claim i,
    /// `result_vector[i]` stores the expected result of the i-th claim.
    result_vector: Vec<F>,
}

impl<F: Copy + Clone + std::fmt::Debug + FieldExt> ClaimGroup<F> {
    /// Builds a ClaimGroup<F> struct from a vector of claims. Also populates
    /// all the redundant fields for easy access to rows/columns.
    /// If the claims do not all agree on the number of variables, a
    /// `ClaimError::NumVarsMismatch` is returned.
    /// If the claims do not all agree on the `to_layer_id`, a
    /// `ClaimError::LayerIdMismatch` is returned.
    pub fn new(claims: Vec<ClaimMle<F>>) -> Result<Self, ClaimError> {
        let num_claims = claims.len();
        if num_claims == 0 {
            return Ok(Self {
                claims: vec![],
                src_layer_id: None,
                claim_points_matrix: vec![],
                claim_points_transpose: vec![],
                result_vector: vec![],
            });
        }
        let num_vars = claims[0].get_num_vars();

        // Check all claims match on the number of variables.
        if !claims
            .clone()
            .into_iter()
            .all(|claim| claim.get_num_vars() == num_vars)
        {
            return Err(ClaimError::NumVarsMismatch);
        }

        // Check all claims match on the `to_layer_id` field.
        let layer_id = claims[0].get_to_layer_id();
        if !claims
            .clone()
            .into_iter()
            .all(|claim| claim.get_to_layer_id() == layer_id)
        {
            return Err(ClaimError::LayerIdMismatch);
        }

        // Populate the points_matrix
        let points_matrix: Vec<_> = claims
            .clone()
            .into_iter()
            .map(|claim| -> Vec<F> { claim.get_point().clone() })
            .collect();

        // Compute the claim points transpose.
        let claim_points_transpose: Vec<Vec<F>> = (0..num_vars)
            .map(|j| (0..num_claims).map(|i| claims[i].get_point()[j]).collect())
            .collect();

        // Compute the result vector.
        let result_vector: Vec<F> = (0..num_claims).map(|i| claims[i].get_result()).collect();

        Ok(Self {
            claims,
            src_layer_id: layer_id,
            claim_points_matrix: points_matrix,
            claim_points_transpose,
            result_vector,
        })
    }

    /// Returns the number of claims stored in this group.
    pub fn get_num_claims(&self) -> usize {
        self.claims.len()
    }

    /// Returns true if the group contains no claims.
    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    /// Returns the number of indices of the claims stored.
    /// Panics if no claims present.
    pub fn get_num_vars(&self) -> usize {
        self.claims[0].get_num_vars()
    }

    /// Returns the common destination layer ID of all the claims stored in this
    /// group.
    pub fn get_layer_id(&self) -> Option<LayerId> {
        self.src_layer_id
    }

    /// Returns the i-th result.
    /// # Panics
    /// If i is not in the range 0 <= i < `self.get_num_claims()`.
    pub fn get_result(&self, i: usize) -> F {
        self.claims[i].get_result()
    }

    /// Returns reference to the i-th point vector.
    /// # Panics
    /// When i is not in the range 0 <= i < `self.get_num_claims()`.
    pub fn get_challenge(&self, i: usize) -> &Vec<F> {
        &self.claims[i].get_point()
    }

    /// Returns a reference to a vector of `self.get_num_claims()` elements, the
    /// j-th entry of which is the i-th coordinate of the j-th claim's point. In
    /// other words, it returns the i-th column of the matrix containing the
    /// claim points as its rows.
    /// # Panics
    /// When i is not in the range: 0 <= i < `self.get_num_vars()`.
    pub fn get_points_column(&self, i: usize) -> &Vec<F> {
        &self.claim_points_transpose[i]
    }

    /// Returns a reference to an "m x n" matrix where n = `self.get_num_vars()`
    /// and m = `self.get_num_claims()` with the claim points as its rows.
    pub fn get_claim_points_matrix(&self) -> &Vec<Vec<F>> {
        &self.claim_points_matrix
    }

    /// Returns a reference to an "n x m" matrix where n = `self.get_num_vars()`
    /// and m = `self.get_num_claims()`, containing the claim points as its
    /// columns.
    pub fn get_claim_points_transpose(&self) -> &Vec<Vec<F>> {
        &self.claim_points_transpose
    }

    /// Returns a reference to a vector with m = `self.get_num_claims()`
    /// elements containing the results of all claims.
    pub fn get_results(&self) -> &Vec<F> {
        &self.result_vector
    }

    /// Returns a reference to the i-th claim.
    pub fn get_claim(&self, i: usize) -> &ClaimMle<F> {
        &self.claims[i]
    }

    /// Returns a reference to a vector of claims contained in this group.
    pub fn get_claim_vector(&self) -> &Vec<ClaimMle<F>> {
        &self.claims
    }

    /// Temporary hacky fix -- claims store their respective (unmodified)
    /// MleRefs instead of the layer itself. This simply looks at the list
    /// of all claims within a single ClaimGroup and grabs the associated
    /// MleRefs from there.
    pub fn get_claim_mle_refs(&self) -> Vec<MleEnum<F>> {
        self.claims
            .clone()
            .into_iter()
            .map(|claim| claim.mle_ref.unwrap())
            .collect_vec()
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
pub fn prover_aggregate_claims<F: FieldExt, Tr: TranscriptSponge<F>>(
    claims: &ClaimGroup<F>,
    compute_wlx_fn: &mut impl FnMut(
        &ClaimGroup<F>,
        usize,
        Option<&Vec<MleEnum<F>>>,
    ) -> Result<Vec<F>, GKRError>,
    transcript_writer: &mut TranscriptWriter<F, Tr>,
) -> Result<(Claim<F>, Vec<Vec<F>>), GKRError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("High-level claim aggregation on {num_claims} claims.");

    let claim_preproc_timer = start_timer!(|| format!("Claim preprocessing"));

    let layer_mle_refs = get_og_mle_refs(claims.get_claim_mle_refs());

    // Holds a sequence of relevant wlx evaluations, one for each claim
    // group that is being aggregated.
    let mut group_wlx_evaluations: Vec<Vec<F>> = vec![];

    let claims = preprocess_claims(claims.get_claim_vector().clone());
    let claim_groups = form_claim_groups(&claims);

    let num_claim_groups = claim_groups.len();

    debug!("Grouped claims for aggregation: ");
    for group in &claim_groups {
        debug!("GROUP:");
        for claim in group.get_claim_vector() {
            debug!("{:#?}", claim);
        }
    }

    end_timer!(claim_preproc_timer);
    let intermediate_timer = start_timer!(|| format!("Intermediate claim aggregation."));

    // TODO(Makis): Parallelize
    let intermediate_results: Result<Vec<(ClaimMle<F>, Vec<Vec<F>>)>, GKRError> = claim_groups
        .into_iter()
        .enumerate()
        .map(|(idx, claim_group)| {
            prover_aggregate_claims_in_one_round(
                &claim_group,
                &layer_mle_refs,
                compute_wlx_fn,
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
    let mut intermediate_wlx_evals: Vec<Vec<F>> = intermediate_results
        .into_iter()
        .map(|result| result.1)
        .flatten()
        .collect();

    // Gather all wlx evaluations into one place.
    group_wlx_evaluations.append(&mut intermediate_wlx_evals);

    end_timer!(intermediate_timer);
    let final_timer = start_timer!(|| format!("Final stage aggregation."));

    // Finally, aggregate all intermediate claims.
    let (claim, mut wlx_evals_option) = prover_aggregate_claims_in_one_round(
        &ClaimGroup::new(intermediate_claims).unwrap(),
        &layer_mle_refs,
        compute_wlx_fn,
        num_claim_groups, // Should be the final prover-supplied V_i(l(x)) evaluations
        transcript_writer,
    )?;

    group_wlx_evaluations.append(&mut wlx_evals_option);

    end_timer!(final_timer);
    Ok((claim.claim, group_wlx_evaluations))
}

pub fn verifier_aggregate_claims<F: FieldExt, Tr: TranscriptSponge<F>>(
    claims: &ClaimGroup<F>,
    transcript_reader: &mut TranscriptReader<F, Tr>,
) -> Result<(Claim<F>, Vec<Vec<F>>), TranscriptReaderError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("High-level claim aggregation on {num_claims} claims.");

    let claim_preproc_timer = start_timer!(|| format!("Claim preprocessing"));

    let layer_mle_refs = get_og_mle_refs(claims.get_claim_mle_refs());

    // Holds a sequence of relevant wlx evaluations, one for each claim
    // group that is being aggregated.
    let mut group_wlx_evaluations: Vec<Vec<F>> = vec![];

    let claims = preprocess_claims(claims.get_claim_vector().clone());
    let claim_groups = form_claim_groups(&claims);

    let num_claim_groups = claim_groups.len();

    debug!("Grouped claims for aggregation: ");
    for group in &claim_groups {
        debug!("GROUP:");
        for claim in group.get_claim_vector() {
            debug!("{:#?}", claim);
        }
    }

    end_timer!(claim_preproc_timer);
    let intermediate_timer = start_timer!(|| format!("Intermediate claim aggregation."));

    // TODO(Makis): Parallelize
    let intermediate_results: Result<Vec<(ClaimMle<F>, Vec<Vec<F>>)>, _> = claim_groups
        .into_iter()
        .enumerate()
        .map(|(idx, claim_group)| {
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
    let mut intermediate_wlx_evals: Vec<Vec<F>> = intermediate_results
        .into_iter()
        .map(|result| result.1)
        .flatten()
        .collect();

    // Gather all wlx evaluations into one place.
    group_wlx_evaluations.append(&mut intermediate_wlx_evals);

    end_timer!(intermediate_timer);
    let final_timer = start_timer!(|| format!("Final stage aggregation."));

    // Finally, aggregate all intermediate claims.
    let (claim, mut wlx_evals_option) = verifier_aggregate_claims_in_one_round(
        &ClaimGroup::new(intermediate_claims).unwrap(),
        transcript_reader,
    )?;

    group_wlx_evaluations.append(&mut wlx_evals_option);

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

/// Part of the claim aggregation process. It returns a vector of evaluations
/// [W(l(0)), W(l(1)), ..., W(l(k))] where W : F^n -> F is the layer MLE stored
/// in `layer`, l : F -> F^n is the interpolated polynomial on the claim points
/// in `claims` (see `compute_aggregated_challenges()` for a definition) and `k`
/// is *at least* the degree of the univariate polynomial W(l(x)) : F -> F.
/// A ClaimError::ClaimAggroError is returned if `claims` is empty.
/// TODO(Makis): Rename this function avoiding the term "wlx".
/// TODO(Makis): Due to the `ClaimGroup<F>` refactor, this function is now just
/// wrapper around `Layer<F>::get_wlx_evaluations()`. Consider removing it.
/*
pub(crate) fn compute_claim_wlx<F: FieldExt>(
    claims: &ClaimGroup<F>,
    layer: &impl Layer<F>,
) -> Result<Vec<F>, ClaimError> {
    if claims.is_empty() {
        return Err(ClaimError::ClaimAggroError);
    }
    let num_claims = claims.get_num_claims();
    let num_vars = claims.get_num_vars();
    let claim_mle_refs = claims.get_claim_mle_refs();

    let results = claims.get_results();
    let points_matrix = claims.get_claim_points_matrix();

    debug_assert_eq!(points_matrix.len(), num_claims);
    debug_assert_eq!(points_matrix[0].len(), num_vars);

    // Get the evals [W(l(0)), W(l(1)), ..., W(l(degree_upper_bound)) ]
    let wlx = layer.get_wlx_evaluations(points_matrix, results, claim_mle_refs, num_claims, num_vars)?;

    Ok(wlx)
}
*/

/// Sorts claims by `from_layer_id` to prepare them for grouping. Also performs
/// claim de-duplication if the `ENABLE_CLAIM_DEDUPLICATION` flag it set.
pub fn preprocess_claims<F: FieldExt>(mut claims: Vec<ClaimMle<F>>) -> Vec<ClaimMle<F>> {
    if !ENABLE_CLAIM_DEDUPLICATION && !ENABLE_CLAIM_GROUPING {
        // There is no need to sort the claims if no optimizations are enabled.
        return claims;
    }

    // Sort claims on the `from_layer_id` field.
    // A trivial total order is imposed which includes `None` values.
    claims.sort_by(|claim1, claim2| {
        match (claim1.get_from_layer_id(), claim2.get_from_layer_id()) {
            (Some(id1), Some(id2)) => match id1.cmp(&id2) {
                // Ties are broken by point value.
                // Ordering::Equal => claim1.get_point().cmp(claim2.get_point()),
                ordering => ordering,
            },
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            // (None, None) => claim1.get_point().cmp(claim2.get_point()),
            (None, None) => Ordering::Equal,
        }
    });

    // Perform claim de-duplication if flag is set.
    if !ENABLE_CLAIM_DEDUPLICATION {
        claims
    } else {
        info!("Performing claim de-duplication.");
        debug!("Num claims BEFORE dedup: {}", claims.len());
        // Remove duplicates.
        // TODO(Makis): Parallelize.
        let claims = claims
            .into_iter()
            .unique_by(|c| c.get_point().clone())
            .collect_vec();
        debug!("\nNum claims AFTER dedup: {}", claims.len());
        claims
    }
}

/// Assign `claims` into groups to be aggregated together.  The naive version of
/// this function assigns all claims to a single group.  On the other hand, if
/// the `ENABLE_CLAIM_GROUPING` flag is set, it groups the claims based on the
/// `from_layer_id` field of each claim.
/// # Requires
/// All claims with the same `from_layer_id` should appear consecutively in the
/// `claims` vector. For example, `claims` can be sorted by `from_layer_id`.
pub fn form_claim_groups<F: FieldExt>(claims: &[ClaimMle<F>]) -> Vec<ClaimGroup<F>> {
    info!("Forming claim group...");

    if !ENABLE_CLAIM_GROUPING {
        // Naive claim grouping: everything goes into the same group.
        vec![ClaimGroup::new(claims.to_vec()).unwrap()]
    } else {
        let num_claims = claims.len();
        let mut claim_group_vec: Vec<ClaimGroup<F>> = vec![];

        // Identify runs of claims with the same `from_layer_id` field.
        let mut start_index = 0;
        for idx in 1..num_claims {
            if claims[idx].get_from_layer_id() != claims[idx - 1].get_from_layer_id() {
                let end_index = idx;
                claim_group_vec
                    .push(ClaimGroup::new(claims[start_index..end_index].to_vec()).unwrap());
                start_index = idx;
            }
        }

        // Process the last group.
        let end_index = num_claims;
        claim_group_vec.push(ClaimGroup::new(claims[start_index..end_index].to_vec()).unwrap());

        claim_group_vec
    }
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
pub fn prover_aggregate_claims_in_one_round<F: FieldExt, Tr: TranscriptSponge<F>>(
    claims: &ClaimGroup<F>,
    layer_mle_refs: &Vec<MleEnum<F>>,
    compute_wlx_fn: &mut impl FnMut(
        &ClaimGroup<F>,
        usize,
        Option<&Vec<MleEnum<F>>>,
    ) -> Result<Vec<F>, GKRError>,
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

    let wlx_evaluations =
        compute_wlx_fn(claims, prover_supplied_wlx_group_idx, Some(layer_mle_refs))?;
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
    for c in claims.get_claim_vector() {
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

pub fn verifier_aggregate_claims_in_one_round<F: FieldExt, Tr: TranscriptSponge<F>>(
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
    let relevant_wlx_evaluations = transcript_reader
        .consume_elements(
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
    let agg_chal = transcript_reader
        .get_challenge("Challenge for claim aggregation")?;
    debug!("Aggregate challenge: {:#?}", agg_chal);

    let aggregated_challenges = compute_aggregated_challenges(claims, agg_chal).unwrap();
    let claimed_val = evaluate_at_a_point(&wlx_evaluations, agg_chal).unwrap();

    debug!("Aggregating claims: ");
    for c in claims.get_claim_vector() {
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

    if !ENABLE_REDUCED_WLX_EVALS {
        // A naive upper bound on the number of evaluations derived as follows:
        // Let n = `num_vars` and m = `num_claims`. Each monomial in W(x1, ...,
        // x_n) contains up to n variables, each of which will be substituted by
        // `l_i(x)`, a canonical line passing through m points and hence of
        // degree up to m-1, resulting in a polynomial W(l(x)) of degree at most
        // n * (m-1) which needs at least n * (m-1) + 1 evaluations to be fully
        // specified.
        debug!("Dummy num_evals");
        (max(num_vars * (num_claims - 1) + 1, num_claims), None)
    } else {
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
}

/// Aggregates `claims` into a single claim on challenge point
/// `r_star` given the `wlx` evaluations of W(l(x)).
/// This function is used by the verifier in the process of verifying
/// claim aggregation.
/*
#[instrument(level = "trace", err)]
#[instrument(level = "debug", skip_all, err)]
pub(crate) fn verify_aggregate_claim<F: FieldExt>(
    wlx: &Vec<F>, // synonym for q(x)
    claims: &ClaimGroup<F>,
    r_star: F,
) -> Result<Claim<F>, ClaimError> {
    if claims.is_empty() {
        return Err(ClaimError::ClaimAggroError);
    }



    let num_claims = claims.get_num_claims();

    // Check q(0), q(1) equals the claimed value -- i.e. W(l(0)),
    // W(l(1)), etc.
    for idx in 0..num_claims {
        if claims.get_result(idx) != wlx[idx] {
            return Err(ClaimError::ClaimAggroError);
        }
    }

    let r = compute_aggregated_challenges(claims, r_star)?;
    let q_rstar = evaluate_at_a_point(wlx, r_star).unwrap();

    let aggregated_claim: Claim<F> = Claim::new(r, q_rstar, claims.get_layer_id(), None, None);
    Ok(aggregated_claim)
}
*/

#[cfg(test)]
// Makis: Making this public so that I can access some of the helper functions
// from "sumcheck/tests.rs".
pub(crate) mod tests {
    use crate::expression::generic_expr::Expression;
    use crate::expression::prover_expr::ProverExpr;
    use crate::layer::{from_mle, GKRLayer, LayerId};
    use crate::mle::dense::DenseMle;
    use rand::Rng;
    use remainder_shared_types::transcript::{poseidon_transcript::PoseidonSponge, Transcript};

    use super::*;
    use ark_std::test_rng;
    use remainder_shared_types::Fr;

    #[test]
    fn test_get_claim() {
        // [1, 1, 1, 1] \oplus (1 - (1 * (1 + V[1, 1, 1, 1]))) * 2
        let expression1: Expression<Fr, ProverExpr> = Expression::constant(Fr::one());
        let mle = DenseMle::<_, Fr>::new_from_raw(
            vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()],
            LayerId::Input(0),
            None,
        );
        let expression3 = Expression::mle(mle.mle_ref());
        let expression = expression1.clone() + expression3.clone();
        // let expression = expression1.clone() * expression;
        let expression = expression1 - expression;
        let expression = expression * Fr::from(2);
        let _expression = expression3.concat_expr(expression);

        // TODO(ryancao): Need to create a layer and fix all the MLE variables...
    }

    // ------- Helper functions for claim aggregation tests -------

    /// Builds `ClaimGroup<Fr>` by evaluation an expression `expr` on
    /// each point in `points`.
    fn claims_from_expr_and_points(
        expr: &Expression<Fr, ProverExpr>,
        points: &Vec<Vec<Fr>>,
    ) -> ClaimGroup<Fr> {
        let claims_vector: Vec<ClaimMle<Fr>> = cfg_into_iter!(points)
            .map(|point| {
                let mut exp = expr.clone();
                exp.index_mle_indices(0);
                let result = exp.evaluate_expr(point.clone()).unwrap();
                ClaimMle::new_raw(point.clone(), result)
            })
            .collect();
        ClaimGroup::new(claims_vector).unwrap()
    }

    /// Builds GKR layer whose MLE is the function whose evaluations
    /// on the boolean hypercube are given by `mle_evals`.
    fn layer_from_evals(mle_evals: Vec<Fr>) -> GKRLayer<Fr> {
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_evals, LayerId::Input(0), None);

        let layer = from_mle(
            mle,
            |mle| mle.mle_ref().expression(),
            |_, _, _| unimplemented!(),
        );

        let layer: GKRLayer<_> = GKRLayer::new(layer, LayerId::Input(0));


        layer
    }

    /// Returns a random MLE expression with an associated GKR layer.
    fn build_random_mle_layer(num_vars: usize) -> GKRLayer<Fr> {
        let mut rng = test_rng();
        let mle_evals: Vec<Fr> = (0..num_vars).map(|_| Fr::from(rng.gen::<u64>())).collect();
        layer_from_evals(mle_evals)
    }

    fn compute_claim_wlx<F: FieldExt>(claims: &ClaimGroup<F>, layer: &impl YieldWLXEvals<F>) -> Vec<F> {
        let num_claims = claims.get_num_claims();
        let num_vars = claims.get_num_vars();

        let results = claims.get_results();
        let points_matrix = claims.get_claim_points_matrix();

        debug_assert_eq!(points_matrix.len(), num_claims);
        debug_assert_eq!(points_matrix[0].len(), num_vars);

        // Get the evals [W(l(0)), W(l(1)), ..., W(l(degree_upper_bound)) ]

        let claim_mle_refs = claims.get_claim_mle_refs();
        layer
            .get_wlx_evaluations(points_matrix, results, claim_mle_refs, num_claims, num_vars)
            .unwrap()
    }

    /// Wraps around low-level claim aggregation WITHOUT Layer ID
    /// information.
    fn claim_aggregation_back_end_wrapper(
        layer: &impl YieldWLXEvals<Fr>,
        claims: &ClaimGroup<Fr>,
        r_star: Fr,
    ) -> ClaimMle<Fr> {
        let r = compute_aggregated_challenges(claims, r_star).unwrap();
        let wlx = compute_claim_wlx(claims, layer);
        let claimed_val = evaluate_at_a_point(&wlx, r_star).unwrap();
        ClaimMle::new_raw(r, claimed_val)
    }

    /// Compute l* = l(r*).
    fn compute_l_star(claims: &ClaimGroup<Fr>, r_star: &Fr) -> Vec<Fr> {
        let num_vars = claims.get_num_vars();

        (0..num_vars)
            .map(|i| {
                let evals: &Vec<Fr> = claims.get_points_column(i);
                evaluate_at_a_point(evals, r_star.clone()).unwrap()
            })
            .collect()
    }

    /// Wraps around high-level claim aggregation with Layer ID
    /// information.
    pub(crate) fn claim_aggregation_testing_wrapper(
        layer: &impl YieldWLXEvals<Fr>,
        claims: &ClaimGroup<Fr>,
    ) -> (Claim<Fr>, Vec<Vec<Fr>>) {
        let mut transcript_writer =
            TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("Dummy transcript for testing");
        prover_aggregate_claims(
            claims,
            &mut |claim, _, mle_refs| Ok(compute_claim_wlx(claims, layer)),
            &mut transcript_writer,
        )
        .unwrap()
    }

    // Returns expected aggregated claim of `expr` on l(r_star) = `l_star`.
    fn compute_expected_claim(
        layer: &GKRLayer<Fr>,
        l_star: &Vec<Fr>,
    ) -> ClaimMle<Fr> {
        let mut expr = layer.expression().clone();
        expr.index_mle_indices(0);
        let result = expr.evaluate_expr(l_star.clone()).unwrap();
        ClaimMle::new_raw(l_star.clone(), result)
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

        // ---------------

        let layer = layer_from_evals(mle_evals);
        let claims = claims_from_expr_and_points(layer.expression(), &points);

        let l_star = compute_l_star(&claims, &r_star);

        // Compare to l(10) computed by hand.
        assert_eq!(l_star, vec![Fr::from(7).neg(), Fr::from(43)]);

        let aggregated_claim = claim_aggregation_back_end_wrapper(&layer, &claims, r_star.clone());
        let expected_claim = compute_expected_claim(&layer, &l_star);

        // Compare to W(l_star) computed by hand.
        assert_eq!(expected_claim.get_result(), Fr::from(551).neg());

        // assert_eq!(aggregated_claim, expected_claim);
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

        // ---------------

        let layer = layer_from_evals(mle_evals);
        let claims = claims_from_expr_and_points(layer.expression(), &points);

        let l_star = compute_l_star(&claims, &r_star);

        // TODO: Assert l_star was computed correctly.

        let aggregated_claim = claim_aggregation_back_end_wrapper(&layer, &claims, r_star.clone());
        let expected_claim = compute_expected_claim(&layer, &l_star);

        assert_eq!(aggregated_claim.get_result(), expected_claim.get_result());
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
        let mut rng = test_rng();
        let r_star = Fr::from(rng.gen::<u64>());

        // ---------------

        let layer = build_random_mle_layer(3);
        let claims = claims_from_expr_and_points(layer.expression(), &points);

        let l_star = compute_l_star(&claims, &r_star);

        let aggregated_claim = claim_aggregation_back_end_wrapper(&layer, &claims, r_star.clone());
        let expected_claim = compute_expected_claim(&layer, &l_star);

        assert_eq!(aggregated_claim.get_result(), expected_claim.get_result());
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

        let mut rng = test_rng();
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle1_evals, LayerId::Input(0), None);
        let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle2_evals, LayerId::Input(0), None);
        let mle_ref = mle1.mle_ref();
        let mle_ref2 = mle2.mle_ref();

        let expr = Expression::products(vec![mle_ref, mle_ref2]);
        let mut expr_copy = expr.clone();

        let layer = from_mle(
            (mle1, mle2),
            |mle| Expression::products(vec![mle.0.mle_ref(), mle.1.mle_ref()]),
            |_, _, _| unimplemented!(),
        );
        let layer: GKRLayer<_> = GKRLayer::new(layer, LayerId::Input(0));

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

        let claim1: ClaimMle<Fr> = ClaimMle::new_raw(chals1, valchal[0]);
        let claim2: ClaimMle<Fr> = ClaimMle::new_raw(chals2, valchal[1]);
        let claim3: ClaimMle<Fr> = ClaimMle::new_raw(chals3, valchal[2] + Fr::one());

        let rchal = Fr::from(rng.gen::<u64>());

        let claims: Vec<ClaimMle<Fr>> = vec![claim1, claim2, claim3];
        let claim_group = ClaimGroup::new(claims).unwrap();
        let res: ClaimMle<Fr> = claim_aggregation_back_end_wrapper(&layer, &claim_group, rchal);

        let transpose1 = vec![Fr::from(2).neg(), Fr::from(123), Fr::from(92108)];
        let transpose2 = vec![Fr::from(192013).neg(), Fr::from(482), Fr::from(29014)];
        let transpose3 = vec![Fr::from(2148), Fr::from(241), Fr::from(524)];

        let fix_vars: Vec<Fr> = vec![transpose1, transpose2, transpose3]
            .into_iter()
            .map(|evals| evaluate_at_a_point(&evals, rchal).unwrap())
            .collect();

        expr_copy.index_mle_indices(0);

        let eval_fixed_vars = expr_copy.evaluate_expr(fix_vars.clone()).unwrap();
        let claim_fixed_vars: ClaimMle<Fr> = ClaimMle::new_raw(fix_vars, eval_fixed_vars);
        assert_ne!(res.get_result(), claim_fixed_vars.get_result());
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
        let expr = Expression::mle(mle_ref);
        let mut expr_copy = expr.clone();

        let layer = from_mle(
            mle1,
            |mle| mle.mle_ref().expression(),
            |_, _, _| unimplemented!(),
        );
        let layer: GKRLayer<_> = GKRLayer::new(layer, LayerId::Input(0));

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

        let claim1: ClaimMle<Fr> = ClaimMle::new_raw(chals1, valchal[0] - Fr::one());
        let claim2: ClaimMle<Fr> = ClaimMle::new_raw(chals2, valchal[1]);
        let claim3: ClaimMle<Fr> = ClaimMle::new_raw(chals3, valchal[2]);

        let rchal = Fr::from(rng.gen::<u64>());

        let claims_vec: Vec<ClaimMle<Fr>> = vec![claim1, claim2, claim3];
        let claim_group = ClaimGroup::new(claims_vec).unwrap();
        let res: ClaimMle<Fr> = claim_aggregation_back_end_wrapper(&layer, &claim_group, rchal);

        let transpose1 = vec![Fr::from(2).neg(), Fr::from(123), Fr::from(92108)];
        let transpose2 = vec![Fr::from(192013).neg(), Fr::from(482), Fr::from(29014)];
        let transpose3 = vec![Fr::from(2148), Fr::from(241), Fr::from(524)];

        let fix_vars: Vec<Fr> = vec![transpose1, transpose2, transpose3]
            .into_iter()
            .map(|evals| evaluate_at_a_point(&evals, rchal).unwrap())
            .collect();

        expr_copy.index_mle_indices(0);

        let eval_fixed_vars = expr_copy.evaluate_expr(fix_vars.clone()).unwrap();
        let claim_fixed_vars: ClaimMle<Fr> = ClaimMle::new_raw(fix_vars, eval_fixed_vars);
        assert_ne!(res.get_result(), claim_fixed_vars.get_result());
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
        let expr = Expression::mle(mle_ref);
        let mut expr_copy = expr.clone();

        let layer = from_mle(
            mle1,
            |mle| mle.mle_ref().expression(),
            |_, _, _| unimplemented!(),
        );
        let layer: GKRLayer<_> = GKRLayer::new(layer, LayerId::Input(0));

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

        let claim1: ClaimMle<Fr> = ClaimMle::new_raw(chals1, valchal[0]);
        let claim2: ClaimMle<Fr> = ClaimMle::new_raw(chals2, valchal[1]);
        let claim3: ClaimMle<Fr> = ClaimMle::new_raw(chals3, valchal[2] + Fr::one());

        let rchal = Fr::from(rng.gen::<u64>());

        let claims_vec: Vec<ClaimMle<Fr>> = vec![claim1, claim2, claim3];
        let claim_group = ClaimGroup::new(claims_vec).unwrap();
        let res: ClaimMle<Fr> = claim_aggregation_back_end_wrapper(&layer, &claim_group, rchal);

        let transpose1 = vec![Fr::from(2).neg(), Fr::from(123), Fr::from(92108)];
        let transpose2 = vec![Fr::from(192013).neg(), Fr::from(482), Fr::from(29014)];
        let transpose3 = vec![Fr::from(2148), Fr::from(241), Fr::from(524)];

        let fix_vars: Vec<Fr> = vec![transpose1, transpose2, transpose3]
            .into_iter()
            .map(|evals| evaluate_at_a_point(&evals, rchal).unwrap())
            .collect();

        expr_copy.index_mle_indices(0);

        let eval_fixed_vars = expr_copy.evaluate_expr(fix_vars.clone()).unwrap();

        let claim_fixed_vars: ClaimMle<Fr> = ClaimMle::new_raw(fix_vars, eval_fixed_vars);
        assert_ne!(res.get_result(), claim_fixed_vars.get_result());
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

        let layer = layer_from_evals(mle_evals);
        let claims = claims_from_expr_and_points(layer.expression(), &points);

        // W(l(0)), W(l(1)) computed by hand.
        assert_eq!(claims.get_result(0), Fr::from(163));
        assert_eq!(claims.get_result(1), Fr::from(1015));

        let l_star = compute_l_star(&claims, &r_star);

        // Compare to l(10) computed by hand.
        assert_eq!(l_star, vec![Fr::from(11), Fr::from(13), Fr::from(5)]);

        let wlx = compute_claim_wlx(&claims, &layer);
        assert_eq!(wlx, vec![Fr::from(163), Fr::from(1015), Fr::from(2269)]);

        let aggregated_claim = claim_aggregation_back_end_wrapper(&layer, &claims, r_star.clone());
        let expected_claim = compute_expected_claim(&layer, &l_star);

        // Compare to W(l_star) computed by hand.
        assert_eq!(expected_claim.get_result(), Fr::from(26773));

        assert_eq!(aggregated_claim.get_result(), expected_claim.get_result());
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

        let layer = layer_from_evals(mle_evals);
        let claims = claims_from_expr_and_points(layer.expression(), &points);

        // W(l(0)), W(l(1)) computed by hand.
        assert_eq!(claims.get_result(0), Fr::from(163));
        assert_eq!(claims.get_result(1), Fr::from(767));

        let l_star = compute_l_star(&claims, &r_star);

        let wlx = compute_claim_wlx(&claims, &layer);
        assert_eq!(wlx, vec![Fr::from(163), Fr::from(767)]);

        // Compare to l(10) computed by hand.
        assert_eq!(l_star, vec![Fr::from(11), Fr::from(3), Fr::from(5)]);

        let aggregated_claim = claim_aggregation_back_end_wrapper(&layer, &claims, r_star.clone());
        let expected_claim = compute_expected_claim(&layer, &l_star);

        // Compare to W(l_star) computed by hand.
        assert_eq!(expected_claim.get_result(), Fr::from(6203));

        assert_eq!(aggregated_claim.get_result(), expected_claim.get_result());
    }

    // To run this, we need to be able to control the optimization flags which
    // are currently constants.
    /*
    #[test]
    fn test_aggro_claim_smart_merge1() {
        // MLE on 3 variables (2^3 = 8 evals)
        let mle_evals: Vec<Fr> = vec![1, 2, 42, 4, 5, 6, 7, 17]
            .into_iter()
            .map(Fr::from)
            .collect();
        let points = vec![
            vec![Fr::from(1), Fr::from(2), Fr::from(3)],
            vec![Fr::from(2), Fr::from(2), Fr::from(3)],
            vec![Fr::from(4), Fr::from(5), Fr::from(6)],
            vec![Fr::from(5), Fr::from(5), Fr::from(6)],
        ];
        let r_star = Fr::from(10);

        // ---------------

        let layer = layer_from_evals(mle_evals);
        let mut claims = claims_from_expr_and_points(layer.expression(), &points);

        // ---- FOR TESTING ONLY ----
        claims.claims[0].from_layer_id = Some(LayerId::Layer(0));
        claims.claims[1].from_layer_id = Some(LayerId::Layer(0));
        claims.claims[2].from_layer_id = Some(LayerId::Layer(1));
        claims.claims[3].from_layer_id = Some(LayerId::Layer(1));
        // --------------------------

        // W(l(0)), W(l(1)) computed by hand.
        assert_eq!(claims.get_result(0), Fr::from(72));
        assert_eq!(claims.get_result(1), Fr::from(283));
        assert_eq!(claims.get_result(2), Fr::from(4044));
        assert_eq!(claims.get_result(3), Fr::from(5290));

        let l_star = compute_l_star(&claims, &r_star);

        // Compare to l(10) computed by hand.
        assert_eq!(l_star, vec![Fr::from(11), Fr::from(13), Fr::from(5)]);

        let wlx = compute_claim_wlx(&claims, &layer);
        // assert_eq!(wlx, vec![Fr::from(163), Fr::from(1015), Fr::from(2269)]);

        let aggregated_claim = claim_aggregation_testing_wrapper(&layer, &claims);
        let expected_claim = compute_expected_claim(&layer, &l_star);

        // Compare to W(l_star) computed by hand.
        // assert_eq!(expected_claim.get_result(), Fr::from(26773));

        assert_eq!(aggregated_claim.0.get_result(), expected_claim.get_result());
    }
    */

    // TODO(Makis): Fix this outdated test.
    /*
    #[test]
    fn test_verify_claim_aggro() {
        let mle_v1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);
        let mle_ref = mle1.mle_ref();
        let mut expr = Expression::mle(mle_ref);
        let _expr_copy = expr.clone();

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

        let claim1: Claim<Fr> = Claim::new_raw(chals1, valchal[0]);
        let claim2: Claim<Fr> = Claim::new_raw(chals2, valchal[1]);
        let claim3: Claim<Fr> = Claim::new_raw(chals3, valchal[2]);
        let claims = vec![claim1, claim2, claim3];

        let rchal = Fr::from(2).neg();

        let claim_group = ClaimGroup::new(claims).unwrap();
        let (res, wlx) = claim_aggregation_testing_wrapper(&layer, &claim_group);
        let rounds = expr.index_mle_indices(0);
        // let verify_result = verify_aggregate_claim(&wlx.unwrap(), &claim_group, rchal).unwrap();

        // Makis; This test isn't verifying anything apart form the
        // fact that no errors are produced :/
    }
    */
}
