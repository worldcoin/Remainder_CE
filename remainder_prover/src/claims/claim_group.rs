use ark_std::cfg_into_iter;
use itertools::Itertools;
use remainder_shared_types::{
    config::global_config::global_verifier_claim_agg_constant_column_optimization,
    transcript::{ProverTranscript, TranscriptReaderError, VerifierTranscript},
    Field,
};
use tracing::{debug, info};

use crate::{
    claims::{
        claim_aggregation::{get_num_wlx_evaluations, get_wlx_evaluations},
        ClaimError,
    },
    mle::dense::DenseMle,
    prover::GKRError,
    sumcheck::evaluate_at_a_point,
};

use super::{Claim, RawClaim};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Stores a collection of claims and provides an API running claim aggregation
/// algorithms on them.
/// The current implementation introduces up to 3x memory redundancy in order to
/// achieve faster access times.
/// Invariant: All claims are on the same number of variables.
#[derive(Clone, Debug)]
pub struct ClaimGroup<F: Field> {
    /// A vector of raw claims in F^n.
    pub claims: Vec<RawClaim<F>>,

    /// A 2D matrix with the claim's points as its rows.
    claim_points_matrix: Vec<Vec<F>>,

    /// The points in `claims` is effectively a matrix of elements in F. We also
    /// store the transpose of this matrix for convenient access.
    claim_points_transpose: Vec<Vec<F>>,

    /// A vector of `self.get_num_claims()` elements. For each claim i,
    /// `result_vector[i]` stores the expected result of the i-th claim.
    result_vector: Vec<F>,
}

impl<F: Field> ClaimGroup<F> {
    /// Generates a [ClaimGroup] from a collection of [Claim]s.
    /// All claims agree on the [Claim::to_layer_id] field and returns
    /// [ClaimError::LayerIdMismatch] otherwise.  Returns
    /// [ClaimError::NumVarsMismatch] if the collection of claims do not all
    /// agree on the number of variables.
    pub fn new(claims: Vec<Claim<F>>) -> Result<Self, ClaimError> {
        let num_claims = claims.len();
        if num_claims == 0 {
            return Ok(Self {
                claims: vec![],
                claim_points_matrix: vec![],
                claim_points_transpose: vec![],
                result_vector: vec![],
            });
        }
        // Check all claims match on the `to_layer_id` field.
        let layer_id = claims[0].get_to_layer_id();
        if !claims
            .iter()
            .all(|claim| claim.get_to_layer_id() == layer_id)
        {
            return Err(ClaimError::LayerIdMismatch);
        }

        Self::new_from_raw_claims(claims.into_iter().map(Into::into).collect())
    }

    /// Generates a new [ClaimGroup] from a collection of [RawClaim]s.
    /// Returns [ClaimError::NumVarsMismatch] if the collection of claims
    /// do not all agree on the number of variables.
    pub fn new_from_raw_claims(claims: Vec<RawClaim<F>>) -> Result<Self, ClaimError> {
        let num_claims = claims.len();

        if num_claims == 0 {
            return Ok(Self {
                claims: vec![],
                claim_points_matrix: vec![],
                claim_points_transpose: vec![],
                result_vector: vec![],
            });
        }

        let num_vars = claims[0].get_num_vars();

        // Check all claims match on the number of variables.
        if !claims.iter().all(|claim| claim.get_num_vars() == num_vars) {
            return Err(ClaimError::NumVarsMismatch);
        }

        // Populate the points_matrix
        let points_matrix: Vec<_> = claims
            .iter()
            .map(|claim| -> Vec<F> { claim.get_point().to_vec() })
            .collect();

        // Compute the claim points transpose.
        let claim_points_transpose: Vec<Vec<F>> = (0..num_vars)
            .map(|j| (0..num_claims).map(|i| claims[i].get_point()[j]).collect())
            .collect();

        // Compute the result vector.
        let result_vector: Vec<F> = (0..num_claims).map(|i| claims[i].get_eval()).collect();

        Ok(Self {
            claims,
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

    /// Returns a reference to a vector with m = `self.get_num_claims()`
    /// elements containing the results of all claims.
    pub fn get_results(&self) -> &Vec<F> {
        &self.result_vector
    }

    /// Returns a reference to the i-th claim.
    pub fn get_raw_claim(&self, i: usize) -> &RawClaim<F> {
        &self.claims[i]
    }

    /// Returns a reference to a vector of claims contained in this group.
    pub fn get_raw_claims(&self) -> &[RawClaim<F>] {
        &self.claims
    }

    /// Returns `claims` sorted by `from_layer_id` to prepare them for grouping.
    /// Also performs claim de-duplication by eliminating copies of claims
    /// on the same point.
    fn preprocess_claims(mut claims: Vec<Claim<F>>) -> Vec<Claim<F>> {
        // Sort claims on the `from_layer_id` field.
        claims.sort_by(|claim1, claim2| {
            claim1
                .get_from_layer_id()
                .partial_cmp(&claim2.get_from_layer_id())
                .unwrap()
        });

        // Perform claim de-duplication
        let claims = claims
            .into_iter()
            .unique_by(|c| c.get_point().to_vec())
            .collect_vec();

        claims
    }

    /// Partition `claims` into groups to be aggregated together.
    pub fn form_claim_groups(claims: Vec<Claim<F>>) -> Vec<Self> {
        // Sort claims by `from_layer_id` and remove duplicates.
        let claims = Self::preprocess_claims(claims);

        let num_claims = claims.len();
        let mut claim_group_vec: Vec<Self> = vec![];

        // Identify runs of claims with the same `from_layer_id` field.
        let mut start_index = 0;
        for idx in 1..num_claims {
            if claims[idx].get_from_layer_id() != claims[idx - 1].get_from_layer_id() {
                let end_index = idx;
                claim_group_vec.push(Self::new(claims[start_index..end_index].to_vec()).unwrap());
                start_index = idx;
            }
        }

        // Process the last group.
        let end_index = num_claims;
        claim_group_vec.push(Self::new(claims[start_index..end_index].to_vec()).unwrap());

        claim_group_vec
    }

    /// Computes the aggregated challenge point by interpolating a polynomial
    /// passing through all the points in the claim group and then evaluating
    /// it at `r_star`.
    /// More precicely, if `self.claims` contains `m` points `[u_0, u_1, ...,
    /// u_{m-1}]` where each `u_i \in F^n`, it computes a univariate polynomial
    /// vector `l : F -> F^n` such that `l(0) = u_0, l(1) = u_1, ..., l(m-1) =
    /// u_{m-1}` using Lagrange interpolation, then evaluates `l` on `r_star`
    /// and returns it.
    ///
    /// # Requires
    /// `self.claims_points` should be non-empty, otherwise a
    /// [ClaimError::ClaimAggroError] is returned.
    /// Using the ClaimGroup abstraction here is not ideal since we are only
    /// operating on the points and not on the results. However, the ClaimGroup API
    /// is convenient for accessing columns and makes the implementation more
    /// readable. We should consider alternative designs.
    fn compute_aggregated_challenges(&self, r_star: F) -> Result<Vec<F>, ClaimError> {
        if self.is_empty() {
            return Err(ClaimError::ClaimAggroError);
        }

        let num_vars = self.get_num_vars();

        // Compute r = l(r*) by performing Lagrange interpolation on each coordinate
        // using `evaluate_at_a_point`.
        let r: Vec<F> = cfg_into_iter!(0..num_vars)
            .map(|idx| {
                let evals = self.get_points_column(idx);
                // Interpolate the value l(r*) from the values
                // l(0), l(1), ..., l(m-1) where m = # of claims.
                evaluate_at_a_point(evals, r_star).unwrap()
            })
            .collect();

        Ok(r)
    }

    /// Performs claim aggregation on the prover side for this claim group in a
    /// single stage -- this is the standard "Thaler13" claim aggregation
    /// without any heuristic optimizations.
    ///
    /// # Parameters
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
    pub fn prover_aggregate(
        &self,
        layer_mles: &[DenseMle<F>],
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<RawClaim<F>, GKRError> {
        let num_claims = self.get_num_claims();
        debug_assert!(num_claims > 0);
        info!("ClaimGroup aggregation on {num_claims} claims.");

        // Do nothing if there is only one claim.
        if num_claims == 1 {
            debug!("Received 1 claim. Doing nothing.");
            return Ok(self.claims[0].clone());
        }
        assert!(self.get_claim_points_matrix().len() > 1);

        // Aggregate claims by performing the claim aggregation protocol.
        // First compute V_i(l(x)).
        let wlx_evaluations = get_wlx_evaluations(
            self.get_claim_points_matrix(),
            self.get_results(),
            layer_mles.to_vec(),
            num_claims,
            self.get_num_vars(),
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

        let aggregated_challenges = self.compute_aggregated_challenges(agg_chal).unwrap();
        let claimed_val = evaluate_at_a_point(&wlx_evaluations, agg_chal).unwrap();

        debug!("Aggregating claims: {:#?}", self.get_raw_claims());

        let claim = RawClaim::new(aggregated_challenges, claimed_val);
        debug!("Low level aggregated claim:\n{:#?}", &claim);

        Ok(claim)
    }

    /// Performs claim aggregation on the verifier side for this claim group in
    /// a single stage -- this is the standard "Thaler13" claim aggregation
    /// without any heuristic optimizations.
    ///
    /// # Parameters
    /// * `transcript_reader`: is used to retrieve wlx evaluations and generate
    ///   challenges.
    ///
    /// # Returns
    /// If successful, returns a single aggregated claim.
    pub fn verifier_aggregate(
        &self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<RawClaim<F>, TranscriptReaderError> {
        let num_claims = self.get_num_claims();
        debug_assert!(num_claims > 0);
        info!("Low-level claim aggregation on {num_claims} claims.");

        // Do nothing if there is only one claim.
        if num_claims == 1 {
            debug!("Received 1 claim. Doing nothing.");
            return Ok(self.get_raw_claim(0).clone());
        }

        // Aggregate claims by performing the claim aggregation protocol.
        // First retrieve V_i(l(x)).

        let num_wlx_evaluations = if global_verifier_claim_agg_constant_column_optimization() {
            let (num_wlx_evaluations, _, _) =
                get_num_wlx_evaluations(self.get_claim_points_matrix());
            num_wlx_evaluations
        } else {
            ((num_claims - 1) * self.get_num_vars()) + 1
        };

        let num_relevant_wlx_evaluations = num_wlx_evaluations - num_claims;
        let relevant_wlx_evaluations = transcript_reader.consume_elements(
            "Claim Aggregation Wlx_evaluations",
            num_relevant_wlx_evaluations,
        )?;
        let wlx_evaluations = self
            .get_results()
            .clone()
            .into_iter()
            .chain(relevant_wlx_evaluations.clone())
            .collect_vec();

        // Next, sample `r^\star` from the transcript.
        let agg_chal = transcript_reader.get_challenge("Challenge for claim aggregation")?;
        debug!("Aggregate challenge: {:#?}", agg_chal);

        let aggregated_challenges = self.compute_aggregated_challenges(agg_chal).unwrap();
        let claimed_val = evaluate_at_a_point(&wlx_evaluations, agg_chal).unwrap();

        debug!("Aggregating claims: {:#?}", self.get_raw_claims());

        let claim = RawClaim::new(aggregated_challenges, claimed_val);
        debug!("Low level aggregated claim:\n{:#?}", &claim);

        Ok(claim)
    }
}
