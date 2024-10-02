use std::cmp::Ordering;

use itertools::Itertools;
use remainder_shared_types::Field;
use tracing::{debug, info};

use crate::claims::ClaimError;

use super::ClaimMle;

/// A collection of claims for the same layer with an API for accessing the
/// matrix of claim points in a multitude of ways (row-wise or column-wise).
/// This struct is useful for claim aggregation.
/// Invariant: All claims have to agree on `to_layer_id` and on the number of
/// variables.
#[derive(Clone, Debug)]
pub struct ClaimGroup<F: Field> {
    /// A vector of claims in F^n.
    pub claims: Vec<ClaimMle<F>>,

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
    pub fn get_claim(&self, i: usize) -> &ClaimMle<F> {
        &self.claims[i]
    }

    /// Returns a reference to a vector of claims contained in this group.
    pub fn get_claims(&self) -> &[ClaimMle<F>] {
        &self.claims
    }
}

/// Sorts claims by `from_layer_id` to prepare them for grouping. Also performs
/// claim de-duplication if the `ENABLE_CLAIM_DEDUPLICATION` flag it set.
fn preprocess_claims<F: Field>(mut claims: Vec<ClaimMle<F>>) -> Vec<ClaimMle<F>> {
    // Sort claims on the `from_layer_id` field.
    // A trivial total order is imposed which includes `None` values.
    claims.sort_by(|claim1, claim2| {
        match (claim1.get_from_layer_id(), claim2.get_from_layer_id()) {
            (Some(id1), Some(id2)) => match id1.partial_cmp(&id2) {
                // Ties are broken by point value.
                // Ordering::Equal => claim1.get_point().cmp(claim2.get_point()),
                Some(ordering) => ordering,
                None => Ordering::Greater,
            },
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            // (None, None) => claim1.get_point().cmp(claim2.get_point()),
            (None, None) => Ordering::Equal,
        }
    });

    // Perform claim de-duplication

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

/// Assign `claims` into groups to be aggregated together.  The naive version of
/// this function assigns all claims to a single group.  On the other hand, if
/// the `ENABLE_CLAIM_GROUPING` flag is set, it groups the claims based on the
/// `from_layer_id` field of each claim.
/// # Requires
/// All claims with the same `from_layer_id` should appear consecutively in the
/// `claims` vector. For example, `claims` can be sorted by `from_layer_id`.
pub fn form_claim_groups<F: Field>(claims: Vec<ClaimMle<F>>) -> Vec<ClaimGroup<F>> {
    info!("Forming claim group...");

    let claims = preprocess_claims(claims);

    let num_claims = claims.len();
    let mut claim_group_vec: Vec<ClaimGroup<F>> = vec![];

    // Identify runs of claims with the same `from_layer_id` field.
    let mut start_index = 0;
    for idx in 1..num_claims {
        if claims[idx].get_from_layer_id() != claims[idx - 1].get_from_layer_id() {
            let end_index = idx;
            claim_group_vec.push(ClaimGroup::new(claims[start_index..end_index].to_vec()).unwrap());
            start_index = idx;
        }
    }

    // Process the last group.
    let end_index = num_claims;
    claim_group_vec.push(ClaimGroup::new(claims[start_index..end_index].to_vec()).unwrap());

    claim_group_vec
}
