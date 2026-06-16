use std::collections::BTreeSet;

use shared_types::{
    Field,
};

use crate::claims::Claim;

/// Remove redundant claims
pub fn dedup_claims<F: Field>(
    all_claims_on_layer: &[Claim<F>],
) -> Vec<Claim<F>> {
    let mut claims = BTreeSet::new();
    for claim in all_claims_on_layer {
        claims.insert(claim.clone());
    }
    claims.into_iter().collect()
}