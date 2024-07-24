#[cfg(test)]
pub mod tests;

mod helpers;

mod claim_group;

use remainder_shared_types::transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter};
use remainder_shared_types::FieldExt;

use crate::claims::wlx_eval::claim_group::ClaimGroup;
use crate::claims::wlx_eval::helpers::{
    prover_aggregate_claims_helper, verifier_aggregate_claims_helper,
};

use crate::mle::mle_enum::MleEnum;

use crate::input_layer::InputLayer;
use crate::prover::GKRError;
use crate::sumcheck::*;

use crate::layer::LayerId;
use crate::layer::{Layer, LayerError};

use serde::{Deserialize, Serialize};

use std::cmp::max;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

use log::debug;

use super::{Claim, ClaimAggregator, ClaimAndProof, ClaimError, YieldClaim};

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
    ) -> Result<ClaimAndProof<F, Self::AggregationProof>, GKRError> {
        let layer_id = layer.id();
        self.prover_aggregate_claims(layer, *layer_id, transcript_writer)
    }

    fn prover_aggregate_claims_input(
        &self,
        layer: &Self::InputLayer,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<ClaimAndProof<F, Self::AggregationProof>, GKRError> {
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
            let ClaimAndProof {
                claim: prev_claim, ..
            } = verifier_aggregate_claims_helper(&claim_group, transcript_reader).map_err(
                |err| GKRError::ErrorWhenVerifyingLayer(layer_id, LayerError::TranscriptError(err)),
            )?;

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
    ) -> Result<ClaimAndProof<F, Vec<Vec<F>>>, GKRError> {
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
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
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

/// Helper functions for types implementing `YieldWlxEvals`
///
///
/// Returns an upper bound on the number of evaluations needed to represent the
/// polynomial `P(x) = W(l(x))` where `W : F^n -> F` is a multilinear polynomial
/// on `n` variables and `l : F -> F^n` is such that:
///  * `l(0) = claim_vecs[0]`,
///  *  `l(1) = `claim_vecs[1]`,
///  *   ...,
///  *  `l(m-1) = `claim_vecs[m-1]`.
///
/// It is guaranteed that the returned value is at least `num_claims =
/// claim_vecs.len()`.
/// # Panics
///  if `claim_vecs` is empty.
pub fn get_num_wlx_evaluations<F: FieldExt>(claim_vecs: &[Vec<F>]) -> (usize, Option<Vec<usize>>) {
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
