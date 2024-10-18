#[cfg(test)]
pub mod tests;

pub(crate) mod helpers;

/// The module which performs claim aggregation algorithms.
pub mod claim_group;
use remainder_shared_types::transcript::{ProverTranscript, VerifierTranscript};
use remainder_shared_types::Field;

use crate::claims::wlx_eval::claim_group::ClaimGroup;
use crate::claims::wlx_eval::helpers::{
    prover_aggregate_claims_helper, verifier_aggregate_claims_helper,
};

use crate::claims::ClaimError;
use crate::layer::layer_enum::LayerEnum;
use crate::mle::dense::DenseMle;

use crate::prover::GKRError;
use crate::sumcheck::*;

use crate::layer::{Layer, LayerError, LayerId};

use serde::{Deserialize, Serialize};

use std::cmp::max;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

use log::debug;

use super::{Claim, ClaimAggregator, YieldClaim};

/// The default ClaimAggregator.
///
/// Keeps tracks of claims using a hashmap with the [LayerId] as the key.
///
/// Aggregates claims using univariate polynomial interpolation.
///
/// Collects additional information in the [ClaimMle] struct to make computation
/// of evaluations easier, most importantly the `original_bookkeeping_table`.
#[derive(Debug)]
pub struct WLXAggregator<F: Field, L> {
    claims: HashMap<LayerId, Vec<ClaimMle<F>>>,
    _marker: std::marker::PhantomData<L>,
}

impl<F: Field, L: Layer<F> + YieldWLXEvals<F> + YieldClaim<ClaimMle<F>>> ClaimAggregator<F>
    for WLXAggregator<F, L>
{
    type Claim = ClaimMle<F>;

    type Layer = L;

    fn new() -> Self {
        Self {
            claims: HashMap::new(),
            _marker: PhantomData,
        }
    }

    fn extract_claims(&mut self, layer: &impl YieldClaim<ClaimMle<F>>) -> Result<(), LayerError> {
        // Ask `layer` to generate claims for other layers.
        let claims = layer.get_claims()?;
        // Assign each claim to the appropriate layer based on the `to_layer_id`
        // field.
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

    fn prover_aggregate_claims(
        &self,
        layer: &LayerEnum<F>,
        output_mles_from_layer: &[DenseMle<F>],
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<Claim<F>, GKRError> {
        let layer_id = layer.layer_id();
        self.prover_aggregate_claims(layer, output_mles_from_layer, layer_id, transcript_writer)
    }

    fn verifier_aggregate_claims(
        &self,
        layer_id: LayerId,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Claim<F>, GKRError> {
        let claims = self
            .get_claims(layer_id)
            .ok_or(GKRError::ErrorWhenVerifyingLayer(
                layer_id,
                LayerError::ClaimError(ClaimError::ClaimAggroError),
            ))?;

        let claim_group = ClaimGroup::new(claims.to_vec()).unwrap();
        debug!("Layer Claim Group for input: {:#?}", claims);

        if claims.len() > 1 {
            let prev_claim = verifier_aggregate_claims_helper(&claim_group, transcript_reader)
                .map_err(|err| {
                    GKRError::ErrorWhenVerifyingLayer(layer_id, LayerError::TranscriptError(err))
                })?;

            Ok(prev_claim)
        } else {
            Ok(claims[0].get_claim().clone())
        }
    }
}

impl<F: Field, L: Layer<F> + YieldWLXEvals<F> + YieldClaim<ClaimMle<F>>> WLXAggregator<F, L> {
    fn prover_aggregate_claims(
        &self,
        layer: &impl YieldWLXEvals<F>,
        output_mles_from_layer: &[DenseMle<F>],
        layer_id: LayerId,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<Claim<F>, GKRError> {
        let claims = self
            .get_claims(layer_id)
            .ok_or(GKRError::ErrorWhenProvingLayer(
                layer_id,
                LayerError::ClaimError(ClaimError::ClaimAggroError),
            ))?;
        let claim_group = ClaimGroup::new(claims.to_vec()).unwrap();
        debug!("Found Layer claims:\n{:#?}", claims);

        prover_aggregate_claims_helper(
            &claim_group,
            layer,
            output_mles_from_layer,
            transcript_writer,
        )
    }
}

///The trait that layers must implement to be compatible with the WLXEval based claim aggregator
pub trait YieldWLXEvals<F: Field> {
    ///Get W(l(x)) evaluations
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        claimed_mles: Vec<DenseMle<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, crate::claims::ClaimError>;
}

/// A claim that can optionally maintain additional source/destination layer
/// information through `from_layer_id` and `to_layer_id`. This information can
/// be used to speed up claim aggregation.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(bound = "F: Field")]
pub struct ClaimMle<F: Field> {
    /// The underlying raw Claim.
    claim: Claim<F>,

    /// The layer ID of the layer that produced this claim (if present); origin
    /// layer.
    pub from_layer_id: Option<LayerId>,

    /// The layer ID of the layer containing the MLE this claim refers to (if
    /// present); destination layer.
    pub to_layer_id: Option<LayerId>,
}

impl<F: Field> ClaimMle<F> {
    /// To be used internally only!
    /// Generate new raw claim without any origin/destination information.
    pub fn new_raw(point: Vec<F>, result: F) -> Self {
        Self {
            claim: Claim::new(point, result),
            from_layer_id: None,
            to_layer_id: None,
        }
    }

    /// Generate new claim, potentially with origin/destination information.
    pub fn new(
        point: Vec<F>,
        result: F,
        from_layer_id: Option<LayerId>,
        to_layer_id: Option<LayerId>,
    ) -> Self {
        Self {
            claim: Claim::new(point, result),
            from_layer_id,
            to_layer_id,
        }
    }

    /// Returns the length of the `point` vector.
    pub fn get_num_vars(&self) -> usize {
        self.claim.get_num_vars()
    }

    /// Returns the point vector in F^n.
    pub fn get_point(&self) -> &Vec<F> {
        self.claim.get_point()
    }

    /// Returns the expected result.
    pub fn get_result(&self) -> F {
        self.claim.get_result()
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

impl<F: fmt::Debug + Field> fmt::Debug for ClaimMle<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Claim")
            .field("point", self.claim.get_point())
            .field("result", &self.claim.get_result())
            .field("from_layer_id", &self.from_layer_id)
            .field("to_layer_id", &self.to_layer_id)
            .finish()
    }
}

/// Helper function for types implementing `YieldWlxEvals`.
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
pub fn get_num_wlx_evaluations<F: Field>(
    claim_vecs: &[Vec<F>],
) -> (usize, Option<Vec<usize>>, Vec<usize>) {
    let num_claims = claim_vecs.len();
    let num_vars = claim_vecs[0].len();

    debug!("Smart num_evals");
    let mut num_constant_columns = num_vars as i64;
    let mut common_idx = vec![];
    let mut non_common_idx = vec![];
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
