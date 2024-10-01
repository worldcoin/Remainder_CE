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
use crate::input_layer::enum_input_layer::InputLayerEnum;
use crate::input_layer::InputLayer;
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

use super::{ClaimAggregator, RawClaim, YieldClaim};

/// The default ClaimAggregator.
///
/// Keeps tracks of claims using a hashmap with the [LayerId] as the key.
///
/// Aggregates claims using univariate polynomial interpolation.
///
/// Collects additional information in the [ClaimMle] struct to make computation
/// of evaluations easier, most importantly the `original_bookkeeping_table`.
#[derive(Debug)]
pub struct WLXAggregator<F: Field, L, LI> {
    claims: HashMap<LayerId, Vec<ClaimMle<F>>>,
    _marker: std::marker::PhantomData<(L, LI)>,
}

impl<
        F: Field,
        L: Layer<F> + YieldWLXEvals<F> + YieldClaim<ClaimMle<F>>,
        LI: InputLayer<F> + YieldWLXEvals<F>,
    > ClaimAggregator<F> for WLXAggregator<F, L, LI>
{
    type Claim = ClaimMle<F>;

    type Layer = L;
    type InputLayer = LI;

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
    ) -> Result<RawClaim<F>, GKRError> {
        let layer_id = layer.layer_id();
        self.prover_aggregate_claims(layer, output_mles_from_layer, layer_id, transcript_writer)
    }

    fn prover_aggregate_claims_input(
        &self,
        layer: &InputLayerEnum<F>,
        output_mles_from_layer: &[DenseMle<F>],
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<RawClaim<F>, GKRError> {
        let layer_id = layer.layer_id();
        self.prover_aggregate_claims(layer, output_mles_from_layer, layer_id, transcript_writer)
    }

    fn verifier_aggregate_claims(
        &self,
        layer_id: LayerId,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<RawClaim<F>, GKRError> {
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

impl<
        F: Field,
        L: Layer<F> + YieldWLXEvals<F> + YieldClaim<ClaimMle<F>>,
        LI: InputLayer<F> + YieldWLXEvals<F>,
    > WLXAggregator<F, L, LI>
{
    fn prover_aggregate_claims(
        &self,
        layer: &impl YieldWLXEvals<F>,
        output_mles_from_layer: &[DenseMle<F>],
        layer_id: LayerId,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<RawClaim<F>, GKRError> {
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
