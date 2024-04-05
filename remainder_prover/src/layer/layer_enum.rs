use remainder_shared_types::transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter};
use serde::{Deserialize, Serialize};

use remainder_shared_types::{transcript::Transcript, FieldExt};
use tracing::instrument;

use crate::claims::wlx_eval::{ClaimMle, YieldWLXEvals};
use crate::{claims::wlx_eval::WLXAggregator, gate::gate::Gate};
use crate::layer_enum;
use crate::mle::dense::DenseMleRef;
use crate::mle::mle_enum::MleEnum;

use super::LayerError;
use super::{empty_layer::EmptyLayer, GKRLayer, Layer};

use crate::claims::{Claim, YieldClaim};

use std::fmt;

layer_enum!(
    LayerEnum,
    (Gkr: GKRLayer<F>),
    (Gate: Gate<F>),
    (EmptyLayer: EmptyLayer<F>)
);

impl<F: FieldExt> fmt::Debug for LayerEnum<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LayerEnum::Gkr(_) => write!(f, "GKR Layer"),
            LayerEnum::Gate(_) => write!(f, "Gate"),
            LayerEnum::EmptyLayer(_) => write!(f, "EmptyLayer"),
        }
    }
}

impl<F: FieldExt> LayerEnum<F> {
    ///Gets the size of the Layer as a whole in terms of number of bits
    pub(crate) fn layer_size(&self) -> usize {
        let expression = match self {
            LayerEnum::Gkr(layer) => &layer.expression,
            LayerEnum::EmptyLayer(layer) => &layer.expr,
            LayerEnum::Gate(_) => unimplemented!(),
        };

        expression.get_expression_size(0)
    }

    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> Box<dyn std::fmt::Display + 'a> {
        match self {
            LayerEnum::Gkr(layer) => Box::new(layer.expression().circuit_description_fmt()),
            LayerEnum::Gate(gate_layer) => Box::new(gate_layer.circuit_description_fmt()),
            LayerEnum::EmptyLayer(empty_layer) => {
                Box::new(empty_layer.expression().circuit_description_fmt())
            }
        }
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for LayerEnum<F> {
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, crate::claims::ClaimError> {
        match self {
            LayerEnum::Gkr(layer) => layer.get_wlx_evaluations(claim_vecs, claimed_vals, claimed_mles, num_claims, num_idx),
            LayerEnum::Gate(layer) => layer.get_wlx_evaluations(claim_vecs, claimed_vals, claimed_mles, num_claims, num_idx),
            LayerEnum::EmptyLayer(layer) => layer.get_wlx_evaluations(claim_vecs, claimed_vals, claimed_mles, num_claims, num_idx),
        }
    }
}

impl<F: FieldExt> YieldClaim<F, ClaimMle<F>> for LayerEnum<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        match self {
            LayerEnum::Gkr(layer) => layer.get_claims(),
            LayerEnum::Gate(layer) => layer.get_claims(),
            LayerEnum::EmptyLayer(layer) => layer.get_claims(),
        }
    }
}
