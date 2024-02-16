use remainder_shared_types::transcript::TranscriptSponge;
use serde::{Deserialize, Serialize};

use remainder_shared_types::{transcript::Transcript, FieldExt};
use tracing::instrument;

use crate::gate::gate::Gate;
use crate::mle::mle_enum::MleEnum;

use super::{empty_layer::EmptyLayer, GKRLayer, Layer};

use super::claims::Claim;

use std::fmt;

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
///An enum representing all the possible kinds of Layers
pub enum LayerEnum<F: FieldExt, Tr: TranscriptSponge<F>> {
    ///A standard `GKRLayer`
    Gkr(GKRLayer<F, Tr>),
    /// Gate Generic
    Gate(Gate<F, Tr>),
    /// Layer with zero variables within it
    EmptyLayer(EmptyLayer<F, Tr>),
}

impl<F: FieldExt, Tr: TranscriptSponge<F>> fmt::Debug for LayerEnum<F, Tr> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LayerEnum::Gkr(_) => write!(f, "GKR Layer"),
            LayerEnum::Gate(_) => write!(f, "Gate"),
            LayerEnum::EmptyLayer(_) => write!(f, "EmptyLayer"),
        }
    }
}

impl<F: FieldExt, Tr: TranscriptSponge<F>> Layer<F> for LayerEnum<F, Tr> {
    type Sponge = Tr;

    #[instrument(skip_all, level = "debug", err)]
    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut Self::Sponge,
    ) -> Result<crate::prover::SumcheckProof<F>, super::LayerError> {
        match self {
            LayerEnum::Gkr(layer) => layer.prove_rounds(claim, transcript),
            LayerEnum::Gate(layer) => layer.prove_rounds(claim, transcript),
            LayerEnum::EmptyLayer(layer) => layer.prove_rounds(claim, transcript),
        }
    }

    #[instrument(skip_all, level = "debug", err)]
    fn verify_rounds(
        &mut self,
        claim: Claim<F>,
        sumcheck_rounds: Vec<Vec<F>>,
        transcript: &mut Self::Sponge,
    ) -> Result<(), super::LayerError> {
        match self {
            LayerEnum::Gkr(layer) => layer.verify_rounds(claim, sumcheck_rounds, transcript),
            LayerEnum::Gate(layer) => layer.verify_rounds(claim, sumcheck_rounds, transcript),
            LayerEnum::EmptyLayer(layer) => layer.verify_rounds(claim, sumcheck_rounds, transcript),
        }
    }

    #[instrument(skip(self), level = "debug", err)]
    fn get_claims(&self) -> Result<Vec<Claim<F>>, super::LayerError> {
        match self {
            LayerEnum::Gkr(layer) => layer.get_claims(),
            LayerEnum::Gate(layer) => layer.get_claims(),
            LayerEnum::EmptyLayer(layer) => layer.get_claims(),
        }
    }

    fn id(&self) -> &super::LayerId {
        match self {
            LayerEnum::Gkr(layer) => layer.id(),
            LayerEnum::Gate(layer) => layer.id(),
            LayerEnum::EmptyLayer(layer) => layer.id(),
        }
    }

    fn new<L: super::LayerBuilder<F>>(builder: L, id: super::LayerId) -> Self
    where
        Self: Sized,
    {
        LayerEnum::Gkr(GKRLayer::new(builder, id))
    }

    fn get_enum(self) -> LayerEnum<F, Self::Sponge> {
        self
    }

    // TODO(Makis): Perhaps refactor to receive a `Claim<F>` instead.
    /// NOTE: This function is effectively deprecated!!!
    #[instrument(skip(self, claim_vecs, claimed_vals), level = "debug", err)]
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, super::claims::ClaimError> {
        match self {
            LayerEnum::Gkr(layer) => layer.get_wlx_evaluations(
                claim_vecs,
                claimed_vals,
                claimed_mles,
                num_claims,
                num_idx,
            ),
            LayerEnum::Gate(layer) => layer.get_wlx_evaluations(
                claim_vecs,
                claimed_vals,
                claimed_mles,
                num_claims,
                num_idx,
            ),
            LayerEnum::EmptyLayer(layer) => layer.get_wlx_evaluations(
                claim_vecs,
                claimed_vals,
                claimed_mles,
                num_claims,
                num_idx,
            ),
        }
    }
}

impl<F: FieldExt, Tr: TranscriptSponge<F>> LayerEnum<F, Tr> {
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
