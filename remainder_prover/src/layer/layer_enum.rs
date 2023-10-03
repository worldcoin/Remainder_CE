use serde::{Deserialize, Serialize};

use remainder_shared_types::{transcript::Transcript, FieldExt};
use tracing::instrument;

use crate::{
    gate::{addgate::{AddGate}, batched_addgate::{AddGateBatched}, mulgate::{MulGate}, batched_mulgate::MulGateBatched},
};

use super::{empty_layer::EmptyLayer, GKRLayer, Layer};

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
///An enum representing all the possible kinds of Layers
pub enum LayerEnum<F: FieldExt, Tr: Transcript<F>> {
    ///A standard `GKRLayer`
    Gkr(GKRLayer<F, Tr>),
    /// A Mulgate
    MulGate(MulGate<F, Tr>),
    /// An Addition Gate
    AddGate(AddGate<F, Tr>),
    /// Batched AddGate
    AddGateBatched(AddGateBatched<F, Tr>),
    /// Batched MulGate
    MulGateBatched(MulGateBatched<F, Tr>),
    /// Layer with zero variables within it
    EmptyLayer(EmptyLayer<F, Tr>),
}

impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for LayerEnum<F, Tr> {
    type Transcript = Tr;

    #[instrument(skip(transcript, self), level = "debug", err)]
    fn prove_rounds(
        &mut self,
        claim: super::Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<crate::prover::SumcheckProof<F>, super::LayerError> {
        match self {
            LayerEnum::Gkr(layer) => layer.prove_rounds(claim, transcript),
            LayerEnum::MulGate(layer) => layer.prove_rounds(claim, transcript),
            LayerEnum::MulGateBatched(layer) => layer.prove_rounds(claim, transcript),
            LayerEnum::AddGate(layer) => layer.prove_rounds(claim, transcript),
            LayerEnum::AddGateBatched(layer) => layer.prove_rounds(claim, transcript),
            LayerEnum::EmptyLayer(layer) => layer.prove_rounds(claim, transcript),
        }
    }

    #[instrument(skip(transcript, self), level = "debug", err)]
    fn verify_rounds(
        &mut self,
        claim: super::Claim<F>,
        sumcheck_rounds: Vec<Vec<F>>,
        transcript: &mut Self::Transcript,
    ) -> Result<(), super::LayerError> {
        match self {
            LayerEnum::Gkr(layer) => layer.verify_rounds(claim, sumcheck_rounds, transcript),
            LayerEnum::MulGate(layer) => layer.verify_rounds(claim, sumcheck_rounds, transcript),
            LayerEnum::MulGateBatched(layer) => {
                layer.verify_rounds(claim, sumcheck_rounds, transcript)
            }
            LayerEnum::AddGate(layer) => layer.verify_rounds(claim, sumcheck_rounds, transcript),
            LayerEnum::AddGateBatched(layer) => {
                layer.verify_rounds(claim, sumcheck_rounds, transcript)
            }
            LayerEnum::EmptyLayer(layer) => layer.verify_rounds(claim, sumcheck_rounds, transcript),
        }
    }

    #[instrument(skip(self), level = "debug", err)]
    fn get_claims(&self) -> Result<Vec<(super::LayerId, super::Claim<F>)>, super::LayerError> {
        match self {
            LayerEnum::Gkr(layer) => layer.get_claims(),
            LayerEnum::MulGate(layer) => layer.get_claims(),
            LayerEnum::MulGateBatched(layer) => layer.get_claims(),
            LayerEnum::AddGate(layer) => layer.get_claims(),
            LayerEnum::AddGateBatched(layer) => layer.get_claims(),
            LayerEnum::EmptyLayer(layer) => layer.get_claims(),
        }
    }

    fn id(&self) -> &super::LayerId {
        match self {
            LayerEnum::Gkr(layer) => layer.id(),
            LayerEnum::MulGate(layer) => layer.id(),
            LayerEnum::MulGateBatched(layer) => layer.id(),
            LayerEnum::AddGate(layer) => layer.id(),
            LayerEnum::AddGateBatched(layer) => layer.id(),
            LayerEnum::EmptyLayer(layer) => layer.id(),
        }
    }

    fn new<L: super::LayerBuilder<F>>(builder: L, id: super::LayerId) -> Self
    where
        Self: Sized,
    {
        LayerEnum::Gkr(GKRLayer::new(builder, id))
    }

    fn get_enum(self) -> LayerEnum<F, Self::Transcript> {
        self
    }

    /// NOTE: This function is effectively deprecated!!!
    #[instrument(skip(self), level = "debug", err)]
    fn get_wlx_evaluations(
        &self,
        claim_vecs: Vec<Vec<F>>,
        claimed_vals: &mut Vec<F>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, super::claims::ClaimError> {
        match self {
            LayerEnum::Gkr(layer) => {
                layer.get_wlx_evaluations(claim_vecs, claimed_vals, num_claims, num_idx)
            }
            LayerEnum::MulGate(layer) => {
                layer.get_wlx_evaluations(claim_vecs, claimed_vals, num_claims, num_idx)
            }
            LayerEnum::MulGateBatched(layer) => {
                layer.get_wlx_evaluations(claim_vecs, claimed_vals, num_claims, num_idx)
            }
            LayerEnum::AddGate(layer) => {
                layer.get_wlx_evaluations(claim_vecs, claimed_vals, num_claims, num_idx)
            }
            LayerEnum::AddGateBatched(layer) => {
                layer.get_wlx_evaluations(claim_vecs, claimed_vals, num_claims, num_idx)
            }
            LayerEnum::EmptyLayer(layer) => {
                layer.get_wlx_evaluations(claim_vecs, claimed_vals, num_claims, num_idx)
            }
        }
    }
}

impl<F: FieldExt, Tr: Transcript<F>> LayerEnum<F, Tr> {
    ///Gets the size of the Layer as a whole in terms of number of bits
    pub(crate) fn layer_size(&self) -> usize {
        let expression = match self {
            LayerEnum::Gkr(layer) => &layer.expression,
            LayerEnum::EmptyLayer(layer) => &layer.expr,
            LayerEnum::AddGate(_) | LayerEnum::AddGateBatched(_) | LayerEnum::MulGate(_) | LayerEnum::MulGateBatched(_) => unimplemented!(),
        };

        expression.get_expression_size(0)
    }
}
