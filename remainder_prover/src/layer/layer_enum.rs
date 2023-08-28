use serde::{Serialize, Deserialize};

use remainder_shared_types::{FieldExt, transcript::Transcript};

use super::{GKRLayer, Layer};

#[derive(Serialize, Deserialize)]
///An enum representing all the possible kinds of Layers
pub enum LayerEnum<F: FieldExt, Tr: Transcript<F>> {
    ///A standard `GKRLayer`
    Gkr(GKRLayer<F, Tr>),
    ///GateMLE layer
    Gate(AddGate<F, Tr>),
    ///BatchedGateMLE Layer
    GateBatched(AddGateBatched<F, Tr>),
}

impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for LayerEnum<F, Tr> {
    type Transcript = Tr;

    fn prove_rounds(
        &mut self,
        claim: super::Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<crate::prover::SumcheckProof<F>, super::LayerError> {
        match self {
            LayerEnum::Gkr(layer) => layer.prove_rounds(claim, transcript),
        }
    }

    fn verify_rounds(
        &mut self,
        claim: super::Claim<F>,
        sumcheck_rounds: Vec<Vec<F>>,
        transcript: &mut Self::Transcript,
    ) -> Result<(), super::LayerError> {
        match self {
            LayerEnum::Gkr(layer) => layer.verify_rounds(claim, sumcheck_rounds, transcript),
        }
    }

    fn get_claims(&self) -> Result<Vec<(super::LayerId, super::Claim<F>)>, super::LayerError> {
        match self {
            LayerEnum::Gkr(layer) => layer.get_claims(),
        }
    }

    fn id(&self) -> &super::LayerId {
        match self {
            LayerEnum::Gkr(layer) => layer.id(),
        }
    }

    fn new<L: super::LayerBuilder<F>>(builder: L, id: super::LayerId) -> Self
    where
        Self: Sized {
        LayerEnum::Gkr(GKRLayer::new(builder, id))
    }

    fn get_enum(self) -> LayerEnum<F, Self::Transcript> {
        self
    }

    fn get_wlx_evaluations(&self, claim_vecs: Vec<Vec<F>>,
        claimed_vals: &mut Vec<F>,
        num_claims: usize,
        num_idx: usize) -> Result<Vec<F>, super::claims::ClaimError> {
        match self {
            LayerEnum::Gkr(layer) => layer.get_wlx_evaluations(claim_vecs, claimed_vals,num_claims, num_idx),
        }
    }
}