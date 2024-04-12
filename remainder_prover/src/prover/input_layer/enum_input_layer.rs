//! A wrapper type that makes working with variants of InputLayer easier

use remainder_shared_types::FieldExt;


use crate::{
    claims::{wlx_eval::YieldWLXEvals},
    input_layer_enum,
    layer::LayerId,
};

use super::{
    ligero_input_layer::{LigeroInputLayer},
    public_input_layer::PublicInputLayer,
    random_input_layer::RandomInputLayer,
    InputLayer,
};

input_layer_enum!(
    InputLayerEnum,
    (LigeroInputLayer: LigeroInputLayer<F>),
    (PublicInputLayer: PublicInputLayer<F>),
    (RandomInputLayer: RandomInputLayer<F>)
);

impl<F: FieldExt> InputLayerEnum<F> {
    pub fn set_layer_id(&mut self, layer_id: LayerId) {
        match self {
            InputLayerEnum::LigeroInputLayer(layer) => layer.layer_id = layer_id,
            InputLayerEnum::PublicInputLayer(layer) => layer.layer_id = layer_id,
            InputLayerEnum::RandomInputLayer(layer) => layer.layer_id = layer_id,
        }
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for InputLayerEnum<F> {
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        claimed_mles: Vec<crate::mle::mle_enum::MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, crate::claims::ClaimError> {
        match self {
            InputLayerEnum::LigeroInputLayer(layer) => layer.get_wlx_evaluations(
                claim_vecs,
                claimed_vals,
                claimed_mles,
                num_claims,
                num_idx,
            ),
            InputLayerEnum::PublicInputLayer(layer) => layer.get_wlx_evaluations(
                claim_vecs,
                claimed_vals,
                claimed_mles,
                num_claims,
                num_idx,
            ),
            InputLayerEnum::RandomInputLayer(layer) => layer.get_wlx_evaluations(
                claim_vecs,
                claimed_vals,
                claimed_mles,
                num_claims,
                num_idx,
            ),
        }
    }
}
