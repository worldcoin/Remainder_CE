//! A wrapper type that makes working with variants of InputLayer easier.

use remainder_shared_types::Field;

use crate::{claims::wlx_eval::YieldWLXEvals, input_layer_enum, layer::LayerId};

use super::{
    hyrax_placeholder_input_layer::HyraxPlaceholderInputLayer,
    hyrax_precommit_placeholder_input_layer::HyraxPrecommitPlaceholderInputLayer,
    ligero_input_layer::LigeroInputLayer, public_input_layer::PublicInputLayer,
    random_input_layer::RandomInputLayer, InputLayer,
};

input_layer_enum!(
    InputLayerEnum,
    (LigeroInputLayer: LigeroInputLayer<F>),
    (PublicInputLayer: PublicInputLayer<F>),
    (RandomInputLayer: RandomInputLayer<F>),
    (HyraxPlaceholderInputLayer: HyraxPlaceholderInputLayer<F>),
    (HyraxPrecommitPlaceholderInputLayer: HyraxPrecommitPlaceholderInputLayer<F>)
);

impl<F: Field> InputLayerEnum<F> {
    /// This function sets the layer ID of the corresponding input layer.
    pub fn set_layer_id(&mut self, layer_id: LayerId) {
        match self {
            InputLayerEnum::LigeroInputLayer(layer) => layer.layer_id = layer_id,
            InputLayerEnum::PublicInputLayer(layer) => layer.layer_id = layer_id,
            InputLayerEnum::RandomInputLayer(layer) => layer.layer_id = layer_id,
            InputLayerEnum::HyraxPlaceholderInputLayer(layer) => layer.layer_id = layer_id,
            InputLayerEnum::HyraxPrecommitPlaceholderInputLayer(layer) => layer.layer_id = layer_id,
        }
    }
}

impl<F: Field> YieldWLXEvals<F> for InputLayerEnum<F> {
    /// Get the evaluations of the bookkeeping table of this layer over enumerated points
    /// in order to perform claim aggregation.
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
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
            InputLayerEnum::HyraxPlaceholderInputLayer(layer) => layer.get_wlx_evaluations(
                claim_vecs,
                claimed_vals,
                claimed_mles,
                num_claims,
                num_idx,
            ),
            InputLayerEnum::HyraxPrecommitPlaceholderInputLayer(layer) => layer
                .get_wlx_evaluations(claim_vecs, claimed_vals, claimed_mles, num_claims, num_idx),
        }
    }
}
