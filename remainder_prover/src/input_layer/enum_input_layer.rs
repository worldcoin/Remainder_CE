//! A wrapper type that makes working with variants of InputLayer easier.

use remainder_shared_types::Field;

use crate::{
    claims::wlx_eval::YieldWLXEvals,
    input_layer_enum,
    mle::dense::DenseMle,
};

use super::{
    ligero_input_layer::LigeroInputLayer,
    public_input_layer::PublicInputLayer, InputLayerTrait
};


input_layer_enum!(
    InputLayerEnum,
    (LigeroInputLayer: LigeroInputLayer<F>),
    (PublicInputLayer: PublicInputLayer<F>)
);

impl<F: Field> YieldWLXEvals<F> for InputLayerEnum<F> {
    /// Get the evaluations of the bookkeeping table of this layer over enumerated points
    /// in order to perform claim aggregation.
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        claimed_mles: Vec<DenseMle<F>>,
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
        }
    }
}
