//! Helper struct that combines multiple `Layer` implementations into
//! a single struct that can represent many types of `Layer`

use remainder_shared_types::FieldExt;

use crate::claims::wlx_eval::{ClaimMle, YieldWLXEvals};
use crate::layer_enum;

use crate::gate_mle::gate::Gate;
use crate::mle::mle_enum::MleEnum;

use super::LayerError;
use super::{regular_layer::RegularLayer, Layer};

use crate::claims::YieldClaim;

layer_enum!(LayerEnum, (Gkr: RegularLayer<F>), (Gate: Gate<F>));

impl<F: FieldExt> LayerEnum<F> {
    ///Gets the size of the Layer as a whole in terms of number of bits
    pub(crate) fn layer_size(&self) -> usize {
        let expression = match self {
            LayerEnum::Gkr(layer) => &layer.expression,
            LayerEnum::Gate(_) => unimplemented!(),
        };

        expression.get_expression_size(0)
    }

    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> Box<dyn std::fmt::Display + 'a> {
        match self {
            LayerEnum::Gkr(layer) => Box::new(layer.expression().circuit_description_fmt()),
            LayerEnum::Gate(gate_layer) => Box::new(gate_layer.circuit_description_fmt()),
        }
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for LayerEnum<F> {
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, crate::claims::ClaimError> {
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
        }
    }
}

impl<F: FieldExt> YieldClaim<F, ClaimMle<F>> for LayerEnum<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        match self {
            LayerEnum::Gkr(layer) => layer.get_claims(),
            LayerEnum::Gate(layer) => layer.get_claims(),
        }
    }
}
