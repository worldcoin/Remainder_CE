//! Helper struct that combines multiple `Layer` implementations into
//! a single struct that can represent many types of `Layer`

use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};

use crate::claims::wlx_eval::{ClaimMle, YieldWLXEvals};
use crate::claims::YieldClaim;
use crate::layer_enum;

use super::gate::{CircuitGateLayer, GateLayer, VerifierGateLayer};
use super::identity_gate::{CircuitIdentityGateLayer, IdentityGate, VerifierIdentityGateLayer};
use super::matmult::{CircuitMatMultLayer, MatMult, VerifierMatMultLayer};
use super::regular_layer::{CircuitRegularLayer, RegularLayer, VerifierRegularLayer};
use crate::mle::mle_enum::MleEnum;

use super::LayerError;

layer_enum!(LayerEnum, (Regular: RegularLayer<F>), (Gate: GateLayer<F>), (IdentityGate: IdentityGate<F>), (MatMult: MatMult<F>));

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "F: FieldExt")]
pub enum CircuitLayerEnum<F: FieldExt> {
    Regular(CircuitRegularLayer<F>),
    Gate(CircuitGateLayer<F>),
    IdentityGate(CircuitIdentityGateLayer<F>),
    MatMult(CircuitMatMultLayer<F>),
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "F: FieldExt")]
pub enum VerifierLayerEnum<F: FieldExt> {
    Regular(VerifierRegularLayer<F>),
    Gate(VerifierGateLayer<F>),
    IdentityGate(VerifierIdentityGateLayer<F>),
    MatMult(VerifierMatMultLayer<F>),
}

impl<F: FieldExt> LayerEnum<F> {
    ///Gets the size of the Layer as a whole in terms of number of bits
    pub fn layer_size(&self) -> usize {
        let expression = match self {
            LayerEnum::Regular(layer) => &layer.expression,
            LayerEnum::Gate(_) => unimplemented!(),
            LayerEnum::IdentityGate(_) => unimplemented!(),
            LayerEnum::MatMult(_) => unimplemented!(),
        };

        expression.get_expression_size(0)
    }

    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> Box<dyn std::fmt::Display + 'a> {
        match self {
            LayerEnum::Regular(layer) => Box::new(layer.expression().circuit_description_fmt()),
            LayerEnum::Gate(gate_layer) => Box::new(gate_layer.circuit_description_fmt()),
            LayerEnum::IdentityGate(id_gate_layer) => {
                Box::new(id_gate_layer.circuit_description_fmt())
            }
            LayerEnum::MatMult(mat_mult_layer) => {
                Box::new(mat_mult_layer.circuit_description_fmt())
            }
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
            LayerEnum::Regular(layer) => layer.get_wlx_evaluations(
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
            LayerEnum::IdentityGate(layer) => layer.get_wlx_evaluations(
                claim_vecs,
                claimed_vals,
                claimed_mles,
                num_claims,
                num_idx,
            ),
            LayerEnum::MatMult(layer) => layer.get_wlx_evaluations(
                claim_vecs,
                claimed_vals,
                claimed_mles,
                num_claims,
                num_idx,
            ),
        }
    }
}

impl<F: FieldExt> YieldClaim<ClaimMle<F>> for LayerEnum<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        match self {
            LayerEnum::Regular(layer) => layer.get_claims(),
            LayerEnum::Gate(layer) => layer.get_claims(),
            LayerEnum::IdentityGate(layer) => layer.get_claims(),
            LayerEnum::MatMult(layer) => layer.get_claims(),
        }
    }
}
