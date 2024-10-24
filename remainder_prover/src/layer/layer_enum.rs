//! Helper struct that combines multiple `Layer` implementations into
//! a single struct that can represent many types of `Layer`

use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

use crate::layer_enum;

use super::gate::{GateLayer, GateLayerDescription, VerifierGateLayer};
use super::identity_gate::{IdentityGate, IdentityGateLayerDescription, VerifierIdentityGateLayer};
use super::matmult::{MatMult, MatMultLayerDescription, VerifierMatMultLayer};
use super::regular_layer::{RegularLayer, RegularLayerDescription, VerifierRegularLayer};

layer_enum!(Layer, (Regular: RegularLayer<F>), (Gate: GateLayer<F>), (IdentityGate: IdentityGate<F>), (MatMult: MatMult<F>));

#[derive(Serialize, Deserialize, Debug, Hash)]
#[serde(bound = "F: Field")]
/// An enum representing the different types of descriptions for each layer,
/// each description containing the shape information of the corresponding layer.
pub enum LayerDescriptionEnum<F: Field> {
    /// The circuit description for a regular layer variant.
    Regular(RegularLayerDescription<F>),
    /// The circuit description for a gate layer variant.
    Gate(GateLayerDescription<F>),
    /// The circuit description for a identity gate layer variant.
    IdentityGate(IdentityGateLayerDescription<F>),
    /// The circuit description for a matmult layer variant.
    MatMult(MatMultLayerDescription<F>),
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "F: Field")]
/// An enum representing the different types of fully bound layers.
pub enum VerifierLayerEnum<F: Field> {
    /// The fully bound representation of a regular layer.
    Regular(VerifierRegularLayer<F>),
    /// The fully bound representation of a gate layer.
    Gate(VerifierGateLayer<F>),
    /// The fully bound representation of an identity gate layer.
    IdentityGate(VerifierIdentityGateLayer<F>),
    /// The fully bound representation of a matmult layer.
    MatMult(VerifierMatMultLayer<F>),
}

impl<F: Field> LayerEnum<F> {
    ///Gets the size of the Layer as a whole in terms of number of bits
    pub fn layer_size(&self) -> usize {
        let expression = match self {
            LayerEnum::Regular(layer) => &layer.expression,
            LayerEnum::Gate(_) => unimplemented!(),
            LayerEnum::IdentityGate(_) => unimplemented!(),
            LayerEnum::MatMult(_) => unimplemented!(),
        };

        expression.get_expression_size()
    }
}
