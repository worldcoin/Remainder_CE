//! Helper struct that combines multiple `Layer` implementations into
//! a single struct that can represent many types of `Layer`

use remainder_shared_types::extension_field::ExtensionField;
use serde::{Deserialize, Serialize};

use crate::layer_enum;

use super::gate::{GateLayer, GateLayerDescription, VerifierGateLayer};
use super::identity_gate::{IdentityGate, IdentityGateLayerDescription, VerifierIdentityGateLayer};
use super::matmult::{MatMult, MatMultLayerDescription, VerifierMatMultLayer};
use super::regular_layer::{RegularLayer, RegularLayerDescription, VerifierRegularLayer};

layer_enum!(Layer, (Regular: RegularLayer<E>), (Gate: GateLayer<E>), (IdentityGate: IdentityGate<E>), (MatMult: MatMult<E>));

#[derive(Serialize, Deserialize, Debug, Hash, Clone)]
#[serde(bound = "E: ExtensionField")]
/// An enum representing the different types of descriptions for each layer,
/// each description containing the shape information of the corresponding layer.
pub enum LayerDescriptionEnum<E: ExtensionField> {
    /// The circuit description for a regular layer variant.
    Regular(RegularLayerDescription<E>),
    /// The circuit description for a gate layer variant.
    Gate(GateLayerDescription<E>),
    /// The circuit description for a identity gate layer variant.
    IdentityGate(IdentityGateLayerDescription<E>),
    /// The circuit description for a matmult layer variant.
    MatMult(MatMultLayerDescription<E>),
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "E: ExtensionField")]
/// An enum representing the different types of fully bound layers.
pub enum VerifierLayerEnum<E: ExtensionField> {
    /// The fully bound representation of a regular layer.
    Regular(VerifierRegularLayer<E>),
    /// The fully bound representation of a gate layer.
    Gate(VerifierGateLayer<E>),
    /// The fully bound representation of an identity gate layer.
    IdentityGate(VerifierIdentityGateLayer<E>),
    /// The fully bound representation of a matmult layer.
    MatMult(VerifierMatMultLayer<E>),
}
