use rayon::iter::empty;
use serde::{Deserialize, Serialize};

use remainder_shared_types::{transcript::Transcript, FieldExt};
use tracing::instrument;

use crate::gate::{
    addgate::AddGate, batched_addgate::AddGateBatched, batched_mulgate::MulGateBatched,
    mulgate::MulGate,
};
use crate::layer_enum;
use crate::mle::dense::DenseMleRef;
use crate::mle::mle_enum::MleEnum;

use super::{empty_layer::EmptyLayer, GKRLayer, Layer};

use super::claims::Claim;

use std::fmt;

// #[derive(Serialize, Deserialize, Clone)]
// #[serde(bound = "F: FieldExt")]
// ///An enum representing all the possible kinds of Layers
// pub enum LayerEnum<F: FieldExt> {
//     ///A standard `GKRLayer`
//     Gkr(GKRLayer<F>),
//     /// A Mulgate
//     MulGate(MulGate<F>),
//     /// An Addition Gate
//     AddGate(AddGate<F>),
//     /// Batched AddGate
//     AddGateBatched(AddGateBatched<F>),
//     /// Batched MulGate
//     MulGateBatched(MulGateBatched<F>),
//     /// Layer with zero variables within it
//     EmptyLayer(EmptyLayer<F>),
// }

layer_enum!(LayerEnum, 
    (Gkr: GKRLayer<F>),
    (MulGate: MulGate<F>),
    (AddGate: AddGate<F>),
    (AddGateBatched: AddGateBatched<F>),
    (MulGateBatched: MulGateBatched<F>),
    (EmptyLayer: EmptyLayer<F>)
);

impl<F: FieldExt> fmt::Debug for LayerEnum<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LayerEnum::Gkr(_) => write!(f, "GKR Layer"),
            LayerEnum::MulGate(_) => write!(f, "MulGate"),
            LayerEnum::AddGate(_) => write!(f, "AddGate"),
            LayerEnum::AddGateBatched(_) => write!(f, "AddGateBatched"),
            LayerEnum::MulGateBatched(_) => write!(f, "MulGateBatched"),
            LayerEnum::EmptyLayer(_) => write!(f, "EmptyLayer"),
        }
    }
}

impl<F: FieldExt> LayerEnum<F> {
    ///Gets the size of the Layer as a whole in terms of number of bits
    pub(crate) fn layer_size(&self) -> usize {
        let expression = match self {
            LayerEnum::Gkr(layer) => &layer.expression,
            LayerEnum::EmptyLayer(layer) => &layer.expr,
            LayerEnum::AddGate(_)
            | LayerEnum::AddGateBatched(_)
            | LayerEnum::MulGate(_)
            | LayerEnum::MulGateBatched(_) => unimplemented!(),
        };

        expression.get_expression_size(0)
    }

    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> Box<dyn std::fmt::Display + 'a> {
        match self {
            LayerEnum::Gkr(layer) => Box::new(layer.expression().circuit_description_fmt()),
            LayerEnum::MulGate(mulgate_layer) => Box::new(mulgate_layer.circuit_description_fmt()),
            LayerEnum::AddGate(addgate_layer) => Box::new(addgate_layer.circuit_description_fmt()),
            LayerEnum::AddGateBatched(addgate_layer_batched) => {
                Box::new(addgate_layer_batched.circuit_description_fmt())
            }
            LayerEnum::MulGateBatched(mulgate_layer_batched) => {
                Box::new(mulgate_layer_batched.circuit_description_fmt())
            }
            LayerEnum::EmptyLayer(empty_layer) => {
                Box::new(empty_layer.expression().circuit_description_fmt())
            }
        }
    }
}
