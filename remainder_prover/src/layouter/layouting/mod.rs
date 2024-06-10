pub mod core_layouter;
use std::collections::HashMap;

use ndarray::Array2;
use remainder_shared_types::FieldExt;

use crate::{
    layer::LayerId,
    prover::{proof_system::ProofSystem, Witness},
};

use super::{
    component::Component,
    nodes::{CircuitNode, NodeId},
};

// pub struct CircuitMap {
//     input_layers: Array2<CircuitLocation>,
//     layers: Array2<CircuitLocation>,
// }

pub struct CircuitMap(HashMap<NodeId, CircuitLocation>);

pub struct CircuitLocation {
    layer_id: LayerId,
    prefix_bits: Vec<bool>,
}

pub trait Layouter<F: FieldExt, N: CircuitNode> {
    type ProofSystem: ProofSystem<F>;
    fn layout<C: Component<N>>(pre_circuit: C) -> Witness<F, Self::ProofSystem>;
}
