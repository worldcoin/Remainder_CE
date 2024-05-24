use ndarray::Array2;

use super::nodes::NodeId;

pub struct CircuitMap {
    input_layers: Array2<CircuitLocation>,
    layers: Array2<CircuitLocation>,
}

pub struct CircuitLocation {
    id: NodeId,
    prefix_bits: Vec<bool>,
}
