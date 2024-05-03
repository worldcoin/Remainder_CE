//! A default Enum for a representation of all possible DAG Nodes

use remainder_shared_types::FieldExt;

use crate::node_enum;

use super::{
    circuit_inputs::{InputLayerNode, InputShred},
    circuit_outputs::OutputNode,
    debug::DebugNode,
};

node_enum!(NodeEnum: FieldExt, (InputShred: InputShred<F>), (InputLayer: InputLayerNode<F>), (Output: OutputNode<F>), (Debug: DebugNode));
