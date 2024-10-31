//! Nodes that represent inputs to a circuit in the circuit DAG.

/// The module which contains functions that combine input data in order
/// to create one layerwise bookkeeping table.
pub mod compile_inputs;

use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

use crate::{layer::LayerId, mle::evals::MultilinearExtension};

use super::{CircuitNode, NodeId};

/// A struct that represents input data that will be used to instantiate a
/// [remainder::prover::GKRCircuitDescription] in order to generate a full circuit.
#[derive(Debug, Clone)]
pub struct InputLayerNodeData<F: Field> {
    /// The [InputLayerNode] ID in the circuit building process that corresponds to
    /// this data.
    pub corresponding_input_node_id: NodeId,
    /// The vector of data that goes in this input layer, as [InputShredData].
    pub data: Vec<InputShredData<F>>,
}

impl<F: Field> InputLayerNodeData<F> {
    /// Constructor for [InputLayerNodeData], using the corresponding fields as
    /// parameters.
    pub fn new(corresponding_input_node_id: NodeId, data: Vec<InputShredData<F>>) -> Self {
        Self {
            corresponding_input_node_id,
            data,
        }
    }
}

#[derive(Debug, Clone)]
/// A struct that represents data that corresponds to an [InputShred].
pub struct InputShredData<F: Field> {
    /// The corresponding input shred ID that this data belongs to.
    pub corresponding_input_shred_id: NodeId,
    /// The data itself, as a [MultilinearExtension].
    pub data: MultilinearExtension<F>,
}

impl<F: Field> InputShredData<F> {
    /// Constructor for [InputShredData], with the fields as parameters.
    pub fn new(corresponding_input_shred_id: NodeId, data: MultilinearExtension<F>) -> Self {
        Self {
            corresponding_input_shred_id,
            data,
        }
    }
}

/// A struct that represents the description of the data (shape in terms of `num_vars`)
/// that will be added to an [InputLayerNode].
#[derive(Debug, Clone)]
pub struct InputShred {
    id: NodeId,
    parent: NodeId,
    num_vars: usize,
}

impl CircuitNode for InputShred {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![]
    }

    fn get_num_vars(&self) -> usize {
        self.num_vars
    }
}

impl InputShred {
    /// Creates a new InputShred from data
    ///
    /// Specifying a source indicates to the layouter that this
    /// InputShred should be appended to the source when laying out
    pub fn new(num_vars: usize, source: &InputLayerNode) -> Self {
        let id = NodeId::new();
        let parent = source.id();

        InputShred {
            id,
            parent,
            num_vars,
        }
    }

    /// Gets the parent NodeId of this InputShred. The InputLayerNode this InputShred will eventually be added to
    pub fn get_parent(&self) -> NodeId {
        self.parent
    }
}

/// An enum representing the data type that can go in
/// a Hyrax input layer, if it is not scalar field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HyraxInputDType {
    /// If the input MLE coefficients are u8s.
    U8,
    /// If the input MLE coefficients are i8s.
    I8,
}

#[derive(Debug, Clone)]
/// A node that represents an InputLayer
pub struct InputLayerNode {
    id: NodeId,
    input_layer_id: LayerId,
    input_shreds: Vec<InputShred>,
}

impl CircuitNode for InputLayerNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn subnodes(&self) -> Option<Vec<NodeId>> {
        Some(self.input_shreds.iter().map(CircuitNode::id).collect())
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![]
    }

    fn get_num_vars(&self) -> usize {
        todo!()
    }
}

impl InputLayerNode {
    /// A constructor for an InputLayerNode. Can either be initialized empty
    /// or with some InputShreds.
    pub fn new(input_shreds: Option<Vec<InputShred>>) -> Self {
        InputLayerNode {
            id: NodeId::new(),
            input_layer_id: LayerId::next_input_layer_id(),
            input_shreds: input_shreds.unwrap_or_default(),
        }
    }

    /// A method to add an InputShred to this InputLayerNode
    pub fn add_shred(&mut self, shred: InputShred) {
        self.input_shreds.push(shred);
    }

    /// Get the input layer id of this InputLayerNode.
    pub fn input_layer_id(&self) -> LayerId {
        self.input_layer_id
    }
}
