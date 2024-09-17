//! Nodes that represent inputs to a circuit in the circuit DAG\

pub mod compile_inputs;

use remainder_shared_types::Field;

use crate::{
    input_layer::CommitmentEnum,
    mle::{dense::DenseMle, evals::MultilinearExtension},
};

use super::{CircuitNode, Context, NodeId};

/// A struct that represents input data that will be used to populate a
/// [GKRCircuitDescription] in order to generate a full circuit.
#[derive(Debug, Clone)]
pub struct InputLayerData<F: Field> {
    pub corresponding_input_node_id: NodeId,
    pub data: Vec<InputShredData<F>>,
    pub precommit: Option<CommitmentEnum<F>>,
}

impl<F: Field> InputLayerData<F> {
    pub fn new(
        corresponding_input_node_id: NodeId,
        data: Vec<InputShredData<F>>,
        precommit: Option<CommitmentEnum<F>>,
    ) -> Self {
        Self {
            corresponding_input_node_id,
            data,
            precommit,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InputShredData<F: Field> {
    pub corresponding_input_shred_id: NodeId,
    pub data: MultilinearExtension<F>,
}

impl<F: Field> InputShredData<F> {
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
    pub fn new(ctx: &Context, num_vars: usize, source: &InputLayerNode) -> Self {
        let id = ctx.get_new_id();
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

/// An enum representing the different types
/// of InputLayer an InputLayerNode can be compiled into
#[derive(Debug, Clone)]
pub enum InputLayerType {
    /// An InputLayer that will be compiled into a [LigeroInputLayer]
    LigeroInputLayer((u8, f64)),
    /// An InputLayer that will be compiled into a [PublicInputLayer]
    PublicInputLayer,
}

#[derive(Debug, Clone)]
/// A node that represents an InputLayer
///
/// TODO! probably split this up into more node types
/// that indicate different things to the layouter
pub struct InputLayerNode {
    id: NodeId,
    children: Vec<InputShred>,
    pub(in crate::layouter) input_layer_type: InputLayerType,
}

impl CircuitNode for InputLayerNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn children(&self) -> Option<Vec<NodeId>> {
        Some(self.children.iter().map(CircuitNode::id).collect())
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
    /// or with some children.
    pub fn new(
        ctx: &Context,
        children: Option<Vec<InputShred>>,
        input_layer_type: InputLayerType,
    ) -> Self {
        InputLayerNode {
            id: ctx.get_new_id(),
            children: children.unwrap_or_default(),
            input_layer_type,
        }
    }

    /// A method to add an InputShred to this InputLayerNode
    pub fn add_shred(&mut self, new_shred: InputShred) {
        self.children.push(new_shred);
    }
}
