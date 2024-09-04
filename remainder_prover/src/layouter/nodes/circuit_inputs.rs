//! Nodes that represent inputs to a circuit in the circuit DAG\

mod compile_inputs;

use std::marker::PhantomData;

use remainder_shared_types::FieldExt;

use super::{CircuitNode, Context, NodeId};

/// A node that represents some Data that will eventually be added to an InputLayer
#[derive(Debug, Clone)]
pub struct InputShred<F> {
    id: NodeId,
    parent: NodeId,
    num_vars: usize,
    _marker: PhantomData<F>,
}

impl<F: FieldExt> CircuitNode for InputShred<F> {
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

impl<F: FieldExt> InputShred<F> {
    /// Creates a new InputShred from data
    ///
    /// Specifying a source indicates to the layouter that this
    /// InputShred should be appended to the source when laying out
    pub fn new(ctx: &Context, num_vars: usize, source: &InputLayerNode<F>) -> Self {
        let id = ctx.get_new_id();
        let parent = source.id();

        InputShred {
            id,
            parent,
            num_vars,
            _marker: PhantomData,
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
pub struct InputLayerNode<F: FieldExt> {
    id: NodeId,
    children: Vec<InputShred<F>>,
    pub(in crate::layouter) input_layer_type: InputLayerType,
    _marker: PhantomData<F>,
}

impl<F: FieldExt> CircuitNode for InputLayerNode<F> {
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

impl<F: FieldExt> InputLayerNode<F> {
    /// A constructor for an InputLayerNode. Can either be initialized empty
    /// or with some children.
    pub fn new(
        ctx: &Context,
        children: Option<Vec<InputShred<F>>>,
        input_layer_type: InputLayerType,
    ) -> Self {
        InputLayerNode {
            id: ctx.get_new_id(),
            children: children.unwrap_or_default(),
            input_layer_type,
            _marker: PhantomData,
        }
    }

    /// A method to add an InputShred to this InputLayerNode
    pub fn add_shred(&mut self, new_shred: InputShred<F>) {
        self.children.push(new_shred);
    }
}
