//! Nodes that represent inputs to a circuit in the circuit DAG

use std::marker::PhantomData;

use remainder_shared_types::FieldExt;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    mle::evals::MultilinearExtension,
};

use super::{CircuitNode, ClaimableNode, Context, NodeId};

/// A node that represents some Data that will eventually be added to an InputLayer
#[derive(Debug, Clone)]
pub struct InputShred<F> {
    id: NodeId,
    parent: Option<NodeId>,
    data: MultilinearExtension<F>,
}

impl<F: FieldExt> CircuitNode for InputShred<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        if let Some(parent) = self.parent {
            vec![parent]
        } else {
            vec![]
        } // ende: I think this is correct, but I'm not sure
    }
}

impl<F: FieldExt> ClaimableNode for InputShred<F> {
    type F = F;

    fn get_data(&self) -> &MultilinearExtension<Self::F> {
        &self.data
    }

    fn get_expr(&self) -> Expression<Self::F, AbstractExpr> {
        Expression::<F, AbstractExpr>::mle(self.id)
    }
}

impl<F: FieldExt> InputShred<F> {
    /// Creates a new InputShred from data
    ///
    /// Specifying a source indicates to the layouter that this
    /// InputShred should be appended to the source when laying out
    pub fn new(
        ctx: &Context,
        data: MultilinearExtension<F>,
        source: Option<&InputLayerNode<F>>,
    ) -> Self {
        let id = ctx.get_new_id();
        let parent = source.map(CircuitNode::id);

        InputShred { id, parent, data }
    }
}

/// An enum representing the different types
/// of InputLayer an InputLayerNode can be compiled into
#[derive(Debug, Clone)]
pub enum InputLayerType {
    ///An InputLayer that will be compiled into a `LigeroInputLayer`
    LigeroInputLayer,
    PublicInputLayer,
    Default,
}

#[derive(Debug, Clone)]
/// A node that represents an InputLayer
///
/// TODO! probably split this up into more node types
/// that indicate different things to the layouter
pub struct InputLayerNode<F> {
    id: NodeId,
    children: Vec<InputShred<F>>,
    pub(in crate::layouter) input_layer_type: InputLayerType,
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
}

impl<F: FieldExt> ClaimableNode for InputLayerNode<F> {
    type F = F;

    fn get_data(&self) -> &MultilinearExtension<Self::F> {
        todo!()
        // ende's comment here: maybe InputLayerNode also needs a data field?
        // i.e. after combining the shreds, appending all the necessary prefix bits, etc.
    }

    fn get_expr(&self) -> Expression<Self::F, AbstractExpr> {
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
        }
    }

    /// A method to add an InputShred to this InputLayerNode
    pub fn add_shred(&mut self, new_shred: InputShred<F>) {
        self.children.push(new_shred);
    }
}
