//! The nodes that represent circuit outputs in the circuit DAG

use std::marker::PhantomData;

use remainder_shared_types::FieldExt;

use super::{CircuitNode, ClaimableNode, Context, NodeId};

#[derive(Debug, Clone)]
/// the node that represents the output of a circuit
pub struct OutputNode<F> {
    id: NodeId,
    source: NodeId,
    _marker: PhantomData<F>,
}

impl<F: FieldExt> CircuitNode for OutputNode<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.source]
    }
}

impl<F: FieldExt> OutputNode<F> {
    /// Creates a new OutputNode from a source
    pub fn new(ctx: &Context, source: &impl ClaimableNode<F = F>) -> Self {
        Self {
            id: ctx.get_new_id(),
            source: source.id(),
            _marker: PhantomData,
        }
    }
}
