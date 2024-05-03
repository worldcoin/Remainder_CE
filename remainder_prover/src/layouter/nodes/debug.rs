//! Contains nodes that apply debug labels to the circuit DAG

use super::{CircuitNode, Context, NodeId};

/// A node in the DAG used for debugging purposes only which
/// applies a label to a given set of nodes
#[derive(Debug, Clone)]
pub struct DebugNode {
    id: NodeId,
    label: String,
    sources: Vec<NodeId>,
}

impl CircuitNode for DebugNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        self.sources.clone()
    }
}

impl DebugNode {
    /// Creates a new DebugNode w/ a label applied to the given set of Nodes
    pub fn new(ctx: &Context, label: String, nodes: &[&dyn CircuitNode]) -> Self {
        Self {
            id: ctx.get_new_id(),
            label,
            sources: nodes.into_iter().map(|node| node.id()).collect(),
        }
    }

    /// Adds a new CircuitNode to the set of Nodes this Debug label applies to
    pub fn add_node(&mut self, node: &impl CircuitNode) {
        self.sources.push(node.id());
    }

    /// Appends an additional set of CircuitNodes to the set of Nodes this Debug label applies to
    pub fn add_nodes(&mut self, nodes: &[&dyn CircuitNode]) {
        self.sources.extend(nodes.into_iter().map(|node| node.id()))
    }
}
