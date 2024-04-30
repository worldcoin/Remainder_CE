//! Module for nodes that can be added to a circuit DAG

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct NodeId(u64);

pub trait CircuitNode {
    fn id(&self) -> NodeId;

    fn children(&self) -> Option<Vec<NodeId>> {
        None
    }

    fn sources(&self) -> Vec<NodeId>;
}
