//! Module for supplying utilities for creating and combining sets of Nodes
pub mod equality_check;
use super::nodes::CircuitNode;

/// A type that represents a collection of CircuitNodes, ready to be yielded
pub trait Component<N: CircuitNode> {
    /// Yields all CircuitNodes this Component owns so that they can be combined or modified
    fn yield_nodes(self) -> Vec<N>;
}

/// Default implementor of `Component`
///
/// keeps track of an unordered list of nodes,
/// and can consume other components to add to this list
#[derive(Clone, Debug, Default)]
pub struct ComponentSet<N> {
    nodes: Vec<N>,
}

impl<N: CircuitNode> ComponentSet<N> {
    /// Creates an empty `ComponentSet`
    pub fn new() -> Self {
        ComponentSet { nodes: vec![] }
    }

    /// Adds the nodes a `Component` yields to this `ComponentSet`
    pub fn add_component<C: Component<N>>(&mut self, other: C) {
        self.nodes.append(&mut other.yield_nodes())
    }

    /// Creates a new ComponentSet with some Nodes added explicitely
    pub fn new_raw(nodes: Vec<N>) -> Self {
        Self { nodes }
    }
}

impl<N: CircuitNode> Component<N> for ComponentSet<N> {
    fn yield_nodes(self) -> Vec<N> {
        self.nodes
    }
}
