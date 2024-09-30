//! Module for nodes that can be added to a circuit DAG

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub use itertools::Either;
pub use remainder_shared_types::{Field, Fr};
use serde::{Deserialize, Serialize};

use crate::expression::{abstract_expr::AbstractExpr, generic_expr::Expression};
use crate::layer::layer_enum::LayerDescriptionEnum;
use crate::layer::LayerId;

use super::layouting::{CircuitDescriptionMap, DAGError};

pub mod circuit_inputs;
pub mod circuit_outputs;
pub mod debug;
pub mod fiat_shamir_challenge;
pub mod gate;
pub mod identity_gate;
pub mod lookup;
pub mod matmult;
pub mod node_enum;
pub mod sector;
pub mod split_node;

/// Container of global context for node creation
///
/// Contains a consistently incrementing Id to prevent
/// collisions in node id creation
#[derive(Debug, Default, Clone)]
pub struct Context(Arc<AtomicU64>);

impl Context {
    /// Creates an empty Context
    pub fn new() -> Self {
        Self(Arc::new(AtomicU64::new(0)))
    }

    /// Retrieves a new node id from the context
    /// that is guaranteed to be unique.
    pub fn get_new_id(&self) -> NodeId {
        let id = self.0.fetch_add(1, Ordering::Relaxed);
        NodeId(id)
    }
}

/// The circuit-unique ID for each node
#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy, Ord, PartialOrd, Serialize, Deserialize)]
pub struct NodeId(u64);

impl NodeId {
    /// Creates a new NodeId from a Context
    pub fn new(ctx: &Context) -> Self {
        ctx.get_new_id()
    }

    /// Creates a new NodeId from a u64, for testing only
    #[cfg(test)]
    pub fn new_unsafe(id: u64) -> Self {
        Self(id)
    }

    /// creates an [Expression<F, AbstractExpr>] from this NodeId
    pub fn expr<F: Field>(self) -> Expression<F, AbstractExpr> {
        Expression::<F, AbstractExpr>::mle(self)
    }
}

/// A Node in the directed acyclic graph (DAG). The directed edges in the DAG model the dependencies
/// between nodes, with the dependent node being the target of the edge and the dependency being the
/// source.
pub trait CircuitNode {
    /// The unique ID of this node
    fn id(&self) -> NodeId;

    /// Return the "sub-nodes" of this node.  These are nodes that are "owned" by this node.  Note
    /// that this is not a relationship in the DAG.
    /// e.g. [InputLayerNode] owns the [InputShredNode]s that are its subnodes.
    fn subnodes(&self) -> Option<Vec<NodeId>> {
        None
    }
    /// Return the ids of the nodes that this node depends upon, i.e. nodes whose values must be
    /// known before the values of this node can be node.  These are the source nodes of the
    /// directed edges of the DAG that terminate at this node.
    fn sources(&self) -> Vec<NodeId>;

    /// Get the number of variables used to represent the data in this node.
    fn get_num_vars(&self) -> usize;
}

/// A Node that contains the information neccessary to Compile itself
///
/// Implement this for any node that does not need additional Layingout before compilation
pub trait CompilableNode<F: Field>: CircuitNode {
    /// Generate the circuit description of a node, which represents the
    /// shape of a certain layer.
    fn generate_circuit_description(
        &self,
        layer_id: &mut LayerId,
        circuit_description_map: &mut CircuitDescriptionMap,
    ) -> Result<Vec<LayerDescriptionEnum<F>>, DAGError>;
}

/// An organized grouping of many node types
pub trait NodeGroup {
    /// The set of nodes this `NodeGroup` supports
    type NodeEnum: CircuitNode;

    /// Sorts a set of `Self::NodeEnum` into a `NodeGroup`
    fn new(nodes: Vec<Self::NodeEnum>) -> Self;
}

/// A Node that can yield a specific node type
pub trait YieldNode<N: CircuitNode>: NodeGroup {
    /// Gets all nodes of the specified type from the `NodeGroup`
    fn get_nodes(&mut self) -> Vec<N>;
}

#[macro_export]
///This macro generates a layer enum that represents all the possible layers
/// Every layer variant of the enum needs to implement Layer, and the enum will also implement Layer and pass methods to it's variants
///
/// Usage:
///
/// layer_enum(EnumName, (FirstVariant: LayerType), (SecondVariant: SecondLayerType), ..)
macro_rules! node_enum {
    ($type_name:ident: $bound:tt, $(($var_name:ident: $variant:ty)),+) => {
        #[derive(Clone, Debug)]
        #[doc = r"Remainder generated trait enum"]
        pub enum $type_name<F: $bound> {
            $(
                #[doc = "Remainder generated node variant"]
                $var_name($variant),
            )*
        }


        impl<F: $bound> $crate::layouter::nodes::CircuitNode for $type_name<F> {
            fn id(&self) -> $crate::layouter::nodes::NodeId {
                match self {
                    $(
                        Self::$var_name(node) => node.id(),
                    )*
                }
            }

            fn subnodes(&self) -> Option<Vec<$crate::layouter::nodes::NodeId>> {
                match self {
                    $(
                        Self::$var_name(node) => node.subnodes(),
                    )*
                }
            }

            fn sources(&self) -> Vec<$crate::layouter::nodes::NodeId> {
                match self {
                    $(
                        Self::$var_name(node) => node.sources(),
                    )*
                }
            }

            fn get_num_vars(&self) -> usize {
                match self {
                    $(
                        Self::$var_name(node) => node.get_num_vars(),
                    )*
                }
            }
        }

        $(
            impl<F: $bound> From<$variant> for $type_name<F> {
                fn from(var: $variant) -> $type_name<F> {
                    Self::$var_name(var)
                }
            }
        )*
    }
}
