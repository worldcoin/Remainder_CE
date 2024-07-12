//! Module for nodes that can be added to a circuit DAG

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub use itertools::Either;
pub use remainder_shared_types::{FieldExt, Fr};
use serde::{Deserialize, Serialize};

use crate::prover::proof_system::ProofSystem;
use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    mle::evals::MultilinearExtension,
};

use super::compiling::WitnessBuilder;
use super::layouting::{CircuitMap, DAGError};

pub mod circuit_inputs;
pub mod circuit_outputs;
pub mod debug;
pub mod gate;
pub mod identity_gate;
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
    pub fn expr<F: FieldExt>(self) -> Expression<F, AbstractExpr> {
        Expression::<F, AbstractExpr>::mle(self)
    }
}

/// A Node that can exist w/ dependencies in the circuit DAG
pub trait CircuitNode {
    /// The unique ID of this node
    fn id(&self) -> NodeId;

    /// The children of this node
    fn children(&self) -> Option<Vec<NodeId>> {
        None
    }
    /// The sources of this node in the DAG
    fn sources(&self) -> Vec<NodeId>;
}

/// A circuit node that can have a Claim made against it.
///
/// Yields the MLE that any claim made on this node would be the evaluation of
pub trait ClaimableNode: CircuitNode {
    /// The Field this node uses
    type F: FieldExt;
    /// A function for getting the MLE that this node generates in the circuit
    ///
    /// Any claim made against this node will be evaluated on this MLE
    fn get_data(&self) -> &MultilinearExtension<Self::F>;

    /// An abstract expression node that will make a claim on this node
    fn get_expr(&self) -> Expression<Self::F, AbstractExpr>;
}

/// A Node that contains the information neccessary to Compile itself
///
/// Implement this for any node that does not need additional Layingout before compilation
pub trait CompilableNode<F: FieldExt, Pf: ProofSystem<F>>: CircuitNode {
    /// Compiles the node by adding any layers neccessary to the `WitnessBuilder`
    ///
    /// If any `ClaimableNode` is added to the witness it is the responsibility
    /// of this function to add that `NodeId` to the `CircuitMap`
    fn compile<'a>(
        &'a self,
        witness_builder: &mut WitnessBuilder<F, Pf>,
        circuit_map: &mut CircuitMap<'a, F>,
    ) -> Result<(), DAGError>;
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

            fn children(&self) -> Option<Vec<$crate::layouter::nodes::NodeId>> {
                match self {
                    $(
                        Self::$var_name(node) => node.children(),
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
