//! Module for nodes that can be added to a circuit DAG
pub use itertools::Either;
pub use remainder_shared_types::{Field, Fr};
use serde::{Deserialize, Serialize};

use crate::abstract_expr::AbstractExpression;
use crate::layer::layer_enum::LayerDescriptionEnum;
use crate::layouter::builder::CircuitMap;
use remainder::layer::layer_enum::LayerDescriptionEnum;

use remainder::circuit_building_context::CircuitBuildingContext;

use anyhow::Result;

pub mod circuit_inputs;
pub mod circuit_outputs;
pub mod fiat_shamir_challenge;
pub mod gate;
pub mod identity_gate;
pub mod lookup;
pub mod matmult;
pub mod node_enum;
pub mod sector;
pub mod split_node;

/// The circuit-unique ID for each node
#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy, Ord, PartialOrd, Serialize, Deserialize)]
pub struct NodeId(usize);

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeId {
    /// Creates a new NodeId from the global circuit context.
    pub fn new() -> Self {
        Self(CircuitBuildingContext::next_node_id())
    }

    /// Creates a new NodeId from a usize, for testing only
    #[cfg(test)]
    pub fn new_unsafe(id: usize) -> Self {
        Self(id)
    }

    /// creates an [AbstractExpression<F>] from this NodeId
    pub fn expr<F: Field>(self) -> AbstractExpression<F> {
        AbstractExpression::<F>::mle(self)
<<<<<<< HEAD:remainder_prover/src/layouter/nodes.rs
=======
    }

    /// Obtain the integer value, for printing GkrError messages
    pub fn get_id(self) -> usize {
        self.0
>>>>>>> benny/extract_frontend:remainder_frontend/src/layouter/nodes.rs
    }
}

/// Implement Display for NodeId, so that we can use it in error messages
impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#x}", self.0)
    }
}

/// A Node in the directed acyclic graph (DAG). The directed edges in the DAG model the dependencies
/// between nodes, with the dependent node being the target of the edge and the dependency being the
/// source.
///
/// All "contiguous" MLEs, e.g. sector, gate, matmul, output, input shred
pub trait CircuitNode {
    /// The unique ID of this node
    fn id(&self) -> NodeId;

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
///
/// TODO: Merge this with `circuitnode`
pub trait CompilableNode<F: Field>: CircuitNode {
    /// Generate the circuit description of a node, which represents the
    /// shape of a certain layer.
    fn generate_circuit_description(
        &self,
        circuit_map: &mut CircuitMap,
    ) -> Result<Vec<LayerDescriptionEnum<F>>>;
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
