//! Module for nodes that can be added to a circuit DAG

use std::sync::{Arc, Mutex};

use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    mle::evals::MultilinearExtension,
};

pub mod circuit_inputs;
pub mod circuit_outputs;
pub mod debug;
pub mod node_enum;
pub mod sector;

/// Container of global context for node creation
///
/// Contains a consistently incrementing Id to prevent
/// collisions in node id creation
#[derive(Clone, Debug)]
pub struct Context(Arc<Mutex<u64>>);

impl Context {
    /// Creates an empty Context
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(0)))
    }

    /// Retrieves a new node id from the context
    /// that is guaranteed to be unique.
    ///
    /// Will block thread on the Mutex lock
    pub fn get_new_id(&self) -> NodeId {
        // Getting the lock will only return Err if another thread
        // panicked instead of dropping the lock. Since we're not interested
        // in recovery in such a case, unwrapping here is fine.
        let mut id = self.0.lock().unwrap();
        *id += 1;
        NodeId(*id)
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
            fn id(&self) -> crate::layouter::nodes::NodeId {
                match self {
                    $(
                        Self::$var_name(node) => node.id(),
                    )*
                }
            }

            fn children(&self) -> Option<Vec<crate::layouter::nodes::NodeId>> {
                match self {
                    $(
                        Self::$var_name(node) => node.children(),
                    )*
                }
            }

            fn sources(&self) -> Vec<crate::layouter::nodes::NodeId> {
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
