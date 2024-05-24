//! Module for nodes that can be added to a circuit DAG

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

pub use itertools::Either;
use remainder_shared_types::{FieldExt, Fr};

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    mle::evals::MultilinearExtension,
};

use self::{debug::DebugNode, node_enum::NodeEnum};

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
#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy, PartialOrd)]
pub struct NodeId(u64);

impl NodeId {
    pub fn new(ctx: &Context) -> Self {
        ctx.get_new_id()
    }
}

/// A Node that can exist w/ dependencies in the circuit DAG
pub trait CircuitNode {
    fn id(&self) -> NodeId;

    fn children(&self) -> Option<Vec<NodeId>> {
        None
    }

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
        pub enum $type_name<F> {
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

        $(
            impl<F: $bound> $crate::layouter::nodes::MaybeInto<$variant> for $type_name<F> {
                fn maybe_into(self) -> $crate::layouter::nodes::Either<$variant, $type_name<F>> {
                    match self {
                        Self::$var_name(node) => $crate::layouter::nodes::Either::Left(node),
                        _ => $crate::layouter::nodes::Either::Right(self)
                    }
                }

                fn maybe_into_ref(&self) -> Option<&$variant> {
                    match self {
                        Self::$var_name(node) => Some(node),
                        _ => None
                    }
                }

                fn maybe_into_mut(&mut self) -> Option<&mut $variant> {
                    match self {
                        Self::$var_name(node) => Some(node),
                        _ => None
                    }
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

pub trait MaybeInto<T>: Sized {
    fn maybe_into(self) -> Either<T, Self>;
    fn maybe_into_ref(&self) -> Option<&T>;
    fn maybe_into_mut(&mut self) -> Option<&mut T>;
}

pub trait MaybeFrom<T>: Sized {
    fn maybe_from(other: T) -> Either<Self, T>;
    fn maybe_from_ref(other: &T) -> Option<&Self>;
    fn maybe_from_mut(other: &mut T) -> Option<&mut Self>;
}

impl<U, T> MaybeFrom<U> for T
where
    U: MaybeInto<T>,
{
    fn maybe_from(other: U) -> Either<Self, U> {
        other.maybe_into()
    }

    fn maybe_from_ref(other: &U) -> Option<&Self> {
        other.maybe_into_ref()
    }

    fn maybe_from_mut(other: &mut U) -> Option<&mut Self> {
        other.maybe_into_mut()
    }
}
