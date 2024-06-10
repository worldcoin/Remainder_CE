//! A module for defining how certain nodes can be Compiled into a GKR Witness

pub mod input_layer;

use itertools::Either;
use remainder_shared_types::FieldExt;

use crate::{
    input_layer::InputLayer,
    layer::LayerId,
    prover::{proof_system::ProofSystem, Witness},
};

use super::{
    layouting::CircuitLocation,
    nodes::{sector::Sector, CircuitNode, MaybeFrom, MaybeInto, NodeId},
};

#[derive(Clone, Debug)]
pub struct DAG<N>(Vec<Option<N>>);

impl<N: CircuitNode> DAG<N> {
    pub fn new(nodes: Vec<N>) -> Self {
        Self(nodes.into_iter().map(Option::Some).collect())
    }

    pub fn get_node_type<Other: CircuitNode + MaybeFrom<N>>(&mut self) -> Vec<Other> {
        self.0
            .iter_mut()
            .filter_map(|item| {
                let node = item.take()?;
                let node = Other::maybe_from(node);
                match node {
                    Either::Left(node) => Some(node),
                    Either::Right(node) => {
                        item.replace(node);
                        None
                    }
                }
            })
            .collect()
    }

    pub fn get_node_refs<Other: CircuitNode + MaybeFrom<N>>(&self) -> Vec<&Other> {
        self.0
            .iter()
            .filter_map(|item| <Other as MaybeFrom<N>>::maybe_from_ref(item.as_ref()?))
            .collect()
    }
}
