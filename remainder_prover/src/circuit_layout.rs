//! Defines the utilities for taking a list of nodes and turning it into a
//! layedout circuit
use std::collections::{hash_map::Entry, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

use crate::{
    layer::LayerId,
    mle::{dense::DenseMle, evals::MultilinearExtension, mle_description::MleDescription},
};

use anyhow::{anyhow, Result};

/// A HashMap that records during circuit compilation where nodes live in the
/// circuit and what data they yield.
#[derive(Debug)]
pub struct CircuitEvalMap<F: Field>(pub(crate) HashMap<CircuitLocation, MultilinearExtension<F>>);
/// A map that maps layer ID to all the MLEs that are output from that layer.
/// Together these MLEs are combined along with the information from their
/// prefix bits to form the layerwise bookkeeping table.
pub type LayerMap<F> = HashMap<LayerId, Vec<DenseMle<F>>>;

impl<F: Field> CircuitEvalMap<F> {
    /// Create a new circuit map, which maps circuit location to the data stored
    /// at that location.removing
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Using the circuit location, which is a layer_id and prefix_bits tuple,
    /// get the data that exists here.
    pub fn get_data_from_circuit_mle(
        &self,
        circuit_mle: &MleDescription<F>,
    ) -> Result<&MultilinearExtension<F>> {
        let circuit_location =
            CircuitLocation::new(circuit_mle.layer_id(), circuit_mle.prefix_bits());
        let result = self
            .0
            .get(&circuit_location)
            .ok_or(anyhow!("Circuit location not found!"));
        if let Ok(actual_result) = result {
            assert_eq!(actual_result.num_vars(), circuit_mle.num_free_vars());
        }
        result
    }

    /// Adds a new node to the CircuitMap
    pub fn add_node(&mut self, circuit_location: CircuitLocation, value: MultilinearExtension<F>) {
        self.0.insert(circuit_location, value);
    }

    /// Destructively convert this into a map that maps LayerId to the
    /// [MultilinearExtension]s that generate claims on this area. This is to
    /// aid in claim aggregation, so we know the parts of the layerwise
    /// bookkeeping table in order to aggregate claims on this layer.
    pub fn convert_to_layer_map(mut self) -> LayerMap<F> {
        let mut layer_map = HashMap::<LayerId, Vec<DenseMle<F>>>::new();
        self.0.drain().for_each(|(circuit_location, data)| {
            let corresponding_mle = DenseMle::new_with_prefix_bits(
                data,
                circuit_location.layer_id,
                circuit_location.prefix_bits,
            );
            if let Entry::Vacant(e) = layer_map.entry(circuit_location.layer_id) {
                e.insert(vec![corresponding_mle]);
            } else {
                layer_map
                    .get_mut(&circuit_location.layer_id)
                    .unwrap()
                    .push(corresponding_mle);
            }
        });
        layer_map
    }
}

impl<F: Field> Default for CircuitEvalMap<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// The location of a Node in the circuit
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CircuitLocation {
    /// The LayerId this node has been placed into
    pub layer_id: LayerId,
    /// Any prefix_bits neccessary to differenciate claims made onto this node
    /// from other nodes in the same layer
    pub prefix_bits: Vec<bool>,
}

impl CircuitLocation {
    /// Creates a new CircuitLocation
    pub fn new(layer_id: LayerId, prefix_bits: Vec<bool>) -> Self {
        Self {
            layer_id,
            prefix_bits,
        }
    }
}
