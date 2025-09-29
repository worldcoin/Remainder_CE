//! Defines the utilities for taking a list of nodes and turning it into a
//! layedout circuit
use std::collections::{hash_map::Entry, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

use itertools::Itertools;
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

use crate::circuit_building_context;
use crate::input_layer::ligero_input_layer::{
    LigeroInputLayerDescription, LigeroInputLayerDescriptionWithPrecommit,
};
use crate::prover::GKRCircuitDescription;
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

/// A circuit, along with all of its input data, ready to be proven using the vanila GKR proving
/// system which uses Ligero as a PCS for private input layers, and provides no zero-knowledge
/// guarantees.
#[derive(Clone, Debug)]
pub struct ProvableCircuit<F: Field> {
    circuit_description: GKRCircuitDescription<F>,
    inputs: HashMap<LayerId, MultilinearExtension<F>>,
    private_inputs: HashMap<LayerId, LigeroInputLayerDescriptionWithPrecommit<F>>,
}

impl<F: Field> ProvableCircuit<F> {
    /// Constructor
    pub fn new(
        circuit_description: GKRCircuitDescription<F>,
        inputs: HashMap<LayerId, MultilinearExtension<F>>,
        private_inputs: HashMap<LayerId, LigeroInputLayerDescriptionWithPrecommit<F>>,
    ) -> Self {
        Self {
            circuit_description,
            inputs,
            private_inputs,
        }
    }

    /// # WARNING
    /// To be used only for testing and debugging.
    ///
    /// Constructs a form of this circuit that can be verified when a proof is provided.
    /// This is done by erasing all input data associated with private input layers, along with any
    /// commitments (the latter can be found in the proof).
    pub fn _gen_verifiable_circuit(&self) -> VerifiableCircuit<F> {
        let public_inputs: HashMap<LayerId, MultilinearExtension<F>> = self
            .inputs
            .clone()
            .into_iter()
            .filter(|(layer_id, _)| !self.private_inputs.contains_key(layer_id))
            .collect();
        let private_inputs: HashMap<LayerId, LigeroInputLayerDescription<F>> = self
            .private_inputs
            .clone()
            .into_iter()
            .map(|(layer_id, (desc, _))| (layer_id, desc))
            .collect();

        // Ensure the union of public input Layer IDs with private input layer IDs equal
        // the set of all input layer IDs.
        debug_assert_eq!(
            public_inputs
                .keys()
                .chain(private_inputs.keys())
                .cloned()
                .sorted()
                .collect_vec(),
            self.inputs.keys().cloned().sorted().collect_vec()
        );

        VerifiableCircuit {
            circuit_description: self.circuit_description.clone(),
            public_inputs,
            private_inputs,
        }
    }

    /// Returns a reference to the [GKRCircuitDescription] of this circuit.
    ///
    /// TODO: This is only used by the back end. Do _not_ expose it to the circuit developer.
    pub fn get_gkr_circuit_description_ref(&self) -> &GKRCircuitDescription<F> {
        &self.circuit_description
    }

    /// Returns a vector of the [LayerId]s of all input layers with visibility [LayerVisibility::Public].
    ///
    /// TODO: Consider returning an iterator instead.
    pub fn get_public_input_layer_ids(&self) -> Vec<LayerId> {
        self.inputs
            .keys()
            .filter(|layer_id| !self.private_inputs.contains_key(layer_id))
            .cloned()
            .collect()
    }

    /// Returns a vector of the [LayerId]s of all input layers with visibility [LayerVisibility::Private].
    ///
    /// TODO: Consider returning an iterator instead.
    pub fn get_private_input_layer_ids(&self) -> Vec<LayerId> {
        self.private_inputs.keys().cloned().collect()
    }

    /// Returns the data associated with the input layer with ID `layer_id`, or an error if there is
    /// no input layer with this ID.
    pub fn get_input_mle(&self, layer_id: LayerId) -> Result<MultilinearExtension<F>> {
        self.inputs
            .get(&layer_id)
            .ok_or(anyhow!("Unrecognized Layer ID '{layer_id}'"))
            .cloned()
    }

    /// Returns the description of the private input layer with ID `layer_id`, or an error if no
    /// such private input layer exists.
    pub fn get_private_input_layer(
        &self,
        layer_id: LayerId,
    ) -> Result<LigeroInputLayerDescriptionWithPrecommit<F>> {
        self.private_inputs
            .get(&layer_id)
            .ok_or(anyhow!("Unrecognized Layer ID '{layer_id}'"))
            .cloned()
    }

    /// Get a reference to the mapping that maps a [LayerId] of layer (either public or private) to
    /// the data that are associated with it.
    ///
    /// TODO: This is too transparent. Replace this with methods that answer the queries of the
    /// prover directly, and do _not_ expose it to the circuit developer.
    pub fn get_inputs_ref(&self) -> &HashMap<LayerId, MultilinearExtension<F>> {
        &self.inputs
    }
}

/// A circuit that contains a [GKRCircuitDescription], a description of the private input layers,
/// and the data for all the public input layers, ready to be verified against a proof.
#[derive(Clone, Debug)]
pub struct VerifiableCircuit<F: Field> {
    circuit_description: GKRCircuitDescription<F>,
    public_inputs: HashMap<LayerId, MultilinearExtension<F>>,
    private_inputs: HashMap<LayerId, LigeroInputLayerDescription<F>>,
}

impl<F: Field> VerifiableCircuit<F> {
    /// Returns a [VerifiableCircuit] initialized with the given data.
    pub fn new(
        circuit_description: GKRCircuitDescription<F>,
        public_inputs: HashMap<LayerId, MultilinearExtension<F>>,
        private_inputs: HashMap<LayerId, LigeroInputLayerDescription<F>>,
    ) -> Self {
        Self {
            circuit_description,
            public_inputs,
            private_inputs,
        }
    }

    /// Returns a reference to the mapping which maps a [LayerId] of a public input layer to the
    /// data its associated with.
    ///
    /// TODO: This is too transparent. Replace this with methods that answer the queries of the
    /// prover directly, and do _not_ expose it to the circuit developer.
    pub fn get_public_inputs_ref(&self) -> &HashMap<LayerId, MultilinearExtension<F>> {
        &self.public_inputs
    }

    /// Returns a reference to the mapping which maps a [LayerId] of a private input layer to its
    /// description.
    ///
    /// TODO: This is too transparent. Replace this with methods that answer the queries of the
    /// prover directly, and do _not_ expose it to the circuit developer.
    pub fn get_private_inputs_ref(&self) -> &HashMap<LayerId, LigeroInputLayerDescription<F>> {
        &self.private_inputs
    }

    /// Returns a reference to the circuit description.
    ///
    /// TODO: This is only used by the back end. Do _not_ expose it to the circuit developer.
    pub fn get_gkr_circuit_description_ref(&self) -> &GKRCircuitDescription<F> {
        &self.circuit_description
    }

    /// Returns a vector of the [LayerId]s of all input layers with visibility [LayerVisibility::Public].
    pub fn get_public_input_layer_ids(&self) -> Vec<LayerId> {
        self.public_inputs.keys().cloned().collect_vec()
    }

    /// Returns a vector of the [LayerId]s of all input layers with visibility [LayerVisibility::Private].
    pub fn get_private_input_layer_ids(&self) -> Vec<LayerId> {
        self.private_inputs.keys().cloned().collect_vec()
    }

    /// Returns the data associated with the public input layer with ID `layer_id`, or an error if
    /// no such public input layer exists.
    pub fn get_public_input_mle(&self, layer_id: LayerId) -> Result<MultilinearExtension<F>> {
        self.public_inputs
            .get(&layer_id)
            .ok_or(anyhow!("Unrecognized Layer ID '{layer_id}'"))
            .cloned()
    }
}
