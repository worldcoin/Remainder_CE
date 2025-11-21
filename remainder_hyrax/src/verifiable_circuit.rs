use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use remainder::{layer::LayerId, mle::evals::MultilinearExtension, prover::GKRCircuitDescription};
use remainder_shared_types::curves::PrimeOrderCurve;

use crate::{
    hyrax_gkr::hyrax_input_layer::HyraxVerifierInputCommitment,
    provable_circuit::HyraxInputLayerDescriptionWithOptionalVerifierPrecommit,
};

use anyhow::{anyhow, Result};

/// A circiuit that contans a [GKRCircuitDescription], a description of the committed input layers,
/// and the data for all the public input layers, ready to be verified against a proof generated
/// through [super::provable_circuit::HyraxProvableCircuit].
///
/// Since the Hyrax proving system provides zero-knowledge guarantees, we refer to committed input
/// layers as private input layers.
#[derive(Clone, Debug)]
pub struct HyraxVerifiableCircuit<C: PrimeOrderCurve> {
    circuit_description: GKRCircuitDescription<C::Scalar>,
    /// A partial mapping of public input layers to MLEs.
    /// Some (or all) public input layer IDs may be missing.
    /// The input layers present in this mapping are public input data that the verifier placed in
    /// the circuit, and which will be checked for equality with the respective public inputs in the
    /// proof during verification.
    predetermined_public_inputs: HashMap<LayerId, MultilinearExtension<C::Scalar>>,
    private_inputs: HashMap<LayerId, HyraxInputLayerDescriptionWithOptionalVerifierPrecommit<C>>,
    layer_label_to_layer_id: HashMap<String, LayerId>,
}

impl<C: PrimeOrderCurve> HyraxVerifiableCircuit<C> {
    /// Returns a [HyraxVerifiableCircuit] initialized with the given data.
    pub fn new(
        circuit_description: GKRCircuitDescription<C::Scalar>,
        predetermined_public_inputs: HashMap<LayerId, MultilinearExtension<C::Scalar>>,
        private_inputs: HashMap<
            LayerId,
            HyraxInputLayerDescriptionWithOptionalVerifierPrecommit<C>,
        >,
        layer_label_to_layer_id: HashMap<String, LayerId>,
    ) -> Self {
        Self {
            circuit_description,
            predetermined_public_inputs,
            private_inputs,
            layer_label_to_layer_id,
        }
    }

    /// Returns the input layer description and optional pre-commitment to the input layer with ID
    /// `layer_id`.
    ///
    /// # Panics
    /// If provided with an invalid `layer_id`.
    pub fn get_private_input_layer_data_ref(
        &self,
        layer_id: &LayerId,
    ) -> &HyraxInputLayerDescriptionWithOptionalVerifierPrecommit<C> {
        self.private_inputs
            .get(layer_id)
            .expect("Private Input Layer with ID {layer_id} not found")
    }

    /// Returns the number of private input layers.
    pub fn get_num_private_layers(&self) -> usize {
        self.private_inputs.len()
    }

    /// Returns a reference to the circuit description.
    ///
    /// TODO: This is only used by the back end. Do _not_ expose it to the circuit developer.
    pub fn get_gkr_circuit_description_ref(&self) -> &GKRCircuitDescription<C::Scalar> {
        &self.circuit_description
    }

    /// Returns a vector of the [LayerId]s of all input layers with visibility [LayerVisibility::Public].
    pub fn get_public_input_layer_ids(&self) -> Vec<LayerId> {
        let all_input_layer_ids: HashSet<LayerId> =
            self.layer_label_to_layer_id.values().cloned().collect();

        let private_input_layer_ids: HashSet<LayerId> =
            self.private_inputs.keys().cloned().collect();

        // The public input layer IDs are the set difference of `(all input layer IDs \ (private
        // input layer IDs)`. We need to do it this way because `self.predetermined_public_inputs`
        // might not contain all public input layer IDs.
        all_input_layer_ids
            .difference(&private_input_layer_ids)
            .cloned()
            .collect()
    }

    /// Returns a vector of the [LayerId]s of all input layers with visibility
    /// [LayerVisibility::Committed].
    pub fn get_private_input_layer_ids(&self) -> Vec<LayerId> {
        self.private_inputs.keys().cloned().collect_vec()
    }

    /// Returns the data associated with the public input layer with ID `layer_id`, or None if
    /// no such public input layer exists.
    pub fn get_predetermined_public_input_mle_ref(
        &self,
        layer_id: &LayerId,
    ) -> Option<&MultilinearExtension<C::Scalar>> {
        self.predetermined_public_inputs.get(layer_id)
    }

    /// Returns the [LayerId] of the private input layer with label `label`, or an error if no such
    /// layer exists.
    pub fn get_private_input_layer_id(&self, label: &str) -> Result<LayerId> {
        self.layer_label_to_layer_id
            .get(label)
            .cloned()
            .ok_or(anyhow!("Unrecognized private input layer label"))
    }

    /// Sets the pre-commitment parameters of layer with label `layer_label` to `log_num_cols`.
    ///
    /// # Panics
    /// If `layer_label` is not a valid private input layer label.
    pub fn set_commitment_parameters(
        &mut self,
        layer_label: &str,
        log_num_cols: usize,
    ) -> Result<()> {
        let layer_id = self.layer_label_to_layer_id.get(layer_label).unwrap();

        let (layer_descr, _) = self
            .private_inputs
            .get_mut(layer_id)
            .expect("Layer {layer_id} either does not exist, or is not a private input layer.");

        layer_descr.log_num_cols = log_num_cols;

        Ok(())
    }

    /// Sets the pre-commitment of the input layer with label `layer_label` to `commitment`.
    /// Pre-commitments allow the verifier to check that an input layer commitment in the
    /// [crate::hyrax_gkr::HyraxProof] struct is the same as the expected `commitment` provided
    /// here.
    ///
    /// # Panics
    /// If `layer_label` is not a valid private input layer label.
    pub fn set_pre_commitment(&mut self, layer_label: &str, commitment: Vec<C>) -> Result<()> {
        let layer_id = self.layer_label_to_layer_id.get(layer_label).unwrap();

        let (_, optional_commitment) = self
            .private_inputs
            .get_mut(layer_id)
            .expect("Layer {layer_id} either does not exist, or is not a private input layer.");

        *optional_commitment = Some(HyraxVerifierInputCommitment::new(commitment));

        Ok(())
    }
}
