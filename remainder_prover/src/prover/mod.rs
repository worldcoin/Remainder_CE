//!Modules that orchestrates creating a GKR Proof

/// Includes boilerplate for creating a GKR circuit, i.e. creating a transcript, proving, verifying, etc.
pub mod helpers;

/// Includes various traits that define interfaces of a GKR Prover
pub mod proof_system;

/// Struct for representing a list of layers
pub mod layers;

use self::{layers::Layers, proof_system::ProofSystem};
use crate::claims::wlx_eval::WLXAggregator;
use crate::expression::circuit_expr::CircuitMle;
use crate::expression::verifier_expr::VerifierMle;
use crate::input_layer::enum_input_layer::{
    CircuitInputLayerEnum, InputLayerEnum, InputLayerEnumVerifierCommitment,
};
use crate::input_layer::CircuitInputLayer;
use crate::layer::layer_enum::{CircuitLayerEnum, VerifierLayerEnum};
use crate::layer::CircuitLayer;
use crate::layouter::layouting::{CircuitDescriptionMap, InputLayerHintMap, InputNodeMap};
use crate::layouter::nodes::circuit_inputs::InputLayerData;
use crate::layouter::nodes::NodeId;
use crate::mle::Mle;
use crate::output_layer::mle_output_layer::{CircuitMleOutputLayer, MleOutputLayer};
use crate::output_layer::{CircuitOutputLayer, OutputLayer};
use crate::{
    claims::ClaimAggregator,
    input_layer::{InputLayer, InputLayerError},
    layer::{layer_enum::LayerEnum, Layer, LayerError, LayerId},
    mle::MleIndex,
    utils::hash_layers,
};
use ark_std::{end_timer, start_timer};
use itertools::Itertools;
use proof_system::DefaultProofSystem;
use remainder_shared_types::transcript::{ProverTranscript, VerifierTranscript};
use remainder_shared_types::transcript::{
    Transcript, TranscriptReader, TranscriptReaderError, TranscriptWriter,
};
use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;
use thiserror::Error;
use tracing::{debug, info};
use tracing::{instrument, span, Level};

#[derive(Error, Debug, Clone)]
/// Errors relating to the proving of a GKR circuit
pub enum GKRError {
    #[error("No claims were found for layer {0:?}")]
    /// No claims were found for layer
    NoClaimsForLayer(LayerId),
    #[error("Transcript during verifier's interaction with the transcript.")]
    /// Errors when reading from the transcript
    TranscriptError(TranscriptReaderError),
    #[error("Error when proving layer {0:?}: {1}")]
    /// Error when proving layer
    ErrorWhenProvingLayer(LayerId, LayerError),
    #[error("Error when proving input layer {0:?}: {1}")]
    /// Error when proving input layer
    ErrorWhenProvingInputLayer(LayerId, InputLayerError),
    #[error("Error when verifying layer {0:?}: {1}")]
    /// Error when verifying layer
    ErrorWhenVerifyingLayer(LayerId, LayerError),

    /// Error when verifying input layer
    #[error("Error when verifying input layer {0:?}: {1}")]
    ErrorWhenVerifyingInputLayer(LayerId, InputLayerError),

    #[error("Error when verifying output layer")]
    /// Error when verifying output layer
    ErrorWhenVerifyingOutputLayer,
    /// Error for input layer commitment
    #[error("Error when commiting to InputLayer {0}")]
    InputLayerError(InputLayerError),
    #[error("Error when verifying circuit hash.")]
    /// Error when verifying circuit hash
    ErrorWhenVerifyingCircuitHash(TranscriptReaderError),

    /// Error generating the Verifier Key.
    #[error("Error generating the Verifier Key")]
    ErrorGeneratingVerifierKey,
}

/// A proof of the sumcheck protocol; Outer vec is rounds, inner vec is evaluations
/// this inner vec is none if there is no sumcheck proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckProof<F>(pub Vec<Vec<F>>);

impl<F: FieldExt> From<Vec<Vec<F>>> for SumcheckProof<F> {
    fn from(value: Vec<Vec<F>>) -> Self {
        Self(value)
    }
}

/// The witness of a GKR circuit, used to actually prove the circuit
#[derive(Debug)]
pub struct InstantiatedCircuit<F: FieldExt> {
    /// The intermediate layers of the circuit, as defined by the ProofSystem
    pub layers: Layers<F, LayerEnum<F>>,
    /// The output layers of the circuit, as defined by the ProofSystem
    pub output_layers: Vec<MleOutputLayer<F>>,
    /// The input layers of the circuit, as defined by the ProofSystem
    pub input_layers: Vec<InputLayerEnum<F>>,
}

impl<F: FieldExt> InstantiatedCircuit<F> {
    /// Returns the circuit description associated with this Witness to be used
    /// by the verifier.
    pub fn generate_verifier_key(&self) -> Result<GKRCircuitDescription<F>, GKRError> {
        let input_layers: Vec<_> = self
            .input_layers
            .iter()
            .map(|input_layer| input_layer.into_verifier_input_layer())
            .collect();

        let intermediate_layers: Vec<_> = self
            .layers
            .layers
            .iter()
            .map(|layer| {
                layer
                    .into_circuit_layer()
                    .map_err(|_| GKRError::ErrorGeneratingVerifierKey)
            })
            .collect::<Result<Vec<_>, GKRError>>()?;

        let output_layers: Vec<_> = self
            .output_layers
            .iter()
            .map(|output_layer| output_layer.into_circuit_output_layer())
            .collect();

        Ok(GKRCircuitDescription::<F> {
            input_layers,
            intermediate_layers,
            output_layers,
        })
    }
}

/// Controls claim aggregation behavior.
pub const ENABLE_OPTIMIZATION: bool = true;

pub type WitnessAndCircuitDescription<F> = (InstantiatedCircuit<F>, GKRCircuitDescription<F>);

/// The Verifier Key associated with a GKR proof of a [ProofSystem].
/// It consists of consice GKR Circuit description to be use by the Verifier.
#[derive(Debug)]
pub struct GKRCircuitDescription<F: FieldExt> {
    pub input_layers: Vec<CircuitInputLayerEnum<F>>,
    pub intermediate_layers: Vec<CircuitLayerEnum<F>>,
    pub output_layers: Vec<CircuitMleOutputLayer<F>>,
}

impl<F: FieldExt> GKRCircuitDescription<F> {
    /// Constructs a new `GKRCircuitDescription` via circuit description layers
    pub fn new(
        input_layers: Vec<CircuitInputLayerEnum<F>>,
        intermediate_layers: Vec<CircuitLayerEnum<F>>,
        output_layers: Vec<CircuitMleOutputLayer<F>>,
    ) -> Self {
        Self {
            input_layers,
            intermediate_layers,
            output_layers,
        }
    }

    pub fn index_mle_indices(&mut self, start_index: usize) {
        let GKRCircuitDescription {
            input_layers: _,
            intermediate_layers,
            output_layers,
        } = self;
        intermediate_layers
            .iter_mut()
            .for_each(|intermediate_layer| {
                intermediate_layer.index_mle_indices(start_index);
            });
        output_layers.iter_mut().for_each(|output_layer| {
            output_layer.index_mle_indices(start_index);
        })
    }

    /// Verifies a GKR proof produced by the `prove` method.
    /// # Arguments
    /// * `transcript_reader`: servers as the proof.
    #[instrument(skip_all, err)]
    fn verify(
        &mut self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<(), GKRError> {
        // TODO(Makis): Add circuit hash to Transcript.
        /*
        if let Some(circuit_hash) = maybe_circuit_hash {
            let transcript_circuit_hash = transcript_reader
                .consume_element("Circuit Hash")
                .map_err(GKRError::ErrorWhenVerifyingCircuitHash)?;
            assert_eq!(transcript_circuit_hash, circuit_hash);
        }
        */
        self.index_mle_indices(0);
        let input_layer_commitments_timer = start_timer!(|| "Retrieve Input Layer Commitments");

        let input_layer_commitments: Vec<InputLayerEnumVerifierCommitment<F>> = self
            .input_layers
            .iter()
            .map(|input_layer| {
                input_layer
                    .get_commitment_from_transcript(transcript_reader)
                    .unwrap()
            })
            //.collect::<Result<Vec<Commitment>, GKRError>>()?;
            .collect();

        end_timer!(input_layer_commitments_timer);

        // Claim aggregator to keep track of GKR-style claims across all layers.
        let mut aggregator = WLXAggregator::<F, LayerEnum<F>, InputLayerEnum<F>>::new();

        // --------- STAGE 1: Output Claim Generation ---------
        let claims_timer = start_timer!(|| "Output claims generation");
        let verifier_output_claims_span =
            span!(Level::DEBUG, "verifier_output_claims_span").entered();

        for circuit_output_layer in self.output_layers.iter() {
            let layer_id = circuit_output_layer.layer_id();
            info!("Verifying Output Layer: {:?}", layer_id);

            let verifier_output_layer = circuit_output_layer
                .retrieve_mle_from_transcript_and_fix_layer(transcript_reader)
                .map_err(|_| GKRError::ErrorWhenVerifyingOutputLayer)?;
            dbg!("checkpoint_1");

            aggregator
                .extract_claims(&verifier_output_layer)
                .map_err(|_| GKRError::ErrorWhenVerifyingOutputLayer)?;
            dbg!("checkpoint_2");
        }

        // dbg!(&aggregator);

        end_timer!(claims_timer);
        verifier_output_claims_span.exit();

        // --------- STAGE 2: Verify Intermediate Layers ---------
        let intermediate_layers_timer =
            start_timer!(|| "ALL intermediate layers proof verification");

        for layer in self.intermediate_layers.iter().rev() {
            let layer_id = layer.layer_id();

            info!("Intermediate Layer: {:?}", layer_id);
            let layer_timer =
                start_timer!(|| format!("Proof verification for layer {:?}", layer_id));

            let claim_aggr_timer =
                start_timer!(|| format!("Verify aggregated claim for layer {:?}", layer_id));

            let prev_claim = aggregator.verifier_aggregate_claims(layer_id, transcript_reader)?;
            debug!("Aggregated claim: {:#?}", prev_claim);

            end_timer!(claim_aggr_timer);

            info!("Verifier: about to verify layer");
            let sumcheck_msg_timer =
                start_timer!(|| format!("Verify sumcheck message for layer {:?}", layer_id));

            // Performs the actual sumcheck verification step.
            let verifier_layer: VerifierLayerEnum<F> = layer
                .verify_rounds(prev_claim, transcript_reader)
                .map_err(|err| {
                    GKRError::ErrorWhenVerifyingLayer(layer_id, LayerError::VerificationError(err))
                })?;

            end_timer!(sumcheck_msg_timer);

            dbg!("ehllo");
            aggregator
                .extract_claims(&verifier_layer)
                .map_err(|_| GKRError::ErrorWhenVerifyingOutputLayer)?;

            end_timer!(layer_timer);
        }

        end_timer!(intermediate_layers_timer);

        // --------- STAGE 3: Verify Input Layers ---------
        let input_layers_timer = start_timer!(|| "INPUT layers proof verification");

        for (input_layer, commitment) in
            self.input_layers.iter().zip(input_layer_commitments.iter())
        {
            let layer_timer = start_timer!(|| format!(
                "proof generation for INPUT layer {:?}",
                input_layer.layer_id()
            ));

            let input_layer_id = input_layer.layer_id();
            info!("--- Input Layer: {:?} ---", input_layer_id);

            let claim_aggr_timer = start_timer!(|| format!(
                "verify aggregated claim for INPUT layer {:?}",
                input_layer.layer_id()
            ));

            let input_layer_claim =
                aggregator.verifier_aggregate_claims(input_layer_id, transcript_reader)?;

            debug!("Input layer claim: {:#?}", input_layer_claim);
            end_timer!(claim_aggr_timer);

            let sumcheck_msg_timer = start_timer!(|| format!(
                "verify sumcheck message for INPUT layer {:?}",
                input_layer.layer_id()
            ));

            input_layer
                .verify(commitment, input_layer_claim, transcript_reader)
                .map_err(GKRError::InputLayerError)?;

            end_timer!(sumcheck_msg_timer);

            end_timer!(layer_timer);
        }

        end_timer!(input_layers_timer);

        Ok(())
    }
}
