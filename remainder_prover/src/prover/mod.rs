//!Modules that orchestrates creating a GKR Proof

/// Includes boilerplate for creating a GKR circuit, i.e. creating a transcript, proving, verifying, etc.
pub mod helpers;

/// Includes various traits that define interfaces of a GKR Prover
pub mod proof_system;

/// Struct for representing a list of layers
pub mod layers;

use self::{layers::Layers, proof_system::ProofSystem};
use crate::expression::verifier_expr::VerifierMle;
use crate::input_layer::VerifierInputLayer;
use crate::layer::CircuitLayer;
use crate::mle::Mle;
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
use remainder_shared_types::transcript::ProverTranscript;
use remainder_shared_types::transcript::{
    Transcript, TranscriptReader, TranscriptReaderError, TranscriptWriter,
};
use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};
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
pub struct Witness<F: FieldExt, Pf: ProofSystem<F>> {
    /// The intermediate layers of the circuit, as defined by the ProofSystem
    pub layers: Layers<F, Pf::Layer>,
    /// The output layers of the circuit, as defined by the ProofSystem
    pub output_layers: Vec<Pf::OutputLayer>,
    /// The input layers of the circuit, as defined by the ProofSystem
    pub input_layers: Vec<Pf::InputLayer>,
}

impl<F: FieldExt, Pf: ProofSystem<F>> Witness<F, Pf> {
    /// Returns the circuit description associated with this Witness to be used
    /// by the verifier.
    pub fn generate_verifier_key(&self) -> Result<GKRVerifierKey<F, Pf>, GKRError> {
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

        Ok(GKRVerifierKey::<F, Pf> {
            input_layers,
            intermediate_layers,
            output_layers,
        })
    }
}

/// Controls claim aggregation behavior.
pub const ENABLE_OPTIMIZATION: bool = true;

/// A helper type for easier reference to a circuit's Transcript
pub type CircuitTranscript<F, C> =
    <<C as GKRCircuit<F>>::ProofSystem as ProofSystem<F>>::Transcript;

/// A helper type alias for easier reference to a circuits InputLayer
pub type CircuitInputLayer<F, C> =
    <<C as GKRCircuit<F>>::ProofSystem as ProofSystem<F>>::InputLayer;

/// A helper type alias for easier reference to a circuits ClaimAggregator
pub type CircuitClaimAggregator<F, C> =
    <<C as GKRCircuit<F>>::ProofSystem as ProofSystem<F>>::ClaimAggregator;

type WitnessCommitmentsKey<F, C> = (
    Witness<F, <C as GKRCircuit<F>>::ProofSystem>,
    Vec<<CircuitInputLayer<F, C> as InputLayer<F>>::Commitment>,
    GKRVerifierKey<F, <C as GKRCircuit<F>>::ProofSystem>,
);

/// A GKRCircuit ready to be proven
pub trait GKRCircuit<F: FieldExt> {
    /// The ProofSystem that describes the allowed cryptographic operations this Circuit uses
    type ProofSystem: ProofSystem<F>;

    /// The hash of the circuit, use to uniquely identify the circuit
    const CIRCUIT_HASH: Option<F::Repr> = None;

    /// The forward pass, defining the layer relationships and generating the layers
    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem>;

    /// Calls `synthesize` and also generates commitments from each of the input layers
    #[instrument(skip_all, err)]
    fn synthesize_and_commit(
        &mut self,
        transcript: &mut impl ProverTranscript<F>,
    ) -> Result<WitnessCommitmentsKey<F, Self>, GKRError> {
        let mut witness = self.synthesize();

        let verifier_key = witness.generate_verifier_key()?;

        let commitments = witness
            .input_layers
            .iter_mut()
            .map(|input_layer| {
                let commitment = input_layer.commit().map_err(GKRError::InputLayerError)?;
                CircuitInputLayer::<F, Self>::append_commitment_to_transcript(
                    &commitment,
                    transcript,
                );
                Ok(commitment)
            })
            .try_collect()?;

        Ok((witness, commitments, verifier_key))
    }

    /// The backwards pass, creating the GKRProof.
    #[instrument(skip_all, err)]
    fn prove(
        &mut self,
        mut transcript_writer: TranscriptWriter<F, CircuitTranscript<F, Self>>,
    ) -> Result<(Transcript<F>, GKRVerifierKey<F, Self::ProofSystem>), GKRError>
    where
        CircuitTranscript<F, Self>: Sync,
    {
        let synthesize_commit_timer = start_timer!(|| "Circuit synthesize and commit");
        info!("Synethesizing circuit...");

        // Add circuit hash to transcript, if exists.
        if let Some(circuit_hash) = Self::get_circuit_hash() {
            transcript_writer.append("Circuit Hash", circuit_hash);
        }

        // TODO(Makis): User getter syntax.
        let (
            Witness {
                input_layers,
                mut output_layers,
                layers,
            },
            commitments,
            verifier_key,
        ) = self.synthesize_and_commit(&mut transcript_writer)?;

        info!("Circuit synthesized and witness generated.");
        end_timer!(synthesize_commit_timer);

        // Claim aggregator to keep track of GKR-style claims across all layers.
        let mut aggregator = <CircuitClaimAggregator<F, Self> as ClaimAggregator<F>>::new();

        // --------- STAGE 1: Output Claim Generation ---------
        let claims_timer = start_timer!(|| "Output claims generation");
        let output_claims_span = span!(Level::DEBUG, "output_claims_span").entered();

        // Go through circuit output layers and grab claims on each.
        for output in output_layers.iter_mut() {
            let layer_id = output.layer_id();
            info!("Output Layer: {:?}", layer_id);

            output.append_mle_to_transcript(&mut transcript_writer);

            output
                .fix_layer(&mut transcript_writer)
                .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

            // Add the claim to either the set of current claims we're proving
            // or the global set of claims we need to eventually prove.
            aggregator
                .extract_claims(output)
                .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;
        }

        end_timer!(claims_timer);
        output_claims_span.exit();

        // --------- STAGE 2: Prove Intermediate Layers ---------
        let intermediate_layers_timer = start_timer!(|| "ALL intermediate layers proof generation");
        let all_layers_sumcheck_proving_span =
            span!(Level::DEBUG, "all_layers_sumcheck_proving_span").entered();

        // Collects all the prover messages for sumchecking over each layer, as
        // well as all the prover messages for claim aggregation at the
        // beginning of proving each layer.
        for mut layer in layers.layers.into_iter().rev() {
            let layer_id = layer.layer_id();

            let layer_timer = start_timer!(|| format!("Generating proof for layer {:?}", layer_id));
            let layer_id_trace_repr = format!("{:?}", layer_id);
            let layer_sumcheck_proving_span = span!(
                Level::DEBUG,
                "layer_sumcheck_proving_span",
                layer_id = layer_id_trace_repr
            )
            .entered();
            info!("Proving Intermediate Layer: {:?}", layer_id);

            info!("Starting claim aggregation...");
            let claim_aggr_timer =
                start_timer!(|| format!("Claim aggregation for layer {:?}", layer_id));

            let layer_claim = aggregator.prover_aggregate_claims(&layer, &mut transcript_writer)?;

            end_timer!(claim_aggr_timer);

            info!("Prove sumcheck message");
            let sumcheck_msg_timer = start_timer!(|| format!(
                "Compute sumcheck message for layer {:?}",
                layer.layer_id()
            ));

            // Compute all sumcheck messages across this particular layer.
            let _prover_sumcheck_messages = layer
                .prove_rounds(layer_claim, &mut transcript_writer)
                .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

            end_timer!(sumcheck_msg_timer);

            aggregator
                .extract_claims(&layer)
                .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

            end_timer!(layer_timer);
            layer_sumcheck_proving_span.exit();
        }

        end_timer!(intermediate_layers_timer);
        all_layers_sumcheck_proving_span.exit();

        // --------- STAGE 3: Prove Input Layers ---------
        let input_layers_timer = start_timer!(|| "INPUT layers proof generation");
        let input_layer_proving_span = span!(Level::DEBUG, "input_layer_proving_span").entered();

        for input_layer in input_layers {
            let layer_id = input_layer.layer_id();

            info!("New Input Layer: {:?}", layer_id);
            let layer_timer =
                start_timer!(|| format!("proof generation for INPUT layer {:?}", layer_id));

            let claim_aggr_timer = start_timer!(|| format!(
                "claim aggregation for INPUT layer {:?}",
                input_layer.layer_id()
            ));

            let layer_claim =
                aggregator.prover_aggregate_claims_input(&input_layer, &mut transcript_writer)?;

            end_timer!(claim_aggr_timer);

            let opening_proof_timer =
                start_timer!(|| format!("opening proof for INPUT layer {:?}", layer_id));

            let opening_proof = input_layer
                .open(&mut transcript_writer, layer_claim)
                .map_err(GKRError::InputLayerError)?;

            end_timer!(opening_proof_timer);

            end_timer!(layer_timer);
        }

        // TODO(Makis): What do we do with the input commitments?
        // Put them into transcript?

        end_timer!(input_layers_timer);
        input_layer_proving_span.exit();

        Ok((transcript_writer.get_transcript(), verifier_key))
    }

    /// Generate the circuit hash
    fn gen_circuit_hash(&mut self) -> F
    where
        Self::ProofSystem: ProofSystem<F, Layer = LayerEnum<F>>,
    {
        let mut transcript_writer =
            TranscriptWriter::<F, CircuitTranscript<F, Self>>::new("Circuit Hash");
        let (Witness { layers, .. }, _, _) =
            self.synthesize_and_commit(&mut transcript_writer).unwrap();

        hash_layers(&layers)
    }

    /// Get the circuit hash
    fn get_circuit_hash() -> Option<F>
where {
        Self::CIRCUIT_HASH.map(|bytes| F::from_repr(bytes).unwrap())
    }
}

/// The Verifier Key associated with a GKR proof of a [ProofSystem].
/// It consists of consice GKR Circuit description to be use by the Verifier.
#[derive(Debug)]
pub struct GKRVerifierKey<F: FieldExt, Pf: ProofSystem<F>> {
    input_layers: Vec<<<Pf as ProofSystem<F>>::InputLayer as InputLayer<F>>::VerifierInputLayer>,
    intermediate_layers: Vec<<<Pf as ProofSystem<F>>::Layer as Layer<F>>::CircuitLayer>,
    output_layers: Vec<<<Pf as ProofSystem<F>>::OutputLayer as OutputLayer<F>>::CircuitOutputLayer>,
}

impl<F: FieldExt, Pf: ProofSystem<F>> GKRVerifierKey<F, Pf> {
    /// Verifies a GKR proof produced by the `prove` method.
    /// # Arguments
    /// * `transcript_reader`: servers as the proof.
    #[instrument(skip_all, err)]
    fn verify(
        &mut self,
        transcript_reader: &mut TranscriptReader<F, <Pf as ProofSystem<F>>::Transcript>,
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

        let input_layer_commitments_timer = start_timer!(|| "Retrieve Input Layer Commitments");

        let input_layer_commitments: Vec<<<<Pf as ProofSystem<F>>::InputLayer as InputLayer<F>>::VerifierInputLayer as VerifierInputLayer<F>>::Commitment> = self
            .input_layers
            .iter()
            .map(|input_layer| {
                input_layer
                    .get_commitment_from_transcript(transcript_reader)
                    .unwrap()
                // .map_err(GKRError::InputLayerError)?
                // .map_err(|err| {
                //     GKRError::ErrorWhenVerifyingInputLayer(input_layer.layer_id(), err)
                // })?
            })
            //.collect::<Result<Vec<Commitment>, GKRError>>()?;
            .collect();

        end_timer!(input_layer_commitments_timer);

        // Claim aggregator to keep track of GKR-style claims across all layers.
        let mut aggregator = <<Pf as ProofSystem<F>>::ClaimAggregator as ClaimAggregator<F>>::new();

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

            aggregator
                .extract_claims(&verifier_output_layer)
                .map_err(|_| GKRError::ErrorWhenVerifyingOutputLayer)?;
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
            let verifier_layer =
                layer
                    .verify_rounds(prev_claim, transcript_reader)
                    .map_err(|err| {
                        GKRError::ErrorWhenVerifyingLayer(
                            layer_id,
                            LayerError::VerificationError(err),
                        )
                    })?;

            end_timer!(sumcheck_msg_timer);

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
