//!Modules that orchestrates creating a GKR Proof

/// For combining sub-circuits(multiple layers) into a single circuit(layer)
pub mod combine_layers;

/// Includes boilerplate for creating a GKR circuit, i.e. creating a transcript, proving, verifying, etc.
pub mod helpers;

/// For the various input layers to the GKR circuit
pub mod input_layer;

/// Includes various traits that define interfaces of a GKR Prover
pub mod proof_system;

/// Includes various builders needed for testing purposes
pub mod test_helper_circuits;

/// Comprehensive tests for the GKR Prover
#[cfg(test)]
pub(crate) mod tests;

use self::{
    input_layer::{InputLayer, InputLayerError},
    proof_system::ProofSystem,
};
use crate::{
    claims::{Claim, ClaimAggregator},
    gate::gate::{BinaryOperation, Gate},
    layer::{
        layer_builder::LayerBuilder, layer_enum::LayerEnum, regular_layer::RegularLayer, Layer,
        LayerError, LayerId,
    },
    mle::{
        dense::{DenseMle, DenseMleRef},
        MleIndex, MleRef,
    },
    utils::hash_layers,
};
use ark_std::{end_timer, start_timer};
use itertools::Itertools;
use remainder_shared_types::transcript::{
    TranscriptReader, TranscriptReaderError, TranscriptWriter,
};
use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use thiserror::Error;
use tracing::{debug, info};
use tracing::{instrument, span, Level};

/// The list of Layers that make up the GKR circuit
pub struct Layers<F: FieldExt, T: Layer<F>> {
    /// A Vec of pointers to various layer types
    pub layers: Vec<T>,
    marker: PhantomData<F>,
}

impl<F: FieldExt, T: Layer<F>> Layers<F, T> {
    /// Add a GKRLayer to a list of layers
    pub fn add_gkr<B: LayerBuilder<F>>(&mut self, new_layer: B) -> B::Successor
    where
        T: From<RegularLayer<F>>,
    {
        let id = LayerId::Layer(self.layers.len());
        let successor = new_layer.next_layer(id, None);
        let layer = RegularLayer::<F>::new(new_layer, id);
        self.layers.push(layer.into());
        successor
    }

    /// Add a batched Add Gate layer to a list of layers
    /// In the batched case, consider a vector of mles corresponding to an mle for each "batch" or "copy".
    /// Add a Gate layer to a list of layers
    /// In the batched case (`num_dataparallel_bits` > 0), consider a vector of mles corresponding to an mle for each "batch" or "copy".
    /// Then we refer to the mle that represents the concatenation of these mles by interleaving as the
    /// flattened mle and each individual mle as a batched mle.
    ///
    /// # Arguments
    /// * `nonzero_gates`: the gate wiring between single-copy circuit (as the wiring for each circuit remains the same)
    /// x is the label on the batched mle `lhs`, y is the label on the batched mle `rhs`, and z is the label on the next layer, batched
    /// * `lhs`: the flattened mle representing the left side of the summation
    /// * `rhs`: the flattened mle representing the right side of the summation
    /// * `num_dataparallel_bits`: the number of bits representing the circuit copy we are looking at
    /// * `gate_operation`: which operation the gate is performing. right now, can either be an 'add' or 'mul' gate
    ///
    /// # Returns
    /// A flattened `DenseMle` that represents the evaluations of the add gate wiring on `lhs` and `rhs` over the boolean hypercube
    pub fn add_gate(
        &mut self,
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMleRef<F>,
        rhs: DenseMleRef<F>,
        num_dataparallel_bits: Option<usize>,
        gate_operation: BinaryOperation,
    ) -> DenseMle<F, F>
    where
        T: From<Gate<F>>,
    {
        let id = LayerId::Layer(self.layers.len());
        // constructor for batched mul gate struct
        let gate: Gate<F> = Gate::new(
            num_dataparallel_bits,
            nonzero_gates.clone(),
            lhs.clone(),
            rhs.clone(),
            gate_operation,
            id,
        );
        let max_gate_val = nonzero_gates
            .clone()
            .into_iter()
            .fold(0, |acc, (z, _, _)| std::cmp::max(acc, z));

        // number of entries in the resulting table is the max gate z value * 2 to the power of the number of dataparallel bits, as we are
        // evaluating over all values in the boolean hypercube which includes dataparallel bits
        let num_dataparallel_vals = 1 << (num_dataparallel_bits.unwrap_or(0));
        let res_table_num_entries = (max_gate_val + 1) * num_dataparallel_vals;
        self.layers.push(gate.into());

        // iterate through each of the indices and perform the binary operation specified
        let mut res_table = vec![F::zero(); res_table_num_entries];
        (0..num_dataparallel_vals).for_each(|idx| {
            nonzero_gates
                .clone()
                .into_iter()
                .for_each(|(z_ind, x_ind, y_ind)| {
                    let f2_val = *lhs
                        .bookkeeping_table()
                        .get(idx + (x_ind * num_dataparallel_vals))
                        .unwrap_or(&F::zero());
                    let f3_val = *rhs
                        .bookkeeping_table()
                        .get(idx + (y_ind * num_dataparallel_vals))
                        .unwrap_or(&F::zero());
                    res_table[idx + (z_ind * num_dataparallel_vals)] =
                        gate_operation.perform_operation(f2_val, f3_val);
                });
        });

        let res_mle: DenseMle<F, F> = DenseMle::new_from_raw(res_table, id, None);

        res_mle
    }

    /// Creates a new Layers
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            marker: PhantomData,
        }
    }

    /// Returns the number of layers in the GKR circuit
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl<F: FieldExt, T: Layer<F>> Default for Layers<F, T> {
    fn default() -> Self {
        Self::new()
    }
}

///An output layer which will have it's bits bound and then evaluated
pub type OutputLayer<F> = Box<dyn MleRef<F = F>>;

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
    #[error("Error when verifying output layer")]
    /// Error when verifying output layer
    ErrorWhenVerifyingOutputLayer,
    /// Error for input layer commitment
    #[error("Error when commiting to InputLayer {0}")]
    InputLayerError(InputLayerError),
    #[error("Error when verifying circuit hash.")]
    /// Error when verifying circuit hash
    ErrorWhenVerifyingCircuitHash(TranscriptReaderError),
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

/// The proof for an individual GKR layer
#[derive(Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct LayerProof<F: FieldExt, Pf: ProofSystem<F>> {
    /// The sumcheck proof of each Layer, could either be a RegularLayer or a Gate
    pub sumcheck_proof: <Pf::Layer as Layer<F>>::Proof,
    /// The layer we are proving over
    pub layer: Pf::Layer,
    /// The proof of the claim aggregation
    pub claim_aggregation_proof: <Pf::ClaimAggregator as ClaimAggregator<F>>::AggregationProof,
}

/// Proof for circuit input layer
#[derive(Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct InputLayerProof<F: FieldExt, Pf: ProofSystem<F>> {
    /// the layer id of the input layer
    pub layer_id: LayerId,
    /// The proof of the claim aggregation for this input layer
    pub input_layer_claim_aggregation_proof:
        <Pf::ClaimAggregator as ClaimAggregator<F>>::AggregationProof,
    /// The commitment to the input layer
    pub input_commitment: <Pf::InputLayer as InputLayer<F>>::Commitment,
    /// The opening proof for the commitment
    pub input_opening_proof: <Pf::InputLayer as InputLayer<F>>::OpeningProof,
}

/// All the elements to be passed to the verifier for the succinct non-interactive sumcheck proof
#[derive(Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct GKRProof<F: FieldExt, Pf: ProofSystem<F>> {
    /// The sumcheck proof of each GKR Layer, along with the fully bound expression.
    /// In reverse order (i.e. layer closest to the output layer is first)
    pub layer_sumcheck_proofs: Vec<LayerProof<F, Pf>>,
    /// All the output layers that this circuit yields
    pub output_layers: Vec<Pf::OutputLayer>,
    /// Proofs for each input layer (e.g. `LigeroInputLayer` or `PublicInputLayer`).
    pub input_layer_proofs: Vec<InputLayerProof<F, Pf>>,
    /// Hash of the entire circuit description, to be used in the FS transcript
    pub maybe_circuit_hash: Option<F>,
}

/// The witness of a GKR circuit, used to actually prove the circuit
pub struct Witness<F: FieldExt, Pf: ProofSystem<F>> {
    /// The intermediate layers of the circuit, as defined by the ProofSystem
    pub layers: Layers<F, Pf::Layer>,
    /// The output layers of the circuit, as defined by the ProofSystem
    pub output_layers: Vec<Pf::OutputLayer>,
    /// The input layers of the circuit, as defined by the ProofSystem
    pub input_layers: Vec<Pf::InputLayer>,
}

/// Controls claim aggregation behavior.
pub const ENABLE_OPTIMIZATION: bool = true;

#[allow(type_alias_bounds)]
/// A helper type for easier reference to a circuit's Transcript
pub type CircuitTranscript<F, C: GKRCircuit<F>> = <C::ProofSystem as ProofSystem<F>>::Transcript;

#[allow(type_alias_bounds)]
/// A helper type alias for easier reference to a circuits Layer
pub type CircuitLayer<F, C: GKRCircuit<F>> = <C::ProofSystem as ProofSystem<F>>::Layer;

#[allow(type_alias_bounds)]
/// A helper type alias for easier reference to a circuits InputLayer
pub type CircuitInputLayer<F, C: GKRCircuit<F>> = <C::ProofSystem as ProofSystem<F>>::InputLayer;

#[allow(type_alias_bounds)]
/// A helper type alias for easier reference to a circuits ClaimAggregator
pub type CircuitClaimAggregator<F, C: GKRCircuit<F>> =
    <C::ProofSystem as ProofSystem<F>>::ClaimAggregator;

/// A GKRCircuit ready to be proven
pub trait GKRCircuit<F: FieldExt> {
    /// The ProofSystem that describes the allowed cryptographic operations this Circuit uses
    type ProofSystem: ProofSystem<F>;

    /// The hash of the circuit, use to uniquely identify the circuit
    const CIRCUIT_HASH: Option<[u8; 32]> = None;

    /// The forward pass, defining the layer relationships and generating the layers
    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem>;

    /// Calls `synthesize` and also generates commitments from each of the input layers
    #[instrument(skip_all, err)]
    fn synthesize_and_commit(
        &mut self,
        transcript: &mut TranscriptWriter<F, CircuitTranscript<F, Self>>,
    ) -> Result<
        (
            Witness<F, Self::ProofSystem>,
            Vec<<CircuitInputLayer<F, Self> as InputLayer<F>>::Commitment>,
        ),
        GKRError,
    > {
        let mut witness = self.synthesize();

        let commitments = witness
            .input_layers
            .iter_mut()
            .map(|input_layer| {
                let commitment = input_layer.commit().map_err(GKRError::InputLayerError)?;
                CircuitInputLayer::<F, Self>::prover_append_commitment_to_transcript(
                    &commitment,
                    transcript,
                );
                Ok(commitment)
            })
            .try_collect()?;

        Ok((witness, commitments))
    }

    /// The backwards pass, creating the GKRProof
    #[instrument(skip_all, err)]
    fn prove(
        &mut self,
        transcript_writer: &mut TranscriptWriter<F, CircuitTranscript<F, Self>>,
    ) -> Result<GKRProof<F, Self::ProofSystem>, GKRError>
    where
        CircuitTranscript<F, Self>: Sync,
    {
        let synthesize_commit_timer = start_timer!(|| "synthesize and commit");
        // --- Synthesize the circuit, using LayerBuilders to create internal, output, and input layers ---
        // --- Also commit and add those commitments to the transcript
        info!("Synethesizing circuit...");

        // --- Add circuit hash to transcript, if exists ---
        if let Some(circuit_hash) = Self::get_circuit_hash() {
            transcript_writer.append("Circuit Hash", circuit_hash);
        }

        let (
            Witness {
                input_layers,
                mut output_layers,
                layers,
            },
            commitments,
        ) = self.synthesize_and_commit(transcript_writer)?;
        info!("Circuit synthesized and witness generated.");
        end_timer!(synthesize_commit_timer);

        let claims_timer = start_timer!(|| "output claims generation");

        // --- TRACE: grabbing output claims ---
        let output_claims_span = span!(Level::DEBUG, "output_claims_span").entered();

        // --- Keep track of GKR-style claims across all layers ---
        let mut aggregator = <CircuitClaimAggregator<F, Self> as ClaimAggregator<F>>::new();

        // --- Go through circuit output layers and grab claims on each ---
        for output in output_layers.iter_mut() {
            info!("New Output Layer: {:?}", output.get_layer_id());
            let bits = output.index_mle_indices(0);

            if bits != 0 {
                debug!("Bookkeeping table: {:?}", output.bookkeeping_table());
                // --- Evaluate each output MLE at a random challenge point ---
                for bit in 0..bits {
                    let challenge = transcript_writer.get_challenge("Setting Output Layer Claim");
                    output.fix_variable(bit, challenge);
                }
            }
            let layer_id = output.get_layer_id();

            // --- Add the claim to either the set of current claims we're proving ---
            // --- or the global set of claims we need to eventually prove ---
            aggregator
                .add_claims(output)
                .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;
        }

        end_timer!(claims_timer);

        let intermediate_layers_timer = start_timer!(|| "ALL intermediate layers proof generation");

        // --- END TRACE: grabbing output claims ---
        output_claims_span.exit();

        // --- TRACE: Proving intermediate GKR layers ---
        let all_layers_sumcheck_proving_span =
            span!(Level::DEBUG, "all_layers_sumcheck_proving_span").entered();

        // --- Collects all the prover messages for sumchecking over each layer, ---
        // --- as well as all the prover messages for claim aggregation at the ---
        // --- beginning of proving each layer ---
        let layer_sumcheck_proofs = layers
            .layers
            .into_iter()
            .rev()
            .map(|mut layer| {
                let layer_timer =
                    start_timer!(|| format!("proof generation for layer {:?}", *layer.id()));

                // --- TRACE: Proving an individual GKR layer ---
                let layer_id = *layer.id();
                info!("New Intermediate Layer: {:?}", layer_id);

                let layer_id_trace_repr = format!("{}", layer_id);
                let _layer_sumcheck_proving_span = span!(
                    Level::DEBUG,
                    "layer_sumcheck_proving_span",
                    layer_id = layer_id_trace_repr
                )
                .entered();

                // --- For each layer, get the ID and all the claims on that layer ---
                let _layer_claims_vec = aggregator
                    .get_claims(layer_id)
                    .ok_or(GKRError::NoClaimsForLayer(layer_id))?;

                info!("Time for claim aggregation...");
                let claim_aggr_timer =
                    start_timer!(|| format!("claim aggregation for layer {:?}", *layer.id()));

                let (layer_claim, claim_aggregation_proof) =
                    aggregator.prover_aggregate_claims(&layer, transcript_writer)?;

                debug!("Claim Aggregation Proof: {:#?}", claim_aggregation_proof);
                end_timer!(claim_aggr_timer);
                let sumcheck_msg_timer = start_timer!(|| format!(
                    "compute sumcheck message for layer {:?}",
                    *layer.id()
                ));

                // --- Compute all sumcheck messages across this particular layer ---
                let prover_sumcheck_messages =
                    layer
                        .prove_rounds(layer_claim, transcript_writer)
                        .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

                debug!("sumcheck_proof: {:#?}", prover_sumcheck_messages);
                end_timer!(sumcheck_msg_timer);

                aggregator
                    .add_claims(&layer)
                    .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

                end_timer!(layer_timer);

                Ok(LayerProof {
                    sumcheck_proof: prover_sumcheck_messages,
                    layer,
                    claim_aggregation_proof,
                })
            })
            .try_collect()?;

        end_timer!(intermediate_layers_timer);
        // --- END TRACE: Proving intermediate GKR layers ---
        all_layers_sumcheck_proving_span.exit();

        let input_layers_timer = start_timer!(|| "INPUT layers proof generation");

        // --- TRACE: Proving input layer ---
        let input_layer_proving_span = span!(Level::DEBUG, "input_layer_proving_span").entered();

        let input_layer_proofs = input_layers
            .into_iter()
            .zip(commitments)
            .map(|(input_layer, commitment)| {
                let layer_timer = start_timer!(|| format!(
                    "proof generation for INPUT layer {:?}",
                    input_layer.layer_id()
                ));
                let layer_id = input_layer.layer_id();
                info!("New Input Layer: {:?}", layer_id);

                let _layer_claims_vec = aggregator
                    .get_claims(*layer_id)
                    .ok_or(GKRError::NoClaimsForLayer(*layer_id))?;

                let claim_aggr_timer = start_timer!(|| format!(
                    "claim aggregation for INPUT layer {:?}",
                    input_layer.layer_id()
                ));

                let (layer_claim, claim_aggregation_proof) =
                    aggregator.prover_aggregate_claims_input(&input_layer, transcript_writer)?;

                debug!("Claim Aggregation Proof: {:#?}", claim_aggregation_proof);
                end_timer!(claim_aggr_timer);

                let opening_proof_timer = start_timer!(|| format!(
                    "opening proof for INPUT layer {:?}",
                    input_layer.layer_id()
                ));

                let opening_proof = input_layer
                    .open(transcript_writer, layer_claim)
                    .map_err(GKRError::InputLayerError)?;

                end_timer!(opening_proof_timer);

                end_timer!(layer_timer);

                Ok(InputLayerProof {
                    layer_id: *layer_id,
                    input_commitment: commitment,
                    input_layer_claim_aggregation_proof: claim_aggregation_proof,
                    input_opening_proof: opening_proof,
                })
            })
            .try_collect()?;

        end_timer!(input_layers_timer);
        // --- END TRACE: Proving input layer ---
        input_layer_proving_span.exit();

        let gkr_proof = GKRProof {
            layer_sumcheck_proofs,
            output_layers,
            input_layer_proofs,
            maybe_circuit_hash: Self::get_circuit_hash(),
        };

        Ok(gkr_proof)
    }

    /// Verifies the GKRProof produced by fn prove
    ///
    /// Takes in a transcript for FS and re-generates challenges on its own
    #[instrument(skip_all, err)]
    fn verify(
        &mut self,
        transcript_reader: &mut TranscriptReader<F, CircuitTranscript<F, Self>>,
        gkr_proof: GKRProof<F, Self::ProofSystem>,
    ) -> Result<(), GKRError> {
        // --- Unpacking GKR proof + adding input commitments to transcript first ---
        let GKRProof {
            layer_sumcheck_proofs,
            output_layers,
            input_layer_proofs,
            maybe_circuit_hash,
        } = gkr_proof;

        let input_layers_timer = start_timer!(|| "append INPUT commitments to transcript");

        if let Some(circuit_hash) = maybe_circuit_hash {
            let transcript_circuit_hash = transcript_reader
                .consume_element("Circuit Hash")
                .map_err(GKRError::ErrorWhenVerifyingCircuitHash)?;
            debug_assert_eq!(transcript_circuit_hash, circuit_hash);
        }

        for input_layer in input_layer_proofs.iter() {
            CircuitInputLayer::<F, Self>::verifier_append_commitment_to_transcript(
                &input_layer.input_commitment,
                transcript_reader,
            )
            .map_err(GKRError::InputLayerError)?;
        }
        end_timer!(input_layers_timer);

        // --- Verifier keeps track of the claims on its own ---
        let mut aggregator = <CircuitClaimAggregator<F, Self> as ClaimAggregator<F>>::new();

        let claims_timer = start_timer!(|| "output claims generation");
        // --- TRACE: output claims ---
        let verifier_output_claims_span =
            span!(Level::DEBUG, "verifier_output_claims_span").entered();

        // --- NOTE that all the `Expression`s and MLEs contained within `gkr_proof` are already bound! ---
        for output in output_layers.iter() {
            let mle_indices = output.mle_indices();
            let mut claim_chal: Vec<F> = vec![];
            debug!("Bookkeeping table: {:#?}", output.bookkeeping_table());
            for (bit, index) in mle_indices
                .iter()
                .filter(|index| !matches!(index, &&MleIndex::Fixed(_)))
                .enumerate()
            {
                let challenge = transcript_reader
                    .get_challenge("Setting Output Layer Claim")
                    .map_err(|_| GKRError::ErrorWhenVerifyingOutputLayer)?;

                // We assume that all the outputs are zero-valued for now. We should be
                // doing the initial step of evaluating V_1'(z) as specified in Thaler 13 page 14,
                // but given the assumption we have that V_1'(z) = 0 for all z if the prover is honest.
                if MleIndex::Bound(challenge, bit) != *index {
                    dbg!(&(challenge, bit));
                    dbg!(&index);
                    return Err(GKRError::ErrorWhenVerifyingOutputLayer);
                }
                claim_chal.push(challenge);
            }
            let layer_id = output.get_layer_id();
            info!("New Output Layer {:?}", layer_id);

            // --- Append claims to either the claim tracking map OR the first (sumchecked) layer's list of claims ---
            aggregator
                .add_claims(output)
                .map_err(|_| GKRError::ErrorWhenVerifyingOutputLayer)?;
        }

        end_timer!(claims_timer);

        let intermediate_layers_timer =
            start_timer!(|| "ALL intermediate layers proof verification");
        // --- END TRACE: output claims ---
        verifier_output_claims_span.exit();

        // --- Go through each of the layers' sumcheck proofs ---
        for sumcheck_proof_single in layer_sumcheck_proofs {
            let LayerProof {
                sumcheck_proof,
                mut layer,
                claim_aggregation_proof: _,
            } = sumcheck_proof_single;

            let layer_timer =
                start_timer!(|| format!("proof verification for layer {:?}", *layer.id()));

            // --- TRACE: Proving an individual GKR layer ---
            let layer_id = *layer.id();
            info!("Intermediate Layer: {:?}", layer_id);
            debug!("The LayerEnum: {:#?}", layer);
            let layer_id = *layer.id();
            let layer_id_trace_repr = format!("{}", layer_id);
            let _layer_sumcheck_verification_span = span!(
                Level::DEBUG,
                "layer_sumcheck_verification_span",
                layer_id = layer_id_trace_repr
            )
            .entered();

            let claim_aggr_timer =
                start_timer!(|| format!("verify aggregated claim for layer {:?}", *layer.id()));
            // --- Perform the claim aggregation verification, first sampling `r` ---
            // --- Note that we ONLY do this if need be! ---

            let prev_claim = aggregator.verifier_aggregate_claims(layer_id, transcript_reader)?;

            end_timer!(claim_aggr_timer);

            let sumcheck_msg_timer =
                start_timer!(|| format!("verify sumcheck message for layer {:?}", *layer.id()));

            debug!("Aggregated claim: {:#?}", prev_claim);
            info!("Verifier: about to verify layer");

            // --- Performs the actual sumcheck verification step ---
            layer
                .verify_rounds(prev_claim, sumcheck_proof, transcript_reader)
                .map_err(|err| GKRError::ErrorWhenVerifyingLayer(layer_id, err))?;

            end_timer!(sumcheck_msg_timer);

            aggregator
                .add_claims(&layer)
                .map_err(|_| GKRError::ErrorWhenVerifyingOutputLayer)?;

            end_timer!(layer_timer);
        }

        end_timer!(intermediate_layers_timer);

        let input_layers_timer = start_timer!(|| "INPUT layers proof verification");

        for input_layer in input_layer_proofs {
            let layer_timer = start_timer!(|| format!(
                "proof generation for INPUT layer {:?}",
                input_layer.layer_id
            ));

            let input_layer_id = input_layer.layer_id;
            info!("--- Input Layer: {:?} ---", input_layer_id);

            let claim_aggr_timer = start_timer!(|| format!(
                "verify aggregated claim for INPUT layer {:?}",
                input_layer.layer_id
            ));

            let input_layer_claim =
                aggregator.verifier_aggregate_claims(input_layer_id, transcript_reader)?;

            debug!("Input layer claim: {:#?}", input_layer_claim);
            end_timer!(claim_aggr_timer);

            let sumcheck_msg_timer = start_timer!(|| format!(
                "verify sumcheck message for INPUT layer {:?}",
                input_layer.layer_id
            ));

            CircuitInputLayer::<F, Self>::verify(
                &input_layer.input_commitment,
                &input_layer.input_opening_proof,
                input_layer_claim,
                transcript_reader,
            )
            .map_err(GKRError::InputLayerError)?;

            end_timer!(sumcheck_msg_timer);

            end_timer!(layer_timer);
        }

        end_timer!(input_layers_timer);

        Ok(())
    }

    /// Generate the circuit hash
    fn gen_circuit_hash(&mut self) -> F
    where
        Self::ProofSystem: ProofSystem<F, Layer = LayerEnum<F>>,
    {
        let mut transcript_writer =
            TranscriptWriter::<F, CircuitTranscript<F, Self>>::new("Circuit Hash");
        let (Witness { layers, .. }, _) =
            self.synthesize_and_commit(&mut transcript_writer).unwrap();

        hash_layers(&layers)
    }

    /// Get the circuit hash
    fn get_circuit_hash() -> Option<F> {
        Self::CIRCUIT_HASH.map(|bytes| F::from_bytes_le(&bytes))
    }
}
