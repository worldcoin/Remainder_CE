//!Module that orchestrates creating a GKR Proof

pub mod combine_layers;
pub mod helpers;
/// For the input layer to the GKR circuit
pub mod input_layer;
pub mod test_helper_circuits;
#[cfg(test)]
pub(crate) mod tests;

use std::collections::HashMap;

use crate::{
    gate::gate::{BinaryOperation, Gate},
    layer::{
        claims::{aggregate_claims, get_num_wlx_evaluations},
        claims::{Claim, ClaimGroup},
        layer_enum::LayerEnum,
        GKRLayer, Layer, LayerBuilder, LayerError, LayerId,
    },
    mle::{
        dense::{DenseMle, DenseMleRef},
        MleRef,
    },
    mle::{mle_enum::MleEnum, MleIndex},
    sumcheck::evaluate_at_a_point,
    utils::{hash_layers, pad_to_nearest_power_of_two},
};

use tracing::{debug, info, trace};

// use lcpc_2d::{FieldExt, ligero_commit::{remainder_ligero_commit_prove, remainder_ligero_eval_prove, remainder_ligero_verify}, adapter::convert_halo_to_lcpc, LcProofAuxiliaryInfo, poseidon_ligero::PoseidonSpongeHasher, ligero_structs::LigeroEncoding, ligero_ml_helper::naive_eval_mle_at_challenge_point};
// use lcpc_2d::fs_transcript::halo2_remainder_transcript::Transcript;

use ark_std::{end_timer, start_timer};
use remainder_shared_types::transcript::Transcript;
use remainder_shared_types::FieldExt;

// use derive_more::From;
use itertools::{Either, Itertools};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{event, instrument, span, Level};

use self::input_layer::{
    enum_input_layer::{CommitmentEnum, InputLayerEnum, OpeningEnum},
    InputLayer, InputLayerError,
};

use core::cmp::Ordering;

use tracing::warn;

/// New type for containing the list of Layers that make up the GKR circuit
///
/// Literally just a Vec of pointers to various layer types!
pub struct Layers<F: FieldExt, Tr: Transcript<F>>(pub Vec<LayerEnum<F, Tr>>);

impl<F: FieldExt, Tr: Transcript<F> + 'static> Layers<F, Tr> {
    /// Add a layer to a list of layers
    pub fn add<B: LayerBuilder<F>, L: Layer<F, Transcript = Tr> + 'static>(
        &mut self,
        new_layer: B,
    ) -> B::Successor {
        let id = LayerId::Layer(self.0.len());
        let successor = new_layer.next_layer(id, None);
        let layer = L::new(new_layer, id);
        self.0.push(layer.get_enum());
        successor
    }

    /// Add a GKRLayer to a list of layers
    pub fn add_gkr<B: LayerBuilder<F>>(&mut self, new_layer: B) -> B::Successor {
        self.add::<_, GKRLayer<_, Tr>>(new_layer)
    }

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
    ) -> DenseMle<F, F> {
        let id = LayerId::Layer(self.0.len());
        // constructor for batched mul gate struct
        let gate: Gate<F, Tr> = Gate::new(
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
        self.0.push(gate.get_enum());

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
        Self(vec![])
    }

    pub fn next_layer_id(&self) -> usize {
        self.0.len()
    }
}

impl<F: FieldExt, Tr: Transcript<F> + 'static> Default for Layers<F, Tr> {
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
    #[error("Error when proving layer {0:?}: {1}")]
    /// Error when proving layer
    ErrorWhenProvingLayer(LayerId, LayerError),
    #[error("Error when verifying layer {0:?}: {1}")]
    /// Error when verifying layer
    ErrorWhenVerifyingLayer(LayerId, LayerError),
    #[error("Error when verifying output layer")]
    /// Error when verifying output layer
    ErrorWhenVerifyingOutputLayer,
    /// Error for input layer commitment
    #[error("Error when commiting to InputLayer {0}")]
    InputLayerError(InputLayerError),
}

/// A proof of the sumcheck protocol; Outer vec is rounds, inner vec is evaluations
/// this inner vec is none if there is no sumcheck proof -- this means that 
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckProof<F>(pub Option<Vec<Vec<F>>>);

impl<F: FieldExt> From<Option<Vec<Vec<F>>>> for SumcheckProof<F> {
    fn from(value: Option<Vec<Vec<F>>>) -> Self {
        Self(value)
    }
}

/// The proof for an individual GKR layer
#[derive(Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct LayerProof<F: FieldExt, Tr: Transcript<F>> {
    pub sumcheck_proof: SumcheckProof<F>,
    pub layer: LayerEnum<F, Tr>,
    /// When the claim aggregation optimization is on, each Layer produces many
    /// wlx evaluations.
    pub wlx_evaluations: Vec<Vec<F>>,
}

/// Proof for circuit input layer
#[derive(Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct InputLayerProof<F: FieldExt> {
    pub layer_id: LayerId,
    /// When the claim aggregation optimization is on, each Layer produces many
    /// wlx evaluations.
    pub input_layer_wlx_evaluations: Vec<Vec<F>>,
    pub input_commitment: CommitmentEnum<F>,
    pub input_opening_proof: OpeningEnum<F>,
}

/// All the elements to be passed to the verifier for the succinct non-interactive sumcheck proof
#[derive(Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct GKRProof<F: FieldExt, Tr: Transcript<F>> {
    /// The sumcheck proof of each GKR Layer, along with the fully bound expression.
    ///
    /// In reverse order (i.e. layer closest to the output layer is first)
    pub layer_sumcheck_proofs: Vec<LayerProof<F, Tr>>,
    /// All the output layers that this circuit yields
    pub output_layers: Vec<MleEnum<F>>,
    /// Proofs for each input layer (e.g. `LigeroInputLayer` or `PublicInputLayer`).
    pub input_layer_proofs: Vec<InputLayerProof<F>>,
    /// Hash of the entire circuit description, to be used in the FS transcript!
    /// TODO!(%Labs): Actually make this secure!
    pub maybe_circuit_hash: Option<F>,
}

pub struct Witness<F: FieldExt, Tr: Transcript<F>> {
    pub layers: Layers<F, Tr>,
    pub output_layers: Vec<MleEnum<F>>,
    pub input_layers: Vec<InputLayerEnum<F, Tr>>,
}

/// Controls claim aggregation behavior.
pub const ENABLE_OPTIMIZATION: bool = true;

/// Controls

/// A GKRCircuit ready to be proven
pub trait GKRCircuit<F: FieldExt> {
    /// The transcript this circuit uses
    type Transcript: Transcript<F>;

    const CIRCUIT_HASH: Option<[u8; 32]> = None;

    /// The forward pass, defining the layer relationships and generating the layers
    fn synthesize(&mut self) -> Witness<F, Self::Transcript>;

    /// Calls `synthesize` and also generates commitments from each of the input layers
    #[instrument(skip_all, err)]
    fn synthesize_and_commit(
        &mut self,
        transcript: &mut Self::Transcript,
    ) -> Result<(Witness<F, Self::Transcript>, Vec<CommitmentEnum<F>>), GKRError> {
        let mut witness = self.synthesize();

        let commitments = witness
            .input_layers
            .iter_mut()
            .map(|input_layer| {
                let commitment = input_layer.commit().map_err(GKRError::InputLayerError)?;
                InputLayerEnum::append_commitment_to_transcript(&commitment, transcript).unwrap();
                Ok(commitment)
            })
            .try_collect()?;

        Ok((witness, commitments))
    }

    /// The backwards pass, creating the GKRProof
    #[instrument(skip_all, err)]
    fn prove(
        &mut self,
        transcript: &mut Self::Transcript,
    ) -> Result<GKRProof<F, Self::Transcript>, GKRError>
    where
        <Self as GKRCircuit<F>>::Transcript: Sync,
    {
        let synthesize_commit_timer = start_timer!(|| "synthesize and commit");
        // --- Synthesize the circuit, using LayerBuilders to create internal, output, and input layers ---
        // --- Also commit and add those commitments to the transcript
        info!("Synethesizing circuit...");

        // --- Add circuit hash to transcript, if exists ---
        if let Some(circuit_hash) = Self::get_circuit_hash() {
            transcript
                .append_field_element("Circuit Hash", circuit_hash)
                .unwrap();
        }

        let (
            Witness {
                input_layers,
                mut output_layers,
                layers,
            },
            commitments,
        ) = self.synthesize_and_commit(transcript)?;
        info!("Circuit synthesized and witness generated.");
        end_timer!(synthesize_commit_timer);

        let claims_timer = start_timer!(|| "output claims generation");

        // --- TRACE: grabbing output claims ---
        let output_claims_span = span!(Level::DEBUG, "output_claims_span").entered();

        // --- Keep track of GKR-style claims across all layers ---
        let mut claims: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();

        // --- Go through circuit output layers and grab claims on each ---
        for output in output_layers.iter_mut() {
            info!("New Output Layer: {:?}", output.get_layer_id());
            let mut claim = None;
            let bits = output.index_mle_indices(0);

            let claim = if bits != 0 {
                debug!("Bookkeeping table: {:?}", output.bookkeeping_table());
                // --- Evaluate each output MLE at a random challenge point ---
                for bit in 0..bits {
                    let challenge = transcript
                        .get_challenge("Setting Output Layer Claim")
                        .unwrap();
                    claim = output.fix_variable(bit, challenge);
                }

                // --- Gather the claim and layer ID ---
                claim.unwrap()
            } else {
                Claim::new_raw(vec![], output.bookkeeping_table()[0])
            };

            // --- Gather the claim and layer ID ---
            let mut claim = claim;
            let layer_id = output.get_layer_id();
            claim.to_layer_id = Some(layer_id);
            claim.mle_ref = Some(output.clone());
            debug!("Creating a claim: {:#?}", claim);

            // --- Add the claim to either the set of current claims we're proving ---
            // --- or the global set of claims we need to eventually prove ---
            if let Some(curr_claims) = claims.get_mut(&layer_id) {
                curr_claims.push(claim);
            } else {
                claims.insert(layer_id, vec![claim]);
            }
        }

        end_timer!(claims_timer);

        let intermediate_layers_timer = start_timer!(|| "ALL intermediate layers proof generation");

        // Count the total number of wlx evaluations for benchmarking purposes.
        let mut wlx_count = 0;
        // --- END TRACE: grabbing output claims ---
        output_claims_span.exit();

        // --- TRACE: Proving intermediate GKR layers ---
        let all_layers_sumcheck_proving_span =
            span!(Level::DEBUG, "all_layers_sumcheck_proving_span").entered();

        // --- Collects all the prover messages for sumchecking over each layer, ---
        // --- as well as all the prover messages for claim aggregation at the ---
        // --- beginning of proving each layer ---
        let layer_sumcheck_proofs = layers
            .0
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
                let layer_claims_vec = claims
                    .get(&layer_id)
                    .ok_or_else(|| GKRError::NoClaimsForLayer(layer_id.clone()))?
                    .clone();
                let layer_claim_group = ClaimGroup::new(layer_claims_vec.clone()).unwrap();
                debug!("Found Layer claims:\n{:#?}", layer_claim_group);

                // --- Add the claimed values to the FS transcript ---
                for claim in &layer_claims_vec {
                    transcript
                        .append_field_elements("Claimed bits to be aggregated", claim.get_point())
                        .unwrap();
                    transcript
                        .append_field_element("Claimed value to be aggregated", claim.get_result())
                        .unwrap();
                }

                info!("Time for claim aggregation...");
                let claim_aggr_timer =
                    start_timer!(|| format!("claim aggregation for layer {:?}", *layer.id()));

                let layer_init_wlx_count = wlx_count;

                let (layer_claim, relevant_wlx_evaluations) = aggregate_claims(
                    &layer_claim_group,
                    &mut |claims, _, layer_mle_refs| {
                        let wlx_evals = layer
                            .get_wlx_evaluations(
                                claims.get_claim_points_matrix(),
                                claims.get_results(),
                                layer_mle_refs.unwrap().clone(),
                                claims.get_num_claims(),
                                claims.get_num_vars(),
                            )
                            .unwrap();
                        // wlx_count += wlx_evals.len();
                        println!("Layer {:?}: +{} evaluations.", layer_id, wlx_evals.len());
                        Ok(wlx_evals)
                    },
                    transcript,
                )
                .unwrap();
                info!("Done aggregating claims! New claim: {:#?}", layer_claim);

                // println!(
                //     "Total Evaluations for Intermediate Layer {:?}: {}",
                //     layer_id,
                //     wlx_count - layer_init_wlx_count
                // );

                debug!("Relevant wlx evals: {:#?}", relevant_wlx_evaluations);
                end_timer!(claim_aggr_timer);
                let sumcheck_msg_timer = start_timer!(|| format!(
                    "compute sumcheck message for layer {:?}",
                    *layer.id()
                ));

                // --- Compute all sumcheck messages across this particular layer ---
                // dbg!(&layer_claim);
                // dbg!(&claims);
                let prover_sumcheck_messages = layer
                    .prove_rounds(layer_claim, transcript)
                    .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

                debug!("sumcheck_proof: {:#?}", prover_sumcheck_messages);
                end_timer!(sumcheck_msg_timer);

                // --- Grab all the resulting claims from the above sumcheck procedure and add them to the claim tracking map ---
                let post_sumcheck_new_claims = layer
                    .get_claims()
                    .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

                debug!(
                    "After sumcheck, I have the following claims: {:#?}",
                    post_sumcheck_new_claims
                );

                for claim in post_sumcheck_new_claims {
                    if let Some(curr_claims) = claims.get_mut(&claim.get_to_layer_id().unwrap()) {
                        curr_claims.push(claim);
                    } else {
                        claims.insert(claim.get_to_layer_id().unwrap(), vec![claim]);
                    }
                }

                end_timer!(layer_timer);

                Ok(LayerProof {
                    sumcheck_proof: prover_sumcheck_messages,
                    layer,
                    wlx_evaluations: relevant_wlx_evaluations,
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

                let layer_claims_vec = claims
                    .get(&layer_id)
                    .ok_or_else(|| GKRError::NoClaimsForLayer(layer_id.clone()))?;

                let layer_claim_group = ClaimGroup::new(layer_claims_vec.clone()).unwrap();
                trace!(
                    "Layer Claim Group for {:?}:\n{:#?}",
                    layer_id,
                    layer_claim_group
                );

                // --- Add the claimed values to the FS transcript ---
                for claim in layer_claims_vec {
                    transcript
                        .append_field_elements(
                            "Claimed challenge coordinates to be aggregated",
                            claim.get_point(),
                        )
                        .unwrap();
                    transcript
                        .append_field_element("Claimed value to be aggregated", claim.get_result())
                        .unwrap();
                }

                let claim_aggr_timer = start_timer!(|| format!(
                    "claim aggregation for INPUT layer {:?}",
                    input_layer.layer_id()
                ));

                let layer_init_wlx_count = wlx_count;

                let (layer_claim, relevant_wlx_evaluations) = aggregate_claims(
                    &layer_claim_group,
                    &mut |claims, _, _| {
                        let wlx_evals = input_layer.compute_claim_wlx(claims).unwrap();
                        // wlx_count += wlx_evals.len();
                        println!(
                            "Input Layer {:?}: +{} evaluations.",
                            layer_id,
                            wlx_evals.len()
                        );
                        Ok(wlx_evals)
                    },
                    transcript,
                )
                .unwrap();

                // println!(
                //     "Total Evaluations for Input Layer {:?}: {}",
                //     layer_id,
                //     wlx_count - layer_init_wlx_count
                // );
                debug!("Relevant wlx evaluations: {:#?}", relevant_wlx_evaluations);
                end_timer!(claim_aggr_timer);

                let opening_proof_timer = start_timer!(|| format!(
                    "opening proof for INPUT layer {:?}",
                    input_layer.layer_id()
                ));

                let opening_proof = input_layer
                    .open(transcript, layer_claim)
                    .map_err(GKRError::InputLayerError)?;

                end_timer!(opening_proof_timer);

                end_timer!(layer_timer);

                Ok(InputLayerProof {
                    layer_id: *layer_id,
                    input_commitment: commitment,
                    input_layer_wlx_evaluations: relevant_wlx_evaluations,
                    input_opening_proof: opening_proof,
                })
            })
            .try_collect()?;

        end_timer!(input_layers_timer);
        println!("TOTAL EVALUATIONS: {}", wlx_count);
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
        transcript: &mut Self::Transcript,
        gkr_proof: GKRProof<F, Self::Transcript>,
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
            transcript
                .append_field_element("Circuit Hash", circuit_hash)
                .unwrap();
        }

        for input_layer in input_layer_proofs.iter() {
            InputLayerEnum::append_commitment_to_transcript(
                &input_layer.input_commitment,
                transcript,
            )
            .map_err(|err| {
                GKRError::ErrorWhenVerifyingLayer(
                    input_layer.layer_id,
                    LayerError::TranscriptError(err),
                )
            })?;
        }
        end_timer!(input_layers_timer);

        // --- Verifier keeps track of the claims on its own ---
        let mut claims: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();

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
                let challenge = transcript
                    .get_challenge("Setting Output Layer Claim")
                    .unwrap();

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

            let claim = Claim::new(
                mle_indices
                    .iter()
                    .map(|index| index.val().unwrap())
                    .collect(),
                F::zero(),
                None,
                Some(layer_id),
                Some(output.clone()),
            );

            debug!("Generating claim: {:#?}", claim);
            // --- Append claims to either the claim tracking map OR the first (sumchecked) layer's list of claims ---
            if let Some(curr_claims) = claims.get_mut(&layer_id) {
                curr_claims.push(claim);
            } else {
                claims.insert(layer_id, vec![claim]);
            }
        }

        end_timer!(claims_timer);

        let intermediate_layers_timer =
            start_timer!(|| "ALL intermediate layers proof verification");
        // --- END TRACE: output claims ---
        verifier_output_claims_span.exit();

        // --- Go through each of the layers' sumcheck proofs... ---
        for sumcheck_proof_single in layer_sumcheck_proofs {
            let LayerProof {
                sumcheck_proof,
                mut layer,
                wlx_evaluations: relevant_wlx_evaluations,
            } = sumcheck_proof_single;

            let layer_timer =
                start_timer!(|| format!("proof verification for layer {:?}", *layer.id()));

            // --- TRACE: Proving an individual GKR layer ---
            let layer_id = *layer.id();
            info!("Intermediate Layer: {:?}", layer_id);
            debug!("The LayerEnum: {:#?}", layer);
            let layer_claims = claims
                .get(&layer_id)
                .ok_or_else(|| GKRError::NoClaimsForLayer(layer_id.clone()))?;
            let layer_claim_group = ClaimGroup::new(layer_claims.clone()).unwrap();
            debug!("Found Layer claims:\n{:#?}", layer_claim_group);
            let layer_num_claims = layer_claim_group.get_num_claims();
            // --- TRACE: Proving an individual GKR layer ---
            let layer_id = *layer.id();
            let layer_id_trace_repr = format!("{}", layer_id);
            let _layer_sumcheck_verification_span = span!(
                Level::DEBUG,
                "layer_sumcheck_verification_span",
                layer_id = layer_id_trace_repr
            )
            .entered();

            // --- Independently grab the claims which should've been imposed on this layer (based on the verifier's own claim tracking) ---
            let layer_claims = claims
                .get(&layer_id)
                .ok_or(GKRError::NoClaimsForLayer(layer_id))?;

            // --- Append claims to the FS transcript... TODO!(ryancao): Do we actually need to do this??? ---
            for claim in layer_claims {
                transcript
                    .append_field_elements("Claimed bits to be aggregated", claim.get_point())
                    .unwrap();
                transcript
                    .append_field_element("Claimed value to be aggregated", claim.get_result())
                    .unwrap();
            }

            let claim_aggr_timer =
                start_timer!(|| format!("verify aggregated claim for layer {:?}", *layer.id()));
            // --- Perform the claim aggregation verification, first sampling `r` ---
            // --- Note that we ONLY do this if need be! ---
            let mut prev_claim = layer_claims[0].clone();
            if layer_num_claims > 1 {
                info!("Got > 1 claims. Verifying aggregation...");

                // --- Perform the claim aggregation verification ---
                let (claim, _) = aggregate_claims(
                    &layer_claim_group, // This is the "claim group" representing ALL claims on this layer
                    &mut |claim_group: &ClaimGroup<F>,
                          prover_supplied_wlx_evaluations_idx: usize,
                          _|
                     -> Result<Vec<F>, GKRError> {
                        debug!("Compute_wlx was called during claim aggregation verification");

                        let claim_wlx_evaluations = claim_group.get_results().clone();

                        let all_wlx_evaluations: Vec<F> = claim_wlx_evaluations
                            .into_iter()
                            .chain(
                                relevant_wlx_evaluations[prover_supplied_wlx_evaluations_idx]
                                    .clone()
                                    .into_iter(),
                            )
                            .collect();

                        let claim_vecs = claim_group.get_claim_points_matrix();
                        let (expected_num_evals, _) = get_num_wlx_evaluations(claim_vecs);
                        if expected_num_evals != all_wlx_evaluations.len() {
                            return Err(GKRError::ErrorWhenVerifyingLayer(
                                layer_id,
                                LayerError::AggregationError,
                            ));
                        }

                        Ok(all_wlx_evaluations)
                    },
                    transcript,
                )?;
                prev_claim = claim;
            }

            end_timer!(claim_aggr_timer);

            let sumcheck_msg_timer =
                start_timer!(|| format!("verify sumcheck message for layer {:?}", *layer.id()));

            debug!("Aggregated claim: {:#?}", prev_claim);
            info!("Verifier: about to verify layer");

            // --- Performs the actual sumcheck verification step ---
            layer
                .verify_rounds(prev_claim, sumcheck_proof.0, transcript)
                .map_err(|err| GKRError::ErrorWhenVerifyingLayer(layer_id, err))?;

            end_timer!(sumcheck_msg_timer);

            // --- Extract/track claims from the expression independently (this ensures implicitly that the ---
            // --- prover is behaving correctly with respect to claim reduction, as otherwise the claim ---
            // --- aggregation verification step will fail ---
            let other_claims = layer
                .get_claims()
                .map_err(|err| GKRError::ErrorWhenVerifyingLayer(layer_id, err))?;

            debug!(
                "After sumcheck, I have the following claims: {:#?}",
                other_claims
            );

            for claim in other_claims {
                if let Some(curr_claims) = claims.get_mut(&claim.get_to_layer_id().unwrap()) {
                    curr_claims.push(claim);
                } else {
                    claims.insert(claim.get_to_layer_id().unwrap(), vec![claim]);
                }
            }

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
            let relevant_wlx_evaluations = input_layer.input_layer_wlx_evaluations.clone();
            info!("--- Input Layer: {:?} ---", input_layer_id);
            let input_layer_claims = claims
                .get(&input_layer_id)
                .ok_or_else(|| GKRError::NoClaimsForLayer(input_layer_id.clone()))?;
            let input_layer_claim_group = ClaimGroup::new(input_layer_claims.clone()).unwrap();
            debug!("Layer Claim Group for input: {:#?}", input_layer_claims);

            // --- Add the claimed values to the FS transcript ---
            for claim in input_layer_claims {
                transcript
                    .append_field_elements(
                        "Claimed challenge coordinates to be aggregated",
                        claim.get_point(),
                    )
                    .unwrap();
                transcript
                    .append_field_element("Claimed value to be aggregated", claim.get_result())
                    .unwrap();
            }

            let claim_aggr_timer = start_timer!(|| format!(
                "verify aggregated claim for INPUT layer {:?}",
                input_layer.layer_id
            ));

            let input_layer_claim = if input_layer_claims.len() > 1 {
                let (prev_claim, _) = aggregate_claims(
                    &input_layer_claim_group,
                    &mut |claim_group,
                          prover_supplied_wlx_evaluations_idx,
                          _|
                     -> Result<Vec<F>, GKRError> {
                        debug!("Compute_wlx was called during claim aggregation verification in the input layer");

                        let claim_wlx_evaluations = claim_group.get_results().clone();
                        let all_wlx_evaluations: Vec<F> = claim_wlx_evaluations
                            .into_iter()
                            .chain(
                                relevant_wlx_evaluations[prover_supplied_wlx_evaluations_idx]
                                    .clone()
                                    .into_iter(),
                            )
                            .collect();

                        let claim_vecs = claim_group.get_claim_points_matrix();
                        let (expected_num_evals, _) = get_num_wlx_evaluations(claim_vecs);
                        if expected_num_evals != all_wlx_evaluations.len() {
                            debug!("wlx eval lengths don't match:\nexpected = {expected_num_evals}\nfound = {}", all_wlx_evaluations.len());
                            return Err(GKRError::ErrorWhenVerifyingLayer(
                                input_layer_id,
                                LayerError::AggregationError,
                            ));
                        }

                        Ok(all_wlx_evaluations)
                    },
                    transcript,
                )?;

                prev_claim
            } else {
                input_layer_claims[0].clone()
            };

            debug!("Input layer claim: {:#?}", input_layer_claim);
            end_timer!(claim_aggr_timer);

            let sumcheck_msg_timer = start_timer!(|| format!(
                "verify sumcheck message for INPUT layer {:?}",
                input_layer.layer_id
            ));

            InputLayerEnum::verify(
                &input_layer.input_commitment,
                &input_layer.input_opening_proof,
                input_layer_claim,
                transcript,
            )
            .map_err(GKRError::InputLayerError)?;

            end_timer!(sumcheck_msg_timer);

            end_timer!(layer_timer);
        }

        end_timer!(input_layers_timer);

        Ok(())
    }

    ///Gen the circuit hash
    fn gen_circuit_hash(&mut self) -> F {
        let mut transcript = Self::Transcript::new("blah");
        let (Witness { layers, .. }, _) = self.synthesize_and_commit(&mut transcript).unwrap();

        hash_layers(&layers)
    }

    fn get_circuit_hash() -> Option<F> {
        Self::CIRCUIT_HASH.map(|bytes| F::from_bytes_le(&bytes))
    }
}
