//!Module that orchestrates creating a GKR Proof

pub mod combine_layers;
/// For the input layer to the GKR circuit
pub mod input_layer;
pub mod test_helper_circuits;
#[cfg(test)]
pub(crate) mod tests;

use std::collections::HashMap;

use crate::{
    gate::{
        addgate::AddGate, batched_addgate::AddGateBatched, batched_mulgate::MulGateBatched,
        mulgate::MulGate,
    },
    layer::{
        claims::aggregate_claims,
        claims::compute_aggregated_challenges,
        claims::compute_claim_wlx,
        claims::verify_aggregate_claim,
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
    utils::pad_to_nearest_power_of_two,
};

// use lcpc_2d::{FieldExt, ligero_commit::{remainder_ligero_commit_prove, remainder_ligero_eval_prove, remainder_ligero_verify}, adapter::convert_halo_to_lcpc, LcProofAuxiliaryInfo, poseidon_ligero::PoseidonSpongeHasher, ligero_structs::LigeroEncoding, ligero_ml_helper::naive_eval_mle_at_challenge_point};
// use lcpc_2d::fs_transcript::halo2_remainder_transcript::Transcript;

use remainder_shared_types::transcript::Transcript;
use remainder_shared_types::FieldExt;

// use derive_more::From;
use itertools::{Either, Itertools};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use self::input_layer::{
    enum_input_layer::{CommitmentEnum, InputLayerEnum, OpeningEnum},
    InputLayer, InputLayerError,
};

use core::cmp::Ordering;

use log::{debug, info, trace, warn};

// use lcpc_2d::ScalarField;
// use lcpc_2d::adapter::LigeroProof;

///  New  type for containing the list of Layers that make up the GKR circuit
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

    /// Add an Add Gate layer to a list of layers (unbatched version)
    ///
    /// # Arguments
    /// `nonzero_gates`: the gate wiring between `lhs` and `rhs` represented as tuples (z, x, y) where
    /// x is the label on the `lhs`, y is the label on the `rhs`, and z is the label on the next layer
    /// `lhs`: the mle representing the left side of the sum
    /// `rhs`: the mle representing the right side of the sum
    ///
    /// # Returns
    /// A `DenseMle` that represents the evaluations of the add gate wiring on `lhs` and `rhs` over the boolean hypercube
    pub fn add_add_gate(
        &mut self,
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMleRef<F>,
        rhs: DenseMleRef<F>,
    ) -> DenseMle<F, F> {
        let id = LayerId::Layer(self.0.len());
        // use the add gate constructor in order to initialize a new add gate mle
        let gate: AddGate<F, Tr> =
            AddGate::new(id, nonzero_gates.clone(), lhs.clone(), rhs.clone(), 0, None);

        // we want to return an mle representing the evaluations of this over all the points in the boolean hypercube
        // the size of this mle is dependent on the max gate label given in the z coordinate of the tuples (as defined above)
        let max_gate_val = nonzero_gates
            .clone()
            .into_iter()
            .fold(0, |acc, (z, _, _)| std::cmp::max(acc, z));
        self.0.push(gate.get_enum());

        // we use the nonzero add gates in order to evaluate the values at the next layer
        let mut sum_table = vec![F::zero(); max_gate_val + 1];
        nonzero_gates.into_iter().for_each(|(z, x, y)| {
            let sum_val = *lhs.bookkeeping_table().get(x).unwrap_or(&F::zero())
                + *rhs.bookkeeping_table().get(y).unwrap_or(&F::zero());
            sum_table[z] = sum_val;
        });

        let res_mle: DenseMle<F, F> = DenseMle::new_from_raw(sum_table, id, None);
        res_mle
    }

    /// Add a Mul Gate layer to a list of layers (unbatched version)
    ///
    /// # Arguments
    /// `nonzero_gates`: the gate wiring between `lhs` and `rhs` represented as tuples (z, x, y) where
    /// x is the label on the `lhs`, y is the label on the `rhs`, and z is the label on the next layer
    /// `lhs`: the mle representing the left side of the multiplication
    /// `rhs`: the mle representing the right side of the multiplication
    ///
    /// # Returns
    /// A `DenseMle` that represents the evaluations of the mul gate wiring on `lhs` and `rhs` over the boolean hypercube
    pub fn add_mul_gate(
        &mut self,
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMleRef<F>,
        rhs: DenseMleRef<F>,
    ) -> DenseMle<F, F> {
        let id = LayerId::Layer(self.0.len());
        // use the mul gate constructor in order to initialize a new add gate mle
        let gate: MulGate<F, Tr> =
            MulGate::new(id, nonzero_gates.clone(), lhs.clone(), rhs.clone(), 0, None);

        // we want to return an mle representing the evaluations of this over all the points in the boolean hypercube
        // the size of this mle is dependent on the max gate label given in the z coordinate of the tuples (as defined above)
        let max_gate_val = nonzero_gates
            .clone()
            .into_iter()
            .fold(0, |acc, (z, _, _)| std::cmp::max(acc, z));
        self.0.push(gate.get_enum());

        // we use the nonzero mul gates in order to evaluate the values at the next layer
        let mut mul_table = vec![F::zero(); max_gate_val + 1];
        nonzero_gates.into_iter().for_each(|(z, x, y)| {
            let mul_val = *lhs.bookkeeping_table().get(x).unwrap_or(&F::zero())
                * *rhs.bookkeeping_table().get(y).unwrap_or(&F::zero());
            mul_table[z] = mul_val;
        });

        let res_mle: DenseMle<F, F> = DenseMle::new_from_raw(mul_table, id, None);
        res_mle
    }

    /// Add a batched Add Gate layer to a list of layers
    /// In the batched case, consider a vector of mles corresponding to an mle for each "batch" or "copy".
    /// Then we refer to the mle that represents the concatenation of these mles by interleaving as the
    /// flattened mle and each individual mle as a batched mle.
    ///
    /// # Arguments
    /// `nonzero_gates`: the gate wiring between single-copy circuit (as the wiring for each circuit remains the same)
    /// x is the label on the batched mle `lhs`, y is the label on the batched mle `rhs`, and z is the label on the next layer, batched
    /// `lhs`: the flattened mle representing the left side of the summation
    /// `rhs`: the flattened mle representing the right side of the summation
    /// `num_dataparallel_bits`: the number of bits representing the circuit copy we are looking at
    ///
    /// # Returns
    /// A flattened `DenseMle` that represents the evaluations of the add gate wiring on `lhs` and `rhs` over the boolean hypercube
    pub fn add_add_gate_batched(
        &mut self,
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMleRef<F>,
        rhs: DenseMleRef<F>,
        num_dataparallel_bits: usize,
    ) -> DenseMle<F, F> {
        let id = LayerId::Layer(self.0.len());
        // constructor for batched add gate struct
        let gate: AddGateBatched<F, Tr> = AddGateBatched::new(
            num_dataparallel_bits,
            nonzero_gates.clone(),
            lhs.clone(),
            rhs.clone(),
            id,
        );
        let max_gate_val = nonzero_gates
            .clone()
            .into_iter()
            .fold(0, |acc, (z, _, _)| std::cmp::max(acc, z));

        // number of entries in the resulting table is the max gate z value * 2 to the power of the number of dataparallel bits, as we are
        // evaluating over all values in the boolean hypercube which includes dataparallel bits
        let num_dataparallel_vals = 1 << num_dataparallel_bits;
        let sum_table_num_entries = (max_gate_val + 1) * num_dataparallel_vals;
        self.0.push(gate.get_enum());

        // iterate through each of the indices and compute the sum
        let mut sum_table = vec![F::zero(); sum_table_num_entries];
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
                    sum_table[idx + (z_ind * num_dataparallel_vals)] = f2_val + f3_val;
                });
        });
        let res_mle: DenseMle<F, F> = DenseMle::new_from_raw(sum_table, id, None);
        res_mle
    }

    /// Add a batched Mul Gate layer to a list of layers
    /// In the batched case, consider a vector of mles corresponding to an mle for each "batch" or "copy".
    /// Then we refer to the mle that represents the concatenation of these mles by interleaving as the
    /// flattened mle and each individual mle as a batched mle.
    ///
    /// # Arguments
    /// `nonzero_gates`: the gate wiring between single-copy circuit (as the wiring for each circuit remains the same)
    /// x is the label on the batched mle `lhs`, y is the label on the batched mle `rhs`, and z is the label on the next layer, batched
    /// `lhs`: the flattened mle representing the left side of the summation
    /// `rhs`: the flattened mle representing the right side of the summation
    /// `num_dataparallel_bits`: the number of bits representing the circuit copy we are looking at
    ///
    /// # Returns
    /// A flattened `DenseMle` that represents the evaluations of the mul gate wiring on `lhs` and `rhs` over the boolean hypercube
    pub fn add_mul_gate_batched(
        &mut self,
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMleRef<F>,
        rhs: DenseMleRef<F>,
        num_dataparallel_bits: usize,
    ) -> DenseMle<F, F> {
        let id = LayerId::Layer(self.0.len());
        // constructor for batched mul gate struct
        let gate: MulGateBatched<F, Tr> = MulGateBatched::new(
            num_dataparallel_bits,
            nonzero_gates.clone(),
            lhs.clone(),
            rhs.clone(),
            id,
        );
        let max_gate_val = nonzero_gates
            .clone()
            .into_iter()
            .fold(0, |acc, (z, _, _)| std::cmp::max(acc, z));
        let max_gate_val = nonzero_gates
            .clone()
            .into_iter()
            .fold(0, |acc, (z, _, _)| std::cmp::max(acc, z));

        // number of entries in the resulting table is the max gate z value * 2 to the power of the number of dataparallel bits, as we are
        // evaluating over all values in the boolean hypercube which includes dataparallel bits
        let num_dataparallel_vals = 1 << num_dataparallel_bits;
        let sum_table_num_entries = (max_gate_val + 1) * num_dataparallel_vals;
        self.0.push(gate.get_enum());

        // iterate through each of the indices and compute the product
        let mut mul_table = vec![F::zero(); sum_table_num_entries];
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
                    mul_table[idx + (z_ind * num_dataparallel_vals)] = f2_val * f3_val;
                });
        });

        let res_mle: DenseMle<F, F> = DenseMle::new_from_raw(mul_table, id, None);

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
    #[error("Error when commiting to InputLayer {0}")]
    InputLayerError(InputLayerError),
}

/// A proof of the sumcheck protocol; Outer vec is rounds, inner vec is evaluations
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

    pub input_layer_proofs: Vec<InputLayerProof<F>>,
}

pub struct Witness<F: FieldExt, Tr: Transcript<F>> {
    pub layers: Layers<F, Tr>,
    pub output_layers: Vec<MleEnum<F>>,
    pub input_layers: Vec<InputLayerEnum<F, Tr>>,
}

/// Controls claim aggregation behavior.
const ENABLE_OPTIMIZATION: bool = false;

/// A GKRCircuit ready to be proven
pub trait GKRCircuit<F: FieldExt> {
    /// The transcript this circuit uses
    type Transcript: Transcript<F>;

    /// The forward pass, defining the layer relationships and generating the layers
    fn synthesize(&mut self) -> Witness<F, Self::Transcript>;

    /// Calls `synthesize` and also generates commitments from each of the input layers
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

    ///  The backwards pass, creating the GKRProof
    fn prove(
        &mut self,
        transcript: &mut Self::Transcript,
    ) -> Result<GKRProof<F, Self::Transcript>, GKRError> {
        // --- Synthesize the circuit, using LayerBuilders to create internal, output, and input layers ---
        // --- Also commit and add those commitments to the transcript
        info!("Synethesizing circuit...");
        let (
            Witness {
                input_layers,
                mut output_layers,
                layers,
            },
            commitments,
        ) = self.synthesize_and_commit(transcript)?;
        info!("Circuit synthesized and witness generated.");

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
            debug!("Creating a claim: {:#?}", claim);

            // --- Add the claim to either the set of current claims we're proving ---
            // --- or the global set of claims we need to eventually prove ---
            if let Some(curr_claims) = claims.get_mut(&layer_id) {
                curr_claims.push(claim);
            } else {
                claims.insert(layer_id, vec![claim]);
            }
        }

        // --- Collects all the prover messages for sumchecking over each layer, ---
        // --- as well as all the prover messages for claim aggregation at the ---
        // --- beginning of proving each layer ---
        let layer_sumcheck_proofs = layers
            .0
            .into_iter()
            .rev()
            .map(|mut layer| {
                // --- For each layer, get the ID and all the claims on that layer ---
                let layer_id = *layer.id();
                info!("New Intermediate Layer: {:?}", layer_id);
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
                let (layer_claim, relevant_wlx_evaluations) = aggregate_claims(
                    &layer_claim_group,
                    &mut |claims| compute_claim_wlx(claims, &layer).unwrap(),
                    transcript,
                    ENABLE_OPTIMIZATION,
                );
                info!("Done aggregating claims! New claim: {:#?}", layer_claim);
                debug!("Relevant wlx evals: {:#?}", relevant_wlx_evaluations);

                // --- Compute all sumcheck messages across this particular layer ---
                let prover_sumcheck_messages = layer
                    .prove_rounds(layer_claim, transcript)
                    .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

                debug!("sumcheck_proof: {:#?}", prover_sumcheck_messages);

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

                Ok(LayerProof {
                    sumcheck_proof: prover_sumcheck_messages,
                    layer,
                    wlx_evaluations: relevant_wlx_evaluations.unwrap_or_default(),
                })
            })
            .try_collect()?;

        let input_layer_proofs = input_layers
            .into_iter()
            .zip(commitments)
            .map(|(input_layer, commitment)| {
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
                let num_claims = layer_claim_group.get_num_claims();

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

                let (layer_claim, relevant_wlx_evaluations) = aggregate_claims(
                    &layer_claim_group,
                    &mut |claims| input_layer.compute_claim_wlx(claims).unwrap(),
                    transcript,
                    ENABLE_OPTIMIZATION,
                );
                debug!("Relevant wlx evaluations: {:#?}", relevant_wlx_evaluations);

                let opening_proof = input_layer
                    .open(transcript, layer_claim)
                    .map_err(GKRError::InputLayerError)?;

                Ok(InputLayerProof {
                    layer_id: *layer_id,
                    input_commitment: commitment,
                    input_layer_wlx_evaluations: relevant_wlx_evaluations.unwrap_or_default(),
                    input_opening_proof: opening_proof,
                })
            })
            .try_collect()?;

        let gkr_proof = GKRProof {
            layer_sumcheck_proofs,
            output_layers,
            input_layer_proofs,
        };

        Ok(gkr_proof)
    }

    /// Verifies the GKRProof produced by fn prove
    ///
    /// Takes in a transcript for FS and re-generates challenges on its own
    fn verify(
        &mut self,
        transcript: &mut Self::Transcript,
        gkr_proof: GKRProof<F, Self::Transcript>,
    ) -> Result<(), GKRError> {
        let GKRProof {
            layer_sumcheck_proofs,
            output_layers,
            input_layer_proofs,
        } = gkr_proof;

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

        // --- Verifier keeps track of the claims on its own ---
        let mut claims: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();

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
            );

            debug!("Generating claim: {:#?}", claim);
            // --- Append claims to either the claim tracking map OR the first (sumchecked) layer's list of claims ---
            if let Some(curr_claims) = claims.get_mut(&layer_id) {
                curr_claims.push(claim);
            } else {
                claims.insert(layer_id, vec![claim]);
            }
        }

        // --- Go through each of the layers' sumcheck proofs... ---
        for sumcheck_proof_single in layer_sumcheck_proofs {
            let LayerProof {
                sumcheck_proof,
                mut layer,
                wlx_evaluations: relevant_wlx_evaluations,
            } = sumcheck_proof_single;

            // --- Independently grab the claims which should've been imposed on this layer (based on the verifier's own claim tracking) ---
            let layer_id = *layer.id();
            info!("Intermediate Layer: {:?}", layer_id);
            debug!("The LayerEnum: {:#?}", layer);
            let layer_claims = claims
                .get(&layer_id)
                .ok_or_else(|| GKRError::NoClaimsForLayer(layer_id.clone()))?;
            let layer_claim_group = ClaimGroup::new(layer_claims.clone()).unwrap();
            debug!("Found Layer claims:\n{:#?}", layer_claim_group);
            let layer_num_claims = layer_claim_group.get_num_claims();

            // --- Append claims to the FS transcript... TODO!(ryancao): Do we actually need to do this??? ---
            for claim in layer_claims {
                transcript
                    .append_field_elements("Claimed bits to be aggregated", claim.get_point())
                    .unwrap();
                transcript
                    .append_field_element("Claimed value to be aggregated", claim.get_result())
                    .unwrap();
            }

            // --- Perform the claim aggregation verification, first sampling `r` ---
            // --- Note that we ONLY do this if need be! ---
            let mut prev_claim = layer_claims[0].clone();
            if layer_num_claims > 1 {
                info!("Got > 1 claims. Verifying aggregation...");
                // --- Perform the claim aggregation verification ---

                // let all_wlx_evaluations: Vec<F> = layer_claims
                //     .into_iter()
                //     .map(|claim| claim.get_result().clone())
                //     .chain(relevant_wlx_evaluations.clone().into_iter())
                //     .collect();
                // debug!("All wlx evals:\n{:#?}", all_wlx_evaluations);

                let mut idx = 0;
                let (claim, _) = aggregate_claims(
                    &layer_claim_group,
                    &mut |claim_group| {
                        debug!("Compute_wlx was called during claim aggregation verification");
                        idx += 1;
                        let current_wlx_evaluations = relevant_wlx_evaluations[idx - 1].clone();
                        let all_wlx_evaluations: Vec<F> = claim_group
                            .get_claim_vector()
                            .into_iter()
                            .map(|claim| claim.get_result().clone())
                            .chain(current_wlx_evaluations.into_iter())
                            .collect();
                        debug!("All wlx evals returned:\n{:#?}", all_wlx_evaluations);
                        all_wlx_evaluations
                    },
                    transcript,
                    ENABLE_OPTIMIZATION,
                );
                prev_claim = claim;

                if idx != relevant_wlx_evaluations.len() {
                    return Err(GKRError::ErrorWhenVerifyingLayer(
                        layer_id,
                        LayerError::AggregationError,
                    ));
                }

                // transcript
                //     .append_field_elements(
                //         "Claim Aggregation Wlx_evaluations",
                //         &relevant_wlx_evaluations,
                //     )
                //     .unwrap();
                // let agg_chal = transcript
                //     .get_challenge("Challenge for claim aggregation")
                //     .unwrap();
                // debug!("Aggregate challenge: {:#?}", agg_chal);

                // prev_claim =
                //     verify_aggregate_claim(&all_wlx_evaluations, &layer_claim_group, agg_chal)
                //         .map_err(|_err| {
                //             GKRError::ErrorWhenVerifyingLayer(
                //                 layer_id.clone(),
                //                 LayerError::AggregationError,
                //             )
                //         })?;
            }

            debug!("Aggregated claim: {:#?}", prev_claim);
            info!("Verifier: about to verify layer");
            // debug!("sumcheck_proof: {:#?}", sumcheck_proof.0);
            // debug!(
            //     "msg0 + msg1 = {:?}",
            //     sumcheck_proof.0[0][0] + sumcheck_proof.0[0][1]
            // );
            // --- Performs the actual sumcheck verification step ---
            layer
                .verify_rounds(prev_claim, sumcheck_proof.0, transcript)
                .map_err(|err| GKRError::ErrorWhenVerifyingLayer(layer_id, err))?;

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
        }

        for input_layer in input_layer_proofs {
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

            let input_layer_claim = if input_layer_claims.len() > 1 {
                let mut idx = 0;
                let (prev_claim, _) = aggregate_claims(
                    &input_layer_claim_group,
                    &mut |claim_group| {
                        idx += 1;
                        let current_wlx_evaluations = relevant_wlx_evaluations[idx - 1].clone();
                        let all_wlx_evaluations: Vec<F> = claim_group
                            .get_claim_vector()
                            .into_iter()
                            .map(|claim| claim.get_result().clone())
                            .chain(current_wlx_evaluations.into_iter())
                            .collect();
                        debug!("All wlx evals:\n{:#?}", all_wlx_evaluations);
                        all_wlx_evaluations
                    },
                    transcript,
                    ENABLE_OPTIMIZATION,
                );

                if idx != relevant_wlx_evaluations.len() {
                    return Err(GKRError::ErrorWhenVerifyingLayer(
                        input_layer_id,
                        LayerError::AggregationError,
                    ));
                }

                prev_claim
                // let all_input_wlx_evaluations: Vec<F> = input_layer_claims
                //     .into_iter()
                //     .map(|claim| claim.get_result().clone())
                //     .chain(input_layer.input_layer_wlx_evaluations.clone().into_iter())
                //     .collect();
                // debug!("All wlx evaluations: {:#?}", all_input_wlx_evaluations);

                // // --- Add the aggregation step to the transcript ---
                // transcript
                //     .append_field_elements(
                //         "Input claim aggregation Wlx_evaluations",
                //         &input_layer.input_layer_wlx_evaluations,
                //     )
                //     .unwrap();

                // // --- Grab the input claim aggregation challenge ---
                // let input_r_star = transcript
                //     .get_challenge("Challenge for input claim aggregation")
                //     .unwrap();
                // debug!("Aggregation challenge: {:?}", input_r_star);

                // // --- Perform the aggregation verification step and extract the correct input layer claim ---
                // verify_aggregate_claim(
                //     &all_input_wlx_evaluations,
                //     &input_layer_claim_group,
                //     input_r_star,
                // )
                // .map_err(|_err| {
                //     GKRError::ErrorWhenVerifyingLayer(input_layer_id, LayerError::AggregationError)
                // })?
            } else {
                input_layer_claims[0].clone()
            };
            debug!("Input layer claim: {:#?}", input_layer_claim);

            InputLayerEnum::verify(
                &input_layer.input_commitment,
                &input_layer.input_opening_proof,
                input_layer_claim,
                transcript,
            )
            .map_err(GKRError::InputLayerError)?;
        }

        Ok(())
    }
}
