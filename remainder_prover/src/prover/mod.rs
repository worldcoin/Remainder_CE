//!Module that orchestrates creating a GKR Proof

pub mod combine_layers;
/// For the input layer to the GKR circuit
pub mod input_layer;
#[cfg(test)]
pub(crate) mod tests;
pub mod test_helper_circuits;

use std::collections::HashMap;

use crate::{
    layer::{
        claims::compute_aggregated_challenges, claims::compute_claim_wlx,
        claims::verify_aggregate_claim, layer_enum::LayerEnum, Claim, GKRLayer, Layer,
        LayerBuilder, LayerError, LayerId,
    },
    mle::{
        dense::{DenseMle, DenseMleRef},
        MleRef,
    },
    mle::{MleIndex, mle_enum::MleEnum}, sumcheck::evaluate_at_a_point, gate::{addgate::AddGate, mulgate::MulGate, batched_addgate::AddGateBatched, batched_mulgate::MulGateBatched}
};

use remainder_shared_types::transcript::{Transcript};
use remainder_shared_types::FieldExt;

// use derive_more::From;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{instrument, span, event, Level};

use self::input_layer::{
    enum_input_layer::{CommitmentEnum, InputLayerEnum, OpeningEnum},
    InputLayer, InputLayerError,
};

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

    /// Add an Add Gate layer to a list of layers (unbatched version)
    /// 
    /// # Arguments
    /// * `nonzero_gates`: the gate wiring between `lhs` and `rhs` represented as tuples (z, x, y) where 
    /// x is the label on the `lhs`, y is the label on the `rhs`, and z is the label on the next layer
    /// * `lhs`: the mle representing the left side of the sum
    /// * `rhs`: the mle representing the right side of the sum
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
        let gate: AddGate<F, Tr> = AddGate::new(id, nonzero_gates.clone(), lhs.clone(), rhs.clone(), 0, None);
        
        // we want to return an mle representing the evaluations of this over all the points in the boolean hypercube
        // the size of this mle is dependent on the max gate label given in the z coordinate of the tuples (as defined above)
        let max_gate_val = nonzero_gates.clone().into_iter().fold(
            0, 
            |acc, (z, _, _)| {
                std::cmp::max(acc, z)
            }
        );
        self.0.push(gate.get_enum());

        // we use the nonzero add gates in order to evaluate the values at the next layer
        let mut sum_table = vec![F::zero(); max_gate_val + 1];
        nonzero_gates.into_iter().for_each(
            |(z, x, y)| {
                let sum_val = *lhs.bookkeeping_table().get(x).unwrap_or(&F::zero()) + 
                *rhs.bookkeeping_table().get(y).unwrap_or(&F::zero());
                sum_table[z] = sum_val;
                
            }
        );

        let res_mle: DenseMle<F, F> = DenseMle::new_from_raw(sum_table, id, None);
        res_mle
    }

    /// Add a Mul Gate layer to a list of layers (unbatched version)
    /// 
    /// # Arguments
    /// * `nonzero_gates`: the gate wiring between `lhs` and `rhs` represented as tuples (z, x, y) where 
    /// x is the label on the `lhs`, y is the label on the `rhs`, and z is the label on the next layer
    /// * `lhs`: the mle representing the left side of the multiplication
    /// * `rhs`: the mle representing the right side of the multiplication
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
        let gate: MulGate<F, Tr> = MulGate::new(id, nonzero_gates.clone(), lhs.clone(), rhs.clone(), 0, None);
        
        // we want to return an mle representing the evaluations of this over all the points in the boolean hypercube
        // the size of this mle is dependent on the max gate label given in the z coordinate of the tuples (as defined above)
        let max_gate_val = nonzero_gates.clone().into_iter().fold(
            0, 
            |acc, (z, _, _)| {
                std::cmp::max(acc, z)
            }
        );
        self.0.push(gate.get_enum());

        // we use the nonzero mul gates in order to evaluate the values at the next layer
        let mut mul_table = vec![F::zero(); max_gate_val + 1];
        nonzero_gates.into_iter().for_each(
            |(z, x, y)| {
                let mul_val = *lhs.bookkeeping_table().get(x).unwrap_or(&F::zero()) * 
                *rhs.bookkeeping_table().get(y).unwrap_or(&F::zero());
                mul_table[z] = mul_val;
                
            }
        );

        let res_mle: DenseMle<F, F> = DenseMle::new_from_raw(mul_table, id, None);
        res_mle
    }

    /// Add a batched Add Gate layer to a list of layers 
    /// In the batched case, consider a vector of mles corresponding to an mle for each "batch" or "copy".
    /// Then we refer to the mle that represents the concatenation of these mles by interleaving as the 
    /// flattened mle and each individual mle as a batched mle.
    /// 
    /// # Arguments
    /// * `nonzero_gates`: the gate wiring between single-copy circuit (as the wiring for each circuit remains the same)
    /// x is the label on the batched mle `lhs`, y is the label on the batched mle `rhs`, and z is the label on the next layer, batched
    /// * `lhs`: the flattened mle representing the left side of the summation
    /// * `rhs`: the flattened mle representing the right side of the summation
    /// * `num_dataparallel_bits`: the number of bits representing the circuit copy we are looking at
    /// 
    /// # Returns
    /// A flattened `DenseMle` that represents the evaluations of the add gate wiring on `lhs` and `rhs` over the boolean hypercube
    pub fn add_add_gate_batched(&mut self, nonzero_gates: Vec<(usize, usize, usize)>, lhs: DenseMleRef<F>, rhs: DenseMleRef<F>, num_dataparallel_bits: usize) -> DenseMle<F, F> {
        let id = LayerId::Layer(self.0.len());
        // constructor for batched add gate struct
        let gate: AddGateBatched<F, Tr> = AddGateBatched::new(num_dataparallel_bits, nonzero_gates.clone(), lhs.clone(), rhs.clone(), id);
        let max_gate_val = nonzero_gates.clone().into_iter().fold(
            0, 
            |acc, (z, _, _)| {
                std::cmp::max(acc, z)
            }
        );

        // number of entries in the resulting table is the max gate z value * 2 to the power of the number of dataparallel bits, as we are 
        // evaluating over all values in the boolean hypercube which includes dataparallel bits
        let num_dataparallel_vals = 1 << num_dataparallel_bits;
        let sum_table_num_entries = (max_gate_val + 1) * num_dataparallel_vals;
        self.0.push(gate.get_enum());



        // iterate through each of the indices and compute the sum
        let mut sum_table = vec![F::zero(); sum_table_num_entries];
        (0..num_dataparallel_vals).for_each(|idx|
            {
                nonzero_gates.clone().into_iter().for_each(
                    |(z_ind, x_ind, y_ind)| {
                        let f2_val = *lhs.bookkeeping_table().get(idx + (x_ind * num_dataparallel_vals)).unwrap_or(&F::zero());
                        let f3_val = *rhs.bookkeeping_table().get(idx + (y_ind * num_dataparallel_vals)).unwrap_or(&F::zero());
                        sum_table[idx + (z_ind * num_dataparallel_vals)] = f2_val + f3_val;
                    }
                );
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
    /// * `nonzero_gates`: the gate wiring between single-copy circuit (as the wiring for each circuit remains the same)
    /// x is the label on the batched mle `lhs`, y is the label on the batched mle `rhs`, and z is the label on the next layer, batched
    /// * `lhs`: the flattened mle representing the left side of the summation
    /// * `rhs`: the flattened mle representing the right side of the summation
    /// * `num_dataparallel_bits`: the number of bits representing the circuit copy we are looking at
    /// 
    /// # Returns
    /// A flattened `DenseMle` that represents the evaluations of the mul gate wiring on `lhs` and `rhs` over the boolean hypercube
    pub fn add_mul_gate_batched(&mut self, nonzero_gates: Vec<(usize, usize, usize)>, lhs: DenseMleRef<F>, rhs: DenseMleRef<F>, num_dataparallel_bits: usize) -> DenseMle<F, F> {
        let id = LayerId::Layer(self.0.len());
        // constructor for batched mul gate struct
        let gate: MulGateBatched<F, Tr> = MulGateBatched::new(num_dataparallel_bits, nonzero_gates.clone(), lhs.clone(), rhs.clone(), id);
        let max_gate_val = nonzero_gates.clone().into_iter().fold(
            0, 
            |acc, (z, _, _)| {
                std::cmp::max(acc, z)
            }
        );

        // number of entries in the resulting table is the max gate z value * 2 to the power of the number of dataparallel bits, as we are 
        // evaluating over all values in the boolean hypercube which includes dataparallel bits
        let num_dataparallel_vals = 1 << num_dataparallel_bits;
        let sum_table_num_entries = (max_gate_val + 1) * num_dataparallel_vals;
        self.0.push(gate.get_enum());


        // iterate through each of the indices and compute the product
        let mut mul_table = vec![F::zero(); sum_table_num_entries];
        (0..num_dataparallel_vals).for_each(|idx|
            {
                nonzero_gates.clone().into_iter().for_each(
                    |(z_ind, x_ind, y_ind)| {
                        let f2_val = *lhs.bookkeeping_table().get(idx + (x_ind * num_dataparallel_vals)).unwrap_or(&F::zero());
                        let f3_val = *rhs.bookkeeping_table().get(idx + (y_ind * num_dataparallel_vals)).unwrap_or(&F::zero());
                        mul_table[idx + (z_ind * num_dataparallel_vals)] = f2_val * f3_val;
                    }
                );
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
    /// Error for input layer commitment
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
    pub wlx_evaluations: Vec<F>,
}

/// Proof for circuit input layer
#[derive(Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct InputLayerProof<F: FieldExt> {
    pub layer_id: LayerId,
    pub input_layer_wlx_evaluations: Vec<F>,
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

/// A GKRCircuit ready to be proven
pub trait GKRCircuit<F: FieldExt> {
    /// The transcript this circuit uses
    type Transcript: Transcript<F>;

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
                let commitment = input_layer
                    .commit()
                    .map_err(GKRError::InputLayerError)?;
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
    ) -> Result<GKRProof<F, Self::Transcript>, GKRError> {
        // --- Synthesize the circuit, using LayerBuilders to create internal, output, and input layers ---
        // --- Also commit and add those commitments to the transcript
        let (
            Witness {
                input_layers,
                mut output_layers,
                layers,
            },
            commitments,
        ) = self.synthesize_and_commit(transcript)?;

        // --- TRACE: grabbing output claims ---
        let output_claims_span = span!(Level::DEBUG, "output_claims_span").entered();

        // --- Keep track of GKR-style claims across all layers ---
        let mut claims: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();

        // --- Go through circuit output layers and grab claims on each ---
        for output in output_layers.iter_mut() {
            let mut claim = None;
            let bits = output.index_mle_indices(0);

            let claim = if bits != 0 {
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
                (vec![], output.bookkeeping_table()[0])
            };

            let layer_id = output.get_layer_id();

            // --- Add the claim to either the set of current claims we're proving ---
            // --- or the global set of claims we need to eventually prove ---
            if let Some(curr_claims) = claims.get_mut(&layer_id) {
                curr_claims.push(claim);
            } else {
                claims.insert(layer_id, vec![claim]);
            }
        }

        // --- END TRACE: grabbing output claims ---
        output_claims_span.exit();

        // --- TRACE: Proving intermediate GKR layers ---
        let all_layers_sumcheck_proving_span = span!(Level::DEBUG, "all_layers_sumcheck_proving_span").entered();

        // --- Collects all the prover messages for sumchecking over each layer, ---
        // --- as well as all the prover messages for claim aggregation at the ---
        // --- beginning of proving each layer ---
        let layer_sumcheck_proofs = layers
            .0
            .into_iter()
            .rev()
            .map(|mut layer| {

                // --- TRACE: Proving an individual GKR layer ---
                let layer_id = *layer.id();
                let _layer_sumcheck_proving_span = span!(Level::DEBUG, "layer_sumcheck_proving_span", layer_id = layer_id.to_tracing_repr()).entered();

                // --- For each layer, get the ID and all the claims on that layer ---
                let layer_claims = claims
                    .get(&layer_id)
                    .ok_or(GKRError::NoClaimsForLayer(layer_id))?;

                // --- Add the claimed values to the FS transcript ---
                for claim in layer_claims {
                    transcript
                        .append_field_elements("Claimed bits to be aggregated", &claim.0)
                        .unwrap();
                    transcript
                        .append_field_element("Claimed value to be aggregated", claim.1)
                        .unwrap();
                }

                // --- If it's an empty layer, skip the claim aggregation ---
                let empty_layer = layer_claims.iter().all(|claim| claim.0.is_empty());

                // --- Aggregate claims by sampling r^\star from the verifier and performing the ---
                // --- claim aggregation protocol. We ONLY aggregate if need be! ---
                let (layer_claim, relevant_wlx_evaluations) = if layer_claims.len() > 1 && !empty_layer {

                    // --- Aggregate claims by performing the claim aggregation protocol. First compute V_i(l(x)) ---
                    let wlx_evaluations = compute_claim_wlx(layer_claims, &layer).unwrap();
                    let relevant_wlx_evaluations = wlx_evaluations[layer_claims.len()..].to_vec();

                    transcript
                        .append_field_elements(
                            "Claim Aggregation Wlx_evaluations",
                            &relevant_wlx_evaluations,
                        )
                        .unwrap();

                    // --- Next, sample r^\star from the transcript ---
                    let agg_chal = transcript
                        .get_challenge("Challenge for claim aggregation")
                        .unwrap();

                    let aggregated_challenges =
                        compute_aggregated_challenges(layer_claims, agg_chal).unwrap();

                    let claimed_val = evaluate_at_a_point(&wlx_evaluations, agg_chal).unwrap();

                    (
                        (aggregated_challenges, claimed_val),
                        Some(relevant_wlx_evaluations),
                    )
                } else {
                    (layer_claims[0].clone(), None)
                };

                // --- Compute all sumcheck messages across this particular layer ---
                let prover_sumcheck_messages = layer
                    .prove_rounds(layer_claim, transcript)
                    .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

                // --- Grab all the resulting claims from the above sumcheck procedure and add them to the claim tracking map ---
                let post_sumcheck_new_claims = layer
                    .get_claims()
                    .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id, err))?;

                // dbg!(layer_id, &post_sumcheck_new_claims);

                for (layer_id, claim) in post_sumcheck_new_claims {
                    if let Some(curr_claims) = claims.get_mut(&layer_id) {
                        curr_claims.push(claim);
                    } else {
                        claims.insert(layer_id, vec![claim]);
                    }
                }

                Ok(LayerProof {
                    sumcheck_proof: prover_sumcheck_messages,
                    layer,
                    wlx_evaluations: relevant_wlx_evaluations.unwrap_or_default(),
                })
            })
            .try_collect()?;

        // --- END TRACE: Proving intermediate GKR layers ---
        all_layers_sumcheck_proving_span.exit();

        // --- TRACE: Proving input layer ---
        let input_layer_proving_span = span!(Level::DEBUG, "input_layer_proving_span").entered();

        let input_layer_proofs = input_layers
            .into_iter()
            .zip(commitments)
            .map(|(input_layer, commitment)| {
                let layer_id = input_layer.layer_id();

                let layer_claims = claims
                    .get(layer_id)
                    .ok_or(GKRError::NoClaimsForLayer(*layer_id))?;

                // --- Add the claimed values to the FS transcript ---
                for claim in layer_claims {
                    transcript
                        .append_field_elements(
                            "Claimed challenge coordinates to be aggregated",
                            &claim.0,
                        )
                        .unwrap();
                    transcript
                        .append_field_element("Claimed value to be aggregated", claim.1)
                        .unwrap();
                }

                let (layer_claim, relevant_wlx_evaluations) = if layer_claims.len() > 1 {
                    // --- Aggregate claims by performing the claim aggregation protocol. First compute V_i(l(x)) ---
                    let wlx_evaluations =
                        input_layer
                            .compute_claim_wlx(layer_claims)
                            .map_err(|err| {
                                GKRError::ErrorWhenProvingLayer(
                                    *layer_id,
                                    LayerError::ClaimError(err),
                                )
                            })?;
                    let relevant_wlx_evaluations = wlx_evaluations[layer_claims.len()..].to_vec();

                    transcript
                        .append_field_elements(
                            "Claim Aggregation Wlx_evaluations",
                            &relevant_wlx_evaluations,
                        )
                        .unwrap();

                    // --- Next, sample r^\star from the transcript ---
                    let agg_chal = transcript
                        .get_challenge("Challenge for claim aggregation")
                        .unwrap();

                    let aggregated_challenges = input_layer
                        .compute_aggregated_challenges(layer_claims, agg_chal)
                        .map_err(|err| {
                            GKRError::ErrorWhenProvingLayer(*layer_id, LayerError::ClaimError(err))
                        })?;
                    let claimed_val = evaluate_at_a_point(&wlx_evaluations, agg_chal).unwrap();

                    (
                        (aggregated_challenges, claimed_val),
                        Some(relevant_wlx_evaluations),
                    )
                } else {
                    (layer_claims[0].clone(), None)
                };

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

        // --- END TRACE: Proving input layer ---
        input_layer_proving_span.exit();

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

        // --- TRACE: output claims ---
        let verifier_output_claims_span = span!(Level::DEBUG, "verifier_output_claims_span").entered();

        // --- NOTE that all the `Expression`s and MLEs contained within `gkr_proof` are already bound! ---
        for output in output_layers.iter() {
            let mle_indices = output.mle_indices();
            let mut claim_chal: Vec<F> = vec![];
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
            let claim = (
                mle_indices
                    .iter()
                    .map(|index| index.val().unwrap())
                    .collect(),
                F::zero(),
            );
            let layer_id = output.get_layer_id();

            // --- Append claims to either the claim tracking map OR the first (sumchecked) layer's list of claims ---
            if let Some(curr_claims) = claims.get_mut(&layer_id) {
                curr_claims.push(claim);
            } else {
                claims.insert(layer_id, vec![claim]);
            }
        }

        // --- END TRACE: output claims ---
        verifier_output_claims_span.exit();

        // --- Go through each of the layers' sumcheck proofs... ---
        for sumcheck_proof_single in layer_sumcheck_proofs {
            let LayerProof {
                sumcheck_proof,
                mut layer,
                wlx_evaluations,
            } = sumcheck_proof_single;

            // --- TRACE: Proving an individual GKR layer ---
            let layer_id = *layer.id();
            let _layer_sumcheck_verification_span = span!(Level::DEBUG, "layer_sumcheck_verification_span", layer_id = layer_id.to_tracing_repr()).entered();

            // --- Independently grab the claims which should've been imposed on this layer (based on the verifier's own claim tracking) ---
            let layer_claims = claims
                .get(&layer_id)
                .ok_or(GKRError::NoClaimsForLayer(layer_id))?;


            // --- Append claims to the FS transcript... TODO!(ryancao): Do we actually need to do this??? ---
            for claim in layer_claims {
                transcript
                    .append_field_elements("Claimed bits to be aggregated", &claim.0)
                    .unwrap();
                transcript
                    .append_field_element("Claimed value to be aggregated", claim.1)
                    .unwrap();
            }

            // --- Perform the claim aggregation verification, first sampling `r` ---
            // --- Note that we ONLY do this if need be! ---
            let mut prev_claim = layer_claims[0].clone();

            // dbg!(&layer_claims);
            if layer_claims.len() > 1 {
                // --- Perform the claim aggregation verification, first sampling `r` ---
                let all_wlx_evaluations: Vec<F> = layer_claims
                    .iter()
                    .map(|(_, val)| *val)
                    .chain(wlx_evaluations.clone().into_iter())
                    .collect();

                transcript
                    .append_field_elements("Claim Aggregation Wlx_evaluations", &wlx_evaluations)
                    .unwrap();
                let agg_chal = transcript
                    .get_challenge("Challenge for claim aggregation")
                    .unwrap();

                prev_claim = verify_aggregate_claim(&all_wlx_evaluations, layer_claims, agg_chal)
                    .map_err(|_err| {
                    GKRError::ErrorWhenVerifyingLayer(
                        layer_id,
                        LayerError::AggregationError,
                    )
                })?;
            }
            
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

            for (layer_id, claim) in other_claims {
                if let Some(curr_claims) = claims.get_mut(&layer_id) {
                    curr_claims.push(claim);
                } else {
                    claims.insert(layer_id, vec![claim]);
                }
            }
        }


        for input_layer in input_layer_proofs {
            let input_layer_id = input_layer.layer_id;
            let input_layer_claims = claims
                .get(&input_layer_id)
                .ok_or(GKRError::NoClaimsForLayer(input_layer_id))?;

            // dbg!(&input_layer_claims);

            // --- Add the claimed values to the FS transcript ---
            for claim in input_layer_claims {
                transcript
                    .append_field_elements(
                        "Claimed challenge coordinates to be aggregated",
                        &claim.0,
                    )
                    .unwrap();
                transcript
                    .append_field_element("Claimed value to be aggregated", claim.1)
                    .unwrap();
            }

            let input_layer_claim = if input_layer_claims.len() > 1 {
                let all_input_wlx_evaluations: Vec<F> = input_layer_claims
                    .iter()
                    .map(|(_, val)| *val)
                    .chain(input_layer.input_layer_wlx_evaluations.clone().into_iter())
                    .collect();

                // dbg!(&all_input_wlx_evaluations);

                // --- Add the aggregation step to the transcript ---
                transcript
                    .append_field_elements(
                        "Input claim aggregation Wlx_evaluations",
                        &input_layer.input_layer_wlx_evaluations,
                    )
                    .unwrap();

                // --- Grab the input claim aggregation challenge ---
                let input_r_star = transcript
                    .get_challenge("Challenge for input claim aggregation")
                    .unwrap();

                // --- Perform the aggregation verification step and extract the correct input layer claim ---
                verify_aggregate_claim(&all_input_wlx_evaluations, input_layer_claims, input_r_star)
                    .map_err(|_err| {
                        GKRError::ErrorWhenVerifyingLayer(
                            input_layer_id,
                            LayerError::AggregationError,
                        )
                    })?
            } else {
                input_layer_claims[0].clone()
            };

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
