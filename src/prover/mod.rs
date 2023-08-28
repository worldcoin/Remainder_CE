//!Module that orchestrates creating a GKR Proof

/// For the input layer to the GKR circuit
pub mod input_layer_faje;
pub mod input_layer;


use std::collections::HashMap;

use crate::{
    layer::{
        from_mle, claims::aggregate_claims, claims::verify_aggregate_claim, claims::compute_aggregated_challenges, claims::compute_claim_wlx, Claim, GKRLayer, Layer,
        LayerBuilder, LayerError, LayerId, layer_enum::LayerEnum,
    },
    mle::{MleIndex, mle_enum::MleEnum},
    mle::{MleRef, dense::DenseMleRef},
    expression::ExpressionStandard,
    utils::pad_to_nearest_power_of_two, sumcheck::evaluate_at_a_point, prover::input_layer_faje::InputLayerType
};

use lcpc_2d::{FieldExt, ligero_commit::{remainder_ligero_commit_prove, remainder_ligero_eval_prove, remainder_ligero_verify}, adapter::convert_halo_to_lcpc, LcProofAuxiliaryInfo, poseidon_ligero::PoseidonSpongeHasher, ligero_structs::LigeroEncoding, ligero_ml_helper::naive_eval_mle_at_challenge_point};
use lcpc_2d::fs_transcript::halo2_remainder_transcript::Transcript;

// use derive_more::From;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use self::input_layer_faje::{EvalProofType, InputLayer, verify_public_input_layer, get_input_layer_proofs, verify_input_layer_proofs};

use lcpc_2d::ScalarField;
use lcpc_2d::adapter::LigeroProof;

///  New  type for containing the list of Layers that make up the GKR circuit
/// 
/// Literally just a Vec of pointers to various layer types!
pub struct Layers<F: FieldExt, Tr: Transcript<F>>(Vec<LayerEnum<F, Tr>>);

impl<F: FieldExt, Tr: Transcript<F> + 'static> Layers<F, Tr> {
    /// Add a layer to a list of layers
    pub fn add<B: LayerBuilder<F>, L: Layer<F, Transcript = Tr> + 'static>(
        &mut self,
        new_layer: B,
    ) -> B::Successor {
        let id = LayerId::Layer(self.0.len());
        let successor = new_layer.next_layer(id.clone(), None);
        let layer = L::new(new_layer, id);
        self.0.push(layer.get_enum());
        successor
    }

    /// Add a GKRLayer to a list of layers
    pub fn add_gkr<B: LayerBuilder<F>>(&mut self, new_layer: B) -> B::Successor {
        self.add::<_, GKRLayer<_, Tr>>(new_layer)
    }

    /// Creates a new Layers
    pub fn new() -> Self {
        Self(vec![])
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
}

/// A proof of the sumcheck protocol; Outer vec is rounds, inner vec is evaluations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckProof<F: FieldExt>(pub Vec<Vec<F>>);

impl<F: FieldExt> From<Vec<Vec<F>>> for SumcheckProof<F> {
    fn from(value: Vec<Vec<F>>) -> Self {
        Self(value)
    }
}

/// The proof for an individual GKR layer
#[derive(Serialize, Deserialize)]
pub struct LayerProof<F: FieldExt, Tr: Transcript<F>> {
    sumcheck_proof: SumcheckProof<F>,
    layer: LayerEnum<F, Tr>,
    wlx_evaluations: Vec<F>,
}

/// Proof for circuit input layer
//#[derive(Serialize, Deserialize)]
pub struct InputLayerProof<F: FieldExt, F2: ScalarField> {
    input_layer_aggregated_claim_proof: Vec<F>,
    eval_proof: EvalProofType<F, F2>,
    aux_info: Option<LcProofAuxiliaryInfo>,
    layer_id: LayerId,
}

/// All the elements to be passed to the verifier for the succinct non-interactive sumcheck proof
//#[derive(Serialize, Deserialize)]
pub struct GKRProof<F: FieldExt, Tr: Transcript<F>, F2: ScalarField> {
    /// The sumcheck proof of each GKR Layer, along with the fully bound expression.
    /// 
    /// In reverse order (i.e. layer closest to the output layer is first)
    pub layer_sumcheck_proofs: Vec<LayerProof<F, Tr>>,
    /// All the output layers that this circuit yields
    pub output_layers: Vec<Box<dyn MleRef<F = F>>>,
    /// Proof for the circuit input layer
    pub input_layer_proofs: Vec<InputLayerProof<F, F2>>,
}

/// A GKRCircuit ready to be proven
pub trait GKRCircuit<F: FieldExt> {
    /// The transcript this circuit uses
    type Transcript: Transcript<F>;

    /// The forward pass, defining the layer relationships and generating the layers
    fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, Vec<InputLayer<F>>);

    ///  The backwards pass, creating the GKRProof
    fn prove<F2: ScalarField>(
        &mut self,
        transcript: &mut Self::Transcript,
    ) -> Result<GKRProof<F, Self::Transcript, F2>, GKRError> {

        // --- Synthesize the circuit, using LayerBuilders to create internal, output, and input layers ---
        let (layers, mut output_layers, input_layers) = self.synthesize();

        // --- Keep track of GKR-style claims across all layers ---
        let mut claims: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new(); 

        // --- Go through circuit output layers and grab claims on each ---
        for output in output_layers.iter_mut() {
            
            let mut claim = None;
            let bits = output.index_mle_indices(0);

            // --- Evaluate each output MLE at a random challenge point ---
            for bit in 0..bits {
                let challenge = transcript
                    .get_challenge("Setting Output Layer Claim")
                    .unwrap();
                claim = output.fix_variable(bit, challenge);
            }

            // --- Gather the claim and layer ID ---
            let claim = claim.unwrap();
            let layer_id = output.get_layer_id();

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
                let layer_id = layer.id().clone();
                dbg!(&layer_id);
                let layer_claims = claims
                    .get(&layer_id)
                    .ok_or_else(|| GKRError::NoClaimsForLayer(layer_id.clone()))?;
                dbg!(layer_claims);

                // --- Add the claimed values to the FS transcript ---
                for claim in layer_claims {
                    transcript
                        .append_field_elements("Claimed bits to be aggregated", &claim.0)
                        .unwrap();
                    transcript
                        .append_field_element("Claimed value to be aggregated", claim.1)
                        .unwrap();
                }

                // --- Aggregate claims by sampling r^\star from the verifier and performing the ---
                // --- claim aggregation protocol. We ONLY aggregate if need be! ---
                let mut layer_claim = layer_claims[0].clone();
                let mut relevant_wlx_evaluations = vec![];
                if layer_claims.len() > 1 {
                    // --- Aggregate claims by sampling r^\star from the verifier and performing the ---
                    // --- claim aggregation protocol ---
                    let wlx_evaluations = compute_claim_wlx(&layer_claims, &layer).unwrap();
                    relevant_wlx_evaluations = wlx_evaluations[layer_claims.len()..].to_vec();
                    dbg!(&relevant_wlx_evaluations);
                    transcript.append_field_elements("Claim Aggregation Wlx_evaluations", &relevant_wlx_evaluations).unwrap();

                    let agg_chal = transcript.get_challenge("Challenge for claim aggregation").unwrap();

                    let aggregated_challenges = compute_aggregated_challenges(&layer_claims, agg_chal).unwrap();
                    let claimed_val = evaluate_at_a_point(&wlx_evaluations, agg_chal).unwrap();

                    layer_claim = (aggregated_challenges, claimed_val);

                } else {
                    dbg!("Yeah not aggregating claims this time around");
                }
                dbg!(layer_claim.clone());

                // --- Compute all sumcheck messages across this particular layer ---
                let prover_sumcheck_messages = layer
                    .prove_rounds(layer_claim, transcript)
                    .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id.clone(), err))?;

                // --- Grab all the resulting claims from the above sumcheck procedure and add them to the claim tracking map ---
                let post_sumcheck_new_claims = layer
                    .get_claims()
                    .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id.clone(), err))?;

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
                    wlx_evaluations: relevant_wlx_evaluations,
                })
            })
            .try_collect()?;

        let input_layer_proofs: Vec<InputLayerProof<F, F2>> = get_input_layer_proofs(input_layers, claims, transcript);


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
    fn verify<F2: ScalarField>(
        &mut self,
        transcript: &mut Self::Transcript,
        gkr_proof: GKRProof<F, Self::Transcript, F2>,
    ) -> Result<(), GKRError> {
        let GKRProof {
            layer_sumcheck_proofs,
            output_layers,
            input_layer_proofs
        } = gkr_proof;

        // --- Verifier keeps track of the claims on its own ---
        let mut claims: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();

        // --- NOTE that all the `Expression`s and MLEs contained within `gkr_proof` are already bound! ---
        for output in output_layers.iter() {

            let mle_indices = output.mle_indices();
            let mut claim_chal: Vec<F> = vec![];
            for (bit, index) in mle_indices.iter().enumerate() {
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
            let claim = (claim_chal, F::zero());
            let layer_id = output.get_layer_id();

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
                wlx_evaluations,
            } = sumcheck_proof_single;

            // --- Independently grab the claims which should've been imposed on this layer (based on the verifier's own claim tracking) ---
            let layer_id = layer.id().clone();
            let layer_claims = claims
                .get(&layer_id)
                .ok_or_else(|| GKRError::NoClaimsForLayer(layer_id.clone()))?;

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
            if layer_claims.len() > 1 {
                // --- Perform the claim aggregation verification, first sampling `r` ---

                let all_wlx_evaluations: Vec<F> = layer_claims.into_iter().map(
                    |(_, val)| *val
                ).chain(wlx_evaluations.clone().into_iter()).collect();

                transcript
                    .append_field_elements("Claim Aggregation Wlx_evaluations", &wlx_evaluations)
                    .unwrap();
                let agg_chal = transcript
                        .get_challenge("Challenge for claim aggregation")
                        .unwrap();

                prev_claim = verify_aggregate_claim(&all_wlx_evaluations, layer_claims, agg_chal)
                    .map_err(|_err| {
                        GKRError::ErrorWhenVerifyingLayer(
                            layer_id.clone(),
                            LayerError::AggregationError,
                        )
                    })?;

                   
            }

            // --- Performs the actual sumcheck verification step ---
            layer
                .verify_rounds(prev_claim, sumcheck_proof.0, transcript)
                .map_err(|err| GKRError::ErrorWhenVerifyingLayer(layer_id.clone(), err))?;

            // --- Extract/track claims from the expression independently (this ensures implicitly that the ---
            // --- prover is behaving correctly with respect to claim reduction, as otherwise the claim ---
            // --- aggregation verification step will fail ---
            let other_claims = layer
                .get_claims()
                .map_err(|err| GKRError::ErrorWhenVerifyingLayer(layer_id.clone(), err))?;

            for (layer_id, claim) in other_claims {
                if let Some(curr_claims) = claims.get_mut(&layer_id) {
                    curr_claims.push(claim);
                } else {
                    claims.insert(layer_id, vec![claim]);
                }
            }
        }


        verify_input_layer_proofs(input_layer_proofs, claims, transcript);
            
        

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{cmp::max, time::Instant};

    use ark_bn254::Fr;
    use ark_ff::PrimeField;
    use ark_std::{test_rng, UniformRand, log2, One, Zero};

    use crate::{mle::{dense::{DenseMle, Tuple2}, MleRef, Mle, zero::ZeroMleRef, mle_enum::MleEnum}, layer::{LayerBuilder, from_mle, LayerId}, expression::ExpressionStandard, zkdt::{structs::{DecisionNode, LeafNode, BinDecomp16Bit, InputAttribute}, zkdt_layer::{DecisionPackingBuilder, LeafPackingBuilder, ConcatBuilder, RMinusXBuilder, BitExponentiationBuilder, SquaringBuilder, ProductBuilder, SplitProductBuilder, EqualityCheck, AttributeConsistencyBuilder, InputPackingBuilder}}};
    use lcpc_2d::FieldExt;
    use lcpc_2d::fs_transcript::halo2_poseidon_transcript::PoseidonTranscript;
    use lcpc_2d::fs_transcript::halo2_remainder_transcript::Transcript;

    use super::{GKRCircuit, Layers, input_layer_faje::{InputLayer, InputLayerType}};
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr as H2Fr;

    /// This circuit is a 4 --> 2 circuit, such that
    /// [x_1, x_2, x_3, x_4, (y_1, y_2)] --> [x_1 * x_3, x_2 * x_4] --> [x_1 * x_3 - y_1, x_2 * x_4 - y_2]
    /// Note that we also have the difference thingy (of size 2)
    struct SimpleCircuit<F: FieldExt> {
        mle: DenseMle<F, Tuple2<F>>,
    }
    impl<F: FieldExt> GKRCircuit<F> for SimpleCircuit<F> {

        type Transcript = PoseidonTranscript<F>;

        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, Vec<InputLayer<F>>) {

            // --- The input layer should just be the concatenation of `mle` and `output_input` ---
            let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
            let mut input_layer = InputLayer::<F>::new_from_mles(&mut input_mles, Some(5), InputLayerType::LigeroInputLayer, LayerId::Input(0));
            let mle_clone = self.mle.clone();

            // --- Create Layers to be added to ---
            let mut layers = Layers::new();

            // --- Create a SimpleLayer from the first `mle` within the circuit ---
            let mult_builder = from_mle(
                mle_clone,
                // --- The expression is a simple product between the first and second halves ---
                |mle| ExpressionStandard::products(vec![mle.first(), mle.second()]),
                // --- The witness generation simply zips the two halves and multiplies them ---
                |mle, layer_id, prefix_bits| {
                    DenseMle::new_from_iter(
                        mle.into_iter()
                            .map(|Tuple2((first, second))| first * second),
                        layer_id,
                        prefix_bits,
                    )
                },
            );

            // --- Stacks the two aforementioned layers together into a single layer ---
            // --- Then adds them to the overall circuit ---
            let first_layer_output = layers.add_gkr(mult_builder);

            // --- Ahh. So we're doing the thing where we add the "real" circuit output as a circuit input, ---
            // --- then check if the difference between the two is zero ---
            let mut output_input =
                DenseMle::new_from_iter(first_layer_output.into_iter(), LayerId::Input(0), None);

            // --- Index the input-output layer ONLY for the input ---
            input_layer.index_input_output_mle(&mut Box::new(&mut output_input));

            // --- Subtract the computed circuit output from the advice circuit output ---
            let output_diff_builder = from_mle(
                (first_layer_output, output_input.clone()),
                |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
                |(mle1, mle2), layer_id, prefix_bits| {
                    let num_vars = max(mle1.num_iterated_vars(), mle2.num_iterated_vars());
                    ZeroMleRef::new(num_vars, prefix_bits, layer_id)
                },
            );

            // --- Add this final layer to the circuit ---
            let circuit_output = layers.add_gkr(output_diff_builder);

            // --- The input layer should just be the concatenation of `mle` and `output_input` ---
            let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
            input_layer.combine_input_mles(&input_mles, Some(Box::new(&mut output_input)));

            (layers, vec![Box::new(circuit_output)], vec![input_layer])
        }
    }

    /// Circuit which just subtracts its two halves! No input-output layer needed.
    struct SimplestCircuit<F: FieldExt> {
        mle: DenseMle<F, Tuple2<F>>,
    }
    impl<F: FieldExt> GKRCircuit<F> for SimplestCircuit<F> {

        type Transcript = PoseidonTranscript<F>;

        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, Vec<InputLayer<F>>) {

            // --- The input layer should just be the concatenation of `mle` and `output_input` ---
            let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
            let mut input_layer = InputLayer::<F>::new_from_mles(&mut input_mles, None, InputLayerType::LigeroInputLayer, LayerId::Input(0));
            let mle_clone = self.mle.clone();

            // --- Create Layers to be added to ---
            let mut layers = Layers::new();

            // --- Create a SimpleLayer from the first `mle` within the circuit ---
            let diff_builder = from_mle(
                mle_clone,
                // --- The expression is a simple diff between the first and second halves ---
                |mle| {
                    let first_half = Box::new(ExpressionStandard::Mle(mle.first()));
                    let second_half = Box::new(ExpressionStandard::Mle(mle.second()));
                    let negated_second_half = Box::new(ExpressionStandard::Negated(second_half));
                    ExpressionStandard::Sum(first_half, negated_second_half)
                },
                // --- The witness generation simply zips the two halves and subtracts them ---
                |mle, layer_id, prefix_bits| {
                    // DenseMle::new_from_iter(
                    //     mle.into_iter()
                    //         .map(|Tuple2((first, second))| first - second),
                    //     layer_id,
                    //     prefix_bits,
                    // )
                    // --- The output SHOULD be all zeros ---
                    let num_vars = max(mle.first().num_vars(), mle.second().num_vars());
                    ZeroMleRef::new(num_vars, prefix_bits, layer_id)
                },
            );

            // --- Stacks the two aforementioned layers together into a single layer ---
            // --- Then adds them to the overall circuit ---
            let first_layer_output = layers.add_gkr(diff_builder);

            // --- The input layer should just be the concatenation of `mle` and `output_input` ---
            let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
            input_layer.combine_input_mles(&input_mles, None);

            (layers, vec![Box::new(first_layer_output)], vec![input_layer])
        }
    }


    /// This circuit is a 4 --> 2 circuit, such that
    /// [x_1, x_2, x_3, x_4, (y_1, y_2)] --> [x_1 * x_3, x_2 * x_4] --> [x_1 * x_3 - y_1, x_2 * x_4 - y_2]
    /// Note that we also have the difference thingy (of size 2)
    struct SimpleCircuitMultipleInput<F: FieldExt> {
        mle: DenseMle<F, Tuple2<F>>,
    }
    impl<F: FieldExt> GKRCircuit<F> for SimpleCircuitMultipleInput<F> {

        type Transcript = PoseidonTranscript<F>;

        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, Vec<InputLayer<F>>) {

            // --- The input layer should just be the concatenation of `mle` and `output_input` ---
            let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
            let mut input_layer = InputLayer::<F>::new_from_mles(&mut input_mles, None, InputLayerType::LigeroInputLayer, LayerId::Input(0));
            let mle_clone = self.mle.clone();

            // --- Create Layers to be added to ---
            let mut layers = Layers::new();

            // --- Create a SimpleLayer from the first `mle` within the circuit ---
            let mult_builder = from_mle(
                mle_clone,
                // --- The expression is a simple product between the first and second halves ---
                |mle| ExpressionStandard::products(vec![mle.first(), mle.second()]),
                // --- The witness generation simply zips the two halves and multiplies them ---
                |mle, layer_id, prefix_bits| {
                    DenseMle::new_from_iter(
                        mle.into_iter()
                            .map(|Tuple2((first, second))| first * second),
                        layer_id,
                        prefix_bits,
                    )
                },
            );

            // --- Stacks the two aforementioned layers together into a single layer ---
            // --- Then adds them to the overall circuit ---
            let first_layer_output = layers.add_gkr(mult_builder);

            // --- Ahh. So we're doing the thing where we add the "real" circuit output as a circuit input, ---
            // --- then check if the difference between the two is zero ---
            let mut output_input =
                DenseMle::new_from_iter(first_layer_output.into_iter(), LayerId::Input(1), None);

            // --- Index the input-output layer ONLY for the input ---
           // input_layer.index_input_output_mle(&mut Box::new(&mut output_input));

            // --- Subtract the computed circuit output from the advice circuit output ---
            let output_diff_builder = from_mle(
                (first_layer_output, output_input.clone()),
                |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
                |(mle1, mle2), layer_id, prefix_bits| {
                    let num_vars = max(mle1.num_iterated_vars(), mle2.num_iterated_vars());
                    ZeroMleRef::new(num_vars, prefix_bits, layer_id)
                },
            );

            // --- Add this final layer to the circuit ---
            let circuit_output = layers.add_gkr(output_diff_builder);

            // --- The input layer should just be the concatenation of `mle` and `output_input` ---
            let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
            input_layer.combine_input_mles(&input_mles, None);

            let mut input_output_layer = InputLayer::<F>::new_from_mles(&mut vec![Box::new(&mut output_input)], None, InputLayerType::LigeroInputLayer, LayerId::Input(1));
            input_output_layer.combine_input_mles(&mut vec![Box::new(&mut output_input)], None);
            input_output_layer.index_input_output_mle(&mut Box::new(&mut output_input));

            (layers, vec![Box::new(circuit_output)], vec![input_layer, input_output_layer])
        }
    }
    

    /// This circuit is a 4k --> k circuit, such that
    /// [x_1, x_2, x_3, x_4] --> [x_1 * x_3, x_2 + x_4] --> [(x_1 * x_3) - (x_2 + x_4)]
    struct TestCircuit<F: FieldExt> {
        mle: DenseMle<F, Tuple2<F>>,
        mle_2: DenseMle<F, Tuple2<F>>,
    }

    impl<F: FieldExt> GKRCircuit<F> for TestCircuit<F> {

        type Transcript = PoseidonTranscript<F>;

        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, Vec<InputLayer<F>>) {

            // --- The input layer should just be the concatenation of `mle`, `mle_2`, and `output_input` ---
            // let mut self_mle_clone = self.mle.clone();
            // let mut self_mle_2_clone = self.mle_2.clone();
            let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle), Box::new(&mut self.mle_2)];
            let input_layer = InputLayer::<F>::new_from_mles(&mut input_mles, Some(1), InputLayerType::PublicInputLayer, LayerId::Input(0));
            let mle_clone = self.mle.clone();
            let mle_2_clone = self.mle_2.clone();

            // --- Create Layers to be added to ---
            let mut layers = Layers::new();

            // --- Create a SimpleLayer from the first `mle` within the circuit ---
            let builder = from_mle(
                mle_clone,
                // --- The expression is a simple product between the first and second halves ---
                |mle| ExpressionStandard::products(vec![mle.first(), mle.second()]),
                // --- The witness generation simply zips the two halves and multiplies them ---
                |mle, layer_id, prefix_bits| {
                    DenseMle::new_from_iter(
                        mle.into_iter()
                            .map(|Tuple2((first, second))| first * second),
                        layer_id,
                        prefix_bits,
                    )
                },
            );

            // --- Similarly here, but with addition between the two halves ---
            // --- Note that EACH of `mle` and `mle_2` are parts of the input layer ---
            let builder2 = from_mle(
                mle_2_clone,
                |mle| mle.first().expression() + mle.second().expression(),
                |mle, layer_id, prefix_bits| {
                    DenseMle::new_from_iter(
                        mle.into_iter()
                            .map(|Tuple2((first, second))| first + second),
                        layer_id,
                        prefix_bits,
                    )
                },
            );

            // --- Stacks the two aforementioned layers together into a single layer ---
            // --- Then adds them to the overall circuit ---
            let builder3 = builder.concat(builder2);
            let output = layers.add_gkr(builder3);

            // --- Creates a single layer which takes [x_1, ..., x_n, y_1, ..., y_n] and returns [x_1 - y_1, ..., x_n - y_n] ---
            let builder4 = from_mle(
                output,
                |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
                |(mle1, mle2), layer_id, prefix_bits| {
                    DenseMle::new_from_iter(
                        mle1.clone()
                            .into_iter()
                            .zip(mle2.clone().into_iter())
                            .map(|(first, second)| first - second),
                        layer_id,
                        prefix_bits,
                    )
                },
            );

            // --- Appends this to the circuit ---
            let computed_output = layers.add_gkr(builder4);

            // --- Ahh. So we're doing the thing where we add the "real" circuit output as a circuit input, ---
            // --- then check if the difference between the two is zero ---
            let mut output_input =
                DenseMle::new_from_iter(computed_output.into_iter(), LayerId::Input(0), None);
            input_layer.index_input_output_mle(&mut Box::new(&mut output_input));

            // --- Subtract the computed circuit output from the advice circuit output ---
            let builder5 = from_mle(
                (computed_output, output_input.clone().clone()),
                |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
                |(mle1, mle2), layer_id, prefix_bits| {
                    let num_vars = max(mle1.num_iterated_vars(), mle2.num_iterated_vars());
                    ZeroMleRef::new(num_vars, prefix_bits, layer_id)
                },
            );

            // --- Add this final layer to the circuit ---
            let circuit_circuit_output = layers.add_gkr(builder5);

            // --- The input layer should just be the concatenation of `mle`, `mle_2`, and `output_input` ---
            let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle), Box::new(&mut self.mle_2), Box::new(&mut output_input)];
            let mut input_layer = InputLayer::<F>::new_from_mles(&mut input_mles, None, InputLayerType::LigeroInputLayer, LayerId::Input(0));

            // --- The input layer should just be the concatenation of `mle`, `mle_2`, and `output_input` ---
            let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle), Box::new(&mut self.mle_2)];
            input_layer.combine_input_mles(&input_mles, Some(Box::new(&mut output_input)));

            (layers, vec![Box::new(circuit_circuit_output)], vec![input_layer])
        }
    }


    #[test]
    fn test_gkr_simplest_circuit() {
        let mut rng = test_rng();
        let size = 1 << 4;

        // --- This should be 2^2 ---
        let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
            (0..size).map(|_| {
                let num = Fr::rand(&mut rng);
                //let second_num = Fr::rand(&mut rng);
                (num, num).into()
            }),
            LayerId::Input(0),
            None,
        );
        // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        //     (0..size).map(|idx| (Fr::from(idx + 1), Fr::from(idx + 1)).into()),
        //     LayerId::Input(0),
        //     None,
        // );

        let mut circuit: SimplestCircuit<Fr> = SimplestCircuit { mle };

        let mut transcript: PoseidonTranscript<Fr> =
            PoseidonTranscript::new("GKR Prover Transcript");
        let now = Instant::now();

        match circuit.prove::<H2Fr>(&mut transcript) {
            Ok(proof) => {
                println!(
                    "proof generated successfully in {}!",
                    now.elapsed().as_secs_f32()
                );
                let mut transcript: PoseidonTranscript<Fr> =
                    PoseidonTranscript::new("GKR Verifier Transcript");
                let now = Instant::now();
                match circuit.verify(&mut transcript, proof) {
                    Ok(_) => {
                        println!(
                            "Verification succeeded: takes {}!",
                            now.elapsed().as_secs_f32()
                        );
                    }
                    Err(err) => {
                        println!("Verify failed! Error: {err}");
                        panic!();
                    }
                }
            }
            Err(err) => {
                println!("Proof failed! Error: {err}");
                panic!();
            }
        }

        // panic!();
    }

    #[test]
    fn test_gkr_simple_circuit() {
        let mut rng = test_rng();
        let size = 1 << 5;

        // --- This should be 2^2 ---
        let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
            (0..size).map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)).into()),
            LayerId::Input(0),
            None,
        );
        // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        //     (0..size).map(|idx| (Fr::from(idx + 2), Fr::from(idx + 2)).into()),
        //     LayerId::Input(0),
        //     None,
        // );

        let mut circuit: SimpleCircuit<Fr> = SimpleCircuit { mle };

        let mut transcript: PoseidonTranscript<Fr> =
            PoseidonTranscript::new("GKR Prover Transcript");
        let now = Instant::now();

        match circuit.prove::<H2Fr>(&mut transcript) {
            Ok(proof) => {
                println!(
                    "proof generated successfully in {}!",
                    now.elapsed().as_secs_f32()
                );
                let mut transcript: PoseidonTranscript<Fr> =
                    PoseidonTranscript::new("GKR Verifier Transcript");
                let now = Instant::now();
                match circuit.verify(&mut transcript, proof) {
                    Ok(_) => {
                        println!(
                            "Verification succeeded: takes {}!",
                            now.elapsed().as_secs_f32()
                        );
                    }
                    Err(err) => {
                        println!("Verify failed! Error: {err}");
                        panic!();
                    }
                }
            }
            Err(err) => {
                println!("Proof failed! Error: {err}");
                panic!();
            }
        }

        // panic!();
    }

    #[test]
    fn test_gkr() {
        let mut rng = test_rng();
        let size = 1 << 1;
        // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
        // tracing::subscriber::set_global_default(subscriber)
        //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

        // --- This should be 2^2 ---
        let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
            (0..size).map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)).into()),
            LayerId::Input(0),
            None,
        );
        // --- This should be 2^2 ---
        let mle_2: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
            (0..size).map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)).into()),
            LayerId::Input(0),
            None,
        );

        let mut circuit: TestCircuit<Fr> = TestCircuit { mle, mle_2 };

        let mut transcript: PoseidonTranscript<Fr> =
            PoseidonTranscript::new("GKR Prover Transcript");
        let now = Instant::now();

        match circuit.prove::<H2Fr>(&mut transcript) {
            Ok(proof) => {
                println!(
                    "proof generated successfully in {}!",
                    now.elapsed().as_secs_f32()
                );
                let mut transcript: PoseidonTranscript<Fr> =
                    PoseidonTranscript::new("GKR Verifier Transcript");
                let now = Instant::now();
                match circuit.verify(&mut transcript, proof) {
                    Ok(_) => {
                        println!(
                            "Verification succeeded: takes {}!",
                            now.elapsed().as_secs_f32()
                        );
                    }
                    Err(err) => {
                        println!("Verify failed! Error: {err}");
                        panic!();
                    }
                }
            }
            Err(err) => {
                println!("Proof failed! Error: {err}");
                panic!();
            }
        }
    }

    #[test]
    fn test_gkr_simple_circuit_multiple_input_layers() {
        let mut rng = test_rng();
        let size = 1 << 5;

        // --- This should be 2^2 ---
        let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
            (0..size).map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)).into()),
            LayerId::Input(0),
            None,
        );
        // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        //     (0..size).map(|idx| (Fr::from(idx + 2), Fr::from(idx + 2)).into()),
        //     LayerId::Input(0),
        //     None,
        // );

        let mut circuit: SimpleCircuitMultipleInput<Fr> = SimpleCircuitMultipleInput { mle };

        let mut transcript: PoseidonTranscript<Fr> =
            PoseidonTranscript::new("GKR Prover Transcript");
        let now = Instant::now();

        match circuit.prove::<H2Fr>(&mut transcript) {
            Ok(proof) => {
                println!(
                    "proof generated successfully in {}!",
                    now.elapsed().as_secs_f32()
                );
                let mut transcript: PoseidonTranscript<Fr> =
                    PoseidonTranscript::new("GKR Verifier Transcript");
                let now = Instant::now();
                match circuit.verify(&mut transcript, proof) {
                    Ok(_) => {
                        println!(
                            "Verification succeeded: takes {}!",
                            now.elapsed().as_secs_f32()
                        );
                    }
                    Err(err) => {
                        println!("Verify failed! Error: {err}");
                        panic!();
                    }
                }
            }
            Err(err) => {
                println!("Proof failed! Error: {err}");
                panic!();
            }
        }

        // panic!();
    }
}
