//!Module that orchestrates creating a GKR Proof

/// For the input layer to the GKR circuit
pub mod input_layer;

use std::collections::HashMap;

use crate::{
    layer::{
        claims::aggregate_claims, claims::verify_aggregate_claim, Claim, GKRLayer, Layer,
        LayerBuilder, LayerError, LayerId,
    },
    mle::MleIndex,
    mle::MleRef,
    expression::ExpressionStandard,
    utils::pad_to_nearest_power_of_two
};

use lcpc_2d::{FieldExt, ligero_commit::{remainder_ligero_commit_prove, remainder_ligero_eval_prove, remainder_ligero_verify}, adapter::convert_halo_to_lcpc, LcProofAuxiliaryInfo, poseidon_ligero::PoseidonSpongeHasher, ligero_structs::LigeroEncoding, ligero_ml_helper::naive_eval_mle_at_challenge_point};
use lcpc_2d::fs_transcript::halo2_remainder_transcript::Transcript;

// use derive_more::From;
use itertools::Itertools;
use thiserror::Error;

use self::input_layer::InputLayer;

use lcpc_2d::ScalarField;
use lcpc_2d::adapter::LigeroProof;

/// New type for containing the list of Layers that make up the GKR circuit
/// 
/// Literally just a Vec of pointers to various layer types!
pub struct Layers<F: FieldExt, Tr: Transcript<F>>(Vec<Box<dyn Layer<F, Transcript = Tr>>>);

impl<F: FieldExt, Tr: Transcript<F> + 'static> Layers<F, Tr> {
    /// Add a layer to a list of layers
    pub fn add<B: LayerBuilder<F>, L: Layer<F, Transcript = Tr> + 'static>(
        &mut self,
        new_layer: B,
    ) -> B::Successor {
        let id = LayerId::Layer(self.0.len());
        let successor = new_layer.next_layer(id.clone(), None);
        let layer = L::new(new_layer, id);
        dbg!(layer.expression());
        self.0.push(Box::new(layer));
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
#[derive(Clone, Debug)]
pub struct SumcheckProof<F: FieldExt>(Vec<Vec<F>>);

impl<F: FieldExt> From<Vec<Vec<F>>> for SumcheckProof<F> {
    fn from(value: Vec<Vec<F>>) -> Self {
        Self(value)
    }
}

/// The proof for an individual GKR layer
pub struct LayerProof<F: FieldExt, Tr: Transcript<F>> {
    sumcheck_proof: SumcheckProof<F>,
    layer: Box<dyn Layer<F, Transcript = Tr>>,
    wlx_evaluations: Vec<F>,
}

/// Proof for circuit input layer
pub struct InputLayerProof<F: FieldExt> {
    input_layer_aggregated_claim_proof: Vec<F>,
}

/// All the elements to be passed to the verifier for the succinct non-interactive sumcheck proof
pub struct GKRProof<F: FieldExt, Tr: Transcript<F>, F2: ScalarField> {
    /// The sumcheck proof of each GKR Layer, along with the fully bound expression.
    /// 
    /// In reverse order (i.e. layer closest to the output layer is first)
    pub layer_sumcheck_proofs: Vec<LayerProof<F, Tr>>,
    /// All the output layers that this circuit yields
    pub output_layers: Vec<Box<dyn MleRef<F = F>>>,
    /// Proof for the circuit input layer
    pub input_layer_proof: InputLayerProof<F>,
    /// Ligero proof
    pub ligero_commit_eval_proof: LigeroProof<F2>,
    /// Ligero auxiliary info for verifier
    pub ligero_aux: LcProofAuxiliaryInfo,
}

/// A GKRCircuit ready to be proven
pub trait GKRCircuit<F: FieldExt> {
    /// The transcript this circuit uses
    type Transcript: Transcript<F>;

    /// The Halo2 field (for prover transcript)
    type F2: ScalarField;

    /// The forward pass, defining the layer relationships and generating the layers
    fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, InputLayer<F>);

    /// The backwards pass, creating the GKRProof
    fn prove(
        &mut self,
        transcript: &mut Self::Transcript,
    ) -> Result<GKRProof<F, Self::Transcript, Self::F2>, GKRError> {

        // --- Synthesize the circuit, using LayerBuilders to create internal, output, and input layers ---
        let (layers, mut output_layers, input_layer) = self.synthesize();

        // --- Compute the Ligero commitment to the combined input MLE ---
        // TODO!(ryancao): Hard-code this somewhere else!!
        let rho_inv: u8 = 4;
        let orig_input_layer_bookkeeping_table = pad_to_nearest_power_of_two(input_layer.get_combined_mle().unwrap().mle.clone());
        let (_, comm, root, aux) = remainder_ligero_commit_prove(&orig_input_layer_bookkeeping_table, rho_inv);

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
                let mut wlx_evaluations = vec![];
                if layer_claims.len() > 1 {
                    let agg_chal = transcript
                        .get_challenge("Challenge for claim aggregation")
                        .unwrap();

                    (layer_claim, wlx_evaluations) =
                        aggregate_claims(layer_claims, layer.expression(), agg_chal).unwrap();

                    transcript
                        .append_field_elements("Claim Aggregation Wlx_evaluations", &wlx_evaluations)
                        .unwrap();
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
                    wlx_evaluations,
                })
            })
            .try_collect()?;

        // --- Gather all of the claims on the input layer and aggregate them ---
        let input_layer_id = LayerId::Input;
        let input_layer_claims = claims
            .get(&input_layer_id)
            .ok_or_else(|| GKRError::NoClaimsForLayer(input_layer_id.clone()))?;
        dbg!(input_layer_claims);

        // --- Add the claimed values to the FS transcript ---
        for claim in input_layer_claims {
            transcript
                .append_field_elements("Claimed challenge coordinates to be aggregated", &claim.0)
                .unwrap();
            transcript
                .append_field_element("Claimed value to be aggregated", claim.1)
                .unwrap();
        }

        // --- Aggregate challenges ONLY if we have more than one ---
        let mut input_layer_claim = input_layer_claims[0].clone();
        let mut input_wlx_evaluations = vec![];
        if input_layer_claims.len() > 1 {
            dbg!("Aggregating input claims");
            let agg_chal = transcript
                .get_challenge("Challenge for claim aggregation")
                .unwrap();

            let input_layer_expression = ExpressionStandard::Mle(input_layer.get_combined_mle().unwrap().mle_ref());

            (input_layer_claim, input_wlx_evaluations) =
                aggregate_claims(input_layer_claims, &input_layer_expression, agg_chal).unwrap();

            transcript
                .append_field_elements("Claim Aggregation Wlx_evaluations", &input_wlx_evaluations)
                .unwrap();
        } else {
            dbg!("Not aggroing input claims this time around");
        }
        // dbg!(input_layer_claim.clone());
        // dbg!(-input_layer_claim.1);

        // --- Sanitycheck on the un-aggregated input claims ---
        let padded_input_layer_mle = pad_to_nearest_power_of_two(input_layer.get_combined_mle().unwrap().mle);
        dbg!(&padded_input_layer_mle);
        for input_claim in input_layer_claims {
            
            // let input_layer_challenge_coords_big_endian = input_claim.0.clone().into_iter().rev().collect_vec();
            let naive_eval = naive_eval_mle_at_challenge_point(&padded_input_layer_mle, &input_claim.0);
            dbg!(&padded_input_layer_mle);
            dbg!(&input_claim);
            dbg!(&naive_eval);
            assert_eq!(naive_eval, input_claim.1);
        }
        // panic!();

        // --- Sanitycheck (TODO!(ryancao): Remove this) ---
        let padded_input_layer_mle = pad_to_nearest_power_of_two(input_layer.get_combined_mle().unwrap().mle);
        // let input_layer_challenge_coords_big_endian = input_layer_claim.0.clone().into_iter().rev().collect_vec();
        let naive_eval = naive_eval_mle_at_challenge_point(&padded_input_layer_mle, &input_layer_claim.0);
        // dbg!(-naive_eval);
        // dbg!(&input_layer_claim);
        assert_eq!(naive_eval, input_layer_claim.1);

        let input_layer_proof = InputLayerProof {
            input_layer_aggregated_claim_proof: input_wlx_evaluations,
        };

        // --- Finally, the Ligero commit + eval proof ---
        let ligero_commit_eval_proof = remainder_ligero_eval_prove(
            &orig_input_layer_bookkeeping_table,
            &input_layer_claim.0,
            transcript,
            aux.clone(),
            comm,
            root
        );

        let gkr_proof = GKRProof {
            layer_sumcheck_proofs,
            output_layers,
            input_layer_proof,
            ligero_commit_eval_proof,
            ligero_aux: aux,
        };

        Ok(gkr_proof)
    }

    /// Verifies the GKRProof produced by fn prove
    /// 
    /// Takes in a transcript for FS and re-generates challenges on its own
    fn verify(
        &mut self,
        transcript: &mut Self::Transcript,
        gkr_proof: GKRProof<F, Self::Transcript, Self::F2>,
    ) -> Result<(), GKRError> {
        let GKRProof {
            layer_sumcheck_proofs,
            output_layers,
            input_layer_proof,
            ligero_commit_eval_proof,
            ligero_aux
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
                let agg_chal = transcript
                    .get_challenge("Challenge for claim aggregation")
                    .unwrap();

                prev_claim = verify_aggregate_claim(&wlx_evaluations, layer_claims, agg_chal)
                    .map_err(|_err| {
                        GKRError::ErrorWhenVerifyingLayer(
                            layer_id.clone(),
                            LayerError::AggregationError,
                        )
                    })?;

                transcript
                    .append_field_elements("Claim Aggregation Wlx_evaluations", &wlx_evaluations)
                    .unwrap();
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

        // --- Verify the input claim aggregation step ---
        let InputLayerProof {
            input_layer_aggregated_claim_proof,
        } = input_layer_proof;

        // --- Grab the self-tracked input layer claims ---
        let input_layer_id = LayerId::Input;
        let input_layer_claims = claims
            .get(&input_layer_id)
            .ok_or_else(|| GKRError::NoClaimsForLayer(input_layer_id.clone()))?;

        // --- Add the claimed values to the FS transcript ---
        for claim in input_layer_claims {
            transcript
                .append_field_elements("Claimed challenge coordinates to be aggregated", &claim.0)
                .unwrap();
            transcript
                .append_field_element("Claimed value to be aggregated", claim.1)
                .unwrap();
        }

        // --- Do claim aggregation on input layer ONLY if needed ---
        let mut input_layer_claim = input_layer_claims[0].clone();
        if input_layer_claims.len() > 1 {

            // --- Grab the input claim aggregation challenge ---
            let input_r_star = transcript
                .get_challenge("Challenge for input claim aggregation")
                .unwrap();

            // --- Perform the aggregation verification step and extract the correct input layer claim ---
            input_layer_claim = verify_aggregate_claim(&input_layer_aggregated_claim_proof, input_layer_claims, input_r_star)
                .map_err(|_err| {
                    GKRError::ErrorWhenVerifyingLayer(
                        input_layer_id,
                        LayerError::AggregationError,
                    )
                })?;

            // --- Add the aggregation step to the transcript ---
            transcript
            .append_field_elements("Input claim aggregation Wlx_evaluations", &input_layer_aggregated_claim_proof)
            .unwrap();
        }

        // --- The prover interprets the challenge coords as big-endian, but Ligero interprets them as little-endian ---
        // let input_layer_challenge_coords_big_endian = input_layer_claim.0.clone().into_iter().rev().collect_vec();

        dbg!("----- Hahahahahahahaahaha!-----");

        // --- This is broken for now... The prover input claim is not the same as the Ligero proof ---
        let (root, ligero_eval_proof, _) = convert_halo_to_lcpc::<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F, Self::F2>(ligero_aux.clone(), ligero_commit_eval_proof);
        remainder_ligero_verify::<F, Self::F2>(&root, &ligero_eval_proof, ligero_aux, transcript, &input_layer_claim.0, input_layer_claim.1);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{cmp::max, time::Instant};

    use ark_bn254::Fr;
    use ark_std::{test_rng, UniformRand, log2, One, Zero};

    use crate::{mle::{dense::{DenseMle, Tuple2}, MleRef, Mle, zero::ZeroMleRef}, layer::{LayerBuilder, from_mle, LayerId}, expression::ExpressionStandard, zkdt::{structs::{DecisionNode, LeafNode, BinDecomp16Bit, InputAttribute}, zkdt_layer::{DecisionPackingBuilder, LeafPackingBuilder, ConcatBuilder, RMinusXBuilder, BitExponentiationBuilder, SquaringBuilder, ProductBuilder, SplitProductBuilder, DifferenceBuilder, AttributeConsistencyBuilder, InputPackingBuilder}}};
    use lcpc_2d::FieldExt;
    use lcpc_2d::fs_transcript::halo2_poseidon_transcript::PoseidonTranscript;
    use lcpc_2d::fs_transcript::halo2_remainder_transcript::Transcript;
    use lcpc_2d::ScalarField;

    use super::{GKRCircuit, Layers, input_layer::InputLayer};
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr as H2Fr;

    struct PermutationCircuit<F: FieldExt> {
        dummy_input_data_mle_vec: DenseMle<F, InputAttribute<F>>,               // batched
        dummy_permuted_input_data_mle_vec: DenseMle<F, InputAttribute<F>>,      // batched
        r: F,
        r_packing: F,
        input_len: usize,
        num_inputs: usize
    }

    impl<F: FieldExt> GKRCircuit<F> for PermutationCircuit<F> {
        type Transcript = PoseidonTranscript<F>;
        type F2 = H2Fr;
        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, InputLayer<F>) {
            let mut layers = Layers::new();

            // layer 0: packing
            let input_packing_builder = InputPackingBuilder::new(
                self.dummy_input_data_mle_vec.clone(),
                self.r,
                self.r_packing);

            let input_permuted_packing_builder = InputPackingBuilder::new(
                self.dummy_permuted_input_data_mle_vec.clone(),
                self.r,
                self.r_packing);

            let packing_builders = input_packing_builder.concat(input_permuted_packing_builder);
            let (mut input_packed, mut input_permuted_packed) = layers.add_gkr(packing_builders);

            for _ in 0..log2(self.input_len * self.num_inputs) {
                let prod_builder = SplitProductBuilder::new(
                    input_packed
                );
                let prod_permuted_builder = SplitProductBuilder::new(
                    input_permuted_packed
                );
                let split_product_builders = prod_builder.concat(prod_permuted_builder);
                (input_packed, input_permuted_packed) = layers.add_gkr(split_product_builders);
            }

            let difference_builder = DifferenceBuilder::new(
                input_packed,
                input_permuted_packed,
            );

            let difference_mle = layers.add_gkr(difference_builder);

            // --- Input MLEs are just the input data and permuted input data MLEs ---
            let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.dummy_input_data_mle_vec), Box::new(&mut self.dummy_permuted_input_data_mle_vec)];
            let input_layer = InputLayer::new_from_mles(&mut input_mles, None);

            (layers, vec![Box::new(difference_mle.mle_ref())], input_layer)
        }
    }

    struct AttributeConsistencyCircuit<F: FieldExt> {
        dummy_permuted_input_data_mle_vec: DenseMle<F, InputAttribute<F>>, // batched
        dummy_decision_node_paths_mle_vec: DenseMle<F, DecisionNode<F>>,     // batched
        tree_height: usize,
    }

    impl<F: FieldExt> GKRCircuit<F> for AttributeConsistencyCircuit<F> {
        type Transcript = PoseidonTranscript<F>;
        type F2 = H2Fr;
        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, InputLayer<F>) {
            let mut layers = Layers::new();

            let attribute_consistency_builder = AttributeConsistencyBuilder::new(
                self.dummy_permuted_input_data_mle_vec.clone(),
                self.dummy_decision_node_paths_mle_vec.clone(),
                self.tree_height
            );

            let difference_mle = layers.add_gkr(attribute_consistency_builder);

            // --- Input MLEs are just the permuted input data and decision path MLEs ---
            let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.dummy_permuted_input_data_mle_vec), Box::new(&mut self.dummy_decision_node_paths_mle_vec)];
            let input_layer = InputLayer::new_from_mles(&mut input_mles, None);

            (layers, vec![Box::new(difference_mle.mle_ref())], input_layer)
        }
    }

    struct MultiSetCircuit<F: FieldExt> {
        dummy_decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
        dummy_leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
        dummy_multiplicities_bin_decomp_mle: DenseMle<F, BinDecomp16Bit<F>>,
        dummy_decision_node_paths_mle_vec: DenseMle<F, DecisionNode<F>>, // batched
        dummy_leaf_node_paths_mle_vec: DenseMle<F, LeafNode<F>>,         // batched
        r: F,
        r_packings: (F, F),
        tree_height: usize,
        num_inputs: usize,
    }

    impl<F: FieldExt> GKRCircuit<F> for MultiSetCircuit<F> {
        type Transcript = PoseidonTranscript<F>;
        type F2 = H2Fr;
        
        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, InputLayer<F>) {
            let mut layers = Layers::new();

            // layer 0
            let decision_packing_builder = DecisionPackingBuilder::new(
                self.dummy_decision_nodes_mle.clone(), self.r, self.r_packings);

            let leaf_packing_builder = LeafPackingBuilder::new(
                self.dummy_leaf_nodes_mle.clone(), self.r, self.r_packings.0
            );

            let packing_builders = decision_packing_builder.concat(leaf_packing_builder);
            let (decision_packed, leaf_packed) = layers.add_gkr(packing_builders);

            // layer 1
            let decision_leaf_concat_builder = ConcatBuilder::new(
                decision_packed, leaf_packed
            );
            let x_packed = layers.add_gkr(decision_leaf_concat_builder);

            // layer 2
            let r_minus_x_builder =  RMinusXBuilder::new(
                x_packed, self.r
            );
            let mut r_minus_x = layers.add_gkr(r_minus_x_builder);

            // layer 3
            let prev_prod_builder = BitExponentiationBuilder::new(
                self.dummy_multiplicities_bin_decomp_mle.clone(),
                0,
                r_minus_x.clone()
            );
            let mut prev_prod = layers.add_gkr(prev_prod_builder);

            for i in 1..16 {

                // layer 3, or i + 2
                let r_minus_x_square_builder = SquaringBuilder::new(
                    r_minus_x
                );
                let r_minus_x_square = layers.add_gkr(r_minus_x_square_builder);

                // layer 4, or i + 3
                let curr_prod_builder = BitExponentiationBuilder::new(
                    self.dummy_multiplicities_bin_decomp_mle.clone(),
                    i,
                    r_minus_x_square.clone()
                );
                let curr_prod = layers.add_gkr(curr_prod_builder);

                // layer 5, or i + 4
                let prod_builder = ProductBuilder::new(
                    curr_prod,
                    prev_prod
                );
                prev_prod = layers.add_gkr(prod_builder);

                r_minus_x = r_minus_x_square;

            }

            let mut exponentiated_nodes = prev_prod;

            for _ in 0..self.tree_height {

                // layer 20, or i+20
                let prod_builder = SplitProductBuilder::new(
                    exponentiated_nodes
                );
                exponentiated_nodes = layers.add_gkr(prod_builder);
            }

            // **** above is nodes exponentiated ****
            // **** below is all decision nodes on the path multiplied ****


            // layer 0: packing
            let decision_path_packing_builder = DecisionPackingBuilder::new(
                self.dummy_decision_node_paths_mle_vec.clone(),
                self.r,
                self.r_packings
            );

            let leaf_path_packing_builder = LeafPackingBuilder::new(
                self.dummy_leaf_node_paths_mle_vec.clone(),
                self.r,
                self.r_packings.0
            );

            let path_packing_builders = decision_path_packing_builder.concat(leaf_path_packing_builder);
            let (decision_path_packed, leaf_path_packed) = layers.add_gkr(path_packing_builders);

            // layer 1: concat
            let path_decision_leaf_concat_builder = ConcatBuilder::new(
                decision_path_packed, leaf_path_packed
            );
            let x_path_packed = layers.add_gkr(path_decision_leaf_concat_builder);

            // layer 2: r - x
            let r_minus_x_path_builder =  RMinusXBuilder::new(
                x_path_packed, self.r
            );
            let r_minus_x_path = layers.add_gkr(r_minus_x_path_builder);

            let mut path_product = r_minus_x_path;

            // layer remaining: product it together
            for _ in 0..log2(self.tree_height * self.num_inputs) {
                let prod_builder = SplitProductBuilder::new(
                    path_product
                );
                path_product = layers.add_gkr(prod_builder);
            }

            let difference_builder = DifferenceBuilder::new(
                exponentiated_nodes,
                path_product
            );

            let difference = layers.add_gkr(difference_builder);

            // --- Input MLEs are just each of the dummy MLE inputs ---
            let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
                Box::new(&mut self.dummy_decision_nodes_mle), 
                Box::new(&mut self.dummy_leaf_nodes_mle),
                Box::new(&mut self.dummy_multiplicities_bin_decomp_mle),
                Box::new(&mut self.dummy_decision_node_paths_mle_vec),
                Box::new(&mut self.dummy_leaf_node_paths_mle_vec),
            ];
            let input_layer = InputLayer::new_from_mles(&mut input_mles, None);

            (layers, vec![Box::new(difference.mle_ref())], input_layer)
        }
    }

    /// This circuit is a 4 --> 2 circuit, such that
    /// [x_1, x_2, x_3, x_4, (y_1, y_2)] --> [x_1 * x_3, x_2 * x_4] --> [x_1 * x_3 - y_1, x_2 * x_4 - y_2]
    /// Note that we also have the difference thingy (of size 2)
    struct SimpleCircuit<F: FieldExt> {
        mle: DenseMle<F, Tuple2<F>>,
    }
    impl<F: FieldExt> GKRCircuit<F> for SimpleCircuit<F> {

        type Transcript = PoseidonTranscript<F>;
        type F2 = H2Fr;

        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, InputLayer<F>) {

            // --- The input layer should just be the concatenation of `mle` and `output_input` ---
            let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
            let mut input_layer = InputLayer::<F>::new_from_mles(&mut input_mles, Some(1));
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
                DenseMle::new_from_iter(first_layer_output.into_iter(), LayerId::Input, None);

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

            (layers, vec![Box::new(circuit_output)], input_layer)
        }
    }

    /// Circuit which just subtracts its two halves! No input-output layer needed.
    struct SimplestCircuit<F: FieldExt> {
        mle: DenseMle<F, Tuple2<F>>,
    }
    impl<F: FieldExt> GKRCircuit<F> for SimplestCircuit<F> {

        type Transcript = PoseidonTranscript<F>;
        type F2 = H2Fr;

        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, InputLayer<F>) {

            // --- The input layer should just be the concatenation of `mle` and `output_input` ---
            let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
            let mut input_layer = InputLayer::<F>::new_from_mles(&mut input_mles, None);
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

            (layers, vec![Box::new(first_layer_output)], input_layer)
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
        type F2 = H2Fr;

        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, InputLayer<F>) {

            // --- The input layer should just be the concatenation of `mle`, `mle_2`, and `output_input` ---
            // let mut self_mle_clone = self.mle.clone();
            // let mut self_mle_2_clone = self.mle_2.clone();
            let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle), Box::new(&mut self.mle_2)];
            let mut input_layer = InputLayer::<F>::new_from_mles(&mut input_mles, Some(1));
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
                DenseMle::new_from_iter(computed_output.into_iter(), LayerId::Input, None);
            input_layer.index_input_output_mle(&mut Box::new(&mut output_input));

            // --- Subtract the computed circuit output from the advice circuit output ---
            let builder5 = from_mle(
                (computed_output, output_input.clone()),
                |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
                |(mle1, mle2), layer_id, prefix_bits| {
                    let num_vars = max(mle1.num_iterated_vars(), mle2.num_iterated_vars());
                    ZeroMleRef::new(num_vars, prefix_bits, layer_id)
                },
            );

            // --- Add this final layer to the circuit ---
            let circuit_output = layers.add_gkr(builder5);

            // --- The input layer should just be the concatenation of `mle`, `mle_2`, and `output_input` ---
            let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle), Box::new(&mut self.mle_2)];
            input_layer.combine_input_mles(&input_mles, Some(Box::new(&mut output_input)));

            (layers, vec![Box::new(circuit_output)], input_layer)
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
                let second_num = Fr::rand(&mut rng);
                (num, second_num).into()
            }),
            LayerId::Input,
            None,
        );
        // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        //     (0..size).map(|idx| (Fr::from(idx + 1), Fr::from(idx + 1)).into()),
        //     LayerId::Input,
        //     None,
        // );

        let mut circuit: SimplestCircuit<Fr> = SimplestCircuit { mle };

        let mut transcript: PoseidonTranscript<Fr> =
            PoseidonTranscript::new("GKR Prover Transcript");
        let now = Instant::now();

        match circuit.prove(&mut transcript) {
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
        let size = 1 << 1;

        // --- This should be 2^2 ---
        // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        //     (0..size).map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)).into()),
        //     LayerId::Input,
        //     None,
        // );
        let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
            (0..size).map(|idx| (Fr::from(idx + 2), Fr::from(idx + 2)).into()),
            LayerId::Input,
            None,
        );

        let mut circuit: SimpleCircuit<Fr> = SimpleCircuit { mle };

        let mut transcript: PoseidonTranscript<Fr> =
            PoseidonTranscript::new("GKR Prover Transcript");
        let now = Instant::now();

        match circuit.prove(&mut transcript) {
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
            LayerId::Input,
            None,
        );
        // --- This should be 2^2 ---
        let mle_2: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
            (0..size).map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)).into()),
            LayerId::Input,
            None,
        );

        let mut circuit: TestCircuit<Fr> = TestCircuit { mle, mle_2 };

        let mut transcript: PoseidonTranscript<Fr> =
            PoseidonTranscript::new("GKR Prover Transcript");
        let now = Instant::now();

        match circuit.prove(&mut transcript) {
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
}
