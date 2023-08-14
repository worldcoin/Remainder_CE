//!Module that orchestrates creating a GKR Proof

/// For the input layer to the GKR circuit
pub mod input_layer;

use std::collections::HashMap;

use crate::{
    layer::{
        claims::aggregate_claims, claims::verify_aggragate_claim, Claim, GKRLayer, Layer,
        LayerBuilder, LayerError, LayerId,
    },
    mle::MleIndex,
    mle::MleRef,
    transcript::Transcript,
    FieldExt, expression::ExpressionStandard,
};

// use derive_more::From;
use itertools::Itertools;
use thiserror::Error;

use self::input_layer::InputLayer;

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
pub struct GKRProof<F: FieldExt, Tr: Transcript<F>> {
    /// The sumcheck proof of each GKR Layer, along with the fully bound expression.
    /// 
    /// In reverse order (i.e. layer closest to the output layer is first)
    pub layer_sumcheck_proofs: Vec<LayerProof<F, Tr>>,
    /// All the output layers that this circuit yields
    pub output_layers: Vec<Box<dyn MleRef<F = F>>>,
    /// Proof for the circuit input layer
    pub input_layer_proof: InputLayerProof<F>,
}

/// A GKRCircuit ready to be proven
pub trait GKRCircuit<F: FieldExt> {
    /// The transcript this circuit uses
    type Transcript: Transcript<F>;
    /// The forward pass, defining the layer relationships and generating the layers
    fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, InputLayer<F>);

    /// The backwards pass, creating the GKRProof
    fn prove(
        &mut self,
        transcript: &mut Self::Transcript,
    ) -> Result<GKRProof<F, Self::Transcript>, GKRError> {

        // --- Synthesize the circuit, using LayerBuilders to create internal, output, and input layers ---
        let (layers, mut output_layers, input_layer) = self.synthesize();

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
                // --- claim aggregation protocol ---
                let agg_chal = transcript
                    .get_challenge("Challenge for claim aggregation")
                    .unwrap();

                let (layer_claim, wlx_evaluations) =
                    aggregate_claims(layer_claims, layer.expression(), agg_chal).unwrap();

                transcript
                    .append_field_elements("Claim Aggregation Wlx_evaluations", &wlx_evaluations)
                    .unwrap();

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

        // --- Add the claimed values to the FS transcript ---
        for claim in input_layer_claims {
            transcript
                .append_field_elements("Claimed challenge coordinates to be aggregated", &claim.0)
                .unwrap();
            transcript
                .append_field_element("Claimed value to be aggregated", claim.1)
                .unwrap();
        }

        // --- Aggregate claims by sampling r^\star from the verifier and performing the ---
        // --- claim aggregation protocol ---
        let agg_chal = transcript
            .get_challenge("Challenge for input claim aggregation")
            .unwrap();

        // --- The "expression" to aggregate over is simply the input layer's combined MLE, but as an expression ---
        let input_layer_expression = ExpressionStandard::Mle(input_layer.get_combined_mle().mle_ref());
        let (input_layer_claim, input_wlx_evaluations) =
            aggregate_claims(input_layer_claims, &input_layer_expression, agg_chal).unwrap();

        transcript
            .append_field_elements("Input claim aggregation Wlx_evaluations", &input_wlx_evaluations)
            .unwrap();

        let input_layer_proof = InputLayerProof {
            input_layer_aggregated_claim_proof: input_wlx_evaluations,
        };

        let gkr_proof = GKRProof {
            layer_sumcheck_proofs,
            output_layers,
            input_layer_proof,
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
            input_layer_proof
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
            let agg_chal = transcript
                .get_challenge("Challenge for claim aggregation")
                .unwrap();

            let prev_claim = verify_aggragate_claim(&wlx_evaluations, layer_claims, agg_chal)
                .map_err(|_err| {
                    GKRError::ErrorWhenVerifyingLayer(
                        layer_id.clone(),
                        LayerError::AggregationError,
                    )
                })?;

            transcript
                .append_field_elements("Claim Aggregation Wlx_evaluations", &wlx_evaluations)
                .unwrap();

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
    
        let input_r_star = transcript
            .get_challenge("Challenge for input claim aggregation")
            .unwrap();

        // --- Perform the aggregation verification step and extract the correct input layer claim ---
        let input_layer_claim = verify_aggragate_claim(&input_layer_aggregated_claim_proof, input_layer_claims, input_r_star)
        .map_err(|_err| {
            GKRError::ErrorWhenVerifyingLayer(
                input_layer_id,
                LayerError::AggregationError,
            )
        })?;

        // TODO!(ryancao): Verify the final `input_layer_claim` using Ligero!!!

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{cmp::max, time::Instant};

    use ark_bn254::Fr;
    use ark_std::{test_rng, UniformRand};

    use crate::{
        expression::ExpressionStandard,
        layer::{from_mle, LayerBuilder, LayerId},
        mle::{
            dense::{DenseMle, Tuple2},
            zero::ZeroMleRef,
            Mle, MleRef,
        },
        transcript::{poseidon_transcript::PoseidonTranscript, Transcript},
        FieldExt,
    };

    use super::{GKRCircuit, Layers, input_layer::InputLayer};

    /// This circuit is a 4k --> k circuit, such that
    /// [x_1, x_2, x_3, x_4] --> [x_1 * x_3, x_2 + x_4] --> [(x_1 * x_3) - (x_2 + x_4)]
    struct TestCircuit<F: FieldExt> {
        mle: DenseMle<F, Tuple2<F>>,
        mle_2: DenseMle<F, Tuple2<F>>,
    }

    impl<F: FieldExt> GKRCircuit<F> for TestCircuit<F> {
        type Transcript = PoseidonTranscript<F>;
        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, InputLayer<F>) {

            // --- Create Layers to be added to ---
            let mut layers = Layers::new();

            // --- Create a SimpleLayer from the first `mle` within the circuit ---
            let builder = from_mle(
                self.mle.clone(),
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
                self.mle_2.clone(),
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
            let output = layers.add_gkr(builder4);

            // --- Ahh. So we're doing the thing where we add the "real" circuit output as a circuit input, ---
            // --- then check if the difference between the two is zero ---
            let mut output_input =
                DenseMle::new_from_iter(output.into_iter(), LayerId::Input, None);

            // --- Subtract the computed circuit output from the advice circuit output ---
            let builder4 = from_mle(
                (output, output_input.clone()),
                |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
                |(mle1, mle2), layer_id, prefix_bits| {
                    let num_vars = max(mle1.num_vars(), mle2.num_vars());
                    ZeroMleRef::new(num_vars, prefix_bits, layer_id)
                },
            );

            // --- Add this final layer to the circuit ---
            let output = layers.add_gkr(builder4);

            // --- The input layer should just be the concatenation of `mle`, `mle_2`, and `output_input` ---
            let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle), Box::new(&mut self.mle_2), Box::new(&mut output_input)];
            let input_layer = InputLayer::<F>::new_from_mles(&mut input_mles);

            (layers, vec![Box::new(output)], input_layer)
        }
    }

    #[test]
    fn test_gkr() {
        let mut rng = test_rng();
        let size = 2 << 10;
        // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
        // tracing::subscriber::set_global_default(subscriber)
        //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

        let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
            (0..size).map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)).into()),
            LayerId::Input,
            None,
        );
        let mle_2: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
            (0..size).map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)).into()),
            LayerId::Input,
            None,
        );

        let mut circuit = TestCircuit { mle, mle_2 };

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
