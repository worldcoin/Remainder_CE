//!Module that orchestrates creating a GKR Proof

use std::{collections::HashMap};

use crate::{
    layer::{
        claims::aggregate_claims, claims::verify_aggragate_claim, Claim, GKRLayer, Layer,
        LayerBuilder, LayerError, LayerId,
    },
    mle::MleIndex,
    mle::MleRef,
    transcript::Transcript,
    FieldExt,
};

// use derive_more::From;
use itertools::Itertools;
use thiserror::Error;

///Newtype for containing the list of Layers that make up the GKR circuit
pub struct Layers<F: FieldExt, Tr: Transcript<F>>(Vec<Box<dyn Layer<F, Transcript = Tr>>>);

impl<F: FieldExt, Tr: Transcript<F> + 'static> Layers<F, Tr> {
    ///Add a layer to a list of layers
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

    ///Add a GKRLayer to a list of layers
    pub fn add_gkr<B: LayerBuilder<F>>(&mut self, new_layer: B) -> B::Successor {
        self.add::<_, GKRLayer<_, Tr>>(new_layer)
    }

    ///Creates a new Layers
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
///Errors relating to the proving of a GKR circuit
pub enum GKRError {
    #[error("No claims were found for layer {0:?}")]
    ///No claims were found for layer
    NoClaimsForLayer(LayerId),
    #[error("Error when proving layer {0:?}: {1}")]
    ///Error when proving layer
    ErrorWhenProvingLayer(LayerId, LayerError),
    #[error("Error when verifying layer {0:?}: {1}")]
    ///Error when verifying layer
    ErrorWhenVerifyingLayer(LayerId, LayerError),
    #[error("Error when verifying output layer")]
    ///Error when verifying output layer
    ErrorWhenVerifyingOutputLayer,
}

///A proof of the sumcheck protocol; Outer vec is rounds, inner vec is evaluations
#[derive(Clone, Debug)]
pub struct SumcheckProof<F: FieldExt>(Vec<Vec<F>>);

impl<F: FieldExt> From<Vec<Vec<F>>> for SumcheckProof<F> {
    fn from(value: Vec<Vec<F>>) -> Self {
        Self(value)
    }
}

///The proof for an individual GKR layer
pub struct LayerProof<F: FieldExt, Tr: Transcript<F>> {
    sumcheck_proof: SumcheckProof<F>,
    layer: Box<dyn Layer<F, Transcript = Tr>>,
    wlx_evaluations: Vec<F>,
}

///All the elements to be passed to the verifier for the succinct non-interactive sumcheck proof
pub struct GKRProof<F: FieldExt, Tr: Transcript<F>> {
    ///The sumcheck proof of each GKR Layer, along with the fully bound expression.
    ///
    /// In reverse order (e.g. layer closest to the output layer is first)
    pub layer_sumcheck_proofs: Vec<LayerProof<F, Tr>>,
    ///All the output layers that this circuit yields
    pub output_layers: Vec<Box<dyn MleRef<F = F>>>,
}

///A GKRCircuit ready to be proven
pub trait GKRCircuit<F: FieldExt> {
    ///The transcript this circuit uses
    type Transcript: Transcript<F>;
    ///The forward pass, defining the layer relationships and generating the layers
    fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>);

    ///The backwards pass, creating the GKRProof
    fn prove(
        &mut self,
        transcript: &mut Self::Transcript,
    ) -> Result<GKRProof<F, Self::Transcript>, GKRError> {
        let (layers, mut output_layers) = self.synthesize();

        let mut claims: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();

        for output in output_layers.iter_mut() {
            let mut claim = None;
            let bits = output.index_mle_indices(0);
            for bit in 0..bits {
                let challenge = transcript
                    .get_challenge("Setting Output Layer Claim")
                    .unwrap();
                claim = output.fix_variable(bit, challenge);
            }

            let claim = claim.unwrap();

            let layer_id = output.get_layer_id();

            if let Some(curr_claims) = claims.get_mut(&layer_id) {
                curr_claims.push(claim);
            } else {
                claims.insert(layer_id, vec![claim]);
            }
        }

        let layer_sumcheck_proofs = layers
            .0
            .into_iter()
            .rev()
            .map(|mut layer| {
                //Aggregate claims
                let layer_id = layer.id().clone();
                let layer_claims = claims
                    .get(&layer_id)
                    .ok_or_else(|| GKRError::NoClaimsForLayer(layer_id.clone()))?;

                for claim in layer_claims {
                    transcript
                        .append_field_elements("Claimed bits to be aggregated", &claim.0)
                        .unwrap();
                    transcript
                        .append_field_element("Claimed value to be aggregated", claim.1)
                        .unwrap();
                }

                let agg_chal = transcript
                    .get_challenge("Challenge for claim aggregation")
                    .unwrap();

                let (layer_claim, wlx_evaluations) =
                    aggregate_claims(layer_claims, layer.expression(), agg_chal).unwrap();

                transcript
                    .append_field_elements("Claim Aggregation Wlx_evaluations", &wlx_evaluations)
                    .unwrap();

                let sumcheck_rounds = layer
                    .prove_rounds(layer_claim, transcript)
                    .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id.clone(), err))?;

                let other_claims = layer
                    .get_claims()
                    .map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id.clone(), err))?;

                //Add the claims to the claim tracking state
                for (layer_id, claim) in other_claims {
                    if let Some(curr_claims) = claims.get_mut(&layer_id) {
                        curr_claims.push(claim);
                    } else {
                        claims.insert(layer_id, vec![claim]);
                    }
                }

                Ok(LayerProof {
                    sumcheck_proof: sumcheck_rounds,
                    layer,
                    wlx_evaluations,
                })
            })
            .try_collect()?;

        let gkr_proof = GKRProof {
            layer_sumcheck_proofs,
            output_layers,
        };

        Ok(gkr_proof)
    }

    /// Verifies the GKRProof produced by fn prove
    ///
    /// Derive randomness from constructing the Transcript from scratch
    fn verify(
        &mut self,
        transcript: &mut Self::Transcript,
        gkr_proof: GKRProof<F, Self::Transcript>,
    ) -> Result<(), GKRError> {
        let GKRProof {
            layer_sumcheck_proofs,
            output_layers,
        } = gkr_proof;

        let mut claims: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();

        for output in output_layers.iter() {
            let mle_indices = output.mle_indices();
            let bits = mle_indices.len();
            let mut claim_chal: Vec<F> = vec![];
            for bit in 0..bits {
                let challenge = transcript
                    .get_challenge("Setting Output Layer Claim")
                    .unwrap();

                // assume the output layers are zero valued...
                // cannot actually do the initial step of evaluating V_1'(z) as specified in Thaler 13 page 14
                // basically checks all the challenges are correct right now
                if MleIndex::Bound(challenge, bit) != mle_indices[bit] {
                    return Err(GKRError::ErrorWhenVerifyingOutputLayer);
                }
                claim_chal.push(challenge);
            }
            let claim = (claim_chal, F::zero());
            let layer_id = output.get_layer_id();

            if let Some(curr_claims) = claims.get_mut(&layer_id) {
                curr_claims.push(claim);
            } else {
                claims.insert(layer_id, vec![claim]);
            }
        }

        // at this point, verifier knows all the output layer's claims, it recurses down the layers.
        // each layer has an individual sumcheck proof
        for sumcheck_proof_single in layer_sumcheck_proofs {
            // for each layer check check claims are actually the first x values of init_evals

            // expression is used for the one oracle query
            let LayerProof {
                sumcheck_proof,
                mut layer,
                wlx_evaluations,
            } = sumcheck_proof_single;

            let layer_id = layer.id().clone();
            let layer_claims = claims
                .get(&layer_id)
                .ok_or_else(|| GKRError::NoClaimsForLayer(layer_id.clone()))?;

            for claim in layer_claims {
                transcript
                    .append_field_elements("Claimed bits to be aggregated", &claim.0)
                    .unwrap();
                transcript
                    .append_field_element("Claimed value to be aggregated", claim.1)
                    .unwrap();
            }

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

            layer
                .verify_rounds(prev_claim, sumcheck_proof.0, transcript)
                .map_err(|err| GKRError::ErrorWhenVerifyingLayer(layer_id.clone(), err))?;

            // verifier manipulates transcript same way as prover
            let other_claims = layer
                .get_claims()
                .map_err(|err| GKRError::ErrorWhenVerifyingLayer(layer_id.clone(), err))?;

            //Add the claims to the claim tracking state
            for (layer_id, claim) in other_claims {
                if let Some(curr_claims) = claims.get_mut(&layer_id) {
                    curr_claims.push(claim);
                } else {
                    claims.insert(layer_id, vec![claim]);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::{cmp::max, time::Instant};

    use ark_bn254::Fr;
    use ark_std::{test_rng, UniformRand};
    use rayon::{prelude::ParallelIterator};
    

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

    use super::{GKRCircuit, Layers};

    struct TestCircuit<F: FieldExt> {
        mle: DenseMle<F, Tuple2<F>>,
        mle_2: DenseMle<F, Tuple2<F>>,
    }

    impl<F: FieldExt> GKRCircuit<F> for TestCircuit<F> {
        type Transcript = PoseidonTranscript<F>;
        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>) {
            let mut layers = Layers::new();

            let builder = from_mle(
                self.mle.clone(),
                |mle| ExpressionStandard::products(vec![mle.first(), mle.second()]),
                |mle, layer_id, prefix_bits| {
                    let mut output = mle
                        .into_iter()
                        .map(|(first, second)| first * second)
                        .collect::<DenseMle<F, F>>();

                    output.define_layer_id(layer_id);
                    output.add_prefix_bits(prefix_bits);

                    output
                },
            );

            let builder2 = from_mle(
                self.mle_2.clone(),
                |mle| mle.first().expression() + mle.second().expression(),
                |mle, layer_id, prefix_bits| {
                    let mut output = mle
                        .into_iter()
                        .map(|(first, second)| first + second)
                        .collect::<DenseMle<F, F>>();

                    output.define_layer_id(layer_id);
                    output.add_prefix_bits(prefix_bits);

                    output
                },
            );

            let builder3 = builder.concat(builder2);

            let output = layers.add_gkr(builder3);

            let builder4 = from_mle(
                output,
                |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
                |(mle1, mle2), layer_id, prefix_bits| {
                    let mut new_mle = mle1
                        .clone()
                        .into_iter()
                        .zip(mle2.clone().into_iter())
                        .map(|(first, second)| first - second)
                        .collect::<DenseMle<F, F>>();
                    new_mle.define_layer_id(layer_id);
                    new_mle.add_prefix_bits(prefix_bits);
                    new_mle
                },
            );

            let output = layers.add_gkr(builder4);

            let mut output_input = output.clone();
            output_input.define_layer_id(LayerId::Input);

            let builder4 = from_mle(
                (output, output_input),
                |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
                |(mle1, mle2), layer_id, prefix_bits| {
                    let num_vars = max(mle1.num_vars(), mle2.num_vars());
                    ZeroMleRef::new(num_vars, prefix_bits, layer_id)
                },
            );

            let output = layers.add_gkr(builder4);

            (layers, vec![Box::new(output)])
        }
    }

    #[test]
    fn test_gkr() {
        let mut rng = test_rng();
        let size = 2 << 18;
        // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
        // tracing::subscriber::set_global_default(subscriber)
        //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

        let mut mle: DenseMle<Fr, Tuple2<Fr>> = (0..size)
            .map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)).into())
            .collect();
        mle.define_layer_id(LayerId::Input);
        let mut mle_2: DenseMle<Fr, Tuple2<Fr>> = (0..size)
            .map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)).into())
            .collect();

        mle_2.define_layer_id(LayerId::Input);

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
