//!Module that orchestrates creating a GKR Proof

use std::{collections::HashMap, ops::Range, vec::IntoIter};

use crate::{
    layer::{Claim, GKRLayer, Layer, LayerBuilder, LayerError, LayerId, claims::aggregate_claims, claims::verify_aggragate_claim},
    transcript::Transcript,
    FieldExt, expression::ExpressionStandard, mle::MleRef, mle::MleIndex
};

use itertools::Itertools;
use thiserror::Error;

///Newtype for containing the list of Layers that make up the GKR circuit
pub struct Layers<F: FieldExt, Tr: Transcript<F>>(Vec<Box<dyn Layer<F, Transcript = Tr>>>);

impl<F: FieldExt, Tr: Transcript<F> + 'static> Layers<F, Tr> {
    ///Add a layer to a list of layers
    pub fn add<B: LayerBuilder<F>, L: Layer<F, Transcript = Tr> + 'static>(&mut self, new_layer: B) -> B::Successor {
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

#[derive(Error, Debug, Clone)]
pub enum GKRError {
    #[error("No claims were found for layer {0:?}")]
    NoClaimsForLayer(LayerId),
    #[error("Error when proving layer {0:?}: {1}")]
    ErrorWhenProvingLayer(LayerId, LayerError),
    #[error("Error when verifying layer {0:?}: {1}")]
    ErrorWhenVerifyingLayer(LayerId, LayerError),
    #[error("Error when verifying output layer")]
    ErrorWhenVerifyingOutputLayer()
}

///A proof of the sumcheck protocol; Outer vec is rounds, inner vec is evaluations
pub struct SumcheckProof<F: FieldExt>(Vec<Vec<F>>);

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
    pub output_layers: Vec<Box<dyn MleRef<F = F>>>
}

///A GKRCircuit ready to be proven
pub trait GKRCircuit<F: FieldExt> {
    type Transcript: Transcript<F>;
    ///The forward pass, defining the layer relationships and generating the layers
    fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>);

    /// Verifies the GKRProof produced by fn prove,
    /// Derive randomness from constructing the Transcript from scratch
    fn verify(&mut self, transcript: &mut Self::Transcript, gkr_proof: GKRProof<F, Self::Transcript>) -> Result<(), GKRError> {
        let GKRProof{layer_sumcheck_proofs, output_layers} = gkr_proof;

        let mut claims: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();

        for output in output_layers.iter() {
            let bits = output.num_vars();
            let mle_indices = output.mle_indices();
            let mut claim_chal: Vec<F> = vec![];
            for bit in 0..bits {
                let challenge = transcript.get_challenge("Setting Output Layer Claim").unwrap();

                // assume the output layers are zero valued...
                // cannot actually do the initial step of evaluating V_1'(z) as specified in Thaler 13 page 14
                // basically checks all the challenges are correct right now
                if MleIndex::Bound(challenge) != mle_indices[bit] {
                    return Err(GKRError::ErrorWhenVerifyingOutputLayer());
                }
                claim_chal.push(challenge);
            }
            let claim = (claim_chal, F::zero());
            let layer_id = output.get_layer_id().unwrap();

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
            let LayerProof {sumcheck_proof: sumcheck_rounds, layer: mut layer, wlx_evaluations: _} = sumcheck_proof_single;

            let layer_id = layer.get_id().clone();
            let layer_claims = claims.get(&layer_id).ok_or_else(|| GKRError::NoClaimsForLayer(layer_id.clone()))?;

            for claim in layer_claims {
                transcript.append_field_elements("Claimed bits to be aggregated", &claim.0).unwrap();
                transcript.append_field_element("Claimed value to be aggregated", claim.1).unwrap();
            }

            let agg_chal = transcript.get_challenge("Challenge for claim aggregation").unwrap();

            let init_evals = &sumcheck_rounds.0[0];
            let prev_claim = verify_aggragate_claim(init_evals, layer_claims, agg_chal, layer.get_expression()).map_err(|_err| GKRError::ErrorWhenVerifyingLayer(layer_id.clone(), LayerError::AggregationError()))?;

            transcript
            .append_field_elements("Initial Sumcheck evaluations", &init_evals)
            .unwrap();

            let _ = layer.verify_rounds(prev_claim, sumcheck_rounds.0, transcript, layer.get_expression().clone());

            // verifier manipulates transcript same way as prover
            let other_claims = layer.get_claims().map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id.clone(), err))?;

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

    ///The backwards pass, creating the GKRProof
    fn prove(&mut self, transcript: &mut Self::Transcript) -> Result<GKRProof<F, Self::Transcript>, GKRError> {
        let (layers, mut output_layers) = self.synthesize();

        let mut claims: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();

        for output in output_layers.iter_mut() {
            let mut claim = None;
            let bits = output.index_mle_indices(0);
            for bit in 0..bits {
                let challenge = transcript.get_challenge("Setting Output Layer Claim").unwrap();
                claim = output.fix_variable(bit, challenge);
            }

            let claim = claim.unwrap();
            let layer_id = output.get_layer_id().unwrap();

            if let Some(curr_claims) = claims.get_mut(&layer_id) {
                curr_claims.push(claim);
            } else {
                claims.insert(layer_id, vec![claim]);
            }
        }

        let layer_sumcheck_proofs = layers.0.into_iter().rev().map(|mut layer| {
            //Aggregate claims
            let layer_id = layer.get_id().clone();
            let layer_claims = claims.get(&layer_id).ok_or_else(|| GKRError::NoClaimsForLayer(layer_id.clone()))?;

            for claim in layer_claims {
                transcript.append_field_elements("Claimed bits to be aggregated", &claim.0).unwrap();
                transcript.append_field_element("Claimed value to be aggregated", claim.1).unwrap();
            }

            let agg_chal = transcript.get_challenge("Challenge for claim aggregation").unwrap();

            // init_evals are the wlx from aggregate_claims
            let (layer_claim, wlx_evaluations) = aggregate_claims(layer_claims, layer.get_expression(), agg_chal).unwrap();

            // TODO!(ende) init_evals will probably come from aggregate_claims, i.e. the wlx values
            let (init_evals, rounds) = layer.start_sumcheck(layer_claim).map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id.clone(), err))?;
            transcript
                .append_field_elements("Initial Sumcheck evaluations", &init_evals)
                .unwrap();

            let sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(init_evals)).chain((1..=rounds).map(|round_index| {
                let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
                let evals = layer.prove_round(round_index, challenge)?;
                transcript
                    .append_field_elements("Sumcheck evaluations", &evals)
                    .unwrap();
                Ok::<_, LayerError>(evals)
            })).try_collect().map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id.clone(), err))?;

            let sumcheck_rounds = SumcheckProof(sumcheck_rounds);

            let expression = layer.get_expression().clone();

            let other_claims = layer.get_claims().map_err(|err| GKRError::ErrorWhenProvingLayer(layer_id.clone(), err))?;

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
        }).try_collect()?;

        let gkr_proof = GKRProof {
            layer_sumcheck_proofs,
            output_layers
        };

        //This needs to return the full sumcheck proof
        Ok(gkr_proof)
    }
}

#[cfg(test)]
mod test {
    use std::cmp::max;

    use ark_bn254::Fr;
    use ark_std::One;

    use crate::{transcript::{poseidon_transcript::PoseidonTranscript, Transcript}, FieldExt, mle::{dense::{DenseMle, Tuple2}, MleRef, Mle, zero::ZeroMleRef}, layer::{LayerBuilder, from_mle, SimpleLayer, LayerId}, expression::ExpressionStandard};

    use super::{GKRCircuit, Layers};

    struct TestCircuit<F: FieldExt> {
        mle: DenseMle<F, Tuple2<F>>,
        mle_2: DenseMle<F, Tuple2<F>>,
    }

    impl<F: FieldExt> GKRCircuit<F> for TestCircuit<F> {
        type Transcript = PoseidonTranscript<F>;
        fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>) {
            let mut layers = Layers::new();

            let builder = from_mle(self.mle.clone(), |mle| {
                ExpressionStandard::products(vec![mle.first(), mle.second()])
            }, |mle, layer_id, prefix_bits| {
                let mut output = mle.into_iter().map(|(first, second)| first * second).collect::<DenseMle<F, F>>();

                output.define_layer_id(layer_id);
                output.add_prefix_bits(prefix_bits);

                output
            });

            let builder2 = from_mle(self.mle_2.clone(), |mle| {
                mle.first().expression() + mle.second().expression()
            }, |mle, layer_id, prefix_bits| {
                let mut output = mle.into_iter().map(|(first, second)| first + second).collect::<DenseMle<F, F>>();

                output.define_layer_id(layer_id);
                output.add_prefix_bits(prefix_bits);

                output
            });

            let builder3 = builder.concat(builder2);

            let output = layers.add_gkr(builder3);

            let builder4 = from_mle(output, |(mle1, mle2)| {
                mle1.mle_ref().expression() - mle2.mle_ref().expression()
            }, |(mle1, mle2), layer_id, prefix_bits| {
                let num_vars = max(mle1.num_vars(), mle2.num_vars());

                ZeroMleRef::new(num_vars, prefix_bits, layer_id)
            });

            let output = layers.add_gkr(builder4);

            (layers, vec![Box::new(output)])
        }
    }

    #[test]
    fn test_gkr() {
        let mut mle: DenseMle<Fr, Tuple2<Fr>> = vec![(Fr::from(2), Fr::from(8)), (Fr::from(7), Fr::from(3))].into_iter().map(|x| x.into()).collect();
        mle.define_layer_id(LayerId::Input);
        let mut mle_2: DenseMle<Fr, Tuple2<Fr>> = vec![(Fr::from(9), Fr::from(7)), (Fr::from(15), Fr::from(6))].into_iter().map(|x| x.into()).collect();
        mle_2.define_layer_id(LayerId::Input);
        let mut circuit = TestCircuit::<Fr> {
            mle,
            mle_2
        };

        let mut transcript: PoseidonTranscript<Fr> = PoseidonTranscript::new("New Poseidon Test Transcript");

        circuit.prove(&mut transcript).unwrap();
     }
 }
