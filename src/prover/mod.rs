//!Module that orchestrates creating a GKR Proof

use std::{collections::HashMap, ops::Range, vec::IntoIter};

use crate::{
    layer::{Claim, GKRLayer, Layer, LayerBuilder, LayerError, LayerId},
    transcript::Transcript,
    FieldExt, expression::ExpressionStandard,
};

use itertools::Itertools;

///Newtype for containing the list of Layers that make up the GKR circuit
pub struct Layers<F: FieldExt>(Vec<Box<dyn Layer<F>>>);

impl<F: FieldExt> Layers<F> {
    ///Add a layer to a list of layers
    pub fn add<B: LayerBuilder<F>, L: Layer<F> + 'static>(&mut self, new_layer: B) -> B::Successor {
        let id = LayerId::Layer(self.0.len());
        let successor = new_layer.next_layer(id.clone(), None);
        let layer = L::new(new_layer, id);
        self.0.push(Box::new(layer));
        successor
    }

    ///Add a GKRLayer to a list of layers
    pub fn add_gkr<B: LayerBuilder<F>>(&mut self, new_layer: B) -> B::Successor {
        self.add::<_, GKRLayer<_>>(new_layer)
    }
}

///A proof of the sumcheck protocol; Outer vec is rounds, inner vec is evaluations
pub struct SumcheckProof<F: FieldExt>(Vec<Vec<F>>);

///All the elements to be passed to the verifier for the succinct non-interactive sumcheck proof
pub struct GKRProof<F: FieldExt> {
    ///The sumcheck proof of each GKR Layer, along with the fully bound expression.
    /// 
    /// In reverse order (e.g. layer closest to the output layer is first)
    pub layer_sumcheck_proofs: Vec<(SumcheckProof<F>, ExpressionStandard<F>)>,
}

///A GKRCircuit ready to be proven
pub trait GKRCircuit<F: FieldExt> {
    ///The forward pass, defining the layer relationships and generating the layers
    fn synthesize(&mut self) -> Layers<F>;

    ///The backwards pass, creating the GKRProof
    fn prove(&mut self, transcript: &mut impl Transcript<F>) -> Result<GKRProof<F>, LayerError> {
        let layers: Layers<F> = self.synthesize();

        //set up some claim tracking stuff
        let mut curr_claim: Claim<F> = todo!();
        let mut claims: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();

        //Output layers???
        let layer_sumcheck_proofs = layers.0.into_iter().rev().map(|mut layer| {
            //Aggregate claims
            let init_evals = layer.start_sumcheck(curr_claim.clone())?;
            transcript
                .append_field_elements("Initial Sumcheck evaluations", &init_evals)
                .unwrap();

            let rounds: Range<usize> = todo!();

            let sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(init_evals)).chain(rounds.map(|round_index| {
                let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
                let evals = layer.prove_round(round_index, challenge)?;
                transcript
                    .append_field_elements("Sumcheck evaluations", &evals)
                    .unwrap();
                Ok::<_, LayerError>(evals)
            })).try_collect()?;

            let sumcheck_rounds = SumcheckProof(sumcheck_rounds);

            let expression = layer.get_expression().clone();

            let other_claims = layer.get_claims()?;

            //Add the claims to the claim tracking state
            for (layer_id, claim) in other_claims {
                if let Some(curr_claims) = claims.get_mut(&layer_id) {
                    curr_claims.push(claim);
                } else {
                    claims.insert(layer_id, vec![claim]);
                }
            }

            Ok((sumcheck_rounds, expression))
        }).try_collect()?;

        let gkr_proof = GKRProof {
            layer_sumcheck_proofs
        };

        //This needs to return the full sumcheck proof
        Ok(gkr_proof)
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{transcript::poseidon_transcript::PoseidonTranscript, FieldExt};

    use super::{GKRCircuit, Layers};

    struct TestCircuit {}

    impl<F: FieldExt> GKRCircuit<F> for TestCircuit {
        fn synthesize(&mut self) -> Layers<F> {
            todo!()
        }
    }

    //#[test]
    fn test_gkr() {
        let mut circuit = TestCircuit {};

        let transcript: PoseidonTranscript<Fr> = todo!();

        circuit.prove(&mut transcript).unwrap();
    }
}
