//!Module that orchestrates creating a GKR Proof

use std::{collections::HashMap, ops::Range, vec::IntoIter};

use crate::{
    layer::{Claim, GKRLayer, Layer, LayerBuilder, LayerError, LayerId},
    transcript::Transcript,
    FieldExt,
};

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

pub trait GKRCircuit<F: FieldExt> {
    fn synthesize(&mut self) -> Layers<F>;

    fn prove(&mut self, transcript: &mut impl Transcript<F>) -> Result<(), LayerError> {
        let layers: Layers<F> = self.synthesize();

        //set up some claim tracking stuff
        let mut curr_claim: Claim<F> = todo!();
        let mut claims: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();

        //Output layers???
        for mut layer in layers.0.into_iter().rev() {
            //Aggregate claims
            let init_evals = layer.start_sumcheck(curr_claim)?;
            transcript
                .append_field_elements("Initial Sumcheck evaluations", &init_evals)
                .unwrap();

            let rounds: Range<usize> = todo!();

            for round_index in rounds {
                let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
                let evals = layer.prove_round(round_index, challenge)?;
                transcript
                    .append_field_elements("Sumcheck evaluations", &evals)
                    .unwrap();
            }

            let other_claims = layer.get_all_claims()?;

            //Add the claims to the claim tracking state
            for (layer_id, claim) in other_claims {
                if let Some(curr_claims) = claims.get_mut(&layer_id) {
                    curr_claims.push(claim);
                } else {
                    claims.insert(layer_id, vec![claim]);
                }
            }
        }

        //This needs to return the full sumcheck proof
        Ok(())
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
