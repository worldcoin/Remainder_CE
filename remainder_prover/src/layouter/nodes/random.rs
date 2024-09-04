use std::marker::PhantomData;

use ark_std::log2;
use remainder_shared_types::{transcript::ProverTranscript, FieldExt};

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    input_layer::random_input_layer::RandomInputLayer,
    layer::LayerId,
    layouter::layouting::{CircuitLocation, CircuitMap},
    mle::evals::MultilinearExtension,
};

use super::{CircuitNode, Context, NodeId};

#[derive(Debug, Clone)]
/// The node representing the random challenge that the verifier supplies via Fiat-Shamir.
pub struct VerifierChallengeNode<F: FieldExt> {
    id: NodeId,
    num_challenges: usize,
    num_vars: usize,
    _marker: PhantomData<F>,
}

impl<F: FieldExt> CircuitNode for VerifierChallengeNode<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn children(&self) -> Option<Vec<NodeId>> {
        None
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![]
    }

    fn get_num_vars(&self) -> usize {
        todo!()
    }
}

impl<F: FieldExt> VerifierChallengeNode<F> {
    pub fn new(ctx: &Context, num_challenges: usize) -> Self {
        Self {
            id: ctx.get_new_id(),
            num_challenges,
            num_vars: log2(num_challenges),
            _marker: PhantomData,
        }
    }

    pub fn compile<'a>(
        &'a mut self,
        layer_id: &mut LayerId,
        circuit_map: &mut CircuitMap<'a, F>,
        transcript: &mut impl ProverTranscript<F>,
    ) -> RandomInputLayer<F> {
        let random_challenges_vec =
            transcript.get_challenges("Verifier FS Challenges", self.num_challenges);
        let mle = MultilinearExtension::new(random_challenges_vec);
        let random_il_layer_id = layer_id.get_and_inc();
        let verifier_challenge_layer = RandomInputLayer::new(mle.clone(), random_il_layer_id);

        circuit_map.add_node(
            self.id,
            (
                CircuitLocation::new(*layer_id, vec![]),
                &self.data.as_ref().unwrap(),
            ),
        );

        verifier_challenge_layer
    }
}

#[cfg(test)]
mod test {
    use remainder_shared_types::Fr;

    use crate::{
        expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
        layouter::{
            compiling::LayouterCircuit,
            component::ComponentSet,
            nodes::{
                circuit_inputs::{InputLayerNode, InputLayerType, InputShred},
                circuit_outputs::OutputNode,
                node_enum::NodeEnum,
                sector::Sector,
            },
        },
        mle::evals::MultilinearExtension,
        prover::helpers::test_circuit,
    };

    use super::VerifierChallengeNode;

    #[test]
    fn test_verifier_challenge_node_in_circuit() {
        let circuit = LayouterCircuit::new(|ctx| {
            let mle_vec_a = MultilinearExtension::new(vec![
                Fr::from(1),
                Fr::from(2),
                Fr::from(9),
                Fr::from(10),
                Fr::from(13),
                Fr::from(1),
                Fr::from(3),
                Fr::from(10),
            ]);

            let verifier_challenge_node = VerifierChallengeNode::new(ctx, 8);

            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let input_a = InputShred::new(ctx, mle_vec_a, &input_layer);

            let product_sector =
                Sector::new(ctx, &[&input_a, &verifier_challenge_node], |inputs| {
                    Expression::<Fr, AbstractExpr>::mle(inputs[0])
                        - Expression::<Fr, AbstractExpr>::mle(inputs[1])
                });

            let difference_sector =
                Sector::new(ctx, &[&product_sector, &product_sector], |inputs| {
                    Expression::<Fr, AbstractExpr>::mle(inputs[0])
                        - Expression::<Fr, AbstractExpr>::mle(inputs[1])
                });

            let output_node = OutputNode::new_zero(ctx, &difference_sector);

            ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                input_layer.into(),
                input_a.into(),
                verifier_challenge_node.into(),
                product_sector.into(),
                difference_sector.into(),
                output_node.into(),
            ])
        });

        test_circuit(circuit, None);
    }
}
