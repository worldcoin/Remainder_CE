//! A verifier challenge node, the node representing when the
//! verifier has to send challenges to the prover.
use ark_std::log2;

use remainder_shared_types::Field;

use crate::{
    input_layer::verifier_challenge_input_layer::CircuitVerifierChallengeInputLayer,
    layer::LayerId,
    layouter::layouting::{CircuitDescriptionMap, CircuitLocation},
};

use super::{CircuitNode, Context, NodeId};

#[derive(Debug, Clone)]
/// The node representing the random challenge that the verifier supplies via Fiat-Shamir.
pub struct VerifierChallengeNode {
    id: NodeId,
    num_vars: usize,
}

impl CircuitNode for VerifierChallengeNode {
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
        self.num_vars
    }
}

impl VerifierChallengeNode {
    /// Constructor for a [VerifierChallengeNode].
    pub fn new(ctx: &Context, num_challenges: usize) -> Self {
        Self {
            id: ctx.get_new_id(),
            num_vars: log2(num_challenges) as usize,
        }
    }

    /// Generate a [CircuitVerifierChallengeInputLayer], which is the
    /// circuit description for a [VerifierChallengeNode].
    pub fn generate_circuit_description<F: Field>(
        &self,
        layer_id: &mut LayerId,
        circuit_description_map: &mut CircuitDescriptionMap,
    ) -> CircuitVerifierChallengeInputLayer<F> {
        let verifier_challenge_layer_id = layer_id.get_and_inc();

        let verifier_challenge_layer = CircuitVerifierChallengeInputLayer::new(
            verifier_challenge_layer_id,
            self.get_num_vars(),
        );

        circuit_description_map.add_node_id_and_location_num_vars(
            self.id,
            (
                CircuitLocation::new(verifier_challenge_layer_id, vec![]),
                self.get_num_vars(),
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
                circuit_inputs::{
                    InputLayerData, InputLayerNode, InputLayerType, InputShred, InputShredData,
                },
                circuit_outputs::OutputNode,
                node_enum::NodeEnum,
                sector::Sector,
                CircuitNode,
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
            let input_a = InputShred::new(ctx, mle_vec_a.num_vars(), &input_layer);
            let input_a_data = InputShredData::new(input_a.id(), mle_vec_a);
            let input_data = InputLayerData::new(input_layer.id(), vec![input_a_data], None);

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

            (
                ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                    input_layer.into(),
                    input_a.into(),
                    verifier_challenge_node.into(),
                    product_sector.into(),
                    difference_sector.into(),
                    output_node.into(),
                ]),
                vec![input_data],
            )
        });

        test_circuit(circuit, None);
    }
}
