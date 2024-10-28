//! A verifier challenge node, the node representing when the
//! verifier has to send challenges to the prover.
use ark_std::log2;

use remainder_shared_types::Field;

use crate::{
    input_layer::fiat_shamir_challenge::FiatShamirChallengeDescription,
    layer::LayerId,
    layouter::layouting::{CircuitDescriptionMap, CircuitLocation},
};

use super::{CircuitNode, NodeId};

#[derive(Debug, Clone)]
/// The node representing the random challenge that the verifier supplies via Fiat-Shamir.
pub struct FiatShamirChallengeNode {
    id: NodeId,
    num_vars: usize,
}

impl CircuitNode for FiatShamirChallengeNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn subnodes(&self) -> Option<Vec<NodeId>> {
        None
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![]
    }

    fn get_num_vars(&self) -> usize {
        self.num_vars
    }
}

impl FiatShamirChallengeNode {
    /// Constructor for a [FiatShamirChallengeNode].
    pub fn new(num_challenges: usize) -> Self {
        Self {
            id: NodeId::new(),
            num_vars: log2(num_challenges) as usize,
        }
    }

    /// Generate a [FiatShamirChallengeDescription], which is the
    /// circuit description for a [FiatShamirChallengeNode].
    pub fn generate_circuit_description<F: Field>(
        &self,
        circuit_description_map: &mut CircuitDescriptionMap,
    ) -> FiatShamirChallengeDescription<F> {
        let layer_id = LayerId::new_fiat_shamir_challenge_layer();
        let fsc_layer = FiatShamirChallengeDescription::new(layer_id, self.get_num_vars());
        circuit_description_map.add_node_id_and_location_num_vars(
            self.id,
            (
                CircuitLocation::new(layer_id, vec![]),
                self.get_num_vars(),
            ),
        );
        fsc_layer
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use remainder_shared_types::Fr;

    use crate::{
        expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
        layouter::nodes::{
            circuit_inputs::{InputLayerNode, InputShred},
            circuit_outputs::OutputNode,
            sector::Sector,
            CircuitNode, NodeId,
        },
        mle::evals::MultilinearExtension,
        prover::{generate_circuit_description, helpers::test_circuit_new},
    };

    use super::FiatShamirChallengeNode;

    #[test]
    fn test_verifier_challenge_node_in_circuit() {
        let input_a_data = MultilinearExtension::new(vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(13),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
        ]);

        let verifier_challenge_node = FiatShamirChallengeNode::new(8);

        let input_layer = InputLayerNode::new(None);
        let input_a = InputShred::new(input_a_data.num_vars(), &input_layer);
        let input_a_id = input_a.id();

        let product_sector = Sector::new(&[&input_a, &verifier_challenge_node], |inputs| {
            Expression::<Fr, AbstractExpr>::mle(inputs[0])
                - Expression::<Fr, AbstractExpr>::mle(inputs[1])
        });

        let difference_sector = Sector::new(&[&product_sector, &product_sector], |inputs| {
            Expression::<Fr, AbstractExpr>::mle(inputs[0])
                - Expression::<Fr, AbstractExpr>::mle(inputs[1])
        });

        let output_node = OutputNode::new_zero(&difference_sector);

        let all_nodes = vec![
            input_layer.into(),
            input_a.into(),
            verifier_challenge_node.into(),
            product_sector.into(),
            difference_sector.into(),
            output_node.into(),
        ];

        let (circ_desc, input_builder_from_shred_map) =
            generate_circuit_description(all_nodes).unwrap();

        let input_builder = move |input_data: MultilinearExtension<Fr>| {
            let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> =
                HashMap::new();
            input_shred_id_to_data.insert(input_a_id, input_data);
            input_builder_from_shred_map(input_shred_id_to_data).unwrap()
        };

        let inputs = input_builder(input_a_data);
        test_circuit_new(&circ_desc, HashMap::new(), &inputs);
    }
}
