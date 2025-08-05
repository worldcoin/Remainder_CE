//! The FiatShamirChallengeNode is a node that represents when the verifier has
//! to send challenges to the prover, which it samples via the Fiat-Shamir
//! transformation.
//!
//! Currently, when the circuit builder is generating a circuit, the actual
//! challenge is not sampled yet, so this node serves as a placeholder for an
//! indeterminate that will later be populated by the challenge that the
//! verifier sapmles.
//!
//! For example, one use-case for this node is in LogUp circuits, where the
//! denominator of the left-hand-side terms of the logup summation equation
//! contains an indeterminate, which is later replaced with a random challenge
//! sampled via FiatShamir. In that case, we would use the
//! [FiatShamirChallengeNode] while building the circuit.
use ark_std::log2;

use remainder_shared_types::Field;

use crate::{
    input_layer::fiat_shamir_challenge::FiatShamirChallengeDescription,
    layer::LayerId,
    layouter::{builder::CircuitMap, layouting::CircuitLocation},
};

use super::{CircuitNode, NodeId};

#[derive(Debug, Clone)]
/// The node representing the random challenge that the verifier supplies via
/// Fiat-Shamir.
pub struct FiatShamirChallengeNode {
    id: NodeId,
    num_vars: usize,
}

impl CircuitNode for FiatShamirChallengeNode {
    fn id(&self) -> NodeId {
        self.id
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
        circuit_map: &mut CircuitMap,
    ) -> FiatShamirChallengeDescription<F> {
        let layer_id = LayerId::next_fiat_shamir_challenge_layer_id();
        let fsc_layer = FiatShamirChallengeDescription::new(layer_id, self.get_num_vars());
        circuit_map.add_node_id_and_location_num_vars(
            self.id,
            (CircuitLocation::new(layer_id, vec![]), self.get_num_vars()),
        );
        fsc_layer
    }
}

#[cfg(test)]
mod test {
    use remainder_shared_types::Fr;

    use crate::{
        layouter::builder::{CircuitBuilder, LayerVisibility},
        mle::evals::MultilinearExtension,
        prover::helpers::test_circuit_with_runtime_optimized_config,
    };

    #[test]
    fn test_verifier_challenge_node_in_circuit() {
        let mut builder = CircuitBuilder::<Fr>::new();

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

        let verifier_challenge_node = builder.add_fiat_shamir_challenge_node(8);

        let input_layer = builder.add_input_layer(LayerVisibility::Public);
        let input_a = builder.add_input_shred("input a", input_a_data.num_vars(), &input_layer);

        let product_sector = builder.add_sector(input_a - verifier_challenge_node);

        let difference_sector = builder.add_sector(&product_sector - &product_sector);
        builder.set_output(&difference_sector);

        let mut circuit = builder.build().unwrap();
        circuit.set_input("input a", input_a_data);

        let provable_circuit = circuit.finalize().unwrap();

        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }
}
