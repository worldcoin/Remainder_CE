//! A module providing identity gate functionality. For rerouting values from one layer to another
//! in an arbitrary fashion.

use remainder_shared_types::Field;

use crate::{
    layer::{
        identity_gate::IdentityGateLayerDescription, layer_enum::LayerDescriptionEnum, LayerId,
    },
    layouter::{builder::CircuitMap, layouting::CircuitLocation},
    mle::mle_description::MleDescription,
    utils::mle::get_total_mle_indices,
};

use super::{CircuitNode, CompilableNode, NodeId};

use anyhow::Result;

/// A node that represents an identity gate in the circuit i.e. that wires values unmodified from
/// one layer to another.
#[derive(Clone, Debug)]
pub struct IdentityGateNode {
    id: NodeId,
    num_vars: usize,
    num_dataparallel_vars: Option<usize>,
    nonzero_gates: Vec<(u32, u32)>,
    pre_routed_data: NodeId,
}

impl CircuitNode for IdentityGateNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.pre_routed_data]
    }

    fn get_num_vars(&self) -> usize {
        self.num_vars
    }
}

impl IdentityGateNode {
    /// Constructs a new IdentityGateNode.
    /// Arguments:
    /// * `pre_routed_data`: The Node that is being routed to this layer.
    /// * `nonzero_gates`: A list of tuples representing the gates that are nonzero, in the form `(dest_idx, src_idx)`.
    /// * `num_vars`: The total number of variables in the layer.
    /// * `num_dataparallel_vars`: The number of dataparallel variables to use in this layer.
    pub fn new(
        pre_routed_data: &dyn CircuitNode,
        nonzero_gates: Vec<(u32, u32)>,
        num_vars: usize,
        num_dataparallel_vars: Option<usize>,
    ) -> Self {
        let gate_idx_bound = 1 << (num_vars - num_dataparallel_vars.unwrap_or(0));
        nonzero_gates.iter().for_each(|(dest_idx, _)| {
            assert!(
                *dest_idx < gate_idx_bound,
                "Gate index {dest_idx} too large for layer with {num_vars} variables",
            )
        });
        Self {
            id: NodeId::new(),
            num_vars,
            num_dataparallel_vars,
            nonzero_gates,
            pre_routed_data: pre_routed_data.id(),
        }
    }
}

impl<F: Field> CompilableNode<F> for IdentityGateNode {
    fn generate_circuit_description(
        &self,
        circuit_map: &mut CircuitMap,
    ) -> Result<Vec<LayerDescriptionEnum<F>>> {
        let (pre_routed_data_location, pre_routed_num_vars) =
            circuit_map.get_location_num_vars_from_node_id(&self.pre_routed_data)?;
        let total_mle_indices =
            get_total_mle_indices(&pre_routed_data_location.prefix_bits, *pre_routed_num_vars);
        let pre_routed_mle =
            MleDescription::new(pre_routed_data_location.layer_id, &total_mle_indices);

        let id_gate_layer_id = LayerId::next_layer_id();
        let id_gate_layer = IdentityGateLayerDescription::new(
            id_gate_layer_id,
            self.nonzero_gates.clone(),
            pre_routed_mle,
            self.num_vars,
            self.num_dataparallel_vars,
        );
        circuit_map.add_node_id_and_location_num_vars(
            self.id,
            (
                CircuitLocation::new(id_gate_layer_id, vec![]),
                self.get_num_vars(),
            ),
        );

        Ok(vec![LayerDescriptionEnum::IdentityGate(id_gate_layer)])
    }
}

#[cfg(test)]
mod test {

    use ark_std::test_rng;
    use rand::Rng;
    use remainder_shared_types::{Field, Fr};

    use crate::{
        layouter::builder::{Circuit, CircuitBuilder, LayerVisibility},
        mle::evals::MultilinearExtension,
        prover::helpers::test_circuit_with_runtime_optimized_config,
    };

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving for the identity gate circuit.
    fn build_identity_gate_test_circuit_description<F: Field>(
        mle_and_shifted_mle_num_vars: usize,
    ) -> Circuit<F> {
        let mut builder = CircuitBuilder::<F>::new();

        // Nonzero gates
        let mut nonzero_gates = vec![];
        (1..(1 << mle_and_shifted_mle_num_vars)).for_each(|idx| {
            nonzero_gates.push((idx, idx - 1));
        });

        // All inputs are public inputs
        let public_input_layer_node = builder.add_input_layer(LayerVisibility::Public);

        // Inputs to the circuit include the "primary MLE" and the "shifted MLE"
        let mle_shred = builder.add_input_shred(
            "Input MLE",
            mle_and_shifted_mle_num_vars,
            &public_input_layer_node,
        );
        let shifted_mle_shred = builder.add_input_shred(
            "Shifter Input MLE",
            mle_and_shifted_mle_num_vars,
            &public_input_layer_node,
        );

        // Create the circuit components
        let gate_sector = builder.add_identity_gate_node(
            &mle_shred,
            nonzero_gates,
            mle_and_shifted_mle_num_vars,
            None,
        );
        let diff_sector = builder.add_sector(gate_sector - shifted_mle_shred);

        builder.set_output(&diff_sector);

        builder.build().unwrap()
    }

    #[test]
    fn test_identity_gate_node_in_circuit() {
        const NUM_FREE_BITS: usize = 1;
        let size = 1 << NUM_FREE_BITS;

        let mut rng = test_rng();

        // Define all the input (data) to the circuit
        let mle_vec: Vec<Fr> = (0..size).map(|_| Fr::from(rng.gen::<u64>())).collect();
        let mle = MultilinearExtension::new(mle_vec.clone());
        let shifted_mle_vec = std::iter::once(Fr::zero())
            .chain(mle_vec.into_iter().take(size - 1))
            .collect();
        let shifted_mle = MultilinearExtension::new(shifted_mle_vec);

        // Create circuit description + input helper function
        let mut circuit = build_identity_gate_test_circuit_description(NUM_FREE_BITS);

        circuit.set_input("Input MLE", mle);
        circuit.set_input("Shifter Input MLE", shifted_mle);

        let provable_circuit = circuit.finalize().unwrap();

        // Prove/verify the circuit
        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }
}
