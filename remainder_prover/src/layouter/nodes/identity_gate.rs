//! A Module for adding `Gate` Layers to components

use ark_std::log2;
use remainder_shared_types::Field;

use crate::{
    layer::{
        identity_gate::IdentityGateLayerDescription, layer_enum::LayerDescriptionEnum, LayerId,
    },
    layouter::layouting::{CircuitDescriptionMap, CircuitLocation, DAGError},
    mle::mle_description::MleDescription,
    utils::mle::get_total_mle_indices,
};

use super::{CircuitNode, CompilableNode, Context, NodeId};

/// A Node that represents a `Gate` layer
#[derive(Clone, Debug)]
pub struct IdentityGateNode {
    id: NodeId,
    nonzero_gates: Vec<(usize, usize)>,
    pre_routed_data: NodeId,
    num_vars: usize,
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
    /// Constructs a new IdentityGateNode and computes the data it generates
    pub fn new(
        ctx: &Context,
        pre_routed_data: &impl CircuitNode,
        nonzero_gates: Vec<(usize, usize)>,
    ) -> Self {
        let max_gate_val = nonzero_gates
            .clone()
            .into_iter()
            .fold(0, |acc, (z, _)| std::cmp::max(acc, z));

        let remap_table_len = (max_gate_val + 1).next_power_of_two();
        let num_vars = log2(remap_table_len) as usize;

        Self {
            id: ctx.get_new_id(),
            nonzero_gates,
            pre_routed_data: pre_routed_data.id(),
            num_vars,
        }
    }
}

impl<F: Field> CompilableNode<F> for IdentityGateNode {
    fn generate_circuit_description(
        &self,
        layer_id: &mut LayerId,
        circuit_description_map: &mut CircuitDescriptionMap,
    ) -> Result<Vec<LayerDescriptionEnum<F>>, DAGError> {
        let (pre_routed_data_location, pre_routed_num_vars) = circuit_description_map
            .0
            .get(&self.pre_routed_data)
            .ok_or(DAGError::DanglingNodeId(self.pre_routed_data))?;
        let total_mle_indices =
            get_total_mle_indices(&pre_routed_data_location.prefix_bits, *pre_routed_num_vars);
        let pre_routed_mle =
            MleDescription::new(pre_routed_data_location.layer_id, &total_mle_indices);

        let id_gate_layer_id = layer_id.get_and_inc();
        let id_gate_layer = IdentityGateLayerDescription::new(
            id_gate_layer_id,
            self.nonzero_gates.clone(),
            pre_routed_mle,
        );
        circuit_description_map.0.insert(
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

    use std::collections::HashMap;

    use ark_std::test_rng;
    use rand::Rng;
    use remainder_shared_types::{Field, Fr};

    use crate::{
        layer::LayerId,
        layouter::nodes::{
            circuit_inputs::{InputLayerNode, InputShred},
            circuit_outputs::OutputNode,
            identity_gate::IdentityGateNode,
            node_enum::NodeEnum,
            sector::Sector,
            CircuitNode, Context, NodeId,
        },
        mle::evals::MultilinearExtension,
        prover::{generate_circuit_description, helpers::test_circuit_new, GKRCircuitDescription},
    };

    /// Struct which allows for easy "semantic" feeding of inputs into the
    /// test identity gate circuit.
    struct IdentityGateTestInputs<F: Field> {
        mle: MultilinearExtension<F>,
        shifted_mle: MultilinearExtension<F>,
    }

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving for the identity gate circuit.
    fn build_identity_gate_test_circuit_description<F: Field>(
        mle_and_shifted_mle_num_vars: usize,
    ) -> (
        GKRCircuitDescription<F>,
        impl Fn(IdentityGateTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
    ) {
        // --- Create global context manager ---
        let context = Context::new();

        // --- Nonzero gates ---
        let mut nonzero_gates = vec![];
        (1..(1 << mle_and_shifted_mle_num_vars)).for_each(|idx| {
            nonzero_gates.push((idx, idx - 1));
        });

        // --- All inputs are public inputs ---
        let public_input_layer_node = InputLayerNode::new(&context, None);

        // --- Inputs to the circuit include the "primary MLE" and the "shifted MLE" ---
        let mle_shred = InputShred::new(
            &context,
            mle_and_shifted_mle_num_vars,
            &public_input_layer_node,
        );
        let shifted_mle_shred = InputShred::new(
            &context,
            mle_and_shifted_mle_num_vars,
            &public_input_layer_node,
        );

        // --- Save IDs to be used later ---
        let mle_shred_id = mle_shred.id();
        let shifted_mle_shred_id = shifted_mle_shred.id();

        // --- Create the circuit components ---
        let gate_sector = IdentityGateNode::new(&context, &mle_shred, nonzero_gates);
        let diff_sector = Sector::new(
            &context,
            &[&gate_sector, &shifted_mle_shred],
            |input_nodes| {
                assert_eq!(input_nodes.len(), 2);
                let mle_1_id = input_nodes[0];
                let mle_2_id = input_nodes[1];

                mle_1_id.expr() - mle_2_id.expr()
            },
        );

        let output = OutputNode::new_zero(&context, &diff_sector);

        // --- Generate the circuit description ---
        let all_circuit_nodes: Vec<NodeEnum<F>> = vec![
            public_input_layer_node.into(),
            mle_shred.into(),
            shifted_mle_shred.into(),
            gate_sector.into(),
            diff_sector.into(),
            output.into(),
        ];

        let (circuit_description, convert_input_shreds_to_input_layers, _) =
            generate_circuit_description(all_circuit_nodes).unwrap();

        // --- Write closure which allows easy usage of circuit inputs ---
        let circuit_data_fn = move |identity_gate_test_inputs: IdentityGateTestInputs<F>| {
            let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
                (mle_shred_id, identity_gate_test_inputs.mle),
                (shifted_mle_shred_id, identity_gate_test_inputs.shifted_mle),
            ]
            .into_iter()
            .collect();
            convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
        };

        (circuit_description, circuit_data_fn)
    }

    #[test]
    fn test_identity_gate_node_in_circuit() {
        const NUM_FREE_BITS: usize = 1;
        let size = 1 << NUM_FREE_BITS;

        let mut rng = test_rng();

        // --- Define all the input (data) to the circuit ---
        let mle_vec: Vec<Fr> = (0..size).map(|_| Fr::from(rng.gen::<u64>())).collect();
        let mle = MultilinearExtension::new(mle_vec.clone());
        let shifted_mle_vec = std::iter::once(Fr::zero())
            .chain(mle_vec.into_iter().take(size - 1))
            .collect();
        let shifted_mle = MultilinearExtension::new(shifted_mle_vec);

        // --- Create circuit description + input helper function ---
        let (identity_gate_test_circuit_desc, input_helper_fn) =
            build_identity_gate_test_circuit_description(NUM_FREE_BITS);

        // --- Convert input data into circuit inputs which are assignable by prover ---
        let circuit_inputs = input_helper_fn(IdentityGateTestInputs { mle, shifted_mle });

        // --- Specify private input layers (+ description and precommit), if any ---
        let private_input_layers = HashMap::new();

        // --- Prove/verify the circuit ---
        test_circuit_new(
            &identity_gate_test_circuit_desc,
            private_input_layers,
            &circuit_inputs,
        );
    }
}
