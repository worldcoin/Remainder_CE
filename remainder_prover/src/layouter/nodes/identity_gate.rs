//! A Module for adding `Gate` Layers to components

use ark_std::log2;
use remainder_shared_types::Field;

use crate::{
    expression::circuit_expr::MleDescription,
    layer::{
        identity_gate::IdentityGateLayerDescription, layer_enum::LayerDescriptionEnum, LayerId,
    },
    layouter::layouting::{CircuitDescriptionMap, CircuitLocation, DAGError},
    utils::mle::get_total_mle_indices,
};

use super::{CircuitNode, CompilableNode, Context, NodeId};

/// A Node that represents a `Gate` layer
#[derive(Clone, Debug)]
pub struct IdentityGateNode {
    id: NodeId,
    num_dataparallel_bits: Option<usize>,
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
        num_dataparallel_bits: Option<usize>,
    ) -> Self {
        let max_gate_val = nonzero_gates
            .clone()
            .into_iter()
            .fold(0, |acc, (z, _)| std::cmp::max(acc, z));

        // number of entries in the resulting table is the max gate z value * 2 to the power of the number of dataparallel bits, as we are
        // evaluating over all values in the boolean hypercube which includes dataparallel bits
        let num_dataparallel_vals = 1 << (num_dataparallel_bits.unwrap_or(0));
        let remap_table_len = (max_gate_val + 1).next_power_of_two() * num_dataparallel_vals;

        let num_vars = log2(remap_table_len) as usize;

        Self {
            id: ctx.get_new_id(),
            num_dataparallel_bits,
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
            self.num_dataparallel_bits,
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

    use ark_std::test_rng;
    use rand::Rng;
    use remainder_shared_types::Fr;

    use crate::{
        layouter::{
            compiling::LayouterCircuit,
            component::ComponentSet,
            nodes::{
                circuit_inputs::{
                    InputLayerData, InputLayerNode, InputLayerType, InputShred, InputShredData,
                },
                circuit_outputs::OutputNode,
                identity_gate::IdentityGateNode,
                node_enum::NodeEnum,
                sector::Sector,
                CircuitNode,
            },
        },
        mle::evals::MultilinearExtension,
        prover::helpers::test_circuit,
    };

    #[test]
    fn test_identity_gate_node_in_circuit() {
        let circuit = LayouterCircuit::new(|ctx| {
            const NUM_FREE_BITS: usize = 4;

            let mut rng = test_rng();
            let size = 1 << NUM_FREE_BITS;

            let mle_vec: Vec<Fr> = (0..size).map(|_| Fr::from(rng.gen::<u64>())).collect();
            let mle = MultilinearExtension::new(mle_vec.clone());
            let shifted_mle_vec = std::iter::once(Fr::zero())
                .chain(mle_vec.into_iter().take(size - 1))
                .collect();
            let shifted_mle = MultilinearExtension::new(shifted_mle_vec);

            let mut nonzero_gates = vec![];

            (1..size).for_each(|idx| {
                nonzero_gates.push((idx, idx - 1));
            });

            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let input_shred_pre_routed = InputShred::new(ctx, mle.num_vars(), &input_layer);
            let input_shred_expected = InputShred::new(ctx, shifted_mle.num_vars(), &input_layer);
            let input_shred_pre_routed_data = InputShredData::new(input_shred_pre_routed.id(), mle);
            let input_shred_expected_data =
                InputShredData::new(input_shred_expected.id(), shifted_mle);
            let input_data = InputLayerData::new(
                input_layer.id(),
                vec![input_shred_pre_routed_data, input_shred_expected_data],
                None,
            );

            let gate_sector =
                IdentityGateNode::new(ctx, &input_shred_pre_routed, nonzero_gates, None);
            let diff_sector =
                Sector::new(ctx, &[&gate_sector, &input_shred_expected], |input_nodes| {
                    assert_eq!(input_nodes.len(), 2);
                    let mle_1_id = input_nodes[0];
                    let mle_2_id = input_nodes[1];

                    mle_1_id.expr() - mle_2_id.expr()
                });

            let output = OutputNode::new_zero(ctx, &diff_sector);

            (
                ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                    input_layer.into(),
                    input_shred_pre_routed.into(),
                    input_shred_expected.into(),
                    gate_sector.into(),
                    diff_sector.into(),
                    output.into(),
                ]),
                vec![input_data],
            )
        });

        test_circuit(circuit, None);
    }
}
