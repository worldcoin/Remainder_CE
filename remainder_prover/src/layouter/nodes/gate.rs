//! A Module for adding `Gate` Layers to components

use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder_shared_types::Field;

use crate::{
    expression::circuit_expr::CircuitMle,
    layer::{
        gate::{BinaryOperation, CircuitGateLayer},
        layer_enum::CircuitLayerEnum,
        LayerId,
    },
    layouter::layouting::{CircuitLocation, DAGError},
    mle::MleIndex,
};

use super::{CircuitNode, CompilableNode, Context, NodeId};

/// A Node that represents a `Gate` layer
#[derive(Clone, Debug)]
pub struct GateNode {
    id: NodeId,
    num_dataparallel_bits: Option<usize>,
    nonzero_gates: Vec<(usize, usize, usize)>,
    lhs: NodeId,
    rhs: NodeId,
    gate_operation: BinaryOperation,
    num_vars: usize,
}

impl CircuitNode for GateNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.lhs, self.rhs]
    }

    fn get_num_vars(&self) -> usize {
        self.num_vars
    }
}

impl GateNode {
    /// Constructs a new GateNode and computes the data it generates
    pub fn new(
        ctx: &Context,
        lhs: &dyn CircuitNode,
        rhs: &dyn CircuitNode,
        nonzero_gates: Vec<(usize, usize, usize)>,
        gate_operation: BinaryOperation,
        num_dataparallel_bits: Option<usize>,
    ) -> Self {
        let max_gate_val = nonzero_gates
            .clone()
            .into_iter()
            .fold(0, |acc, (z, _, _)| std::cmp::max(acc, z));

        // number of entries in the resulting table is the max gate z value * 2 to the power of the number of dataparallel bits, as we are
        // evaluating over all values in the boolean hypercube which includes dataparallel bits
        let num_dataparallel_vals = 1 << (num_dataparallel_bits.unwrap_or(0));
        let res_table_num_entries = (max_gate_val + 1) * num_dataparallel_vals;

        let num_vars = log2(res_table_num_entries) as usize;

        Self {
            id: ctx.get_new_id(),
            num_dataparallel_bits,
            nonzero_gates,
            gate_operation,
            lhs: lhs.id(),
            rhs: rhs.id(),
            num_vars,
        }
    }
}

impl<F: Field> CompilableNode<F> for GateNode {
    fn generate_circuit_description(
        &self,
        layer_id: &mut LayerId,
        circuit_description_map: &mut crate::layouter::layouting::CircuitDescriptionMap,
    ) -> Result<Vec<CircuitLayerEnum<F>>, DAGError> {
        let (lhs_location, lhs_num_vars) =
            circuit_description_map.get_location_num_vars_from_node_id(&self.lhs)?;
        let total_indices = lhs_location
            .prefix_bits
            .iter()
            .map(|bit| MleIndex::Fixed(*bit))
            .chain(repeat_n(MleIndex::Iterated, *lhs_num_vars))
            .collect_vec();
        let lhs_circuit_mle = CircuitMle::new(lhs_location.layer_id, &total_indices);

        let (rhs_location, rhs_num_vars) =
            circuit_description_map.get_location_num_vars_from_node_id(&self.rhs)?;
        let total_indices = rhs_location
            .prefix_bits
            .iter()
            .map(|bit| MleIndex::Fixed(*bit))
            .chain(repeat_n(MleIndex::Iterated, *rhs_num_vars))
            .collect_vec();
        let rhs_circuit_mle = CircuitMle::new(rhs_location.layer_id, &total_indices);

        let gate_layer_id = layer_id.get_and_inc();
        let gate_circuit_description = CircuitGateLayer::new(
            self.num_dataparallel_bits,
            self.nonzero_gates.clone(),
            lhs_circuit_mle,
            rhs_circuit_mle,
            gate_layer_id,
            self.gate_operation,
        );
        circuit_description_map.add_node_id_and_location_num_vars(
            self.id,
            (
                CircuitLocation::new(gate_layer_id, vec![]),
                self.get_num_vars(),
            ),
        );

        Ok(vec![CircuitLayerEnum::Gate(gate_circuit_description)])
    }
}

#[cfg(test)]
mod test {

    use ark_std::test_rng;
    use itertools::Itertools;
    use rand::Rng;
    use remainder_shared_types::Fr;

    use crate::{
        layer::gate::BinaryOperation,
        layouter::{
            compiling::LayouterCircuit,
            component::ComponentSet,
            nodes::{
                circuit_inputs::{
                    InputLayerData, InputLayerNode, InputLayerType, InputShred, InputShredData,
                },
                circuit_outputs::OutputNode,
                gate::GateNode,
                node_enum::NodeEnum,
                CircuitNode,
            },
        },
        mle::evals::MultilinearExtension,
        prover::helpers::test_circuit,
    };

    #[test]
    fn test_gate_node_in_circuit() {
        let circuit = LayouterCircuit::new(|ctx| {
            const NUM_ITERATED_BITS: usize = 4;

            let mut rng = test_rng();
            let size = 1 << NUM_ITERATED_BITS;

            let mle =
                MultilinearExtension::new((0..size).map(|_| Fr::from(rng.gen::<u64>())).collect());

            let neg_mle = MultilinearExtension::new(
                mle.get_evals_vector()
                    .clone()
                    .into_iter()
                    .map(|elem| -elem)
                    .collect_vec(),
            );

            let mut nonzero_gates = vec![];

            (0..size).for_each(|idx| {
                nonzero_gates.push((idx, idx, idx));
            });

            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);

            let input_shred_pos = InputShred::new(ctx, mle.num_vars(), &input_layer);
            let input_shred_pos_data = InputShredData::new(input_shred_pos.id(), mle);
            let input_shred_neg = InputShred::new(ctx, neg_mle.num_vars(), &input_layer);
            let input_shred_neg_data = InputShredData::new(input_shred_neg.id(), neg_mle);
            let input_data = InputLayerData::new(
                input_layer.id(),
                vec![input_shred_pos_data, input_shred_neg_data],
                None,
            );

            let gate_sector = GateNode::new(
                ctx,
                &input_shred_pos,
                &input_shred_neg,
                nonzero_gates,
                BinaryOperation::Add,
                None,
            );

            let output = OutputNode::new_zero(ctx, &gate_sector);

            (
                ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                    input_layer.into(),
                    input_shred_pos.into(),
                    input_shred_neg.into(),
                    gate_sector.into(),
                    output.into(),
                ]),
                vec![input_data],
            )
        });

        test_circuit(circuit, None);
    }

    #[test]
    fn test_data_parallel_gate_node_in_circuit() {
        let circuit = LayouterCircuit::new(|ctx| {
            const NUM_DATAPARALLEL_BITS: usize = 3;
            const NUM_ITERATED_BITS: usize = 4;

            let mut rng = test_rng();
            let size = 1 << (NUM_DATAPARALLEL_BITS + NUM_ITERATED_BITS);

            let mle =
                MultilinearExtension::new((0..size).map(|_| Fr::from(rng.gen::<u64>())).collect());

            let neg_mle = MultilinearExtension::new(
                mle.get_evals_vector()
                    .clone()
                    .into_iter()
                    .map(|elem| -elem)
                    .collect_vec(),
            );

            let mut nonzero_gates = vec![];
            let table_size = 1 << NUM_ITERATED_BITS;

            (0..table_size).for_each(|idx| {
                nonzero_gates.push((idx, idx, idx));
            });

            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);

            let input_shred_pos = InputShred::new(ctx, mle.num_vars(), &input_layer);
            let input_shred_pos_data = InputShredData::new(input_shred_pos.id(), mle);
            let input_shred_neg = InputShred::new(ctx, neg_mle.num_vars(), &input_layer);
            let input_shred_neg_data = InputShredData::new(input_shred_neg.id(), neg_mle);
            let input_data = InputLayerData::new(
                input_layer.id(),
                vec![input_shred_pos_data, input_shred_neg_data],
                None,
            );

            let gate_sector = GateNode::new(
                ctx,
                &input_shred_pos,
                &input_shred_neg,
                nonzero_gates,
                BinaryOperation::Add,
                Some(NUM_DATAPARALLEL_BITS),
            );

            let output = OutputNode::new_zero(ctx, &gate_sector);

            (
                ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                    input_layer.into(),
                    input_shred_pos.into(),
                    input_shred_neg.into(),
                    gate_sector.into(),
                    output.into(),
                ]),
                vec![input_data],
            )
        });

        test_circuit(circuit, None);
    }
}
