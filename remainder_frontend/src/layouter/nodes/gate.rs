//! A Module for adding `Gate` Layers to components

use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder_shared_types::Field;

use remainder::{
    circuit_layout::CircuitLocation,
    layer::{
        gate::{BinaryOperation, GateLayerDescription},
        layer_enum::LayerDescriptionEnum,
        LayerId,
    },
    mle::{mle_description::MleDescription, MleIndex},
};

use crate::layouter::builder::CircuitMap;

use super::{CircuitNode, CompilableNode, NodeId};

use anyhow::Result;

/// A Node that represents a `Gate` layer
#[derive(Clone, Debug)]
pub struct GateNode {
    id: NodeId,
    num_dataparallel_bits: Option<usize>,
    nonzero_gates: Vec<(u32, u32, u32)>,
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
        lhs: &dyn CircuitNode,
        rhs: &dyn CircuitNode,
        nonzero_gates: Vec<(u32, u32, u32)>,
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

        let num_vars = log2(res_table_num_entries as usize) as usize;

        Self {
            id: NodeId::new(),
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
        circuit_map: &mut CircuitMap,
    ) -> Result<Vec<LayerDescriptionEnum<F>>> {
        let (lhs_location, lhs_num_vars) =
            circuit_map.get_location_num_vars_from_node_id(&self.lhs)?;
        let total_indices = lhs_location
            .prefix_bits
            .iter()
            .map(|bit| MleIndex::Fixed(*bit))
            .chain(repeat_n(MleIndex::Free, *lhs_num_vars))
            .collect_vec();
        let lhs_circuit_mle = MleDescription::new(lhs_location.layer_id, &total_indices);

        let (rhs_location, rhs_num_vars) =
            circuit_map.get_location_num_vars_from_node_id(&self.rhs)?;
        let total_indices = rhs_location
            .prefix_bits
            .iter()
            .map(|bit| MleIndex::Fixed(*bit))
            .chain(repeat_n(MleIndex::Free, *rhs_num_vars))
            .collect_vec();
        let rhs_circuit_mle = MleDescription::new(rhs_location.layer_id, &total_indices);

        let gate_layer_id = LayerId::next_layer_id();
        let gate_circuit_description = GateLayerDescription::new(
            self.num_dataparallel_bits,
            self.nonzero_gates.clone(),
            lhs_circuit_mle,
            rhs_circuit_mle,
            gate_layer_id,
            self.gate_operation,
        );
        circuit_map.add_node_id_and_location_num_vars(
            self.id,
            (
                CircuitLocation::new(gate_layer_id, vec![]),
                self.get_num_vars(),
            ),
        );

        Ok(vec![LayerDescriptionEnum::Gate(gate_circuit_description)])
    }
}

#[cfg(test)]
mod test {

    use ark_std::{rand::Rng, test_rng};
    use itertools::Itertools;
    use remainder_shared_types::Fr;

    use crate::layouter::builder::{CircuitBuilder, LayerVisibility};
    use remainder::{
        layer::gate::BinaryOperation, mle::evals::MultilinearExtension,
        prover::helpers::test_circuit_with_runtime_optimized_config,
    };

    #[test]
    fn test_gate_node_in_circuit() {
        let mut builder = CircuitBuilder::<Fr>::new();

        const NUM_FREE_VARS: usize = 4;

        let mut rng = test_rng();
        let size = 1 << NUM_FREE_VARS;

        let mle =
            MultilinearExtension::new((0..size).map(|_| Fr::from(rng.gen::<u64>())).collect());

        let neg_mle = MultilinearExtension::new(mle.iter().map(|elem| -elem).collect_vec());

        let mut nonzero_gates = vec![];

        (0..size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let input_layer = builder.add_input_layer(LayerVisibility::Public);

        let input_shred_pos =
            builder.add_input_shred("Positive Input", NUM_FREE_VARS, &input_layer);

        let input_shred_neg =
            builder.add_input_shred("Negative Input", NUM_FREE_VARS, &input_layer);

        let gate_sector = builder.add_gate_node(
            &input_shred_pos,
            &input_shred_neg,
            nonzero_gates,
            BinaryOperation::Add,
            None,
        );

        builder.set_output(&gate_sector);

        let mut circuit = builder.build().unwrap();

        circuit.set_input("Positive Input", mle);
        circuit.set_input("Negative Input", neg_mle);

        let provable_circuit = circuit.finalize().unwrap();

        test_circuit_with_runtime_optimized_config::<Fr, Fr>(&provable_circuit);
    }

    #[test]
    fn test_data_parallel_gate_node_in_circuit() {
        let mut builder = CircuitBuilder::<Fr>::new();

        const NUM_DATAPARALLEL_VARS: usize = 3;
        const NUM_FREE_VARS: usize = 4;

        let mut rng = test_rng();
        let size = 1 << (NUM_DATAPARALLEL_VARS + NUM_FREE_VARS);

        let mle =
            MultilinearExtension::new((0..size).map(|_| Fr::from(rng.gen::<u64>())).collect());

        let neg_mle = MultilinearExtension::new(mle.iter().map(|elem| -elem).collect_vec());

        let mut nonzero_gates = vec![];
        let table_size = 1 << NUM_FREE_VARS;

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let input_layer = builder.add_input_layer(LayerVisibility::Public);

        let input_shred_pos = builder.add_input_shred(
            "Positive Input",
            NUM_DATAPARALLEL_VARS + NUM_FREE_VARS,
            &input_layer,
        );

        let input_shred_neg = builder.add_input_shred(
            "Negative Input",
            NUM_DATAPARALLEL_VARS + NUM_FREE_VARS,
            &input_layer,
        );

        let gate_sector = builder.add_gate_node(
            &input_shred_pos,
            &input_shred_neg,
            nonzero_gates,
            BinaryOperation::Add,
            Some(NUM_DATAPARALLEL_VARS),
        );

        let _output = builder.set_output(&gate_sector);

        let mut circuit = builder.build().unwrap();

        circuit.set_input("Positive Input", mle);
        circuit.set_input("Negative Input", neg_mle);

        let provable_circuit = circuit.finalize().unwrap();

        test_circuit_with_runtime_optimized_config::<Fr, Fr>(&provable_circuit);
    }
}
