//! A Module for adding `Gate` Layers to components

use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder_shared_types::Field;

use crate::{
    layer::{
        gate::{BinaryOperation, GateLayerDescription},
        layer_enum::LayerDescriptionEnum,
        LayerId,
    },
    layouter::layouting::{CircuitLocation, DAGError},
    mle::{mle_description::MleDescription, MleIndex},
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
    ) -> Result<Vec<LayerDescriptionEnum<F>>, DAGError> {
        let (lhs_location, lhs_num_vars) =
            circuit_description_map.get_location_num_vars_from_node_id(&self.lhs)?;
        let total_indices = lhs_location
            .prefix_bits
            .iter()
            .map(|bit| MleIndex::Fixed(*bit))
            .chain(repeat_n(MleIndex::Free, *lhs_num_vars))
            .collect_vec();
        let lhs_circuit_mle = MleDescription::new(lhs_location.layer_id, &total_indices);

        let (rhs_location, rhs_num_vars) =
            circuit_description_map.get_location_num_vars_from_node_id(&self.rhs)?;
        let total_indices = rhs_location
            .prefix_bits
            .iter()
            .map(|bit| MleIndex::Fixed(*bit))
            .chain(repeat_n(MleIndex::Free, *rhs_num_vars))
            .collect_vec();
        let rhs_circuit_mle = MleDescription::new(rhs_location.layer_id, &total_indices);

        let gate_layer_id = layer_id.get_and_inc();
        let gate_circuit_description = GateLayerDescription::new(
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

        Ok(vec![LayerDescriptionEnum::Gate(gate_circuit_description)])
    }
}

#[cfg(test)]
mod test {

    use std::collections::HashMap;

    use ark_std::test_rng;
    use itertools::Itertools;
    use rand::Rng;
    use remainder_shared_types::Fr;

    use crate::{
        layer::gate::BinaryOperation,
        layouter::nodes::{
            circuit_inputs::{InputLayerNode, InputShred},
            circuit_outputs::OutputNode,
            gate::GateNode,
            CircuitNode, Context, NodeId,
        },
        mle::evals::MultilinearExtension,
        prover::{generate_circuit_description, helpers::test_circuit_new},
    };

    #[test]
    fn test_gate_node_in_circuit() {
        let ctx = &Context::new();
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

        let input_layer = InputLayerNode::new(ctx, None);

        let input_shred_pos = InputShred::new(ctx, NUM_FREE_VARS, &input_layer);
        let input_shred_pos_id = input_shred_pos.id();

        let input_shred_neg = InputShred::new(ctx, NUM_FREE_VARS, &input_layer);
        let input_shred_neg_id = input_shred_neg.id();

        let gate_sector = GateNode::new(
            ctx,
            &input_shred_pos,
            &input_shred_neg,
            nonzero_gates,
            BinaryOperation::Add,
            None,
        );

        let output = OutputNode::new_zero(ctx, &gate_sector);

        let all_nodes = vec![
            input_layer.into(),
            input_shred_pos.into(),
            input_shred_neg.into(),
            gate_sector.into(),
            output.into(),
        ];

        let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
            generate_circuit_description(all_nodes).unwrap();

        let input_builder =
            move |(mle, neg_mle): (MultilinearExtension<Fr>, MultilinearExtension<Fr>)| {
                let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> =
                    HashMap::new();
                input_shred_id_to_data.insert(input_shred_pos_id, mle);
                input_shred_id_to_data.insert(input_shred_neg_id, neg_mle);
                input_builder_from_shred_map(input_shred_id_to_data).unwrap()
            };

        let inputs = input_builder((mle, neg_mle));
        test_circuit_new(&circ_desc, HashMap::new(), &inputs);
    }

    #[test]
    fn test_data_parallel_gate_node_in_circuit() {
        let ctx = &Context::new();
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

        let input_layer = InputLayerNode::new(ctx, None);

        let input_shred_pos =
            InputShred::new(ctx, NUM_DATAPARALLEL_VARS + NUM_FREE_VARS, &input_layer);
        let input_shred_pos_id = input_shred_pos.id();

        let input_shred_neg =
            InputShred::new(ctx, NUM_DATAPARALLEL_VARS + NUM_FREE_VARS, &input_layer);
        let input_shred_neg_id = input_shred_neg.id();

        let gate_sector = GateNode::new(
            ctx,
            &input_shred_pos,
            &input_shred_neg,
            nonzero_gates,
            BinaryOperation::Add,
            Some(NUM_DATAPARALLEL_VARS),
        );

        let output = OutputNode::new_zero(ctx, &gate_sector);

        let all_nodes = vec![
            input_layer.into(),
            input_shred_pos.into(),
            input_shred_neg.into(),
            gate_sector.into(),
            output.into(),
        ];

        let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
            generate_circuit_description(all_nodes).unwrap();

        let input_builder =
            move |(mle, neg_mle): (MultilinearExtension<Fr>, MultilinearExtension<Fr>)| {
                let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> =
                    HashMap::new();
                input_shred_id_to_data.insert(input_shred_pos_id, mle);
                input_shred_id_to_data.insert(input_shred_neg_id, neg_mle);
                input_builder_from_shred_map(input_shred_id_to_data).unwrap()
            };

        let inputs = input_builder((mle, neg_mle));
        test_circuit_new(&circ_desc, HashMap::new(), &inputs);
    }
}
