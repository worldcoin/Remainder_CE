//! A Module for adding `Gate` Layers to components

use ark_std::log2;
use remainder_shared_types::FieldExt;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layer::gate::{BinaryOperation, Gate},
    layouter::layouting::CircuitLocation,
    mle::{
        dense::DenseMle,
        evals::{Evaluations, MultilinearExtension},
    },
    prover::proof_system::ProofSystem,
};

use super::{CircuitNode, ClaimableNode, CompilableNode, Context, NodeId};

/// A Node that represents a `Gate` layer
#[derive(Clone, Debug)]
pub struct GateNode<F: FieldExt> {
    id: NodeId,
    num_dataparallel_bits: Option<usize>,
    nonzero_gates: Vec<(usize, usize, usize)>,
    lhs: NodeId,
    rhs: NodeId,
    gate_operation: BinaryOperation,
    data: MultilinearExtension<F>,
}

impl<F: FieldExt> CircuitNode for GateNode<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.lhs, self.rhs]
    }
}

impl<F: FieldExt> GateNode<F> {
    /// Constructs a new GateNode and computes the data it generates
    pub fn new(
        ctx: &Context,
        lhs: &dyn ClaimableNode<F = F>,
        rhs: &dyn ClaimableNode<F = F>,
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

        let mut res_table = vec![F::ZERO; res_table_num_entries];
        (0..num_dataparallel_vals).for_each(|idx| {
            nonzero_gates
                .clone()
                .into_iter()
                .for_each(|(z_ind, x_ind, y_ind)| {
                    let f2_val = *lhs
                        .get_data()
                        .get_evals()
                        .get(idx + (x_ind * num_dataparallel_vals))
                        .unwrap_or(&F::ZERO);
                    let f3_val = *rhs
                        .get_data()
                        .get_evals()
                        .get(idx + (y_ind * num_dataparallel_vals))
                        .unwrap_or(&F::ZERO);
                    res_table[idx + (z_ind * num_dataparallel_vals)] =
                        gate_operation.perform_operation(f2_val, f3_val);
                });
        });

        let num_vars = log2(res_table.len()) as usize;

        let data = MultilinearExtension::new_from_evals(Evaluations::new(num_vars, res_table));

        Self {
            id: ctx.get_new_id(),
            num_dataparallel_bits,
            nonzero_gates,
            gate_operation,
            lhs: lhs.id(),
            rhs: rhs.id(),
            data,
        }
    }
}

impl<F: FieldExt> ClaimableNode for GateNode<F> {
    type F = F;

    fn get_data(&self) -> &MultilinearExtension<Self::F> {
        &self.data
    }

    fn get_expr(&self) -> Expression<Self::F, AbstractExpr> {
        Expression::<F, AbstractExpr>::mle(self.id)
    }
}

impl<F: FieldExt, Pf: ProofSystem<F, Layer = L>, L: From<Gate<F>>> CompilableNode<F, Pf>
    for GateNode<F>
{
    fn compile<'a>(
        &'a self,
        witness_builder: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
        circuit_map: &mut crate::layouter::layouting::CircuitMap<'a, F>,
    ) -> Result<(), crate::layouter::layouting::DAGError> {
        let (lhs_location, lhs_data) = circuit_map.get_node(&self.lhs)?;
        let lhs = DenseMle::new_with_prefix_bits(
            (*lhs_data).clone(),
            lhs_location.layer_id,
            lhs_location.prefix_bits.clone(),
        );
        let (rhs_location, rhs_data) = circuit_map.get_node(&self.rhs)?;
        let rhs = DenseMle::new_with_prefix_bits(
            (*rhs_data).clone(),
            rhs_location.layer_id,
            rhs_location.prefix_bits.clone(),
        );
        let layer_id = witness_builder.next_layer();
        let gate_layer = Gate::new(
            self.num_dataparallel_bits,
            self.nonzero_gates.clone(),
            lhs,
            rhs,
            self.gate_operation,
            layer_id,
        );
        witness_builder.add_layer(gate_layer.into());
        circuit_map.add_node(
            self.id,
            (CircuitLocation::new(layer_id, vec![]), &self.data),
        );

        Ok(())
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
                circuit_inputs::{InputLayerNode, InputLayerType, InputShred},
                circuit_outputs::OutputNode,
                gate::GateNode,
                node_enum::NodeEnum,
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

            let input_shred_pos = InputShred::new(ctx, mle, &input_layer);
            let input_shred_neg = InputShred::new(ctx, neg_mle, &input_layer);

            let gate_sector = GateNode::new(
                ctx,
                &input_shred_pos,
                &input_shred_neg,
                nonzero_gates,
                BinaryOperation::Add,
                None,
            );

            let output = OutputNode::new_zero(ctx, &gate_sector);

            ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                input_layer.into(),
                input_shred_pos.into(),
                input_shred_neg.into(),
                gate_sector.into(),
                output.into(),
            ])
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

            let input_shred_pos = InputShred::new(ctx, mle, &input_layer);
            let input_shred_neg = InputShred::new(ctx, neg_mle, &input_layer);

            let gate_sector = GateNode::new(
                ctx,
                &input_shred_pos,
                &input_shred_neg,
                nonzero_gates,
                BinaryOperation::Add,
                Some(NUM_DATAPARALLEL_BITS),
            );

            let output = OutputNode::new_zero(ctx, &gate_sector);

            ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                input_layer.into(),
                input_shred_pos.into(),
                input_shred_neg.into(),
                gate_sector.into(),
                output.into(),
            ])
        });

        test_circuit(circuit, None);
    }
}
