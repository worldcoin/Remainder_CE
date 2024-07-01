//! A Module for adding `Gate` Layers to components

use ark_std::log2;
use remainder_shared_types::FieldExt;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layer::gate::{BinaryOperation, Gate},
    layouter::layouting::{CircuitLocation, DAGError},
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
        lhs: impl ClaimableNode<F = F>,
        rhs: impl ClaimableNode<F = F>,
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

        let data = MultilinearExtension::new(Evaluations::new(num_vars, res_table));

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
        let (lhs_location, lhs_data) = circuit_map
            .0
            .get(&self.lhs)
            .ok_or(DAGError::DanglingNodeId(self.lhs))?;
        let lhs = DenseMle::new_with_prefix_bits(
            (*lhs_data).clone(),
            lhs_location.layer_id,
            lhs_location.prefix_bits.clone(),
        );
        let (rhs_location, rhs_data) = circuit_map
            .0
            .get(&self.lhs)
            .ok_or(DAGError::DanglingNodeId(self.lhs))?;
        let rhs = DenseMle::new_with_prefix_bits(
            (*rhs_data).clone(),
            rhs_location.layer_id,
            rhs_location.prefix_bits.clone(),
        );
        let layer_id = witness_builder.next_layer();
        let gate_layer = Gate::new(
            self.num_dataparallel_bits.clone(),
            self.nonzero_gates.clone(),
            lhs,
            rhs,
            self.gate_operation,
            layer_id,
        );
        witness_builder.add_layer(gate_layer.into());
        circuit_map.0.insert(
            self.id,
            (CircuitLocation::new(layer_id, vec![]), &self.data),
        );

        Ok(())
    }
}
