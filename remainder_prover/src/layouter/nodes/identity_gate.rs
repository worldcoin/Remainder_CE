//! A Module for adding `Gate` Layers to components

use ark_std::log2;
use remainder_shared_types::FieldExt;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layer::identity_gate::IdentityGate,
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
pub struct IdentityGateNode<F: FieldExt> {
    id: NodeId,
    nonzero_gates: Vec<(usize, usize)>,
    pre_routed_data: NodeId,
    data: MultilinearExtension<F>,
}

impl<F: FieldExt> CircuitNode for IdentityGateNode<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.pre_routed_data]
    }
}

impl<F: FieldExt> IdentityGateNode<F> {
    /// Constructs a new IdentityGateNode and computes the data it generates
    pub fn new(
        ctx: &Context,
        pre_routed_data: &impl ClaimableNode<F = F>,
        nonzero_gates: Vec<(usize, usize)>,
    ) -> Self {
        let max_gate_val = nonzero_gates
            .clone()
            .into_iter()
            .fold(0, |acc, (z, _)| std::cmp::max(acc, z));

        let mut remap_table = vec![F::ZERO; max_gate_val + 1];
        nonzero_gates.iter().for_each(|(z, x)| {
            let id_val = *pre_routed_data
                .get_data()
                .get_evals()
                .get(*x)
                .unwrap_or(&F::ZERO);
            remap_table[*z] = id_val;
        });
        let num_vars = log2(remap_table.len()) as usize;

        let data = MultilinearExtension::new_from_evals(Evaluations::new(num_vars, remap_table));

        Self {
            id: ctx.get_new_id(),
            nonzero_gates,
            pre_routed_data: pre_routed_data.id(),
            data,
        }
    }
}

impl<F: FieldExt> ClaimableNode for IdentityGateNode<F> {
    type F = F;

    fn get_data(&self) -> &MultilinearExtension<Self::F> {
        &self.data
    }

    fn get_expr(&self) -> Expression<Self::F, AbstractExpr> {
        Expression::<F, AbstractExpr>::mle(self.id)
    }
}

impl<F: FieldExt, Pf: ProofSystem<F, Layer = L>, L: From<IdentityGate<F>>> CompilableNode<F, Pf>
    for IdentityGateNode<F>
{
    fn compile<'a>(
        &'a self,
        witness_builder: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
        circuit_map: &mut crate::layouter::layouting::CircuitMap<'a, F>,
    ) -> Result<(), crate::layouter::layouting::DAGError> {
        let (pre_routed_data_location, pre_routed_data) = circuit_map
            .0
            .get(&self.pre_routed_data)
            .ok_or(DAGError::DanglingNodeId(self.pre_routed_data))?;
        let pre_routed_mle = DenseMle::new_with_prefix_bits(
            (*pre_routed_data).clone(),
            pre_routed_data_location.layer_id,
            pre_routed_data_location.prefix_bits.clone(),
        );

        let layer_id = witness_builder.next_layer();
        let id_gate_layer = IdentityGate::new(layer_id, self.nonzero_gates.clone(), pre_routed_mle);
        witness_builder.add_layer(id_gate_layer.into());
        circuit_map.0.insert(
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
            const NUM_ITERATED_BITS: usize = 4;

            let mut rng = test_rng();
            let size = 1 << NUM_ITERATED_BITS;

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
            let input_shred_pre_routed = InputShred::new(ctx, mle, &input_layer);
            let input_shred_expected = InputShred::new(ctx, shifted_mle, &input_layer);

            let gate_sector = IdentityGateNode::new(ctx, &input_shred_pre_routed, nonzero_gates);
            let diff_sector = Sector::new(
                ctx,
                &[&gate_sector, &input_shred_expected],
                |input_nodes| {
                    assert_eq!(input_nodes.len(), 2);
                    let mle_1_id = input_nodes[0];
                    let mle_2_id = input_nodes[1];

                    mle_1_id.expr() - mle_2_id.expr()
                },
                |data| {
                    let mle_1_data = data[0];
                    MultilinearExtension::new_sized_zero(mle_1_data.num_vars())
                },
            );

            let output = OutputNode::new_zero(ctx, &diff_sector);

            ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                input_layer.into(),
                input_shred_pre_routed.into(),
                input_shred_expected.into(),
                gate_sector.into(),
                diff_sector.into(),
                output.into(),
            ])
        });

        test_circuit(circuit, None);
    }
}
