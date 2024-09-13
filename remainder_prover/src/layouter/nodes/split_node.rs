//! A node that can alter the claims made on it's source `ClaimableNode`

use itertools::{repeat_n, Itertools};
use remainder_shared_types::Field;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layouter::layouting::CircuitLocation,
    mle::evals::{Evaluations, MultilinearExtension},
    prover::proof_system::ProofSystem,
};

use super::{CircuitNode, ClaimableNode, CompilableNode, Context, NodeId};

/// A Node that derives new `ClaimableNode`s from a single
/// `ClaimableNode`.
///
/// The new nodes represent the input node split
/// by a selector bit.
#[derive(Clone, Debug)]
pub struct SplitNode<F: Field> {
    id: NodeId,
    data: MultilinearExtension<F>,
    source: NodeId,
    prefix_bits: Vec<bool>,
}

impl<F: Field> SplitNode<F> {
    /// Creates 2^num_vars `SplitNodes` from a single ClaimableNode
    pub fn new(ctx: &Context, node: &impl ClaimableNode<F = F>, num_vars: usize) -> Vec<Self> {
        let data = node.get_data();
        let source = node.id();
        let step = 1 << num_vars;
        let max_num_vars = data.num_vars() - num_vars;
        (0..(1 << num_vars))
            .zip(bits_iter(num_vars))
            .map(|(idx, prefix_bits)| {
                let data = data
                    .get_evals_vector()
                    .iter()
                    .skip(idx)
                    .step_by(step)
                    .cloned()
                    .collect_vec();

                let data =
                    MultilinearExtension::new_from_evals(Evaluations::new(max_num_vars, data));
                Self {
                    id: ctx.get_new_id(),
                    source,
                    data,
                    prefix_bits,
                }
            })
            .collect()
    }
}

impl<F: Field> CircuitNode for SplitNode<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.source]
    }
}

impl<F: Field> ClaimableNode for SplitNode<F> {
    type F = F;

    fn get_data(&self) -> &MultilinearExtension<Self::F> {
        &self.data
    }

    fn get_expr(
        &self,
    ) -> crate::expression::generic_expr::Expression<
        Self::F,
        crate::expression::abstract_expr::AbstractExpr,
    > {
        Expression::<F, AbstractExpr>::mle(self.id)
    }
}

impl<F: Field, Pf: ProofSystem<F>> CompilableNode<F, Pf> for SplitNode<F> {
    fn compile<'a>(
        &'a self,
        _: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
        circuit_map: &mut crate::layouter::layouting::CircuitMap<'a, F>,
    ) -> Result<(), crate::layouter::layouting::DAGError> {
        let data = &self.data;
        let (source_location, _) = circuit_map.get_node(&self.source)?;

        let prefix_bits = source_location
            .prefix_bits
            .iter()
            .chain(self.prefix_bits.iter())
            .copied()
            .collect();

        let location = CircuitLocation::new(source_location.layer_id, prefix_bits);

        circuit_map.add_node(self.id, (location, data));
        Ok(())
    }
}

///returns an iterator that wil give permutations of binary bits of size
/// num_bits
///
/// 0,0,0 -> 0,0,1 -> 0,1,0 -> 0,1,1 -> 1,0,0 -> 1,0,1 -> 1,1,0 -> 1,1,1
fn bits_iter(num_bits: usize) -> impl Iterator<Item = Vec<bool>> {
    std::iter::successors(Some(vec![false; num_bits]), move |prev| {
        let mut prev = prev.clone();
        let mut removed_bits = 0;
        for index in (0..num_bits).rev() {
            let curr = prev.remove(index);
            if !curr {
                prev.push(true);
                break;
            } else {
                removed_bits += 1;
            }
        }
        if removed_bits == num_bits {
            None
        } else {
            Some(
                prev.into_iter()
                    .chain(repeat_n(false, removed_bits))
                    .collect_vec(),
            )
        }
    })
}

#[cfg(test)]
mod test {

    use itertools::Itertools;
    use remainder_shared_types::Fr;

    use crate::{
        expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
        layouter::{
            compiling::LayouterCircuit,
            component::ComponentSet,
            nodes::{
                circuit_inputs::{InputLayerNode, InputLayerType, InputShred},
                circuit_outputs::OutputNode,
                node_enum::NodeEnum,
                sector::Sector,
            },
        },
        mle::evals::MultilinearExtension,
        prover::helpers::test_circuit,
    };

    use super::SplitNode;

    #[test]
    fn test_split_node_in_circuit() {
        let circuit = LayouterCircuit::new(|ctx| {
            let mle = MultilinearExtension::<Fr>::interlace_mles(vec![
                MultilinearExtension::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]),
                MultilinearExtension::new(vec![Fr::from(5), Fr::from(6), Fr::from(7), Fr::from(8)]),
            ]);
            // the mle_out = [1*5, 2*6, 3*7, 4*8], the product between the two split nodes
            let mle_out = MultilinearExtension::new(vec![
                Fr::from(1 * 5),
                Fr::from(2 * 6),
                Fr::from(3 * 7),
                Fr::from(4 * 8),
            ]);

            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);

            let input_shred = InputShred::new(ctx, mle, &input_layer);
            let input_shred_out = InputShred::new(ctx, mle_out, &input_layer);

            let split_sectors = SplitNode::new(ctx, &input_shred, 1);
            let sector_prod = Sector::new(
                ctx,
                &[&split_sectors[0], &split_sectors[1]],
                |inputs| Expression::<_, AbstractExpr>::products(vec![inputs[0], inputs[1]]),
                |inputs| {
                    let data: Vec<Fr> = inputs[0]
                        .get_evals_vector()
                        .iter()
                        .zip(inputs[1].get_evals_vector().iter())
                        .map(|(lhs, rhs)| *lhs * *rhs)
                        .collect();

                    MultilinearExtension::new(data)
                },
            );

            let final_sector = Sector::new(
                ctx,
                &[&sector_prod, &input_shred_out],
                |inputs| {
                    Expression::<Fr, AbstractExpr>::mle(inputs[0])
                        - Expression::<Fr, AbstractExpr>::mle(inputs[1])
                },
                |_| MultilinearExtension::new_sized_zero(2),
            );

            let output = OutputNode::new_zero(ctx, &final_sector);

            ComponentSet::<NodeEnum<Fr>>::new_raw(
                vec![
                    input_layer.into(),
                    input_shred.into(),
                    input_shred_out.into(),
                    sector_prod.into(),
                    final_sector.into(),
                    output.into(),
                ]
                .into_iter()
                .chain(
                    split_sectors
                        .into_iter()
                        .map(|split_node| split_node.into())
                        .collect_vec(),
                )
                .collect_vec(),
            )
        });

        test_circuit(circuit, None);
    }
}
