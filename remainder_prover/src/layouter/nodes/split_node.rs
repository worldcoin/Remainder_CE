//! A node that can alter the claims made on it's source `ClaimableNode`

use itertools::{repeat_n, Itertools};
use remainder_shared_types::FieldExt;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layouter::layouting::{CircuitLocation, DAGError},
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
pub struct SplitNode<F: FieldExt> {
    id: NodeId,
    data: MultilinearExtension<F>,
    source: NodeId,
    prefix_bits: Vec<bool>,
}

impl<F: FieldExt> SplitNode<F> {
    /// Creates 2^num_vars `SplitNodes` from a single ClaimableNode
    pub fn new(ctx: &Context, node: impl ClaimableNode<F = F>, num_vars: usize) -> Vec<Self> {
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

                let data = MultilinearExtension::new(Evaluations::new(max_num_vars, data));
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

impl<F: FieldExt> CircuitNode for SplitNode<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.source]
    }
}

impl<F: FieldExt> ClaimableNode for SplitNode<F> {
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

impl<F: FieldExt, Pf: ProofSystem<F>> CompilableNode<F, Pf> for SplitNode<F> {
    fn compile<'a>(
        &'a self,
        _: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
        circuit_map: &mut crate::layouter::layouting::CircuitMap<'a, F>,
    ) -> Result<(), crate::layouter::layouting::DAGError> {
        let data = &self.data;
        let (source_location, _) = circuit_map
            .0
            .get(&self.source)
            .ok_or(DAGError::DanglingNodeId(self.source))?;

        let prefix_bits = source_location
            .prefix_bits
            .iter()
            .chain(self.prefix_bits.iter())
            .copied()
            .collect();

        let location = CircuitLocation::new(source_location.layer_id, prefix_bits);

        circuit_map.0.insert(self.id, (location, data));
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
