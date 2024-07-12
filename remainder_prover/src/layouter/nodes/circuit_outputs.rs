//! The nodes that represent circuit outputs in the circuit DAG

use std::marker::PhantomData;

use remainder_shared_types::FieldExt;

use crate::{
    layouter::layouting::{CircuitLocation, DAGError},
    mle::{dense::DenseMle, zero::ZeroMle, MleIndex},
    prover::proof_system::ProofSystem,
};

use super::{CircuitNode, ClaimableNode, CompilableNode, Context, NodeId};

#[derive(Debug, Clone)]
/// the node that represents the output of a circuit
pub struct OutputNode<F> {
    id: NodeId,
    source: NodeId,
    zero: bool,
    _marker: PhantomData<F>,
}

impl<F: FieldExt> CircuitNode for OutputNode<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.source]
    }
}

impl<F: FieldExt> OutputNode<F> {
    /// Creates a new OutputNode from a source w/ some data
    pub fn new(ctx: &Context, source: &impl ClaimableNode<F = F>) -> Self {
        Self {
            id: ctx.get_new_id(),
            source: source.id(),
            zero: false,
            _marker: PhantomData,
        }
    }

    /// Creates a new ZeroMleRef, which constrains the source to equal a Zero Mle
    pub fn new_zero(ctx: &Context, source: &impl ClaimableNode<F = F>) -> Self {
        Self {
            id: ctx.get_new_id(),
            source: source.id(),
            zero: true,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt, Pf: ProofSystem<F, OutputLayer = O>, O: From<DenseMle<F>> + From<ZeroMle<F>>>
    CompilableNode<F, Pf> for OutputNode<F>
{
    fn compile<'a>(
        &'a self,
        witness_builder: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
        circuit_map: &mut crate::layouter::layouting::CircuitMap<'a, F>,
    ) -> Result<(), crate::layouter::layouting::DAGError> {
        let (circuit_location, data) = circuit_map.get_node(&self.source)?;

        let CircuitLocation {
            prefix_bits,
            layer_id,
        } = circuit_location;

        let prefix_bits_mle_index = prefix_bits
            .iter()
            .map(|bit| MleIndex::Fixed(*bit))
            .collect();

        let out = if self.zero {
            ZeroMle::new(data.num_vars(), Some(prefix_bits_mle_index), *layer_id).into()
        } else {
            DenseMle::new_with_prefix_bits((*data).clone(), *layer_id, prefix_bits.clone()).into()
        };

        witness_builder.add_output_layer(out);
        Ok(())
    }
}
