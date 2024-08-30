//! The nodes that represent circuit outputs in the circuit DAG

use std::marker::PhantomData;

use remainder_shared_types::FieldExt;

use crate::{
    layer::LayerId,
    layouter::layouting::CircuitLocation,
    mle::{dense::DenseMle, mle_enum::MleEnum, zero::ZeroMle, MleIndex},
    output_layer::mle_output_layer::MleOutputLayer,
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
    pub fn new(ctx: &Context, source: &impl ClaimableNode<F>) -> Self {
        Self {
            id: ctx.get_new_id(),
            source: source.id(),
            zero: false,
            _marker: PhantomData,
        }
    }

    /// Creates a new ZeroMleRef, which constrains the source to equal a Zero Mle
    pub fn new_zero(ctx: &Context, source: &impl ClaimableNode<F>) -> Self {
        Self {
            id: ctx.get_new_id(),
            source: source.id(),
            zero: true,
            _marker: PhantomData,
        }
    }

    pub fn compile_output<'a>(
        &'a self,
        circuit_map: &mut crate::layouter::layouting::CircuitMap<'a, F>,
    ) -> Result<MleOutputLayer<F>, crate::layouter::layouting::DAGError> {
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

        Ok(out)
    }
}
