//! The nodes that represent circuit outputs in the circuit DAG

use itertools::{repeat_n, Itertools};
use remainder_shared_types::Field;

use crate::{
    layouter::layouting::{CircuitDescriptionMap, CircuitLocation},
    mle::MleIndex,
    output_layer::mle_output_layer::CircuitMleOutputLayer,
};

use super::{CircuitNode, Context, NodeId};

#[derive(Debug, Clone)]
/// the node that represents the output of a circuit
pub struct OutputNode {
    id: NodeId,
    source: NodeId,
    zero: bool,
}

impl CircuitNode for OutputNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.source]
    }

    fn get_num_vars(&self) -> usize {
        todo!()
    }
}

impl OutputNode {
    /// Creates a new OutputNode from a source w/ some data
    pub fn new(ctx: &Context, source: &impl CircuitNode) -> Self {
        Self {
            id: ctx.get_new_id(),
            source: source.id(),
            zero: false,
        }
    }

    /// Creates a new ZeroMleRef, which constrains the source to equal a Zero Mle
    pub fn new_zero(ctx: &Context, source: &impl CircuitNode) -> Self {
        Self {
            id: ctx.get_new_id(),
            source: source.id(),
            zero: true,
        }
    }

    pub fn compile_output<F: Field>(
        &self,
        circuit_map: &mut CircuitDescriptionMap,
    ) -> Result<CircuitMleOutputLayer<F>, crate::layouter::layouting::DAGError> {
        let (circuit_location, num_vars) = circuit_map.get_node(&self.source)?;

        let CircuitLocation {
            prefix_bits,
            layer_id,
        } = circuit_location;

        let total_indices = prefix_bits
            .iter()
            .map(|bit| MleIndex::Fixed(*bit))
            .chain(repeat_n(MleIndex::Iterated, *num_vars))
            .collect_vec();

        let out = CircuitMleOutputLayer::new_zero(*layer_id, &total_indices);

        Ok(out)
    }
}
