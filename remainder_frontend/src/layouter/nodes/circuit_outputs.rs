//! The nodes that represent circuit outputs in the circuit DAG

use itertools::{repeat_n, Itertools};
use remainder_shared_types::Field;

use remainder::{
    circuit_layout::{CircuitMap, CircuitLocation},
    mle::MleIndex,
    output_layer::OutputLayerDescription,
};

use super::{CircuitNode, NodeId};

use anyhow::Result;

#[derive(Debug, Clone)]
/// the node that represents the output of a circuit
pub struct OutputNode {
    id: NodeId,
    source: NodeId,
    _zero: bool,
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
    pub fn new(source: &impl CircuitNode) -> Self {
        Self {
            id: NodeId::new(),
            source: source.id(),
            _zero: false,
        }
    }

    /// Creates a new ZeroMleRef, which constrains the source to equal a Zero Mle
    pub fn new_zero(source: &dyn CircuitNode) -> Self {
        Self {
            id: NodeId::new(),
            source: source.id(),
            _zero: true,
        }
    }

    /// Using the [CircuitDescriptionMap], create a [OutputLayerDescription] which
    /// represents the circuit description of an [OutputNode].
    pub fn generate_circuit_description<F: Field>(
        &self,
        circuit_map: &mut CircuitMap<F>,
    ) -> Result<OutputLayerDescription<F>> {
        let (circuit_location, num_vars) =
            circuit_map.get_location_num_vars_from_node_id(&self.source)?;

        let CircuitLocation {
            prefix_bits,
            layer_id,
        } = circuit_location;

        let total_indices = prefix_bits
            .iter()
            .map(|bit| MleIndex::Fixed(*bit))
            .chain(repeat_n(MleIndex::Free, *num_vars))
            .collect_vec();

        let out = OutputLayerDescription::new_zero(*layer_id, &total_indices);

        Ok(out)
    }
}
