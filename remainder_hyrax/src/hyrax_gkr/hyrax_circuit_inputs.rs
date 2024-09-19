use remainder::layouter::nodes::{
    circuit_inputs::{HyraxInputDType, InputShredData},
    NodeId,
};
use remainder_shared_types::curves::PrimeOrderCurve;

use super::hyrax_input_layer::HyraxProverCommitmentEnum;

/// A struct that represents input data that will be used to populate a
/// [GKRCircuitDescription] in order to generate a full circuit.
#[derive(Debug, Clone)]
pub struct HyraxInputLayerData<C: PrimeOrderCurve> {
    /// The input node ID in the circuit building process that corresponds to
    /// this data.
    pub corresponding_input_node_id: NodeId,
    /// The vector of data that goes in this input layer, as [InputShredData].
    pub data: Vec<InputShredData<C::Scalar>>,
    /// An option that is None if this layer has no precommit, but otherwise
    /// the precommit of this input layer.
    pub precommit: Option<HyraxProverCommitmentEnum<C>>,
    /// The data type of the input, used for optimizations. None if
    /// they are just scalar field elements.
    pub input_data_type: Option<HyraxInputDType>,
}

impl<C: PrimeOrderCurve> HyraxInputLayerData<C> {
    pub fn new(
        corresponding_input_node_id: NodeId,
        data: Vec<InputShredData<C::Scalar>>,
        precommit: Option<HyraxProverCommitmentEnum<C>>,
        input_data_type: Option<HyraxInputDType>,
    ) -> Self {
        Self {
            corresponding_input_node_id,
            data,
            precommit,
            input_data_type,
        }
    }
}
