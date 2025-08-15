/// FIXME the functions below are uncategorized and probably should be moved to a more appropriate
/// module or submodule.
use remainder_shared_types::Field;

use crate::layouter::nodes::{
    circuit_inputs::{InputLayerNode, InputShred, InputShredData},
    CircuitNode,
};
use remainder::mle::evals::MultilinearExtension;

/// Using the number of variables, get an input shred that represents
/// this information.
pub fn get_input_shred_from_num_vars(num_vars: usize, input_node: &InputLayerNode) -> InputShred {
    InputShred::new(num_vars, input_node)
}

/// Using a data vector, get an [InputShred] which represents its
/// shape, along with [InputShredData] which represents the
/// corresponding data.
pub fn build_input_shred_and_data<F: Field>(
    data: MultilinearExtension<F>,
    input_node: &InputLayerNode,
) -> (InputShred, InputShredData<F>) {
    let input_shred = InputShred::new(data.num_vars(), input_node);
    let input_shred_data = InputShredData::new(input_shred.id(), data);
    (input_shred, input_shred_data)
}
