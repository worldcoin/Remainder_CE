use crate::input_layer::InputLayerDescription;
use crate::layer::LayerId;
use crate::mle::evals::MultilinearExtension;
use crate::prover::GKRCircuitDescription;
use crate::worldcoin::circuits::build_iriscode_circuit_description;
use crate::worldcoin::data::{
    build_iriscode_circuit_data, load_worldcoin_data_v2, load_worldcoin_data_v3, wirings_to_reroutings
};
use crate::worldcoin::parameters::decode_wirings;
use ndarray::Array2;
use remainder_shared_types::Fr;
use std::collections::HashMap;
use std::path::Path;

/// Return the circuit description, "private" input layer description and inputs for a trivial 2x2
/// identity matrix circuit.
pub fn small_circuit_description_and_inputs() -> (GKRCircuitDescription<Fr>, InputLayerDescription, HashMap<LayerId, MultilinearExtension<Fr>>) {
    // rewirings for the 2x2 identity matrix
    let wirings = &vec![(0, 0, 0, 0), (0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)];
    let reroutings = wirings_to_reroutings(wirings, 2, 2);
    let (circuit_desc, input_builder, private_input_layer_desc) = build_iriscode_circuit_description::<Fr, 2, 1, 1, 1, 16, 2>(reroutings);
    let data = build_iriscode_circuit_data::<Fr, 1, 1, 1, 16, 2>(
        Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap(),
        &vec![1, 0, 6, -1],
        &vec![1, 0, 1, 0],
        wirings,
    );
    let inputs = input_builder(data);
    (circuit_desc, private_input_layer_desc, inputs)
}

/// Return the circuit description, "private" input layer description and inputs for the v2 iris
/// code circuit, in either the mask (true) or iris (false) case.
/// # Example:
/// ```
/// use remainder::worldcoin::tests::v2_circuit_description_and_inputs;
/// let (circuit_desc, _, inputs) = v2_circuit_description_and_inputs(false);
/// ```
pub fn v2_circuit_description_and_inputs(mask: bool) -> (GKRCircuitDescription<Fr>, InputLayerDescription, HashMap<LayerId, MultilinearExtension<Fr>>) {
    use super::parameters_v2::{
        BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS, MATMULT_ROWS_NUM_VARS,
        NUM_DIGITS, WIRINGS, TO_REROUTE_NUM_VARS, IM_NUM_ROWS, IM_NUM_COLS
    };
    let image_path = if mask {
        Path::new("src/worldcoin/constants/v2/mask/test_image.npy").to_path_buf()
    } else {
        Path::new("src/worldcoin/constants/v2/iris/test_image.npy").to_path_buf()
    };
    let data = load_worldcoin_data_v2::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
        IM_NUM_ROWS,
        IM_NUM_COLS,
    >(image_path, mask);
    let wirings = &decode_wirings(WIRINGS);
    let reroutings = wirings_to_reroutings(wirings, IM_NUM_COLS, 1 << MATMULT_INTERNAL_DIM_NUM_VARS);
    let (circuit_desc, input_builder, private_input_layer_desc) = build_iriscode_circuit_description::<
        Fr,
        TO_REROUTE_NUM_VARS,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(reroutings);
    let inputs = input_builder(data);
    (circuit_desc, private_input_layer_desc, inputs)
}

/// Return the circuit description, "private" input layer description and inputs for the v3 iris
/// code circuit, in either the mask (true) or iris (false) case.
/// # Example:
/// ```
/// use remainder::worldcoin::tests::v3_circuit_description_and_inputs;
/// let (circuit_desc, _, inputs) = v3_circuit_description_and_inputs(false);
/// ```
pub fn v3_circuit_description_and_inputs(mask: bool) -> (GKRCircuitDescription<Fr>, InputLayerDescription, HashMap<LayerId, MultilinearExtension<Fr>>) {
    use super::parameters_v3::{
        BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS, MATMULT_ROWS_NUM_VARS,
        NUM_DIGITS, WIRINGS, TO_REROUTE_NUM_VARS, IM_NUM_ROWS, IM_NUM_COLS
    };
    let image_path = if mask {
        Path::new("src/worldcoin/constants/v3/mask/test_image.npy").to_path_buf()
    } else {
        Path::new("src/worldcoin/constants/v3/iris/test_image.npy").to_path_buf()
    };
    let data = load_worldcoin_data_v3::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
        IM_NUM_ROWS,
        IM_NUM_COLS,
    >(image_path, mask);
    let wirings = &decode_wirings(WIRINGS);
    let reroutings = wirings_to_reroutings(wirings, IM_NUM_COLS, 1 << MATMULT_INTERNAL_DIM_NUM_VARS);
    let (circuit_desc, input_builder, private_input_layer_desc) = build_iriscode_circuit_description::<
        Fr,
        TO_REROUTE_NUM_VARS,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(reroutings);
    let inputs = input_builder(data);
    (circuit_desc, private_input_layer_desc, inputs)
}