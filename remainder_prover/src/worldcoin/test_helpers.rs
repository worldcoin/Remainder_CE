use crate::input_layer::InputLayerDescription;
use crate::layer::LayerId;
use crate::mle::evals::MultilinearExtension;
use crate::prover::GKRCircuitDescription;
use crate::worldcoin::circuits::build_iriscode_circuit_description;
use crate::worldcoin::data::{
    build_iriscode_circuit_data, load_worldcoin_data_v2, load_worldcoin_data_v3, wirings_to_reroutings
};
use crate::worldcoin::io::read_bytes_from_file;
use crate::worldcoin::parameters::decode_wirings;
use ndarray::Array2;
use remainder_shared_types::Fr;
use std::collections::HashMap;

/// Return the circuit description, "private" input layer description and inputs for a trivial 2x2
/// identity matrix circuit.
pub fn small_circuit_description_and_inputs() -> (GKRCircuitDescription<Fr>, InputLayerDescription, HashMap<LayerId, MultilinearExtension<Fr>>) {
    // rewirings for the 2x2 identity matrix
    let wirings = &[(0, 0, 0, 0), (0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)];
    let reroutings = wirings_to_reroutings(wirings, 2, 2);
    let (circuit_desc, input_builder, private_input_layer_desc) = build_iriscode_circuit_description::<Fr, 2, 1, 1, 1, 16, 2>(reroutings);
    let data = build_iriscode_circuit_data::<Fr, 1, 1, 1, 16, 2>(
        Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap(),
        &[1, 0, 6, -1],
        &[1, 0, 1, 0],
        wirings,
    );
    let inputs = input_builder(data);
    (circuit_desc, private_input_layer_desc, inputs)
}

/// Return the circuit description, "private" input layer description and inputs for either version
/// of the iris code circuit, in either the mask (true) or iris (false) case.
/// If `image_bytes` is `None`, the default test image is used.
/// # Example:
/// ```
/// use remainder::worldcoin::test_helpers::circuit_description_and_inputs;
/// let (circuit_desc, _, inputs) = circuit_description_and_inputs(2, false, None);
/// ```
pub fn circuit_description_and_inputs(version: u8, mask: bool, image_bytes: Option<Vec<u8>>) -> (GKRCircuitDescription<Fr>, InputLayerDescription, HashMap<LayerId, MultilinearExtension<Fr>>) {
    let image_bytes = if let Some(image_bytes) = image_bytes {
        image_bytes
    } else {
        let image_type = if mask { "mask" } else { "iris" };
        let image_path = format!("src/worldcoin/constants/v{version}/{image_type}/test_image.bin");
        read_bytes_from_file(&image_path)
    };
    match version {
        2 => v2_circuit_description_and_inputs(mask, image_bytes),
        3 => v3_circuit_description_and_inputs(mask, image_bytes),
        _ => panic!(),
    }
}

/// Return the circuit description, "private" input layer description and inputs for the v2 iris
/// code circuit, in either the mask (true) or iris (false) case.
fn v2_circuit_description_and_inputs(mask: bool, image_bytes: Vec<u8>) -> (GKRCircuitDescription<Fr>, InputLayerDescription, HashMap<LayerId, MultilinearExtension<Fr>>) {
    use super::parameters_v2::{
        BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS, MATMULT_ROWS_NUM_VARS,
        NUM_DIGITS, WIRINGS, TO_REROUTE_NUM_VARS, IM_NUM_ROWS, IM_NUM_COLS
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
    >(image_bytes, mask);
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
fn v3_circuit_description_and_inputs(mask: bool, image_bytes: Vec<u8>) -> (GKRCircuitDescription<Fr>, InputLayerDescription, HashMap<LayerId, MultilinearExtension<Fr>>) {
    use super::parameters_v3::{
        BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS, MATMULT_ROWS_NUM_VARS,
        NUM_DIGITS, WIRINGS, TO_REROUTE_NUM_VARS, IM_NUM_ROWS, IM_NUM_COLS
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
    >(image_bytes, mask);
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