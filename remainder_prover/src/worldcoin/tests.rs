use crate::prover::helpers::test_circuit;
use crate::worldcoin::circuits::{build_circuit, build_circuit_description};
use crate::worldcoin::data::{
    load_worldcoin_data_v2, load_worldcoin_data_v3, wirings_to_reroutings, CircuitData
};
use ndarray::Array2;
use remainder_shared_types::Fr;
use std::path::Path;

#[test]
fn test_trivial_wiring_2x2_circuit_data() {
    // rewirings for the 2x2 identity matrix
    let wirings = &vec![(0, 0, 0, 0), (0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)];
    let reroutings = wirings_to_reroutings(wirings, 2, 2);
    let data: CircuitData<Fr, 1, 1, 1, 16, 2> = CircuitData::build_worldcoin_circuit_data(
        Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap(),
        &vec![1, 0, 6, -1],
        &vec![1, 0, 1, 0],
        wirings,
    );
    dbg!(&data);
    let (circuit_desc, input_node_map, public_input_node, private_input_node) = build_circuit_description::<Fr, 2, 1, 1, 1, 16, 2>(reroutings);
    let circuit = todo!();
    test_circuit(circuit, None);
}

//FIXME(Ben)
// #[test]
// fn test_trivial_wiring_2x2_odd_kernel_dims_circuit_data() {
//     // rewirings for the 2x2 identity matrix
//     let wirings = &vec![(0, 0, 0, 0), (0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)];
//     let reroutings = wirings_to_reroutings(wirings, 2, 2);
//     let data: CircuitData<Fr, 1, 1, 2, 16, 2> = CircuitData::build_worldcoin_circuit_data(
//         Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap(),
//         &vec![1, 0, -4, 6, -1, 3, 0, 0],
//         &vec![1, 0, 1, 0],
//         wirings,
//     );
//     dbg!(&data);
//     let (circuit_desc, input_node_map, public_input_node, private_input_node) = build_circuit_description::<Fr, 2, 1, 1, 2, 16, 2>(reroutings);
//     let circuit = todo!();
//     test_circuit(circuit, None);
// }

#[ignore] // takes 90 seconds!
#[test]
fn test_worldcoin_circuit_iris_v2() {
    use super::parameters_v2::{
        BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    let image_path = Path::new("src/worldcoin/constants/v2/iris/test_image.npy").to_path_buf();
    let data = load_worldcoin_data_v2::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(image_path, false);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

#[ignore] // takes 90 seconds!
#[test]
fn test_worldcoin_circuit_mask_v2() {
    use super::parameters_v2::{
        BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    let image_path = Path::new("src/worldcoin/constants/v2/mask/test_image.npy").to_path_buf();
    let data = load_worldcoin_data_v2::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(image_path, true);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}


#[ignore] // takes 90 seconds!
#[test]
fn test_worldcoin_circuit_iris_v3() {
    use super::parameters_v3::{
        BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    let image_path = Path::new("src/worldcoin/constants/v3/iris/test_image.npy").to_path_buf();
    let data = load_worldcoin_data_v3::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(image_path, false);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

#[ignore] // takes 90 seconds!
#[test]
fn test_worldcoin_circuit_mask_v3() {
    use super::parameters_v3::{
        BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    let image_path = Path::new("src/worldcoin/constants/v3/mask/test_image.npy").to_path_buf();
    let data = load_worldcoin_data_v3::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(image_path, true);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

#[test]
/// Simply checks that the test files for v2 are available (in both the iris and mask case) and that
/// the CircuitData instance can be constructed.
fn test_load_worldcoin_data_v2() {
    use super::parameters_v2::{
        BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    let image_path = Path::new("src/worldcoin/constants/v2/iris/test_image.npy").to_path_buf();
    let data = load_worldcoin_data_v2::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(image_path, false);
    data.ensure_guarantees();
    // mask
    let image_path = Path::new("src/worldcoin/constants/v2/mask/test_image.npy").to_path_buf();
    let data = load_worldcoin_data_v2::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(image_path, true);
    data.ensure_guarantees();
}

#[test]
/// Simply checks that the test files for v3 are available (in both the iris and mask case) and that
/// the CircuitData instance can be constructed.
fn test_load_worldcoin_data_v3() {
    use super::parameters_v3::{
        BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    let image_path = Path::new("src/worldcoin/constants/v3/iris/test_image.npy").to_path_buf();
    let data = load_worldcoin_data_v3::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(image_path, false);
    data.ensure_guarantees();
    // mask
    let image_path = Path::new("src/worldcoin/constants/v3/mask/test_image.npy").to_path_buf();
    let data = load_worldcoin_data_v3::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(image_path, true);
    data.ensure_guarantees();
}