use crate::prover::helpers::test_circuit;
use crate::worldcoin::circuits::build_circuit;
use crate::worldcoin::data::{
    load_worldcoin_data, trivial_wiring_2x2_circuit_data,
    trivial_wiring_2x2_odd_kernel_dims_circuit_data,
};
use remainder_shared_types::Fr;
use std::path::Path;

#[test]
fn test_trivial_wiring_2x2_circuit_data() {
    let data = trivial_wiring_2x2_circuit_data::<Fr>();
    dbg!(&data);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

#[test]
fn test_trivial_wiring_2x2_odd_kernel_dims_circuit_data() {
    let data = trivial_wiring_2x2_odd_kernel_dims_circuit_data::<Fr>();
    dbg!(&data);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

#[ignore] // takes 90 seconds!
#[test]
fn test_worldcoin_circuit_iris_v2() {
    use super::parameters_v2::{
        BASE, CONSTANT_DATA_FOLDER, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    let path = Path::new("../").join(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("iris/test_image.npy");
    let data = load_worldcoin_data::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(path.clone(), image_path, false);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

#[ignore] // takes 90 seconds!
#[test]
fn test_worldcoin_circuit_mask_v2() {
    use super::parameters_v2::{
        BASE, CONSTANT_DATA_FOLDER, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    let path = Path::new("../").join(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("mask/test_image.npy");
    let data = load_worldcoin_data::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(path.clone(), image_path, true);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

#[ignore] // takes 90 seconds!
#[test]
fn test_worldcoin_circuit_iris_v3() {
    use super::parameters_v3::{
        BASE, CONSTANT_DATA_FOLDER, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    let path = Path::new("../").join(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("iris/test_image.npy");
    let data = load_worldcoin_data::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(path.clone(), image_path, false);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

#[ignore] // takes 90 seconds!
#[test]
fn test_worldcoin_circuit_mask_v3() {
    use super::parameters_v3::{
        BASE, CONSTANT_DATA_FOLDER, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    let path = Path::new("../").join(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("mask/test_image.npy");
    let data = load_worldcoin_data::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(path.clone(), image_path, true);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

#[test]
/// Simply checks that the test files for v2 are available (in both the iris and mask case) and that
/// the CircuitData instance can be constructed.
fn test_load_worldcoin_data_v2() {
    use super::parameters_v2::{
        BASE, CONSTANT_DATA_FOLDER, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    use remainder_shared_types::Fr;
    use std::path::Path;
    // iris
    let path = Path::new("../").join(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("iris/test_image.npy");
    let data = load_worldcoin_data::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(path.clone(), image_path, false);
    data.ensure_guarantees();
    // mask
    let image_path = path.join("mask/test_image.npy");
    let data = load_worldcoin_data::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(path.clone(), image_path, true);
    data.ensure_guarantees();
}

#[test]
/// Simply checks that the test files for v3 are available (in both the iris and mask case) and that
/// the CircuitData instance can be constructed.
fn test_load_worldcoin_data_v3() {
    use super::parameters_v3::{
        BASE, CONSTANT_DATA_FOLDER, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    use remainder_shared_types::Fr;
    use std::path::Path;
    // iris
    let path = Path::new("../").join(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("iris/test_image.npy");
    let data = load_worldcoin_data::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(path.clone(), image_path, false);
    data.ensure_guarantees();
    // mask
    let image_path = path.join("mask/test_image.npy");
    let data = load_worldcoin_data::<
        Fr,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(path.clone(), image_path, true);
    data.ensure_guarantees();
}
