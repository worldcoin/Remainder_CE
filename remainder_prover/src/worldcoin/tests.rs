use crate::prover::helpers::test_circuit;
use crate::worldcoin::circuits::build_circuit;
use crate::worldcoin::data::{
    load_worldcoin_data, trivial_wiring_2x2_circuit_data, trivial_wiring_2x2_odd_kernel_dims_circuit_data
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
    use super::parameters_v2::{CONSTANT_DATA_FOLDER, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS};
    let path = Path::new(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("iris/test_image.npy");
    let data = load_worldcoin_data::<Fr, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS>(path.clone(), image_path, false);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

#[ignore] // takes 90 seconds!
#[test]
fn test_worldcoin_circuit_mask_v2() {
    use super::parameters_v2::{CONSTANT_DATA_FOLDER, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS};
    let path = Path::new(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("mask/test_image.npy");
    let data = load_worldcoin_data::<Fr, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS>(path.clone(), image_path, true);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

#[ignore] // takes 90 seconds!
#[test]
fn test_worldcoin_circuit_iris_v3() {
    use super::parameters_v3::{CONSTANT_DATA_FOLDER, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS};
    let path = Path::new(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("iris/test_image.npy");
    let data = load_worldcoin_data::<Fr, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS>(path.clone(), image_path, false);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

#[ignore] // takes 90 seconds!
#[test]
fn test_worldcoin_circuit_mask_v3() {
    use super::parameters_v3::{CONSTANT_DATA_FOLDER, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS};
    let path = Path::new(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("mask/test_image.npy");
    let data = load_worldcoin_data::<Fr, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS>(path.clone(), image_path, true);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}