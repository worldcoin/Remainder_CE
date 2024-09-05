use crate::prover::helpers::test_circuit;
use crate::worldcoin::circuits::build_circuit;
use crate::worldcoin::data::{
    load_worldcoin_data
};
use remainder_shared_types::Fr;
use std::path::Path;

// #[test]
// fn test_worldcoin_circuit_tiny() {
//     let data = tiny_worldcoin_data::<Fr>();
//     dbg!(&data);
//     let circuit = build_circuit(data);
//     test_circuit(circuit, None);
// }

// #[test]
// fn test_worldcoin_circuit_tiny_non_power_of_two() {
//     let data = tiny_worldcoin_data_non_power_of_two::<Fr>();
//     dbg!(&data);
//     let circuit = build_circuit(data);
//     test_circuit(circuit, None);
// }

// #[test]
// fn test_worldcoin_circuit_medium() {
//     let data = medium_worldcoin_data::<Fr>();
//     let circuit = build_circuit(data);
//     test_circuit(circuit, None);
// }

#[ignore] // takes 90 seconds!
#[test]
fn test_worldcoin_circuit_iris_v3() {
    use super::parameters_v3::{CONSTANT_DATA_FOLDER, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS};
    // iris
    let path = Path::new(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("iris/test_image.npy");
    let data = load_worldcoin_data::<Fr, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS>(path.clone(), image_path, false);
    let circuit = build_circuit(data);
    test_circuit(circuit, None);
}

// #[ignore] // takes 90 seconds!
// #[test]
// fn test_worldcoin_circuit_mask() {
//     let path = Path::new(CONSTANT_DATA_FOLDER).to_path_buf();
//     let image_path = path.clone().join("mask/image.npy");
//     let data: WorldcoinCircuitData<Fr, WC_BASE, WC_NUM_DIGITS> = load_data(path.clone(), image_path, true);
//     let circuit = build_circuit(data);
//     test_circuit(circuit, None);
// }
