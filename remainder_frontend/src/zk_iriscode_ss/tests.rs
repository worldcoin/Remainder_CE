use remainder::prover::helpers::test_circuit_with_runtime_optimized_config;
use remainder_shared_types::Bn256Point;

use crate::zk_iriscode_ss::test_helpers::{
    small_circuit_with_private_inputs, small_circuit_with_public_inputs,
};

use super::v3::circuit_description_and_inputs;

#[test]
fn test_small_circuit_both_layers_public() {
    let provable_circuit = small_circuit_with_public_inputs().unwrap();
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

#[test]
fn test_small_circuit_with_ligero_layers() {
    let provable_circuit = small_circuit_with_private_inputs().unwrap();
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

#[ignore]
#[test]
fn test_worldcoin_circuit_iris_v3_public_inputs() {
    let circuit = circuit_description_and_inputs(false, None).unwrap();
    let provable_circuit = circuit.finalize().unwrap();
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
