use remainder::prover::helpers::test_circuit_with_runtime_optimized_config;
use remainder_shared_types::Fr;

use super::test_helpers::small_circuit_description_and_inputs;
use super::v3::circuit_description_and_inputs;

#[test]
fn test_small_circuit_both_layers_public() {
    let provable_circuit = small_circuit_description_and_inputs(false).unwrap();
    test_circuit_with_runtime_optimized_config::<Fr, Fr>(&provable_circuit);
}

#[test]
fn test_small_circuit_with_ligero_layers() {
    let provable_circuit = small_circuit_description_and_inputs(true).unwrap();
    test_circuit_with_runtime_optimized_config::<Fr, Fr>(&provable_circuit);
}

#[ignore]
#[test]
fn test_worldcoin_circuit_iris_v3_public_inputs() {
    let provable_circuit = circuit_description_and_inputs(false, None).unwrap();
    test_circuit_with_runtime_optimized_config::<Fr, Fr>(&provable_circuit);
}
