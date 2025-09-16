use remainder_shared_types::Fr;

use crate::{
    layouter::builder::{Circuit, LayerVisibility},
    worldcoin_mpc::{
        circuits::build_circuit,
        data::{fetch_inversed_test_data, generate_trivial_test_data},
    },
};

use super::circuits::mpc_attach_data;

/// Return the circuit description, "private" input layer description and inputs for a trivial
/// mpc circuit.
pub fn small_circuit_description_and_inputs<
    const NUM_IRIS_4_CHUNKS: usize,
    const PARTY_IDX: usize,
>(
    layer_visibility: LayerVisibility,
) -> Circuit<Fr> {
    let mut circuit = build_circuit::<Fr, NUM_IRIS_4_CHUNKS>(layer_visibility);

    let test_trivial_data = generate_trivial_test_data::<Fr, NUM_IRIS_4_CHUNKS, PARTY_IDX>();

    mpc_attach_data(&mut circuit, test_trivial_data);

    circuit
}

/// Return the circuit description, "private" input layer description and inputs for a
/// mpc circuit, with data from Inversed
pub fn inversed_circuit_description_and_inputs<
    const NUM_IRIS_4_CHUNKS: usize,
    const PARTY_IDX: usize,
>(
    test_idx: usize,
    layer_visibility: LayerVisibility,
) -> Circuit<Fr> {
    let mut circuit = build_circuit::<Fr, NUM_IRIS_4_CHUNKS>(layer_visibility);

    let test_inversed_data = fetch_inversed_test_data::<Fr, NUM_IRIS_4_CHUNKS, PARTY_IDX>(test_idx);

    mpc_attach_data(&mut circuit, test_inversed_data);

    circuit
}
