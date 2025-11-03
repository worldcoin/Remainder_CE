use crate::utils::TestUtilComponents;
pub mod utils;

use remainder::mle::evals::MultilinearExtension;
use remainder::prover::helpers::test_circuit_with_runtime_optimized_config;
use remainder_frontend::layouter::builder::{
    Circuit, CircuitBuilder, FSNodeRef, LayerVisibility, NodeRef,
};
use remainder_shared_types::{Field, Fr};

// A simple component which takes a FS Challenge node of 0 variables
// and subtracts an MLE from it. This is similar to the "r - MLE" step
// in a LogUp circuit.
fn fs_challenge<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    fs_challenge_mle: &FSNodeRef<F>,
    mle: &NodeRef<F>,
) -> NodeRef<F> {
    builder_ref.add_sector(fs_challenge_mle - mle)
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
fn build_fs_challenge_sub_test_circuit<F: Field>(num_free_vars: usize) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // The input mle is public...
    let public_input_layer_node =
        builder.add_input_layer("Public Input Layer", LayerVisibility::Public);
    let mle_shred = builder.add_input_shred("MLE", num_free_vars, &public_input_layer_node);
    let fs_challenge_mle_shred = builder.add_fiat_shamir_challenge_node(1);

    // Create the circuit components
    let checker = fs_challenge(&mut builder, &fs_challenge_mle_shred, &mle_shred);
    let _output = TestUtilComponents::difference(&mut builder, &checker);

    builder.build().unwrap()
}

#[test]
fn test_fs_challenge_sub_circuit() {
    const NUM_FREE_VARS: usize = 1;

    let mle = MultilinearExtension::new(vec![Fr::from(3), Fr::from(2)]);

    // Create circuit description + input helper function
    let mut circuit = build_fs_challenge_sub_test_circuit(NUM_FREE_VARS);

    circuit.set_input("MLE", mle);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
