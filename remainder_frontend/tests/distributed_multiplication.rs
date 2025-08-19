use ark_std::test_rng;
use itertools::Itertools;
use remainder::{
    mle::{dense::DenseMle, Mle},
    prover::helpers::test_circuit_with_runtime_optimized_config,
};
use remainder_frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef};
use remainder_shared_types::{Field, Fr};
use utils::get_dummy_random_mle;
pub mod utils;

pub fn data_parallel_distributed_mult<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    smaller_mle: &NodeRef<F>,
    bigger_mle: &NodeRef<F>,
) -> NodeRef<F> {
    builder_ref.add_sector(bigger_mle * smaller_mle)
}

pub fn diff_two_inputs<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    mle_1: &NodeRef<F>,
    mle_2: &NodeRef<F>,
) -> NodeRef<F> {
    let first_layer_sector = builder_ref.add_sector(mle_1 - mle_2);

    builder_ref.set_output(&first_layer_sector);

    first_layer_sector
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
fn build_dataparallel_wraparound_multiplication_test_circuit<F: Field>(
    num_dataparallel_vars: usize,
    num_free_vars_smaller: usize,
    num_free_vars_bigger: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // All inputs are public
    let public_input_layer_node = builder.add_input_layer("Public Input Layer", LayerVisibility::Public);

    // "Semantic" circuit inputs
    let dataparallel_mle_smaller_shred = builder.add_input_shred(
        "Dataparallel Smaller MLE",
        num_dataparallel_vars + num_free_vars_smaller,
        &public_input_layer_node,
    );
    let dataparallel_mle_bigger_shred = builder.add_input_shred(
        "Dataparallel Bigger MLE",
        num_dataparallel_vars + num_free_vars_bigger,
        &public_input_layer_node,
    );
    let dataparallel_mle_combined_shred = builder.add_input_shred(
        "Dataparallel Combined MLE",
        num_dataparallel_vars + num_free_vars_bigger,
        &public_input_layer_node,
    );

    // Create the circuit components
    let component_1 = data_parallel_distributed_mult(
        &mut builder,
        &dataparallel_mle_smaller_shred,
        &dataparallel_mle_bigger_shred,
    );
    let _component_2 =
        diff_two_inputs(&mut builder, &component_1, &dataparallel_mle_combined_shred);

    builder.build().unwrap()
}

#[test]
fn test_dataparallel_wraparound_multiplication_circuit() {
    const NUM_FREE_VARS_SMALLER: usize = 1;
    const NUM_FREE_VARS_BIGGER: usize = 2;
    const NUM_DATAPARALLEL_VARS: usize = 2;
    let mut rng = test_rng();

    let smaller_mles_vec: Vec<DenseMle<Fr>> = (0..(1 << NUM_DATAPARALLEL_VARS))
        .map(|_| get_dummy_random_mle(NUM_FREE_VARS_SMALLER, &mut rng))
        .collect_vec();

    let bigger_mles_vec: Vec<DenseMle<Fr>> = (0..(1 << NUM_DATAPARALLEL_VARS))
        .map(|_| get_dummy_random_mle(NUM_FREE_VARS_BIGGER, &mut rng))
        .collect_vec();

    let prod_mles = smaller_mles_vec
        .iter()
        .zip(bigger_mles_vec.iter())
        .map(|(small_mle, big_mle)| {
            let big_mle_bt_iter = big_mle.iter();
            let diff = big_mle.len() / small_mle.len();
            DenseMle::new_from_iter(
                big_mle_bt_iter.enumerate().map(|(idx, elem)| {
                    let small_mle_idx = idx / diff;
                    elem * small_mle.get(small_mle_idx).unwrap()
                }),
                small_mle.layer_id,
            )
        })
        .collect_vec();

    // --- Dataparallel-ize + grab inputs from above ---
    let dataparallel_mle_combined = DenseMle::combine_mles(prod_mles).mle; // This works
                                                                           // let dataparallel_mle_combined = DenseMle::batch_mles(prod_mles); // This fails

    let dataparallel_mle_smaller = DenseMle::combine_mles(smaller_mles_vec).mle; // This works
                                                                                 // let dataparallel_mle_smaller = DenseMle::batch_mles(smaller_mles_vec); // This fails

    let dataparallel_mle_bigger = DenseMle::combine_mles(bigger_mles_vec).mle; // This works
                                                                               // let dataparallel_mle_bigger = DenseMle::batch_mles(bigger_mles_vec); // This fails

    // Create circuit description + input helper function
    let mut circuit = build_dataparallel_wraparound_multiplication_test_circuit(
        NUM_DATAPARALLEL_VARS,
        NUM_FREE_VARS_SMALLER,
        NUM_FREE_VARS_BIGGER,
    );

    circuit.set_input("Dataparallel Smaller MLE", dataparallel_mle_smaller);
    circuit.set_input("Dataparallel Bigger MLE", dataparallel_mle_bigger);
    circuit.set_input("Dataparallel Combined MLE", dataparallel_mle_combined);

    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
