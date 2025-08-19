use ark_std::test_rng;
use remainder::{
    mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
    prover::helpers::test_circuit_with_runtime_optimized_config,
};
use remainder_frontend::{
    layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef},
    sel_expr,
};
use remainder_shared_types::{Field, Fr};
use utils::get_dummy_random_mle;
pub mod utils;

fn data_parallel_recombination_interleave<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    mle_1: &NodeRef<F>,
    mle_2: &NodeRef<F>,
    mle_3: &NodeRef<F>,
    mle_4: &NodeRef<F>,
) -> NodeRef<F> {
    builder_ref.add_sector(sel_expr!(sel_expr!(mle_1, mle_2), sel_expr!(mle_3, mle_4)))
}

fn _data_parallel_recombination_stack<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    mle_1: &NodeRef<F>,
    mle_2: &NodeRef<F>,
    mle_3: &NodeRef<F>,
    mle_4: &NodeRef<F>,
) -> NodeRef<F> {
    builder_ref.add_sector(sel_expr!(sel_expr!(mle_1, mle_2), sel_expr!(mle_3, mle_4)))
}

fn diff_two_inputs<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    mle_1: &NodeRef<F>,
    mle_2: &NodeRef<F>,
) -> NodeRef<F> {
    let first_layer_sector = builder_ref.add_sector(mle_1 - mle_2);

    builder_ref.set_output(&first_layer_sector);
    first_layer_sector
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the combined dataparallel circuit.
fn build_dataparallel_recombination_test_circuit<F: Field>(
    num_dataparallel_vars: usize,
    num_free_vars: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // All inputs are public
    let public_input_layer_node = builder.add_input_layer("Public Input Layer", LayerVisibility::Public);

    // "Semantic" circuit inputs
    let mle_1_shred = builder.add_input_shred("MLE 1", num_free_vars, &public_input_layer_node);
    let mle_2_shred = builder.add_input_shred("MLE 2", num_free_vars, &public_input_layer_node);
    let mle_3_shred = builder.add_input_shred("MLE 3", num_free_vars, &public_input_layer_node);
    let mle_4_shred = builder.add_input_shred("MLE 4", num_free_vars, &public_input_layer_node);
    let combined_dataparallel_mle_shred = builder.add_input_shred(
        "Combined Dataparallel MLE",
        num_dataparallel_vars + num_free_vars,
        &public_input_layer_node,
    );

    // Create the circuit components
    // Stack currently fails at layer 0, because expr and witgen for the first component is inconsistent.
    // But if you change from stack to interleave, then it fails at layer 1, because the subtraction of the dataparallel
    // mle from the output mle is not actually 0.
    let component_1 = data_parallel_recombination_interleave(
        &mut builder,
        &mle_1_shred,
        &mle_2_shred,
        &mle_3_shred,
        &mle_4_shred,
    );

    let _component_2 =
        diff_two_inputs(&mut builder, &component_1, &combined_dataparallel_mle_shred);

    builder.build().unwrap()
}

#[test]
fn test_dataparallel_recombination_newmainder() {
    const NUM_FREE_VARS: usize = 2;
    const NUM_DATAPARALLEL_VARS: usize = 2;
    let mut rng = test_rng();

    let (mles_vec, vecs_vec): (Vec<DenseMle<Fr>>, Vec<Vec<Fr>>) = (0..(1 << NUM_DATAPARALLEL_VARS))
        .map(|_| {
            let mle = get_dummy_random_mle(NUM_FREE_VARS, &mut rng);
            let mle_copy = mle.clone();
            let mle_vec = mle_copy.iter().collect();
            (mle, mle_vec)
        })
        .unzip();

    let combined_mle = DenseMle::combine_mles(mles_vec); // This works
                                                         // let combined_mle = DenseMle::batch_mles(mles_vec); // This fails

    // Grab inputs from the above
    let mle_1 = MultilinearExtension::new(vecs_vec[0].clone());
    let mle_2 = MultilinearExtension::new(vecs_vec[1].clone());
    let mle_3 = MultilinearExtension::new(vecs_vec[2].clone());
    let mle_4 = MultilinearExtension::new(vecs_vec[3].clone());
    let dataparallel_combined_mle = combined_mle.mle;

    // Create circuit description + input helper function
    let mut circuit =
        build_dataparallel_recombination_test_circuit(NUM_DATAPARALLEL_VARS, NUM_FREE_VARS);

    circuit.set_input("MLE 1", mle_1);
    circuit.set_input("MLE 2", mle_2);
    circuit.set_input("MLE 3", mle_3);
    circuit.set_input("MLE 4", mle_4);
    circuit.set_input("Combined Dataparallel MLE", dataparallel_combined_mle);

    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
