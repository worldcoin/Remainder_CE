use ark_std::test_rng;

use remainder::{
    mle::{dense::DenseMle, Mle},
    prover::helpers::test_circuit_with_runtime_optimized_config,
    utils::mle::get_dummy_random_mle_vec,
};
use remainder_frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef};
use remainder_shared_types::{Field, Fr};
use utils::TestUtilComponents;

pub mod utils;

#[allow(dead_code)]
struct DataparallelTripleNestedSelectorNodes<F: Field> {
    first_layer_component: NodeRef<F>,
    output_component: NodeRef<F>,
}

impl<F: Field> DataparallelTripleNestedSelectorNodes<F> {
    /// A simple wrapper around the [TestUtilComponents::triple_nested_selector] which
    /// additionally contains a [TestUtilComponents::difference] for zero output
    pub fn new(
        builder_ref: &mut CircuitBuilder<F>,
        mle_1_input: &NodeRef<F>,
        mle_2_input: &NodeRef<F>,
        mle_3_input: &NodeRef<F>,
    ) -> Self {
        let first_layer_component = TestUtilComponents::triple_nested_selector(
            builder_ref,
            mle_1_input,
            mle_2_input,
            mle_3_input,
        );

        let output_component = TestUtilComponents::difference(builder_ref, &first_layer_component);

        Self {
            first_layer_component,
            output_component,
        }
    }
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
fn build_dataparallel_selector_test_circuit<F: Field>(
    num_dataparallel_vars: usize,
    num_vars_mle_1_2: usize,
    num_vars_mle_3: usize,
    num_vars_mle_4: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // All inputs are public
    let public_input_layer_node =
        builder.add_input_layer("Public Input Layer", LayerVisibility::Public);

    // Inputs to the circuit include the four dataparallel MLEs
    let dataparallel_mle_1_shred = builder.add_input_shred(
        "Dataprallel MLE 1",
        num_dataparallel_vars + num_vars_mle_1_2,
        &public_input_layer_node,
    );
    let dataparallel_mle_2_shred = builder.add_input_shred(
        "Dataprallel MLE 2",
        num_dataparallel_vars + num_vars_mle_1_2,
        &public_input_layer_node,
    );
    let dataparallel_mle_3_shred = builder.add_input_shred(
        "Dataprallel MLE 3",
        num_dataparallel_vars + num_vars_mle_3,
        &public_input_layer_node,
    );
    let dataparallel_mle_4_shred = builder.add_input_shred(
        "Dataprallel MLE 4",
        num_dataparallel_vars + num_vars_mle_4,
        &public_input_layer_node,
    );

    // Create the circuit components
    let component_1 = TestUtilComponents::product_scaled(
        &mut builder,
        &dataparallel_mle_1_shred,
        &dataparallel_mle_2_shred,
    );
    let _component_2 = DataparallelTripleNestedSelectorNodes::new(
        &mut builder,
        &component_1,
        &dataparallel_mle_3_shred,
        &dataparallel_mle_4_shred,
    );

    builder.build_with_layer_combination().unwrap()
}

/// A circuit which does the following:
/// * Layer 0: [ProductScaledBuilder] with `mle_1_vec`, `mle_2_vec`
/// * Layer 1: [TripleNestedSelectorBuilder] with output of Layer 0, `mle_3_vec`, `mle_4_vec`
/// * Layer 2: [ZeroBuilder] with the output of Layer 1 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec`, `mle_2_vec` - inputs to [ProductScaledBuilder] both arbitrary bookkeeping
/// table values, same size.
/// * `mle_3_vec`, `mle_4_vec` - inputs to [TripleNestedSelectorBuilder], both arbitrary bookkeeping table values,
/// `mle_3_vec` mles have one more variable than in `mle_1_vec`, `mle_2_vec`, and `mle_4_vec` mles
/// have one more variable than in `mle_3_vec`.
#[test]
fn test_dataparallel_selector_alt_newmainder() {
    const NUM_DATAPARALLEL_VARS: usize = 3;
    const NUM_VARS_MLE_1_2: usize = 2;
    const NUM_VARS_MLE_3: usize = NUM_VARS_MLE_1_2 + 1;
    const NUM_VARS_MLE_4: usize = NUM_VARS_MLE_3 + 1;
    let mut rng = test_rng();

    // This is not strictly necessary; the setup of `DenseMle` -->
    // `batch_mles()` --> `bookkeeping_table` is just to emulate what
    // batching *would* look like
    let mle_1_vec: Vec<DenseMle<Fr>> =
        get_dummy_random_mle_vec(NUM_VARS_MLE_1_2, NUM_DATAPARALLEL_VARS, &mut rng);
    let mle_2_vec: Vec<DenseMle<Fr>> =
        get_dummy_random_mle_vec(NUM_VARS_MLE_1_2, NUM_DATAPARALLEL_VARS, &mut rng);
    let mle_3_vec: Vec<DenseMle<Fr>> =
        get_dummy_random_mle_vec(NUM_VARS_MLE_3, NUM_DATAPARALLEL_VARS, &mut rng);
    let mle_4_vec: Vec<DenseMle<Fr>> =
        get_dummy_random_mle_vec(NUM_VARS_MLE_4, NUM_DATAPARALLEL_VARS, &mut rng);

    // These checks can possibly be done with the newly designed batching bits/system
    assert_eq!(mle_1_vec.len(), mle_2_vec.len());
    assert_eq!(mle_3_vec.len(), mle_2_vec.len());
    assert_eq!(mle_3_vec.len(), mle_4_vec.len());
    assert_eq!(mle_1_vec.len(), 1 << NUM_DATAPARALLEL_VARS);
    let all_num_vars_1_2: Vec<usize> = mle_1_vec
        .iter()
        .chain(mle_2_vec.iter())
        .map(|mle| mle.num_free_vars())
        .collect();
    let all_vars_same_1_2 = all_num_vars_1_2.iter().fold(true, |acc, elem| {
        (*elem == mle_3_vec[0].num_free_vars() - 1) & acc
    });
    assert!(all_vars_same_1_2);
    let all_num_vars_3: Vec<usize> = mle_3_vec.iter().map(|mle| mle.num_free_vars()).collect();
    let all_vars_same_3 = all_num_vars_3.iter().fold(true, |acc, elem| {
        (*elem == mle_4_vec[0].num_free_vars() - 1) & acc
    });
    assert!(all_vars_same_3);
    let all_num_vars_4: Vec<usize> = mle_4_vec.iter().map(|mle| mle.num_free_vars()).collect();
    let all_vars_same_4 = all_num_vars_4.iter().fold(true, |acc, elem| {
        (*elem == mle_4_vec[0].num_free_vars()) & acc
    });
    assert!(all_vars_same_4);
    // These checks can possibly be done with the newly designed batching bits/system

    // TODO(%): the batched mle should be able to demonstrate that there's NUM_DATA_PARALLEL_BITS of batch bits
    let dataparallel_mle_1 = DenseMle::combine_mles(mle_1_vec).mle;
    let dataparallel_mle_2 = DenseMle::combine_mles(mle_2_vec).mle;
    let dataparallel_mle_3 = DenseMle::combine_mles(mle_3_vec).mle;
    let dataparallel_mle_4 = DenseMle::combine_mles(mle_4_vec).mle;

    // Create circuit description + input helper function
    let mut circuit = build_dataparallel_selector_test_circuit(
        NUM_DATAPARALLEL_VARS,
        NUM_VARS_MLE_1_2,
        NUM_VARS_MLE_3,
        NUM_VARS_MLE_4,
    );

    circuit.set_input("Dataprallel MLE 1", dataparallel_mle_1);
    circuit.set_input("Dataprallel MLE 2", dataparallel_mle_2);
    circuit.set_input("Dataprallel MLE 3", dataparallel_mle_3);
    circuit.set_input("Dataprallel MLE 4", dataparallel_mle_4);

    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
