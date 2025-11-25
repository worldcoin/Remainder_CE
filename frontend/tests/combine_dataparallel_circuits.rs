use ark_std::test_rng;

use frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef};
use itertools::Itertools;
use remainder::{
    mle::{dense::DenseMle, Mle},
    prover::helpers::test_circuit_with_memory_optimized_config,
    utils::mle::get_random_mle,
};
use shared_types::{Field, Fr};
use utils::TestUtilComponents;

pub mod utils;

#[allow(dead_code)]
struct DataParallelConstantScaledCircuitAltNodes<F: Field> {
    first_layer: NodeRef<F>,
    second_layer: NodeRef<F>,
    output: NodeRef<F>,
}

impl<F: Field> DataParallelConstantScaledCircuitAltNodes<F> {
    /// A circuit which takes in two vectors of MLEs of the same size:
    /// * Layer 0: [TestUtilComponents::constant_scaled_sum] with the two inputs
    /// * Layer 1: [TestUtilComponents::product_scaled] with the output of Layer 0 and `mle_1_vec`
    /// * Layer 2: [TestUtilComponents::difference] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
    pub fn new(
        builder_ref: &mut CircuitBuilder<F>,
        mle_1_input: &NodeRef<F>,
        mle_2_input: &NodeRef<F>,
    ) -> Self {
        let first_layer =
            TestUtilComponents::constant_scaled_sum(builder_ref, mle_1_input, mle_2_input);

        let second_layer =
            TestUtilComponents::product_scaled(builder_ref, &first_layer, mle_1_input);

        let output = TestUtilComponents::difference(builder_ref, &second_layer);

        Self {
            first_layer,
            second_layer,
            output,
        }
    }
}

#[allow(dead_code)]
struct DataParallelSumConstantCircuitAltNodes<F: Field> {
    first_layer: NodeRef<F>,
    second_layer: NodeRef<F>,
    output: NodeRef<F>,
}

impl<F: Field> DataParallelSumConstantCircuitAltNodes<F> {
    /// A circuit which takes in two vectors of MLEs of the same size:
    /// * Layer 0: [TestUtilComponents::product_sum] with the two inputs
    /// * Layer 1: [TestUtilComponents::constant_scaled_sum] with the output of Layer 0 and `mle_1_vec`
    /// * Layer 2: [TestUtilComponents::difference] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
    pub fn new(
        builder_ref: &mut CircuitBuilder<F>,
        mle_1_input: &NodeRef<F>,
        mle_2_input: &NodeRef<F>,
    ) -> Self {
        let first_layer = TestUtilComponents::product_sum(builder_ref, mle_1_input, mle_2_input);

        let second_layer =
            TestUtilComponents::constant_scaled_sum(builder_ref, &first_layer, mle_1_input);

        let output = TestUtilComponents::difference(builder_ref, &second_layer);

        Self {
            first_layer,
            second_layer,
            output,
        }
    }
}

#[allow(dead_code)]
struct DataParallelProductScaledSumCircuitAltNodes<F: Field> {
    first_layer: NodeRef<F>,
    second_layer: NodeRef<F>,
    output: NodeRef<F>,
}

impl<F: Field> DataParallelProductScaledSumCircuitAltNodes<F> {
    /// A circuit which takes in two vectors of MLEs of the same size:
    /// * Layer 0: [TestUtilComponents::product_scaled] with the two inputs
    /// * Layer 1: [TestUtilComponents::product_sum] with the output of Layer 0 and `mle_1_vec`
    /// * Layer 2: [TestUtilComponents::difference] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
    fn new(
        builder_ref: &mut CircuitBuilder<F>,
        mle_1_input: &NodeRef<F>,
        mle_2_input: &NodeRef<F>,
    ) -> Self {
        let first_layer = TestUtilComponents::product_scaled(builder_ref, mle_1_input, mle_2_input);

        let second_layer = TestUtilComponents::product_sum(builder_ref, &first_layer, mle_1_input);

        let output = TestUtilComponents::difference(builder_ref, &second_layer);

        Self {
            first_layer,
            second_layer,
            output,
        }
    }
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the combined dataparallel circuit.
fn build_combined_dataparallel_test_circuit<F: Field>(
    num_dataparallel_vars: usize,
    mle_1_and_2_num_vars: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // All inputs are public inputs
    let public_input_layer_node =
        builder.add_input_layer("Public Input Layer", LayerVisibility::Public);

    // Inputs to the circuit include the "dataparallel MLE 1" and the "dataparallel MLE 2"
    let dataparallel_mle_1_shred = builder.add_input_shred(
        "Dataparallel MLE 1",
        num_dataparallel_vars + mle_1_and_2_num_vars,
        &public_input_layer_node,
    );
    let dataparallel_mle_2_shred = builder.add_input_shred(
        "Dataparallel MLE 2",
        num_dataparallel_vars + mle_1_and_2_num_vars,
        &public_input_layer_node,
    );

    // Create the circuit components
    let _component_1 = DataParallelProductScaledSumCircuitAltNodes::new(
        &mut builder,
        &dataparallel_mle_1_shred,
        &dataparallel_mle_2_shred,
    );
    let _component_2 = DataParallelSumConstantCircuitAltNodes::new(
        &mut builder,
        &dataparallel_mle_1_shred,
        &dataparallel_mle_2_shred,
    );
    let _component_3 = DataParallelConstantScaledCircuitAltNodes::new(
        &mut builder,
        &dataparallel_mle_1_shred,
        &dataparallel_mle_2_shred,
    );

    builder.build_with_layer_combination().unwrap()
}

#[test]
fn test_combined_dataparallel_circuit_alt_newmainder() {
    const NUM_DATAPARALLEL_VARS: usize = 1;
    const VARS_MLE_1_2: usize = 2;
    let mut rng = test_rng();

    // This is not strictly necessary; the setup of `DenseMle` -->
    // `batch_mles()` --> `bookkeeping_table` is just to emulate what
    // batching *would* look like
    let mle_1_vec: Vec<DenseMle<Fr>> = (0..1 << NUM_DATAPARALLEL_VARS)
        .map(|_| get_random_mle(VARS_MLE_1_2, &mut rng))
        .collect_vec();
    let mle_2_vec: Vec<DenseMle<Fr>> = (0..1 << NUM_DATAPARALLEL_VARS)
        .map(|_| get_random_mle(VARS_MLE_1_2, &mut rng))
        .collect_vec();

    let mle_1_vec_batched = DenseMle::combine_mles(mle_1_vec.clone());
    let mle_2_vec_batched = DenseMle::combine_mles(mle_2_vec.clone());

    // These checks can possibly be done with the newly designed batching bits/system
    let all_num_vars: Vec<usize> = mle_1_vec
        .iter()
        .chain(mle_2_vec.iter())
        .map(|mle| mle.num_free_vars())
        .collect();
    let all_vars_same = all_num_vars.iter().fold(true, |acc, elem| {
        (*elem == mle_1_vec[0].num_free_vars()) & acc
    });
    assert!(all_vars_same);
    assert_eq!(mle_1_vec.len(), mle_2_vec.len());
    assert_eq!(mle_1_vec.len(), 1 << NUM_DATAPARALLEL_VARS);

    // Pull input data from above combination process
    let dataparallel_mle_1 = mle_1_vec_batched.mle;
    let dataparallel_mle_2 = mle_2_vec_batched.mle;

    // Create circuit description + input helper function
    let mut circuit = build_combined_dataparallel_test_circuit(NUM_DATAPARALLEL_VARS, VARS_MLE_1_2);

    circuit.set_input("Dataparallel MLE 1", dataparallel_mle_1);
    circuit.set_input("Dataparallel MLE 2", dataparallel_mle_2);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    // Prove/verify the circuit
    test_circuit_with_memory_optimized_config(&provable_circuit);
}
