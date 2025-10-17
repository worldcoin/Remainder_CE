use ark_std::test_rng;
use itertools::Itertools;

use remainder::{
    mle::dense::DenseMle, prover::helpers::test_circuit_with_runtime_optimized_config,
};
use remainder_frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef};
use remainder_shared_types::Field;

pub mod utils;

use utils::TestUtilComponents;

use crate::utils::get_dummy_random_mle;

#[allow(dead_code)]
struct DataParallelNodes<F: Field> {
    first_layer: NodeRef<F>,
    second_layer: NodeRef<F>,
    output: NodeRef<F>,
}

impl<F: Field> DataParallelNodes<F> {
    /// A circuit which takes in two vectors of MLEs of the same size:
    /// * Layer 0: [TestUtilComponents::product_scaled] with the two inputs
    /// * Layer 1: [TestUtilComponents::product_scaled] with the output of Layer 0 and output of Layer 0.
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
        let product_scaled =
            TestUtilComponents::product_scaled(builder_ref, mle_1_input, mle_2_input);

        let product_scaled_meta =
            TestUtilComponents::product_scaled(builder_ref, &product_scaled, &product_scaled);
        let output = TestUtilComponents::difference(builder_ref, &product_scaled_meta);

        Self {
            first_layer: product_scaled,
            second_layer: product_scaled_meta,
            output,
        }
    }
}

#[allow(dead_code)]
struct TripleNestedSelectorNodes<F: Field> {
    first_layer: NodeRef<F>,
    output: NodeRef<F>,
}

impl<F: Field> TripleNestedSelectorNodes<F> {
    /// A circuit in which:
    /// * Layer 0: [TestUtilComponents::triple_nested_selector] with the three inputs
    /// * Layer 1: [TestUtilComponents::difference] with output of Layer 0 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `inner_inner_sel_mle` - An MLE with arbitrary bookkeeping table values.
    /// * `inner_sel_mle` - An MLE with arbitrary bookkeeping table values, but double
    /// the size of `inner_inner_sel_mle`
    /// * `outer_sel_mle` - An MLE with arbitrary bookkeeping table values, but double
    /// the size of `inner_sel_mle`
    pub fn new(
        builder_ref: &mut CircuitBuilder<F>,
        inner_inner_sel: &NodeRef<F>,
        inner_sel: &NodeRef<F>,
        outer_sel: &NodeRef<F>,
    ) -> Self {
        let triple_nested_selector = TestUtilComponents::triple_nested_selector(
            builder_ref,
            inner_inner_sel,
            inner_sel,
            outer_sel,
        );
        let output = TestUtilComponents::difference(builder_ref, &triple_nested_selector);

        Self {
            first_layer: triple_nested_selector,
            output,
        }
    }
}

#[allow(dead_code)]
struct ScaledProductNodes<F: Field> {
    first_layer: NodeRef<F>,
    output: NodeRef<F>,
}

impl<F: Field> ScaledProductNodes<F> {
    /// A circuit in which:
    /// * Layer 0: [TestUtilComponents::product_scaled] with the two inputs
    /// * Layer 1: [TestUtilComponents::difference] with output of Layer 0 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1` - An MLE with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE with arbitrary bookkeeping table values, same size as `mle_1`.
    pub fn new(
        builder_ref: &mut CircuitBuilder<F>,
        mle_1_input: &NodeRef<F>,
        mle_2_input: &NodeRef<F>,
    ) -> Self {
        let product_scaled =
            TestUtilComponents::product_scaled(builder_ref, mle_1_input, mle_2_input);

        let output = TestUtilComponents::difference(builder_ref, &product_scaled);

        Self {
            first_layer: product_scaled,
            output,
        }
    }
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the combined
/// dataparallel/non-dataparallel circuit.
fn build_combined_dataparallel_nondataparallel_test_circuit<F: Field>(
    num_dataparallel_vars: usize,
    mle_1_2_3_4_num_vars: usize,
    mle_5_num_vars: usize,
    mle_6_num_vars: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::new();

    // All inputs are public
    let public_input_layer_node =
        builder.add_input_layer("Public Input Layer", LayerVisibility::Public);

    // "Semantic" circuit inputs
    let dataparallel_mle_1_shred = builder.add_input_shred(
        "Dataparallel MLE 1",
        num_dataparallel_vars + mle_1_2_3_4_num_vars,
        &public_input_layer_node,
    );
    let dataparallel_mle_2_shred = builder.add_input_shred(
        "Dataparallel MLE 2",
        num_dataparallel_vars + mle_1_2_3_4_num_vars,
        &public_input_layer_node,
    );
    let mle_3_shred =
        builder.add_input_shred("MLE 3", mle_1_2_3_4_num_vars, &public_input_layer_node);
    let mle_4_shred =
        builder.add_input_shred("MLE 4", mle_1_2_3_4_num_vars, &public_input_layer_node);
    let mle_5_shred = builder.add_input_shred("MLE 5", mle_5_num_vars, &public_input_layer_node);
    let mle_6_shred = builder.add_input_shred("MLE 6", mle_6_num_vars, &public_input_layer_node);

    // Create the circuit components
    let _component_1 = DataParallelNodes::new(
        &mut builder,
        &dataparallel_mle_1_shred,
        &dataparallel_mle_2_shred,
    );
    let _component_2 =
        TripleNestedSelectorNodes::new(&mut builder, &mle_4_shred, &mle_5_shred, &mle_6_shred);
    let _component_3 = ScaledProductNodes::new(&mut builder, &mle_3_shred, &mle_4_shred);

    builder.build_with_layer_combination().unwrap()
}

/// A circuit which combines the [DataParallelNodes], [TripleNestedSelectorNodes],
/// and [ScaledProductNodes].
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec`, `mle_2_vec` - inputs to [DataParallelNodes] both arbitrary bookkeeping
/// table values, same size.
/// * `mle_4`, `mle_5`, `mle_6` - inputs to [TripleNestedSelectorNodes], `mle_4` has the same
/// size as the mles in `mle_1_vec`, arbitrary bookkeeping table values. `mle_5` has one more
/// variable than `mle_4`, `mle_6` has one more variable than `mle_5`, both arbitrary bookkeeping
/// table values.
/// * `mle_3`, `mle_4` - inputs to [ScaledProductNodes], both arbitrary bookkeeping table values,
/// same size.
/// * `num_dataparallel_bits` - The number of bits that represent which copy index the circuit is.

#[test]
fn test_combined_dataparallel_nondataparallel_circuit_newmainder() {
    const VARS_MLE_1_2_3_4: usize = 1;
    const VARS_MLE_5: usize = VARS_MLE_1_2_3_4 + 1;
    const VARS_MLE_6: usize = VARS_MLE_5 + 1;
    const NUM_DATAPARALLEL_VARS: usize = 1;
    let mut rng = test_rng();

    let mle_1_vec = (0..1 << NUM_DATAPARALLEL_VARS)
        .map(|_| get_dummy_random_mle(VARS_MLE_1_2_3_4, &mut rng))
        .collect_vec();
    let mle_2_vec = (0..1 << NUM_DATAPARALLEL_VARS)
        .map(|_| get_dummy_random_mle(VARS_MLE_1_2_3_4, &mut rng))
        .collect_vec();

    let mle_1_vec_batched = DenseMle::combine_mles(mle_1_vec);
    let mle_2_vec_batched = DenseMle::combine_mles(mle_2_vec);

    // Grab inputs from the above
    let dataparallel_mle_1 = mle_1_vec_batched.mle;
    let dataparallel_mle_2 = mle_2_vec_batched.mle;
    let mle_3 = get_dummy_random_mle(VARS_MLE_1_2_3_4, &mut rng).mle;
    let mle_4 = get_dummy_random_mle(VARS_MLE_1_2_3_4, &mut rng).mle;
    let mle_5 = get_dummy_random_mle(VARS_MLE_5, &mut rng).mle;
    let mle_6 = get_dummy_random_mle(VARS_MLE_6, &mut rng).mle;

    // Create circuit description + input helper function
    let mut circuit = build_combined_dataparallel_nondataparallel_test_circuit(
        NUM_DATAPARALLEL_VARS,
        VARS_MLE_1_2_3_4,
        VARS_MLE_5,
        VARS_MLE_6,
    );

    circuit.set_input("Dataparallel MLE 1", dataparallel_mle_1);
    circuit.set_input("Dataparallel MLE 2", dataparallel_mle_2);
    circuit.set_input("MLE 3", mle_3);
    circuit.set_input("MLE 4", mle_4);
    circuit.set_input("MLE 5", mle_5);
    circuit.set_input("MLE 6", mle_6);

    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
