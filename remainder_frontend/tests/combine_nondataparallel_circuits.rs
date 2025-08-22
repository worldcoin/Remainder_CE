use ark_std::test_rng;

use remainder::prover::helpers::test_circuit_with_runtime_optimized_config;
use remainder_frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef};
use remainder_shared_types::{Field, Fr};

use utils::{get_dummy_random_mle, TestUtilComponents};

pub mod utils;

#[allow(dead_code)]
struct ConstantScaledCircuitNodes<F: Field> {
    first_layer_component: NodeRef<F>,
    second_layer_component: NodeRef<F>,
    output_component: NodeRef<F>,
}

impl<F: Field> ConstantScaledCircuitNodes<F> {
    /// A circuit which takes in two MLEs of the same size:
    /// * Layer 0: [TestUtilComponents::constant_scaled_sum] with the two inputs
    /// * Layer 1: [TestUtilComponents::product_scaled] with the output of Layer 0 and `mle_1`
    /// * Layer 2: [TestUtilComponents::difference] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1`.
    pub fn new(
        builder_ref: &mut CircuitBuilder<F>,
        mle_1_input: &NodeRef<F>,
        mle_2_input: &NodeRef<F>,
    ) -> Self {
        let first_layer_component =
            TestUtilComponents::constant_scaled_sum(builder_ref, mle_1_input, mle_2_input);

        let second_layer_component =
            TestUtilComponents::product_scaled(builder_ref, &first_layer_component, mle_1_input);

        let output_component = TestUtilComponents::difference(builder_ref, &second_layer_component);

        Self {
            first_layer_component,
            second_layer_component,
            output_component,
        }
    }
}

#[allow(dead_code)]
struct SumConstantCircuitNodes<F: Field> {
    first_layer_component: NodeRef<F>,
    second_layer_component: NodeRef<F>,
    output_component: NodeRef<F>,
}

impl<F: Field> SumConstantCircuitNodes<F> {
    /// A circuit which takes in two MLEs of the same size:
    /// * Layer 0: [TestUtilComponents::product_sum] with the two inputs
    /// * Layer 1: [TestUtilComponents::constant_scaled_sum] with the output of Layer 0 and `mle_1`
    /// * Layer 2: [TestUtilComponents::difference] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1`.
    pub fn new(
        builder_ref: &mut CircuitBuilder<F>,
        mle_1_input: &NodeRef<F>,
        mle_2_input: &NodeRef<F>,
    ) -> Self {
        let first_layer_component =
            TestUtilComponents::product_sum(builder_ref, mle_1_input, mle_2_input);

        let second_layer_component = TestUtilComponents::constant_scaled_sum(
            builder_ref,
            &first_layer_component,
            mle_1_input,
        );

        let output_component = TestUtilComponents::difference(builder_ref, &second_layer_component);

        Self {
            first_layer_component,
            second_layer_component,
            output_component,
        }
    }
}

#[allow(dead_code)]
struct ProductScaledSumCircuitNodes<F: Field> {
    first_layer_component: NodeRef<F>,
    second_layer_component: NodeRef<F>,
    output_component: NodeRef<F>,
}

impl<F: Field> ProductScaledSumCircuitNodes<F> {
    /// A circuit which takes in two MLEs of the same size:
    /// * Layer 0: [TestUtilComponents::product_scaled] with the two inputs
    /// * Layer 1: [TestUtilComponents::product_sum] with the output of Layer 0 and `mle_1`
    /// * Layer 2: [TestUtilComponents::difference] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1`  An MLE with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE with arbitrary bookkeeping table values, same size as `mle_1`.
    pub fn new(
        builder_ref: &mut CircuitBuilder<F>,
        mle_1_input: &NodeRef<F>,
        mle_2_input: &NodeRef<F>,
    ) -> Self {
        let first_layer_component =
            TestUtilComponents::product_scaled(builder_ref, mle_1_input, mle_2_input);

        let second_layer_component =
            TestUtilComponents::product_sum(builder_ref, &first_layer_component, mle_1_input);

        let output_component = TestUtilComponents::difference(builder_ref, &second_layer_component);

        Self {
            first_layer_component,
            second_layer_component,
            output_component,
        }
    }
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the combined
/// dataparallel/non-dataparallel circuit.
fn build_combined_nondataparallel_circuit<F: Field>(mle_1_2_vars: usize) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // All inputs are public
    let public_input_layer_node = builder.add_input_layer(LayerVisibility::Public);

    // "Semantic" circuit inputs
    let mle_1_shred = builder.add_input_shred("MLE 1", mle_1_2_vars, &public_input_layer_node);
    let mle_2_shred = builder.add_input_shred("MLE 2", mle_1_2_vars, &public_input_layer_node);

    // Create the circuit components
    let _component_1 = ProductScaledSumCircuitNodes::new(&mut builder, &mle_1_shred, &mle_2_shred);
    let _component_2 = SumConstantCircuitNodes::new(&mut builder, &mle_1_shred, &mle_2_shred);
    let _component_3 = ConstantScaledCircuitNodes::new(&mut builder, &mle_1_shred, &mle_2_shred);

    builder.build().unwrap()
}

#[test]
fn test_combined_nondataparallel_circuit_newmainder() {
    const VARS_MLE_1_2: usize = 2;
    let mut rng = test_rng();

    // Generate circuit inputs
    let mle_1 = get_dummy_random_mle(VARS_MLE_1_2, &mut rng).mle;
    let mle_2 = get_dummy_random_mle(VARS_MLE_1_2, &mut rng).mle;

    // Create circuit description + input helper function
    let mut circuit = build_combined_nondataparallel_circuit(VARS_MLE_1_2);

    circuit.set_input("MLE 1", mle_1);
    circuit.set_input("MLE 2", mle_2);

    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config::<Fr, Fr>(&provable_circuit);
}
