use ark_std::test_rng;

use remainder::{
<<<<<<< HEAD:remainder_prover/tests/dataparallel_simple.rs
    layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef},
    mle::{dense::DenseMle, AbstractMle},
=======
    mle::{dense::DenseMle, Mle},
>>>>>>> benny/extract_frontend:remainder_frontend/tests/dataparallel_simple.rs
    prover::helpers::test_circuit_with_runtime_optimized_config,
    utils::mle::get_dummy_random_mle_vec,
};
use remainder_frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef};
use remainder_shared_types::{Field, Fr};
use utils::TestUtilComponents;

pub mod utils;

/// A circuit which does the following:
/// * Layer 0: [TestUtilComponents::product_scaled] with `mle_1_vec`, `mle_2_vec`
/// * Layer 1: [TestUtilComponents::difference] with the output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec`, `mle_2_vec` - inputs to [TestUtilComponents::product_scaled] both arbitrary bookkeeping
/// table values, same size.
///
/// * `num_dataparallel_bits` - The number of bits that represent which copy index the circuit is.

#[allow(dead_code)]
struct NonSelectorDataparallelNodes<F: Field> {
    first_layer_component: NodeRef<F>,
    output_component: NodeRef<F>,
}

impl<F: Field> NonSelectorDataparallelNodes<F> {
    /// A simple wrapper around the [TestUtilComponents::product_scaled] which
    /// additionally contains a [TestUtilComponents::difference] for zero output
    pub fn new(
        builder_ref: &mut CircuitBuilder<F>,
        mle_1_input: &NodeRef<F>,
        mle_2_input: &NodeRef<F>,
    ) -> Self {
        let first_layer_component =
            TestUtilComponents::product_scaled(builder_ref, mle_1_input, mle_2_input);

        let output_component = TestUtilComponents::difference(builder_ref, &first_layer_component);

        Self {
            first_layer_component,
            output_component,
        }
    }
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
fn build_dataparallel_simple_test_circuit<F: Field>(
    num_dataparallel_vars: usize,
    num_free_vars: usize,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // All inputs are public
    let public_input_layer_node = builder.add_input_layer(LayerVisibility::Public);

    // "Semantic" circuit inputs
    let dataparallel_mle_1_shred = builder.add_input_shred(
        "Dataparallel MLE 1",
        num_dataparallel_vars + num_free_vars,
        &public_input_layer_node,
    );
    let dataparallel_mle_2_shred = builder.add_input_shred(
        "Dataparallel MLE 2",
        num_dataparallel_vars + num_free_vars,
        &public_input_layer_node,
    );

    // Create the circuit components
    // Stack currently fails at layer 0, because expr and witgen for the first component is inconsistent.
    // But if you change from stack to interleave, then it fails at layer 1, because the subtraction of the dataparallel
    // mle from the output mle is not actually 0.
    let _component_1 = NonSelectorDataparallelNodes::new(
        &mut builder,
        &dataparallel_mle_1_shred,
        &dataparallel_mle_2_shred,
    );

    builder.build().unwrap()
}

#[test]
fn test_dataparallel_simple_newmainder() {
    const NUM_DATAPARALLEL_VARS: usize = 3;
    const NUM_FREE_VARS: usize = 2;
    let mut rng = test_rng();

    let mle_1_vec: Vec<DenseMle<Fr>> =
        get_dummy_random_mle_vec(NUM_FREE_VARS, NUM_DATAPARALLEL_VARS, &mut rng);
    let mle_2_vec: Vec<DenseMle<Fr>> =
        get_dummy_random_mle_vec(NUM_FREE_VARS, NUM_DATAPARALLEL_VARS, &mut rng);

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

    // TODO(%): the batched mle should be able to demonstrate that there's NUM_DATA_PARALLEL_BITS of batch bits
    let dataparallel_mle_1 = DenseMle::combine_mles(mle_1_vec).mle;
    let dataparallel_mle_2 = DenseMle::combine_mles(mle_2_vec).mle;

    // Create circuit description + input helper function
    let mut circuit = build_dataparallel_simple_test_circuit(NUM_DATAPARALLEL_VARS, NUM_FREE_VARS);

    circuit.set_input("Dataparallel MLE 1", dataparallel_mle_1);
    circuit.set_input("Dataparallel MLE 2", dataparallel_mle_2);

    let provable_circuit = circuit.finalize().unwrap();

    // Prove/verify the circuit
    test_circuit_with_runtime_optimized_config(&provable_circuit);
}
