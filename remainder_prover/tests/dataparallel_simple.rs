use std::collections::HashMap;

use ark_std::test_rng;

use itertools::Itertools;
use remainder::{
    layer::LayerId,
    layouter::{
        component::Component,
        nodes::{
            circuit_inputs::{InputLayerNode, InputShred},
            circuit_outputs::OutputNode,
            node_enum::NodeEnum,
            sector::Sector,
            CircuitNode, NodeId,
        },
    },
    mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
    prover::{generate_circuit_description, helpers::test_circuit_new, GKRCircuitDescription},
    utils::mle::get_dummy_random_mle_vec,
};
use remainder_shared_types::{Field, Fr};
use utils::{DifferenceBuilderComponent, ProductScaledBuilderComponent};

pub mod utils;

/// A circuit which does the following:
/// * Layer 0: [ProductScaledBuilder] with `mle_1_vec`, `mle_2_vec`
/// * Layer 1: [ZeroBuilder] with the output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec`, `mle_2_vec` - inputs to [ProductScaledBuilder] both arbitrary bookkeeping
/// table values, same size.
///
/// * `num_dataparallel_bits` - The number of bits that represent which copy index the circuit is.

struct NonSelectorDataparallelComponent<F: Field> {
    first_layer_component: ProductScaledBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> NonSelectorDataparallelComponent<F> {
    /// A simple wrapper around the [TripleNestedBuilderComponent] which
    /// additionally contains a [DifferenceBuilderComponent] for zero output
    pub fn new(
        mle_1_input: &dyn CircuitNode,
        mle_2_input: &dyn CircuitNode,
    ) -> Self {
        let first_layer_component =
            ProductScaledBuilderComponent::new(mle_1_input, mle_2_input);

        let output_component =
            DifferenceBuilderComponent::new(&first_layer_component.get_output_sector());

        Self {
            first_layer_component,
            output_component,
        }
    }
}

impl<F: Field, N> Component<N> for NonSelectorDataparallelComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        self.first_layer_component
            .yield_nodes()
            .into_iter()
            .chain(self.output_component.yield_nodes())
            .collect_vec()
    }
}

/// Struct which allows for easy "semantic" feeding of inputs into the circuit.
struct DataparallelSelectorTestInputs<F: Field> {
    dataparallel_mle_1: MultilinearExtension<F>,
    dataparallel_mle_2: MultilinearExtension<F>,
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
fn build_dataparallel_simple_test_circuit<F: Field>(
    num_dataparallel_vars: usize,
    num_free_vars: usize,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(DataparallelSelectorTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
) {
    // --- All inputs are public ---
    let public_input_layer_node = InputLayerNode::new(None);

    // --- "Semantic" circuit inputs ---
    let dataparallel_mle_1_shred = InputShred::new(
        num_dataparallel_vars + num_free_vars,
        &public_input_layer_node,
    );
    let dataparallel_mle_2_shred = InputShred::new(
        num_dataparallel_vars + num_free_vars,
        &public_input_layer_node,
    );

    // --- Save IDs to be used later ---
    let dataparallel_mle_1_id = dataparallel_mle_1_shred.id();
    let dataparallel_mle_2_id = dataparallel_mle_2_shred.id();

    // --- Create the circuit components ---
    // Stack currently fails at layer 0, because expr and witgen for the first component is inconsistent.
    // But if you change from stack to interleave, then it fails at layer 1, because the subtraction of the dataparallel
    // mle from the output mle is not actually 0.
    let component_1 = NonSelectorDataparallelComponent::new(
        &dataparallel_mle_1_shred,
        &dataparallel_mle_2_shred,
    );

    let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
        public_input_layer_node.into(),
        dataparallel_mle_1_shred.into(),
        dataparallel_mle_2_shred.into(),
    ];
    all_circuit_nodes.extend(component_1.yield_nodes());

    let (circuit_description, convert_input_shreds_to_input_layers) =
        generate_circuit_description(all_circuit_nodes).unwrap();

    // --- Write closure which allows easy usage of circuit inputs ---
    let circuit_data_fn = move |test_inputs: DataparallelSelectorTestInputs<F>| {
        let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
            (dataparallel_mle_1_id, test_inputs.dataparallel_mle_1),
            (dataparallel_mle_2_id, test_inputs.dataparallel_mle_2),
        ]
        .into_iter()
        .collect();
        convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
    };

    (circuit_description, circuit_data_fn)
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
    let dataparallel_mle_1 = DenseMle::batch_mles(mle_1_vec).mle;
    let dataparallel_mle_2 = DenseMle::batch_mles(mle_2_vec).mle;

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn) =
        build_dataparallel_simple_test_circuit(NUM_DATAPARALLEL_VARS, NUM_FREE_VARS);

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(DataparallelSelectorTestInputs {
        dataparallel_mle_1,
        dataparallel_mle_2,
    });

    // --- Specify private input layers (+ description and precommit), if any ---
    let private_input_layers = HashMap::new();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}
