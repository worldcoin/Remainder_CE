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
use utils::DifferenceBuilderComponent;
use utils::{ProductScaledBuilderComponent, TripleNestedBuilderComponent};

pub mod utils;

struct DataparallelTripleNestedSelectorComponent<F: Field> {
    first_layer_component: TripleNestedBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> DataparallelTripleNestedSelectorComponent<F> {
    /// A simple wrapper around the [TripleNestedBuilderComponent] which
    /// additionally contains a [DifferenceBuilderComponent] for zero output
    pub fn new(
        mle_1_input: &dyn CircuitNode,
        mle_2_input: &dyn CircuitNode,
        mle_3_input: &dyn CircuitNode,
    ) -> Self {
        let first_layer_component =
            TripleNestedBuilderComponent::new(mle_1_input, mle_2_input, mle_3_input);

        let output_component =
            DifferenceBuilderComponent::new(&first_layer_component.get_output_sector());

        Self {
            first_layer_component,
            output_component,
        }
    }
}

impl<F: Field, N> Component<N> for DataparallelTripleNestedSelectorComponent<F>
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
    dataparallel_mle_3: MultilinearExtension<F>,
    dataparallel_mle_4: MultilinearExtension<F>,
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
fn build_dataparallel_selector_test_circuit<F: Field>(
    num_dataparallel_vars: usize,
    num_vars_mle_1_2: usize,
    num_vars_mle_3: usize,
    num_vars_mle_4: usize,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(DataparallelSelectorTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
) {
    // --- All inputs are public ---
    let public_input_layer_node = InputLayerNode::new(None);

    // --- Inputs to the circuit include the four dataparallel MLEs ---
    let dataparallel_mle_1_shred = InputShred::new(
        num_dataparallel_vars + num_vars_mle_1_2,
        &public_input_layer_node,
    );
    let dataparallel_mle_2_shred = InputShred::new(
        num_dataparallel_vars + num_vars_mle_1_2,
        &public_input_layer_node,
    );
    let dataparallel_mle_3_shred = InputShred::new(
        num_dataparallel_vars + num_vars_mle_3,
        &public_input_layer_node,
    );
    let dataparallel_mle_4_shred = InputShred::new(
        num_dataparallel_vars + num_vars_mle_4,
        &public_input_layer_node,
    );

    // --- Save IDs to be used later ---
    let dataparallel_mle_1_id = dataparallel_mle_1_shred.id();
    let dataparallel_mle_2_id = dataparallel_mle_2_shred.id();
    let dataparallel_mle_3_id = dataparallel_mle_3_shred.id();
    let dataparallel_mle_4_id = dataparallel_mle_4_shred.id();

    // --- Create the circuit components ---
    let component_1 = ProductScaledBuilderComponent::new(
        &dataparallel_mle_1_shred,
        &dataparallel_mle_2_shred,
    );
    let component_2 = DataparallelTripleNestedSelectorComponent::new(
        &component_1.get_output_sector(),
        &dataparallel_mle_3_shred,
        &dataparallel_mle_4_shred,
    );

    let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
        public_input_layer_node.into(),
        dataparallel_mle_1_shred.into(),
        dataparallel_mle_2_shred.into(),
        dataparallel_mle_3_shred.into(),
        dataparallel_mle_4_shred.into(),
    ];
    all_circuit_nodes.extend(component_1.yield_nodes());
    all_circuit_nodes.extend(component_2.yield_nodes());

    let (circuit_description, convert_input_shreds_to_input_layers) =
        generate_circuit_description(all_circuit_nodes).unwrap();

    // --- Write closure which allows easy usage of circuit inputs ---
    let circuit_data_fn = move |test_inputs: DataparallelSelectorTestInputs<F>| {
        let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
            (dataparallel_mle_1_id, test_inputs.dataparallel_mle_1),
            (dataparallel_mle_2_id, test_inputs.dataparallel_mle_2),
            (dataparallel_mle_3_id, test_inputs.dataparallel_mle_3),
            (dataparallel_mle_4_id, test_inputs.dataparallel_mle_4),
        ]
        .into_iter()
        .collect();
        convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
    };

    (circuit_description, circuit_data_fn)
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
    let dataparallel_mle_1 = DenseMle::batch_mles(mle_1_vec).mle;
    let dataparallel_mle_2 = DenseMle::batch_mles(mle_2_vec).mle;
    let dataparallel_mle_3 = DenseMle::batch_mles(mle_3_vec).mle;
    let dataparallel_mle_4 = DenseMle::batch_mles(mle_4_vec).mle;

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn) = build_dataparallel_selector_test_circuit(
        NUM_DATAPARALLEL_VARS,
        NUM_VARS_MLE_1_2,
        NUM_VARS_MLE_3,
        NUM_VARS_MLE_4,
    );

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(DataparallelSelectorTestInputs {
        dataparallel_mle_1,
        dataparallel_mle_2,
        dataparallel_mle_3,
        dataparallel_mle_4,
    });

    // --- Specify private input layers (+ description and precommit), if any ---
    let private_input_layers = HashMap::new();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}
