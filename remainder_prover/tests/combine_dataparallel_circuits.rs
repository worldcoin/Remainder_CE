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
    utils::mle::get_random_mle,
};
use remainder_shared_types::{Field, Fr};
use utils::{
    ConstantScaledSumBuilderComponent, DifferenceBuilderComponent, ProductScaledBuilderComponent,
    ProductSumBuilderComponent,
};

pub mod utils;

struct DataParallelConstantScaledCircuitAltComponent<F: Field> {
    first_layer_component: ConstantScaledSumBuilderComponent<F>,
    second_layer_component: ProductScaledBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> DataParallelConstantScaledCircuitAltComponent<F> {
    /// A circuit which takes in two vectors of MLEs of the same size:
    /// * Layer 0: [ConstantScaledSumBuilderComponent] with the two inputs
    /// * Layer 1: [ProductScaledBuilderComponent] with the output of Layer 0 and `mle_1_vec`
    /// * Layer 2: [DifferenceBuilderComponent] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
    pub fn new(mle_1_input: &dyn CircuitNode, mle_2_input: &dyn CircuitNode) -> Self {
        let first_layer_component =
            ConstantScaledSumBuilderComponent::new(mle_1_input, mle_2_input);

        let second_layer_component = ProductScaledBuilderComponent::new(
            &first_layer_component.get_output_sector(),
            mle_1_input,
        );

        let output_component =
            DifferenceBuilderComponent::new(&second_layer_component.get_output_sector());

        Self {
            first_layer_component,
            second_layer_component,
            output_component,
        }
    }
}

impl<F: Field, N> Component<N> for DataParallelConstantScaledCircuitAltComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        self.first_layer_component
            .yield_nodes()
            .into_iter()
            .chain(self.second_layer_component.yield_nodes())
            .chain(self.output_component.yield_nodes())
            .collect_vec()
    }
}

struct DataParallelSumConstantCircuitAltComponent<F: Field> {
    first_layer_component: ProductSumBuilderComponent<F>,
    second_layer_component: ConstantScaledSumBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> DataParallelSumConstantCircuitAltComponent<F> {
    /// A circuit which takes in two vectors of MLEs of the same size:
    /// * Layer 0: [ProductSumBuilderComponent] with the two inputs
    /// * Layer 1: [ConstantScaledSumBuilderComponent] with the output of Layer 0 and `mle_1_vec`
    /// * Layer 2: [DifferenceBuilderComponent] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
    pub fn new(mle_1_input: &dyn CircuitNode, mle_2_input: &dyn CircuitNode) -> Self {
        let first_layer_component = ProductSumBuilderComponent::new(mle_1_input, mle_2_input);

        let second_layer_component = ConstantScaledSumBuilderComponent::new(
            &first_layer_component.get_output_sector(),
            mle_1_input,
        );

        let output_component =
            DifferenceBuilderComponent::new(&second_layer_component.get_output_sector());

        Self {
            first_layer_component,
            second_layer_component,
            output_component,
        }
    }
}

impl<F: Field, N> Component<N> for DataParallelSumConstantCircuitAltComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        self.first_layer_component
            .yield_nodes()
            .into_iter()
            .chain(self.second_layer_component.yield_nodes())
            .chain(self.output_component.yield_nodes())
            .collect_vec()
    }
}

struct DataParallelProductScaledSumCircuitAltComponent<F: Field> {
    first_layer_component: ProductScaledBuilderComponent<F>,
    second_layer_component: ProductSumBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> DataParallelProductScaledSumCircuitAltComponent<F> {
    /// A circuit which takes in two vectors of MLEs of the same size:
    /// * Layer 0: [ProductScaledBuilderComponent] with the two inputs
    /// * Layer 1: [ProductSumBuilderComponent] with the output of Layer 0 and `mle_1_vec`
    /// * Layer 2: [DifferenceBuilderComponent] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
    pub fn new(mle_1_input: &dyn CircuitNode, mle_2_input: &dyn CircuitNode) -> Self {
        let first_layer_component = ProductScaledBuilderComponent::new(mle_1_input, mle_2_input);

        let second_layer_component = ProductSumBuilderComponent::new(
            &first_layer_component.get_output_sector(),
            mle_1_input,
        );

        let output_component =
            DifferenceBuilderComponent::new(&second_layer_component.get_output_sector());

        Self {
            first_layer_component,
            second_layer_component,
            output_component,
        }
    }
}

impl<F: Field, N> Component<N> for DataParallelProductScaledSumCircuitAltComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        self.first_layer_component
            .yield_nodes()
            .into_iter()
            .chain(self.second_layer_component.yield_nodes())
            .chain(self.output_component.yield_nodes())
            .collect_vec()
    }
}

/// Struct which allows for easy "semantic" feeding of inputs into the
/// test identity gate circuit.
struct CombinedDataparallelTestInputs<F: Field> {
    dataparallel_mle_1: MultilinearExtension<F>,
    dataparallel_mle_2: MultilinearExtension<F>,
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the combined dataparallel circuit.
fn build_combined_dataparallel_test_circuit<F: Field>(
    num_dataparallel_vars: usize,
    mle_1_and_2_num_vars: usize,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(CombinedDataparallelTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
) {
    // --- All inputs are public inputs ---
    let public_input_layer_node = InputLayerNode::new(None);

    // --- Inputs to the circuit include the "dataparallel MLE 1" and the "dataparallel MLE 2" ---
    let dataparallel_mle_1_shred = InputShred::new(
        num_dataparallel_vars + mle_1_and_2_num_vars,
        &public_input_layer_node,
    );
    let dataparallel_mle_2_shred = InputShred::new(
        num_dataparallel_vars + mle_1_and_2_num_vars,
        &public_input_layer_node,
    );

    // --- Save IDs to be used later ---
    let dataparallel_mle_1_id = dataparallel_mle_1_shred.id();
    let dataparallel_mle_2_id = dataparallel_mle_2_shred.id();

    // --- Create the circuit components ---
    let component_1 = DataParallelProductScaledSumCircuitAltComponent::new(
        &dataparallel_mle_1_shred,
        &dataparallel_mle_2_shred,
    );
    let component_2 = DataParallelSumConstantCircuitAltComponent::new(
        &dataparallel_mle_1_shred,
        &dataparallel_mle_2_shred,
    );
    let component_3 = DataParallelConstantScaledCircuitAltComponent::new(
        &dataparallel_mle_1_shred,
        &dataparallel_mle_2_shred,
    );

    let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
        public_input_layer_node.into(),
        dataparallel_mle_1_shred.into(),
        dataparallel_mle_2_shred.into(),
    ];
    all_circuit_nodes.extend(component_1.yield_nodes());
    all_circuit_nodes.extend(component_2.yield_nodes());
    all_circuit_nodes.extend(component_3.yield_nodes());

    let (circuit_description, convert_input_shreds_to_input_layers) =
        generate_circuit_description(all_circuit_nodes).unwrap();

    // --- Write closure which allows easy usage of circuit inputs ---
    let circuit_data_fn =
        move |combined_dataparallel_test_inputs: CombinedDataparallelTestInputs<F>| {
            let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
                (
                    dataparallel_mle_1_id,
                    combined_dataparallel_test_inputs.dataparallel_mle_1,
                ),
                (
                    dataparallel_mle_2_id,
                    combined_dataparallel_test_inputs.dataparallel_mle_2,
                ),
            ]
            .into_iter()
            .collect();
            convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
        };

    (circuit_description, circuit_data_fn)
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

    let mle_1_vec_batched = DenseMle::batch_mles(mle_1_vec.clone());
    let mle_2_vec_batched = DenseMle::batch_mles(mle_2_vec.clone());

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

    // --- Pull input data from above combination process ---
    let dataparallel_mle_1 = mle_1_vec_batched.mle;
    let dataparallel_mle_2 = mle_2_vec_batched.mle;

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn) =
        build_combined_dataparallel_test_circuit(NUM_DATAPARALLEL_VARS, VARS_MLE_1_2);

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(CombinedDataparallelTestInputs {
        dataparallel_mle_1,
        dataparallel_mle_2,
    });

    // --- Specify private input layers (+ description and precommit), if any ---
    let private_input_layers = HashMap::new();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}
