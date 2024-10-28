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
    mle::evals::MultilinearExtension,
    prover::{generate_circuit_description, helpers::test_circuit_new, GKRCircuitDescription},
};
use remainder_shared_types::Field;

use utils::{
    get_dummy_random_mle, ConstantScaledSumBuilderComponent, DifferenceBuilderComponent,
    ProductScaledBuilderComponent, ProductSumBuilderComponent,
};

pub mod utils;

struct ConstantScaledCircuitComponent<F: Field> {
    first_layer_component: ConstantScaledSumBuilderComponent<F>,
    second_layer_component: ProductScaledBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> ConstantScaledCircuitComponent<F> {
    /// A circuit which takes in two MLEs of the same size:
    /// * Layer 0: [ConstantScaledSumBuilderComponent] with the two inputs
    /// * Layer 1: [ProductScaledBuilderComponent] with the output of Layer 0 and `mle_1`
    /// * Layer 2: [DifferenceBuilderComponent] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1`.
    pub fn new(
        mle_1_input: &dyn CircuitNode,
        mle_2_input: &dyn CircuitNode,
    ) -> Self {
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

impl<F: Field, N> Component<N> for ConstantScaledCircuitComponent<F>
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

struct SumConstantCircuitComponent<F: Field> {
    first_layer_component: ProductSumBuilderComponent<F>,
    second_layer_component: ConstantScaledSumBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> SumConstantCircuitComponent<F> {
    /// A circuit which takes in two MLEs of the same size:
    /// * Layer 0: [ProductSumBuilderComponent] with the two inputs
    /// * Layer 1: [ConstantScaledSumBuilderComponent] with the output of Layer 0 and `mle_1`
    /// * Layer 2: [DifferenceBuilderComponent] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1`.
    pub fn new(
        mle_1_input: &dyn CircuitNode,
        mle_2_input: &dyn CircuitNode,
    ) -> Self {
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

impl<F: Field, N> Component<N> for SumConstantCircuitComponent<F>
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

struct ProductScaledSumCircuitComponent<F: Field> {
    first_layer_component: ProductScaledBuilderComponent<F>,
    second_layer_component: ProductSumBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> ProductScaledSumCircuitComponent<F> {
    /// A circuit which takes in two MLEs of the same size:
    /// * Layer 0: [ProductScaledBuilderComponent] with the two inputs
    /// * Layer 1: [ProductSumBuilderComponent] with the output of Layer 0 and `mle_1`
    /// * Layer 2: [DifferenceBuilderComponent] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1`  An MLE with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE with arbitrary bookkeeping table values, same size as `mle_1`.
    pub fn new(
        mle_1_input: &dyn CircuitNode,
        mle_2_input: &dyn CircuitNode,
    ) -> Self {
        let first_layer_component =
            ProductScaledBuilderComponent::new(mle_1_input, mle_2_input);

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

impl<F: Field, N> Component<N> for ProductScaledSumCircuitComponent<F>
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

/// Struct which allows for easy "semantic" feeding of inputs into the circuit.
struct CombinedNondataparallelTestInputs<F: Field> {
    mle_1: MultilinearExtension<F>,
    mle_2: MultilinearExtension<F>,
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the combined
/// dataparallel/non-dataparallel circuit.
fn build_combined_nondataparallel_circuit<F: Field>(
    mle_1_2_vars: usize,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(CombinedNondataparallelTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
) {
    // --- All inputs are public ---
    let public_input_layer_node = InputLayerNode::new(None);

    // --- "Semantic" circuit inputs ---
    let mle_1_shred = InputShred::new(mle_1_2_vars, &public_input_layer_node);
    let mle_2_shred = InputShred::new(mle_1_2_vars, &public_input_layer_node);

    // --- Save IDs to be used later ---
    let mle_1_id = mle_1_shred.id();
    let mle_2_id = mle_2_shred.id();

    // --- Create the circuit components ---
    let component_1 = ProductScaledSumCircuitComponent::new(&mle_1_shred, &mle_2_shred);
    let component_2 = SumConstantCircuitComponent::new(&mle_1_shred, &mle_2_shred);
    let component_3 = ConstantScaledCircuitComponent::new(&mle_1_shred, &mle_2_shred);

    let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
        public_input_layer_node.into(),
        mle_1_shred.into(),
        mle_2_shred.into(),
    ];
    all_circuit_nodes.extend(component_1.yield_nodes());
    all_circuit_nodes.extend(component_2.yield_nodes());
    all_circuit_nodes.extend(component_3.yield_nodes());

    let (circuit_description, convert_input_shreds_to_input_layers) =
        generate_circuit_description(all_circuit_nodes).unwrap();

    // --- Write closure which allows easy usage of circuit inputs ---
    let circuit_data_fn =
        move |combined_nondataparallel_test_inputs: CombinedNondataparallelTestInputs<F>| {
            let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
                (mle_1_id, combined_nondataparallel_test_inputs.mle_1),
                (mle_2_id, combined_nondataparallel_test_inputs.mle_2),
            ]
            .into_iter()
            .collect();
            convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
        };

    (circuit_description, circuit_data_fn)
}

#[test]
fn test_combined_nondataparallel_circuit_newmainder() {
    const VARS_MLE_1_2: usize = 2;
    let mut rng = test_rng();

    // --- Generate circuit inputs ---
    let mle_1 = get_dummy_random_mle(VARS_MLE_1_2, &mut rng).mle;
    let mle_2 = get_dummy_random_mle(VARS_MLE_1_2, &mut rng).mle;

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn) =
        build_combined_nondataparallel_circuit(VARS_MLE_1_2);

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(CombinedNondataparallelTestInputs { mle_1, mle_2 });

    // --- Specify private input layers (+ description and precommit), if any ---
    let private_input_layers = HashMap::new();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}
