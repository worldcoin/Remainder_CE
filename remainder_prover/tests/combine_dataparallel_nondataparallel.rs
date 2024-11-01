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
    mle::{dense::DenseMle, evals::MultilinearExtension},
    prover::{generate_circuit_description, helpers::test_circuit_new, GKRCircuitDescription},
};
use remainder_shared_types::Field;

pub mod utils;

use utils::{
    DifferenceBuilderComponent, ProductScaledBuilderComponent, TripleNestedBuilderComponent,
};

use crate::utils::get_dummy_random_mle;

struct DataParallelComponent<F: Field> {
    first_layer_component: ProductScaledBuilderComponent<F>,
    second_layer_component: ProductScaledBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> DataParallelComponent<F> {
    /// A circuit which takes in two vectors of MLEs of the same size:
    /// * Layer 0: [ProductScaledBuilderComponent] with the two inputs
    /// * Layer 1: [ProductScaledBuilderComponent] with the output of Layer 0 and output of Layer 0.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
    pub fn new(mle_1_input: &dyn CircuitNode, mle_2_input: &dyn CircuitNode) -> Self {
        let product_scaled_component = ProductScaledBuilderComponent::new(mle_1_input, mle_2_input);

        let product_scaled_meta_component = ProductScaledBuilderComponent::new(
            &product_scaled_component.get_output_sector(),
            &product_scaled_component.get_output_sector(),
        );
        let output_component =
            DifferenceBuilderComponent::new(&product_scaled_meta_component.get_output_sector());

        Self {
            first_layer_component: product_scaled_component,
            second_layer_component: product_scaled_meta_component,
            output_component,
        }
    }
}

impl<F: Field, N> Component<N> for DataParallelComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        self.first_layer_component
            .yield_nodes()
            .into_iter()
            .chain(
                self.second_layer_component
                    .yield_nodes()
                    .into_iter()
                    .chain(self.output_component.yield_nodes()),
            )
            .collect_vec()
    }
}

struct TripleNestedSelectorComponent<F: Field> {
    first_layer_component: TripleNestedBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> TripleNestedSelectorComponent<F> {
    /// A circuit in which:
    /// * Layer 0: [TripleNestedSelectorBuilder] with the three inputs
    /// * Layer 1: [ZeroBuilder] with output of Layer 0 and itself.
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
        inner_inner_sel: &dyn CircuitNode,
        inner_sel: &dyn CircuitNode,
        outer_sel: &dyn CircuitNode,
    ) -> Self {
        let triple_nested_selector_component =
            TripleNestedBuilderComponent::new(inner_inner_sel, inner_sel, outer_sel);
        let output_component =
            DifferenceBuilderComponent::new(&triple_nested_selector_component.get_output_sector());

        Self {
            first_layer_component: triple_nested_selector_component,
            output_component,
        }
    }
}

impl<F: Field, N> Component<N> for TripleNestedSelectorComponent<F>
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

struct ScaledProductComponent<F: Field> {
    first_layer_component: ProductScaledBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> ScaledProductComponent<F> {
    /// A circuit in which:
    /// * Layer 0: [ProductScaledBuilderComponent] with the two inputs
    /// * Layer 1: [DifferenceBuilderComponent] with output of Layer 0 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1` - An MLE with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE with arbitrary bookkeeping table values, same size as `mle_1`.
    pub fn new(mle_1_input: &dyn CircuitNode, mle_2_input: &dyn CircuitNode) -> Self {
        let product_scaled_component = ProductScaledBuilderComponent::new(mle_1_input, mle_2_input);

        let output_component =
            DifferenceBuilderComponent::new(&product_scaled_component.get_output_sector());

        Self {
            first_layer_component: product_scaled_component,
            output_component,
        }
    }
}

impl<F: Field, N> Component<N> for ScaledProductComponent<F>
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
struct CombinedDataparallelNondataparallelTestInputs<F: Field> {
    dataparallel_mle_1: MultilinearExtension<F>,
    dataparallel_mle_2: MultilinearExtension<F>,
    mle_3: MultilinearExtension<F>,
    mle_4: MultilinearExtension<F>,
    mle_5: MultilinearExtension<F>,
    mle_6: MultilinearExtension<F>,
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the combined
/// dataparallel/non-dataparallel circuit.
fn build_combined_dataparallel_nondataparallel_test_circuit<F: Field>(
    num_dataparallel_vars: usize,
    mle_1_2_3_4_num_vars: usize,
    mle_5_num_vars: usize,
    mle_6_num_vars: usize,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(
        CombinedDataparallelNondataparallelTestInputs<F>,
    ) -> HashMap<LayerId, MultilinearExtension<F>>,
) {
    // --- All inputs are public ---
    let public_input_layer_node = InputLayerNode::new(None);

    // --- "Semantic" circuit inputs ---
    let dataparallel_mle_1_shred = InputShred::new(
        num_dataparallel_vars + mle_1_2_3_4_num_vars,
        &public_input_layer_node,
    );
    let dataparallel_mle_2_shred = InputShred::new(
        num_dataparallel_vars + mle_1_2_3_4_num_vars,
        &public_input_layer_node,
    );
    let mle_3_shred = InputShred::new(mle_1_2_3_4_num_vars, &public_input_layer_node);
    let mle_4_shred = InputShred::new(mle_1_2_3_4_num_vars, &public_input_layer_node);
    let mle_5_shred = InputShred::new(mle_5_num_vars, &public_input_layer_node);
    let mle_6_shred = InputShred::new(mle_6_num_vars, &public_input_layer_node);

    // --- Save IDs to be used later ---
    let dataparallel_mle_1_id = dataparallel_mle_1_shred.id();
    let dataparallel_mle_2_id = dataparallel_mle_2_shred.id();
    let mle_3_id = mle_3_shred.id();
    let mle_4_id = mle_4_shred.id();
    let mle_5_id = mle_5_shred.id();
    let mle_6_id = mle_6_shred.id();

    // --- Create the circuit components ---
    let component_1 =
        DataParallelComponent::new(&dataparallel_mle_1_shred, &dataparallel_mle_2_shred);
    let component_2 = TripleNestedSelectorComponent::new(&mle_4_shred, &mle_5_shred, &mle_6_shred);
    let component_3 = ScaledProductComponent::new(&mle_3_shred, &mle_4_shred);

    let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
        public_input_layer_node.into(),
        dataparallel_mle_1_shred.into(),
        dataparallel_mle_2_shred.into(),
        mle_3_shred.into(),
        mle_4_shred.into(),
        mle_5_shred.into(),
        mle_6_shred.into(),
    ];
    all_circuit_nodes.extend(component_1.yield_nodes());
    all_circuit_nodes.extend(component_2.yield_nodes());
    all_circuit_nodes.extend(component_3.yield_nodes());

    let (circuit_description, convert_input_shreds_to_input_layers) =
        generate_circuit_description(all_circuit_nodes).unwrap();

    // --- Write closure which allows easy usage of circuit inputs ---
    let circuit_data_fn =
        move |combined_dataparallel_nondataparallel_test_inputs: CombinedDataparallelNondataparallelTestInputs<
            F,
        >| {
            let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
                (
                    dataparallel_mle_1_id,
                    combined_dataparallel_nondataparallel_test_inputs.dataparallel_mle_1,
                ),
                (
                    dataparallel_mle_2_id,
                    combined_dataparallel_nondataparallel_test_inputs.dataparallel_mle_2,
                ),
                (mle_3_id, combined_dataparallel_nondataparallel_test_inputs.mle_3),
                (mle_4_id, combined_dataparallel_nondataparallel_test_inputs.mle_4),
                (mle_5_id, combined_dataparallel_nondataparallel_test_inputs.mle_5),
                (mle_6_id, combined_dataparallel_nondataparallel_test_inputs.mle_6),
            ]
            .into_iter()
            .collect();
            convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
        };

    (circuit_description, circuit_data_fn)
}

/// A circuit which combines the [DataParallelComponent], [TripleNestedSelectorComponent],
/// and [ScaledProductComponent].
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec`, `mle_2_vec` - inputs to [DataParallelComponent] both arbitrary bookkeeping
/// table values, same size.
/// * `mle_4`, `mle_5`, `mle_6` - inputs to [TripleNestedSelectorComponent], `mle_4` has the same
/// size as the mles in `mle_1_vec`, arbitrary bookkeeping table values. `mle_5` has one more
/// variable than `mle_4`, `mle_6` has one more variable than `mle_5`, both arbitrary bookkeeping
/// table values.
/// * `mle_3`, `mle_4` - inputs to [ScaledProductComponent], both arbitrary bookkeeping table values,
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

    let mle_1_vec_batched = DenseMle::batch_mles(mle_1_vec);
    let mle_2_vec_batched = DenseMle::batch_mles(mle_2_vec);

    // --- Grab inputs from the above ---
    let dataparallel_mle_1 = mle_1_vec_batched.mle;
    let dataparallel_mle_2 = mle_2_vec_batched.mle;
    let mle_3 = get_dummy_random_mle(VARS_MLE_1_2_3_4, &mut rng).mle;
    let mle_4 = get_dummy_random_mle(VARS_MLE_1_2_3_4, &mut rng).mle;
    let mle_5 = get_dummy_random_mle(VARS_MLE_5, &mut rng).mle;
    let mle_6 = get_dummy_random_mle(VARS_MLE_6, &mut rng).mle;

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn) =
        build_combined_dataparallel_nondataparallel_test_circuit(
            NUM_DATAPARALLEL_VARS,
            VARS_MLE_1_2_3_4,
            VARS_MLE_5,
            VARS_MLE_6,
        );

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(CombinedDataparallelNondataparallelTestInputs {
        dataparallel_mle_1,
        dataparallel_mle_2,
        mle_3,
        mle_4,
        mle_5,
        mle_6,
    });

    // --- Specify private input layers (+ description and precommit), if any ---
    let private_input_layers = HashMap::new();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}
