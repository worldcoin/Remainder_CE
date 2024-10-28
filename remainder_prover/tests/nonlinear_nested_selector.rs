use std::collections::HashMap;

use ark_std::test_rng;

use remainder::{
    expression::abstract_expr::ExprBuilder,
    layer::LayerId,
    layouter::{
        component::Component,
        nodes::{
            circuit_inputs::{InputLayerNode, InputShred},
            node_enum::NodeEnum,
            sector::Sector,
            CircuitNode, NodeId,
        },
    },
    mle::evals::MultilinearExtension,
    prover::{generate_circuit_description, helpers::test_circuit_new, GKRCircuitDescription},
};
use remainder_shared_types::Field;
use utils::get_dummy_random_mle;

use crate::utils::DifferenceBuilderComponent;
pub mod utils;

pub struct NonlinearNestedSelectorBuilderComponent<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> NonlinearNestedSelectorBuilderComponent<F> {
    /// A builder which returns the following expression:
    /// ```text
    /// sel(sel(left_inner_sel_mle, right_inner_sel_mle), right_outer_sel_mle)
    ///     + right_prod_mle_1 * right_prod_mle_2
    /// ```
    ///
    /// The idea is that this builder has two selector bits which are nonlinear.
    ///
    /// ## Arguments
    /// * `left_inner_sel_mle` - An MLE with arbitrary bookkeeping table values.
    /// * `right_inner_sel_mle` - An MLE with arbitrary bookkeeping table values, same size as `left_inner_sel_mle`.
    /// * `right_outer_sel_mle` - An MLE with arbitrary bookkeeping table values, one more variable
    /// than `right_inner_sel_mle`.
    /// * `right_prod_mle_1`, `right_prod_mle_2` - MLEs with arbitrary bookkeeping table values, same size,
    /// one more variable than `right_outer_sel_mle`.
    pub fn new(
        left_inner_sel_mle: &dyn CircuitNode,
        right_inner_sel_mle: &dyn CircuitNode,
        right_outer_sel_mle: &dyn CircuitNode,
        right_prod_mle_1: &dyn CircuitNode,
        right_prod_mle_2: &dyn CircuitNode,
    ) -> Self {
        let nonlinear_nested_selector_sector = Sector::new(
            &[
                left_inner_sel_mle,
                right_inner_sel_mle,
                right_outer_sel_mle,
                right_prod_mle_1,
                right_prod_mle_2,
            ],
            |nonlinear_nested_selector_nodes| {
                assert_eq!(nonlinear_nested_selector_nodes.len(), 5);

                let left_inner_sel_mle_id = nonlinear_nested_selector_nodes[0];
                let right_inner_sel_mle_id = nonlinear_nested_selector_nodes[1];
                let right_outer_sel_mle_id = nonlinear_nested_selector_nodes[2];
                let right_prod_mle_1_id = nonlinear_nested_selector_nodes[3];
                let right_prod_mle_2_id = nonlinear_nested_selector_nodes[4];

                let left_inner_sel_side = ExprBuilder::<F>::mle(left_inner_sel_mle_id);
                let right_inner_sel_side = ExprBuilder::<F>::mle(right_inner_sel_mle_id);
                let left_outer_sel_side = left_inner_sel_side.select(right_inner_sel_side);
                let left_sum_side =
                    left_outer_sel_side.select(ExprBuilder::<F>::mle(right_outer_sel_mle_id));
                let right_sum_side =
                    ExprBuilder::<F>::products(vec![right_prod_mle_1_id, right_prod_mle_2_id]);
                left_sum_side + right_sum_side
            },
        );

        Self {
            first_layer_sector: nonlinear_nested_selector_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: Field, N> Component<N> for NonlinearNestedSelectorBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

/// Struct which allows for easy "semantic" feeding of inputs into the circuit.
struct NonlinearNestedSelectorTestInputs<F: Field> {
    left_inner_sel_mle: MultilinearExtension<F>,
    right_inner_sel_mle: MultilinearExtension<F>,
    right_outer_sel_mle: MultilinearExtension<F>,
    right_prod_mle_1: MultilinearExtension<F>,
    right_prod_mle_2: MultilinearExtension<F>,
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
fn build_nonlinear_nested_sel_test_circuit<F: Field>(
    num_vars_product_side: usize,
    num_vars_outer_sel_side: usize,
    num_vars_inner_sel_side: usize,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(NonlinearNestedSelectorTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
) {
    // --- All inputs are public ---
    let public_input_layer_node = InputLayerNode::new(None);

    // --- Inputs to the circuit are the MLEs at the leaves of the nested selector + product ---
    let left_inner_sel_mle_shred =
        InputShred::new(num_vars_inner_sel_side, &public_input_layer_node);
    let right_inner_sel_mle_shred =
        InputShred::new(num_vars_inner_sel_side, &public_input_layer_node);
    let right_outer_sel_mle_shred =
        InputShred::new(num_vars_outer_sel_side, &public_input_layer_node);
    let right_prod_mle_1_shred =
        InputShred::new(num_vars_product_side, &public_input_layer_node);
    let right_prod_mle_2_shred =
        InputShred::new(num_vars_product_side, &public_input_layer_node);

    // --- Save IDs to be used later ---
    let left_inner_sel_mle_id = left_inner_sel_mle_shred.id();
    let right_inner_sel_mle_id = right_inner_sel_mle_shred.id();
    let right_outer_sel_mle_id = right_outer_sel_mle_shred.id();
    let right_prod_mle_1_id = right_prod_mle_1_shred.id();
    let right_prod_mle_2_id = right_prod_mle_2_shred.id();

    // --- Create the circuit components ---
    let component_1 = NonlinearNestedSelectorBuilderComponent::new(
        &left_inner_sel_mle_shred,
        &right_inner_sel_mle_shred,
        &right_outer_sel_mle_shred,
        &right_prod_mle_1_shred,
        &right_prod_mle_2_shred,
    );
    let component_2 = DifferenceBuilderComponent::new(&component_1.get_output_sector());

    let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
        public_input_layer_node.into(),
        left_inner_sel_mle_shred.into(),
        right_inner_sel_mle_shred.into(),
        right_outer_sel_mle_shred.into(),
        right_prod_mle_1_shred.into(),
        right_prod_mle_2_shred.into(),
    ];
    all_circuit_nodes.extend(component_1.yield_nodes());
    all_circuit_nodes.extend(component_2.yield_nodes());

    let (circuit_description, convert_input_shreds_to_input_layers) =
        generate_circuit_description(all_circuit_nodes).unwrap();

    // --- Write closure which allows easy usage of circuit inputs ---
    let circuit_data_fn = move |test_inputs: NonlinearNestedSelectorTestInputs<F>| {
        let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
            (left_inner_sel_mle_id, test_inputs.left_inner_sel_mle),
            (right_inner_sel_mle_id, test_inputs.right_inner_sel_mle),
            (right_outer_sel_mle_id, test_inputs.right_outer_sel_mle),
            (right_prod_mle_1_id, test_inputs.right_prod_mle_1),
            (right_prod_mle_2_id, test_inputs.right_prod_mle_2),
        ]
        .into_iter()
        .collect();
        convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
    };

    (circuit_description, circuit_data_fn)
}

/// A circuit which does the following:
/// * Layer 0: [NonlinearNestedSelectorBuilderComponent] with all inputs.
/// * Layer 1: [DifferenceBuilderComponent] with output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// See [NonlinearNestedSelectorBuilderComponent].
#[test]
fn test_nonlinear_nested_sel_circuit_newmainder() {
    const NUM_VARS_PRODUCT_SIDE: usize = 5;
    const NUM_VARS_OUTER_SEL_SIDE: usize = NUM_VARS_PRODUCT_SIDE - 1;
    const NUM_VARS_INNER_SEL_SIDE: usize = NUM_VARS_OUTER_SEL_SIDE - 1;
    let mut rng = test_rng();

    let left_inner_sel_mle = get_dummy_random_mle(NUM_VARS_INNER_SEL_SIDE, &mut rng).mle;
    let right_inner_sel_mle = get_dummy_random_mle(NUM_VARS_INNER_SEL_SIDE, &mut rng).mle;
    let right_outer_sel_mle = get_dummy_random_mle(NUM_VARS_OUTER_SEL_SIDE, &mut rng).mle;
    let right_prod_mle_1 = get_dummy_random_mle(NUM_VARS_PRODUCT_SIDE, &mut rng).mle;
    let right_prod_mle_2 = get_dummy_random_mle(NUM_VARS_PRODUCT_SIDE, &mut rng).mle;

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn) = build_nonlinear_nested_sel_test_circuit(
        NUM_VARS_PRODUCT_SIDE,
        NUM_VARS_OUTER_SEL_SIDE,
        NUM_VARS_INNER_SEL_SIDE,
    );

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(NonlinearNestedSelectorTestInputs {
        left_inner_sel_mle,
        right_inner_sel_mle,
        right_outer_sel_mle,
        right_prod_mle_1,
        right_prod_mle_2,
    });

    // --- Specify private input layers (+ description and precommit), if any ---
    let private_input_layers = HashMap::new();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}
