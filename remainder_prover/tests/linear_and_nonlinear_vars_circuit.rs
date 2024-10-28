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
use utils::{get_dummy_random_mle, DifferenceBuilderComponent};

pub mod utils;

/// A builder which returns the following expression:
/// `sel(mle_1, mle_1) + mle_2 * mle_2`
///
/// The idea is that the last bit in this expression is linear.
///
/// ## Arguments
/// * `sel_mle` - An MLE with arbitrary bookkeeping table values.
/// * `prod_mle` - An MLE with arbitrary bookkeeping table values; same size as `sel_mle`.

pub struct LastVarLinearBuilderComponent<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> LastVarLinearBuilderComponent<F> {
    pub fn new(sel_node: &dyn CircuitNode, prod_node: &dyn CircuitNode) -> Self {
        let last_bit_linear_sector = Sector::new(&[sel_node, prod_node], |input_nodes| {
            assert_eq!(input_nodes.len(), 2);
            let sel_mle = input_nodes[0];
            let prod_mle = input_nodes[1];

            let lhs_sum_expr = sel_mle.expr().select(sel_mle.expr());
            let rhs_sum_expr = ExprBuilder::<F>::products(vec![prod_mle, prod_mle]);
            lhs_sum_expr + rhs_sum_expr
        });

        Self {
            first_layer_sector: last_bit_linear_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: Field, N> Component<N> for LastVarLinearBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

/// A builder which returns the following expression:
/// `sel(mle_1 * mle_1, mle_1)`
///
/// The idea is that the first bit (selector bit) in this expression is linear.
///
/// ## Arguments
/// * `sel_mle` - An MLE with arbitrary bookkeeping table values.
pub struct FirstVarLinearBuilderComponent<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> FirstVarLinearBuilderComponent<F> {
    pub fn new(sel_node: &dyn CircuitNode) -> Self {
        let last_bit_linear_sector = Sector::new(&[sel_node], |input_nodes| {
            assert_eq!(input_nodes.len(), 1);
            let sel_mle = input_nodes[0];

            ExprBuilder::<F>::products(vec![sel_mle, sel_mle]).select(sel_mle.expr())
        });

        Self {
            first_layer_sector: last_bit_linear_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: Field, N> Component<N> for FirstVarLinearBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

/// Struct which allows for easy "semantic" feeding of inputs into the circuit.
struct LinearNonlinearVarsTestInputs<F: Field> {
    selector_mle: MultilinearExtension<F>,
    product_mle: MultilinearExtension<F>,
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
fn build_linear_and_nonlinear_vars_test_circuit<F: Field>(
    selector_mle_num_vars: usize,
    product_mle_num_vars: usize,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(LinearNonlinearVarsTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
) {
    // --- All inputs are public ---
    let public_input_layer_node = InputLayerNode::new(None);

    // --- "Semantic" circuit inputs ---
    let selector_mle_shred =
        InputShred::new(selector_mle_num_vars, &public_input_layer_node);
    let product_mle_shred =
        InputShred::new(product_mle_num_vars, &public_input_layer_node);

    // --- Save IDs to be used later ---
    let selector_mle_id = selector_mle_shred.id();
    let product_mle_id = product_mle_shred.id();

    // --- Create the circuit components ---
    let component_1 =
        LastVarLinearBuilderComponent::new(&selector_mle_shred, &product_mle_shred);
    let component_2 =
        FirstVarLinearBuilderComponent::new(&component_1.get_output_sector());
    let output_component =
        DifferenceBuilderComponent::new(&component_2.get_output_sector());

    let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
        public_input_layer_node.into(),
        selector_mle_shred.into(),
        product_mle_shred.into(),
    ];
    all_circuit_nodes.extend(component_1.yield_nodes());
    all_circuit_nodes.extend(component_2.yield_nodes());
    all_circuit_nodes.extend(output_component.yield_nodes());

    let (circuit_description, convert_input_shreds_to_input_layers, _) =
        generate_circuit_description(all_circuit_nodes).unwrap();

    // --- Write closure which allows easy usage of circuit inputs ---
    let circuit_data_fn = move |test_inputs: LinearNonlinearVarsTestInputs<F>| {
        let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
            (selector_mle_id, test_inputs.selector_mle),
            (product_mle_id, test_inputs.product_mle),
        ]
        .into_iter()
        .collect();
        convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
    };

    (circuit_description, circuit_data_fn)
}

/// A circuit which does the following:
/// * Layer 0: [LastVarLinearBuilderComponent] with `sel_mle`, `prod_mle`
/// * Layer 1: [FirstVarLinearBuilderComponent] with `sel_mle`
/// * Layer 2: [DifferenceBuilderComponent] with the output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `sel_mle`, `prod_mle` both MLEs with arbitrary bookkeeping table values, same size.

#[test]
fn test_linear_and_nonlinear_vars_circuit_newmainder() {
    const SELECTOR_MLE_NUM_VARS: usize = 2;
    const PRODUCT_MLE_NUM_VARS: usize = SELECTOR_MLE_NUM_VARS;
    let mut rng = test_rng();

    // --- Generate circuit inputs ---
    let selector_mle = get_dummy_random_mle(SELECTOR_MLE_NUM_VARS, &mut rng).mle;
    let product_mle = get_dummy_random_mle(PRODUCT_MLE_NUM_VARS, &mut rng).mle;

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn) =
        build_linear_and_nonlinear_vars_test_circuit(SELECTOR_MLE_NUM_VARS, PRODUCT_MLE_NUM_VARS);

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(LinearNonlinearVarsTestInputs {
        selector_mle,
        product_mle,
    });

    // --- Specify private input layers (+ description and precommit), if any ---
    let private_input_layers = HashMap::new();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}
