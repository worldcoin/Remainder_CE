use std::collections::HashMap;

use remainder::expression::abstract_expr::AbstractExpr;
use remainder::expression::generic_expr::Expression;
use remainder::input_layer::ligero_input_layer::LigeroInputLayerDescription;
use remainder::layer::LayerId;
use remainder::layouter::component::Component;
use remainder::layouter::nodes::circuit_inputs::{InputLayerNode, InputShred};
use remainder::layouter::nodes::circuit_outputs::OutputNode;
use remainder::layouter::nodes::node_enum::NodeEnum;
use remainder::layouter::nodes::sector::Sector;
use remainder::layouter::nodes::{CircuitNode, Context, NodeId};
use remainder::mle::evals::MultilinearExtension;
use remainder::prover::helpers::test_circuit_new;
use remainder::prover::{generate_circuit_description, GKRCircuitDescription};
use remainder_ligero::ligero_structs::LigeroAuxInfo;
use remainder_shared_types::{Field, Fr};

pub struct ProductCheckerComponent<F: Field> {
    pub sector: Sector<F>,
}

impl<F: Field> ProductCheckerComponent<F> {
    /// Checks that factor1 * factor2 - expected_product == 0.
    pub fn new(
        ctx: &Context,
        factor1: &dyn CircuitNode,
        factor2: &dyn CircuitNode,
        expected_product: &dyn CircuitNode,
    ) -> Self {
        let sector = Sector::new(ctx, &[factor1, factor2, expected_product], |input_nodes| {
            assert_eq!(input_nodes.len(), 3);
            Expression::<F, AbstractExpr>::products(vec![input_nodes[0], input_nodes[1]])
                - input_nodes[2].expr()
        });
        Self { sector }
    }
}

impl<F: Field, N> Component<N> for ProductCheckerComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}

/// Struct which allows for easy "semantic" feeding of inputs into the
/// linear/non-linear vars combined test circuit.
struct ProductCheckerTestInputs<F: Field> {
    mle_1: MultilinearExtension<F>,
    mle_2: MultilinearExtension<F>,
    mle_expected: MultilinearExtension<F>,
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
///
/// Note that this additionally returns the [LayerId] of the Ligero input layer!
fn build_product_checker_test_circuit<F: Field>(
    num_free_vars: usize,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(ProductCheckerTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
    LayerId,
) {
    // --- Create global context manager ---
    let context = Context::new();

    // --- The multiplicands are public... ---
    let public_input_layer_node = InputLayerNode::new(&context, None);
    let mle_1_shred = InputShred::new(&context, num_free_vars, &public_input_layer_node);
    let mle_2_shred = InputShred::new(&context, num_free_vars, &public_input_layer_node);

    // --- ...while the expected output is private ---
    let ligero_input_layer_node = InputLayerNode::new(&context, None);
    let mle_expected_shred = InputShred::new(&context, num_free_vars, &ligero_input_layer_node);

    // --- Save IDs to be used later ---
    let mle_1_id = mle_1_shred.id();
    let mle_2_id = mle_2_shred.id();
    let mle_expected_id = mle_expected_shred.id();
    let ligero_input_layer_node_id = ligero_input_layer_node.id();

    // --- Create the circuit components ---
    let checker =
        ProductCheckerComponent::new(&context, &mle_1_shred, &mle_2_shred, &mle_expected_shred);
    let output = OutputNode::new_zero(&context, &checker.sector);

    let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
        public_input_layer_node.into(),
        mle_1_shred.into(),
        mle_2_shred.into(),
        ligero_input_layer_node.into(),
        mle_expected_shred.into(),
        output.into(),
    ];
    all_circuit_nodes.extend(checker.yield_nodes());

    let (
        circuit_description,
        convert_input_shreds_to_input_layers,
        input_layer_node_ids_to_layer_ids,
    ) = generate_circuit_description(all_circuit_nodes).unwrap();

    // --- Write closure which allows easy usage of circuit inputs ---
    let circuit_data_fn = move |test_inputs: ProductCheckerTestInputs<F>| {
        let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
            (mle_1_id, test_inputs.mle_1),
            (mle_2_id, test_inputs.mle_2),
            (mle_expected_id, test_inputs.mle_expected),
        ]
        .into_iter()
        .collect();
        convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
    };

    // --- Grab Ligero input layer ID for later use ---
    let ligero_input_layer_id = *input_layer_node_ids_to_layer_ids
        .get(&ligero_input_layer_node_id)
        .unwrap();

    (circuit_description, circuit_data_fn, ligero_input_layer_id)
}

#[test]
fn test_product_checker() {
    const NUM_FREE_VARS: usize = 2;

    let mle_1 = MultilinearExtension::new(vec![
        Fr::from(3u64),
        Fr::from(2u64),
        Fr::from(3u64),
        Fr::from(2u64),
    ]);
    let mle_2 = MultilinearExtension::new(vec![
        Fr::from(5u64),
        Fr::from(6u64),
        Fr::from(5u64),
        Fr::from(6u64),
    ]);
    let mle_expected = MultilinearExtension::new(vec![
        Fr::from(15u64),
        Fr::from(12u64),
        Fr::from(15u64),
        Fr::from(12u64),
    ]);

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn, ligero_input_layer_id) =
        build_product_checker_test_circuit(NUM_FREE_VARS);

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(ProductCheckerTestInputs {
        mle_1,
        mle_2,
        mle_expected,
    });

    // --- Specify private input layers (+ description and precommit) ---
    let private_input_layers = vec![(
        ligero_input_layer_id,
        (
            LigeroInputLayerDescription {
                layer_id: ligero_input_layer_id,
                num_vars: NUM_FREE_VARS + 2,
                aux: LigeroAuxInfo::new(NUM_FREE_VARS + 2, 4, 1.0, None),
            },
            None,
        ),
    )]
    .into_iter()
    .collect();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}
