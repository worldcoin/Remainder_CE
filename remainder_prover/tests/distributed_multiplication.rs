use std::collections::HashMap;

use ark_std::test_rng;
use itertools::Itertools;
use remainder::{
    expression::abstract_expr::ExprBuilder,
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
};
use remainder_shared_types::{Field, Fr};
use utils::get_dummy_random_mle;
pub mod utils;

pub struct DataparallelDistributedMultiplication<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> DataparallelDistributedMultiplication<F> {
    pub fn new(smaller_mle: &dyn CircuitNode, bigger_mle: &dyn CircuitNode) -> Self {
        let combine_sector = Sector::new(&[smaller_mle, bigger_mle], |input_nodes| {
            assert_eq!(input_nodes.len(), 2);
            let smaller_mle_id = input_nodes[0];
            let bigger_mle_id = input_nodes[1];

            ExprBuilder::<F>::products(vec![bigger_mle_id, smaller_mle_id])
        });

        Self {
            first_layer_sector: combine_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: Field, N> Component<N> for DataparallelDistributedMultiplication<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

pub struct DiffTwoInputsBuilder<F: Field> {
    pub first_layer_sector: Sector<F>,
    pub output_sector: OutputNode,
}

impl<F: Field> DiffTwoInputsBuilder<F> {
    pub fn new(mle_1: &dyn CircuitNode, mle_2: &dyn CircuitNode) -> Self {
        let first_layer_sector = Sector::new(&[mle_1, mle_2], |input_nodes| {
            assert_eq!(input_nodes.len(), 2);
            let mle_1_id = input_nodes[0];
            let mle_2_id = input_nodes[1];

            mle_1_id.expr() - mle_2_id.expr()
        });

        let output_node = OutputNode::new_zero(&first_layer_sector);

        Self {
            first_layer_sector,
            output_sector: output_node,
        }
    }

    pub fn get_output_sector(&self) -> &OutputNode {
        &self.output_sector
    }
}

impl<F: Field, N> Component<N> for DiffTwoInputsBuilder<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into(), self.output_sector.into()]
    }
}

/// Struct which allows for easy "semantic" feeding of inputs into the circuit.
struct DataparallelWraparoundTestInputs<F: Field> {
    dataparallel_mle_smaller: MultilinearExtension<F>,
    dataparallel_mle_bigger: MultilinearExtension<F>,
    dataparallel_mle_combined: MultilinearExtension<F>,
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
fn build_dataparallel_wraparound_multiplication_test_circuit<F: Field>(
    num_dataparallel_vars: usize,
    num_free_vars_smaller: usize,
    num_free_vars_bigger: usize,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(DataparallelWraparoundTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
) {
    // --- All inputs are public ---
    let public_input_layer_node = InputLayerNode::new(None);

    // --- "Semantic" circuit inputs ---
    let dataparallel_mle_smaller_shred = InputShred::new(
        num_dataparallel_vars + num_free_vars_smaller,
        &public_input_layer_node,
    );
    let dataparallel_mle_bigger_shred = InputShred::new(
        num_dataparallel_vars + num_free_vars_bigger,
        &public_input_layer_node,
    );
    let dataparallel_mle_combined_shred = InputShred::new(
        num_dataparallel_vars + num_free_vars_bigger,
        &public_input_layer_node,
    );

    // --- Save IDs to be used later ---
    let dataparallel_mle_smaller_id = dataparallel_mle_smaller_shred.id();
    let dataparallel_mle_bigger_id = dataparallel_mle_bigger_shred.id();
    let dataparallel_mle_combined_id = dataparallel_mle_combined_shred.id();

    // --- Create the circuit components ---
    let component_1 = DataparallelDistributedMultiplication::new(
        &dataparallel_mle_smaller_shred,
        &dataparallel_mle_bigger_shred,
    );
    let component_2 = DiffTwoInputsBuilder::new(
        &component_1.get_output_sector(),
        &dataparallel_mle_combined_shred,
    );

    let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
        public_input_layer_node.into(),
        dataparallel_mle_smaller_shred.into(),
        dataparallel_mle_bigger_shred.into(),
        dataparallel_mle_combined_shred.into(),
    ];
    all_circuit_nodes.extend(component_1.yield_nodes());
    all_circuit_nodes.extend(component_2.yield_nodes());

    let (circuit_description, convert_input_shreds_to_input_layers, _) =
        generate_circuit_description(all_circuit_nodes).unwrap();

    // --- Write closure which allows easy usage of circuit inputs ---
    let circuit_data_fn = move |test_inputs: DataparallelWraparoundTestInputs<F>| {
        let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
            (
                dataparallel_mle_smaller_id,
                test_inputs.dataparallel_mle_smaller,
            ),
            (
                dataparallel_mle_bigger_id,
                test_inputs.dataparallel_mle_bigger,
            ),
            (
                dataparallel_mle_combined_id,
                test_inputs.dataparallel_mle_combined,
            ),
        ]
        .into_iter()
        .collect();
        convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
    };

    (circuit_description, circuit_data_fn)
}

#[test]
fn test_dataparallel_wraparound_multiplication_circuit() {
    const NUM_FREE_VARS_SMALLER: usize = 1;
    const NUM_FREE_VARS_BIGGER: usize = 2;
    const NUM_DATAPARALLEL_VARS: usize = 2;
    let mut rng = test_rng();

    let smaller_mles_vec: Vec<DenseMle<Fr>> = (0..(1 << NUM_DATAPARALLEL_VARS))
        .map(|_| get_dummy_random_mle(NUM_FREE_VARS_SMALLER, &mut rng))
        .collect_vec();

    let bigger_mles_vec: Vec<DenseMle<Fr>> = (0..(1 << NUM_DATAPARALLEL_VARS))
        .map(|_| get_dummy_random_mle(NUM_FREE_VARS_BIGGER, &mut rng))
        .collect_vec();

    let prod_mles = smaller_mles_vec
        .iter()
        .zip(bigger_mles_vec.iter())
        .map(|(small_mle, big_mle)| {
            let small_mle_bt_iter = small_mle.iter();
            let big_mle_bt_iter = big_mle.iter();
            let prod_bt = big_mle_bt_iter
                .zip(small_mle_bt_iter.cycle())
                .map(|(big_elem, small_elem)| big_elem * small_elem);
            DenseMle::new_from_iter(prod_bt, small_mle.layer_id)
        })
        .collect_vec();

    // --- Dataparallel-ize + grab inputs from above ---
    let dataparallel_mle_combined = DenseMle::batch_mles_lil(prod_mles).mle; // This works
                                                                             // let dataparallel_mle_combined = DenseMle::batch_mles(prod_mles); // This fails

    let dataparallel_mle_smaller = DenseMle::batch_mles_lil(smaller_mles_vec).mle; // This works
                                                                                   // let dataparallel_mle_smaller = DenseMle::batch_mles(smaller_mles_vec); // This fails

    let dataparallel_mle_bigger = DenseMle::batch_mles_lil(bigger_mles_vec).mle; // This works
                                                                                 // let dataparallel_mle_bigger = DenseMle::batch_mles(bigger_mles_vec); // This fails

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn) =
        build_dataparallel_wraparound_multiplication_test_circuit(
            NUM_DATAPARALLEL_VARS,
            NUM_FREE_VARS_SMALLER,
            NUM_FREE_VARS_BIGGER,
        );

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(DataparallelWraparoundTestInputs {
        dataparallel_mle_smaller,
        dataparallel_mle_bigger,
        dataparallel_mle_combined,
    });

    // --- Specify private input layers (+ description and precommit), if any ---
    let private_input_layers = HashMap::new();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}
