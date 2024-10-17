use std::collections::HashMap;

use ark_std::test_rng;
use remainder::{
    layer::LayerId,
    layouter::{
        component::Component,
        nodes::{
            circuit_inputs::{InputLayerNode, InputShred},
            circuit_outputs::OutputNode,
            node_enum::NodeEnum,
            sector::Sector,
            CircuitNode, Context, NodeId,
        },
    },
    mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
    prover::{generate_circuit_description, helpers::test_circuit_new, GKRCircuitDescription},
};
use remainder_shared_types::{Field, Fr};
use utils::get_dummy_random_mle;
pub mod utils;

pub struct DataParallelRecombinationInterleaveBuilder<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> DataParallelRecombinationInterleaveBuilder<F> {
    pub fn new(
        ctx: &Context,
        mle_1: &dyn CircuitNode,
        mle_2: &dyn CircuitNode,
        mle_3: &dyn CircuitNode,
        mle_4: &dyn CircuitNode,
    ) -> Self {
        let combine_sector = Sector::new(ctx, &[mle_1, mle_2, mle_3, mle_4], |input_nodes| {
            assert_eq!(input_nodes.len(), 4);
            let mle_1_id = input_nodes[0];
            let mle_2_id = input_nodes[1];
            let mle_3_id = input_nodes[2];
            let mle_4_id = input_nodes[3];

            let lhs = mle_1_id.expr().select(mle_2_id.expr());
            let rhs = mle_3_id.expr().select(mle_4_id.expr());

            lhs.select(rhs)
        });

        Self {
            first_layer_sector: combine_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: Field, N> Component<N> for DataParallelRecombinationInterleaveBuilder<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

pub struct DataParallelRecombinationStackBuilder<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> DataParallelRecombinationStackBuilder<F> {
    pub fn new(
        ctx: &Context,
        mle_1: &dyn CircuitNode,
        mle_2: &dyn CircuitNode,
        mle_3: &dyn CircuitNode,
        mle_4: &dyn CircuitNode,
    ) -> Self {
        let combine_sector = Sector::new(ctx, &[mle_1, mle_2, mle_3, mle_4], |input_nodes| {
            assert_eq!(input_nodes.len(), 4);
            let mle_1_id = input_nodes[0];
            let mle_2_id = input_nodes[1];
            let mle_3_id = input_nodes[2];
            let mle_4_id = input_nodes[3];

            let lhs = mle_1_id.expr().select(mle_2_id.expr());
            let rhs = mle_3_id.expr().select(mle_4_id.expr());

            lhs.select(rhs)
        });

        Self {
            first_layer_sector: combine_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: Field, N> Component<N> for DataParallelRecombinationStackBuilder<F>
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
    pub fn new(ctx: &Context, mle_1: &dyn CircuitNode, mle_2: &dyn CircuitNode) -> Self {
        let first_layer_sector = Sector::new(ctx, &[mle_1, mle_2], |input_nodes| {
            assert_eq!(input_nodes.len(), 2);
            let mle_1_id = input_nodes[0];
            let mle_2_id = input_nodes[1];

            mle_1_id.expr() - mle_2_id.expr()
        });

        let output_node = OutputNode::new_zero(ctx, &first_layer_sector);

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
struct DataparallelRecombinationTestInputs<F: Field> {
    mle_1: MultilinearExtension<F>,
    mle_2: MultilinearExtension<F>,
    mle_3: MultilinearExtension<F>,
    mle_4: MultilinearExtension<F>,
    dataparallel_combined_mle: MultilinearExtension<F>,
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the combined dataparallel circuit.
fn build_dataparallel_recombination_test_circuit<F: Field>(
    num_dataparallel_vars: usize,
    num_free_vars: usize,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(DataparallelRecombinationTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
) {
    // --- Create global context manager ---
    let context = Context::new();

    // --- All inputs are public ---
    let public_input_layer_node = InputLayerNode::new(&context, None);

    // --- "Semantic" circuit inputs ---
    let mle_1_shred = InputShred::new(&context, num_free_vars, &public_input_layer_node);
    let mle_2_shred = InputShred::new(&context, num_free_vars, &public_input_layer_node);
    let mle_3_shred = InputShred::new(&context, num_free_vars, &public_input_layer_node);
    let mle_4_shred = InputShred::new(&context, num_free_vars, &public_input_layer_node);
    let combined_dataparallel_mle_shred = InputShred::new(
        &context,
        num_dataparallel_vars + num_free_vars,
        &public_input_layer_node,
    );

    // --- Save IDs to be used later ---
    let mle_1_id = mle_1_shred.id();
    let mle_2_id = mle_2_shred.id();
    let mle_3_id = mle_3_shred.id();
    let mle_4_id = mle_4_shred.id();
    let combined_dataparallel_mle_id = combined_dataparallel_mle_shred.id();

    // --- Create the circuit components ---
    // Stack currently fails at layer 0, because expr and witgen for the first component is inconsistent.
    // But if you change from stack to interleave, then it fails at layer 1, because the subtraction of the dataparallel
    // mle from the output mle is not actually 0.
    let component_1 = DataParallelRecombinationInterleaveBuilder::new(
        &context,
        &mle_1_shred,
        &mle_2_shred,
        &mle_3_shred,
        &mle_4_shred,
    );

    let component_2 = DiffTwoInputsBuilder::new(
        &context,
        &component_1.get_output_sector(),
        &combined_dataparallel_mle_shred,
    );

    let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
        public_input_layer_node.into(),
        mle_1_shred.into(),
        mle_2_shred.into(),
        mle_3_shred.into(),
        mle_4_shred.into(),
        combined_dataparallel_mle_shred.into(),
    ];
    all_circuit_nodes.extend(component_1.yield_nodes());
    all_circuit_nodes.extend(component_2.yield_nodes());

    let (circuit_description, convert_input_shreds_to_input_layers, _) =
        generate_circuit_description(all_circuit_nodes).unwrap();

    // --- Write closure which allows easy usage of circuit inputs ---
    let circuit_data_fn = move |test_inputs: DataparallelRecombinationTestInputs<F>| {
        let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
            (mle_1_id, test_inputs.mle_1),
            (mle_2_id, test_inputs.mle_2),
            (mle_3_id, test_inputs.mle_3),
            (mle_4_id, test_inputs.mle_4),
            (
                combined_dataparallel_mle_id,
                test_inputs.dataparallel_combined_mle,
            ),
        ]
        .into_iter()
        .collect();
        convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
    };

    (circuit_description, circuit_data_fn)
}

#[test]
fn test_dataparallel_recombination_newmainder() {
    const NUM_FREE_VARS: usize = 2;
    const NUM_DATAPARALLEL_VARS: usize = 2;
    let mut rng = test_rng();

    let (mles_vec, vecs_vec): (Vec<DenseMle<Fr>>, Vec<Vec<Fr>>) = (0..(1 << NUM_DATAPARALLEL_VARS))
        .map(|_| {
            let mle = get_dummy_random_mle(NUM_FREE_VARS, &mut rng);
            let mle_copy = mle.clone();
            let mle_vec = mle_copy.iter().collect();
            (mle, mle_vec)
        })
        .unzip();

    let combined_mle = DenseMle::batch_mles_lil(mles_vec); // This works
                                                           // let combined_mle = DenseMle::batch_mles(mles_vec); // This fails

    // --- Grab inputs from the above ---
    let mle_1 = MultilinearExtension::new(vecs_vec[0].clone());
    let mle_2 = MultilinearExtension::new(vecs_vec[1].clone());
    let mle_3 = MultilinearExtension::new(vecs_vec[2].clone());
    let mle_4 = MultilinearExtension::new(vecs_vec[3].clone());
    let dataparallel_combined_mle = combined_mle.mle;

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn) =
        build_dataparallel_recombination_test_circuit(NUM_DATAPARALLEL_VARS, NUM_FREE_VARS);

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(DataparallelRecombinationTestInputs {
        mle_1,
        mle_2,
        mle_3,
        mle_4,
        dataparallel_combined_mle,
    });

    // --- Specify private input layers (+ description and precommit), if any ---
    let private_input_layers = HashMap::new();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}
