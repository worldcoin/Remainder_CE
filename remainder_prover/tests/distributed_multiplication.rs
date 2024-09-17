use ark_std::test_rng;
use itertools::Itertools;
use remainder::{
    expression::abstract_expr::ExprBuilder,
    layouter::{
        compiling::LayouterCircuit,
        component::{Component, ComponentSet},
        nodes::{
            circuit_inputs::{InputLayerData, InputLayerNode, InputLayerType},
            circuit_outputs::OutputNode,
            node_enum::NodeEnum,
            sector::Sector,
            CircuitNode, Context,
        },
    },
    mle::{dense::DenseMle, Mle},
    prover::helpers::test_circuit,
};
use remainder_shared_types::{Field, Fr};
use utils::{get_dummy_random_mle, get_input_shred_and_data_from_vec};
pub mod utils;

pub struct DataparallelDistributedMultiplication<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> DataparallelDistributedMultiplication<F> {
    pub fn new(ctx: &Context, smaller_mle: &dyn CircuitNode, bigger_mle: &dyn CircuitNode) -> Self {
        let combine_sector = Sector::new(ctx, &[smaller_mle, bigger_mle], |input_nodes| {
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

#[test]
fn test_batching_wraparound_newmainder() {
    const ITERATED_VARS_SMALLER: usize = 1;
    const ITERATED_VARS_BIGGER: usize = 2;
    const DATAPARALLEL_VARS: usize = 2;
    let mut rng = test_rng();

    let smaller_mles_vec: Vec<DenseMle<Fr>> = (0..(1 << DATAPARALLEL_VARS))
        .map(|_| get_dummy_random_mle(ITERATED_VARS_SMALLER, &mut rng))
        .collect_vec();

    let bigger_mles_vec: Vec<DenseMle<Fr>> = (0..(1 << DATAPARALLEL_VARS))
        .map(|_| get_dummy_random_mle(ITERATED_VARS_BIGGER, &mut rng))
        .collect_vec();

    let prod_mles = smaller_mles_vec
        .iter()
        .zip(bigger_mles_vec.iter())
        .map(|(small_mle, big_mle)| {
            let small_mle_bt = small_mle.bookkeeping_table();
            let big_mle_bt = big_mle.bookkeeping_table();
            let prod_bt = big_mle_bt
                .iter()
                .zip(small_mle_bt.iter().cycle())
                .map(|(big_elem, small_elem)| *big_elem * *small_elem);
            DenseMle::new_from_iter(prod_bt, small_mle.layer_id)
        })
        .collect_vec();

    let combined_prod_mle_expected = DenseMle::batch_mles_lil(prod_mles); // This works
                                                                          // let combined_prod_mle_expected = DenseMle::batch_mles(prod_mles); // This fails
    let combined_prod_mle_expected_vec = combined_prod_mle_expected.bookkeeping_table();

    let smaller_combined_mle = DenseMle::batch_mles_lil(smaller_mles_vec); // This works
                                                                           // let smaller_combined_mle = DenseMle::batch_mles(smaller_mles_vec); // This fails
    let smaller_combined_mle_vec = smaller_combined_mle.bookkeeping_table();

    let bigger_combined_mle = DenseMle::batch_mles_lil(bigger_mles_vec); // This works
                                                                         // let bigger_combined_mle = DenseMle::batch_mles(bigger_mles_vec); // This fails
    let bigger_combined_mle_vec = bigger_combined_mle.bookkeeping_table();

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (input_shred_1, input_shred_1_data) =
            get_input_shred_and_data_from_vec(smaller_combined_mle_vec.to_vec(), ctx, &input_layer);
        let (input_shred_2, input_shred_2_data) =
            get_input_shred_and_data_from_vec(bigger_combined_mle_vec.to_vec(), ctx, &input_layer);
        let (input_shred_3, input_shred_3_data) = get_input_shred_and_data_from_vec(
            combined_prod_mle_expected_vec.to_vec(),
            ctx,
            &input_layer,
        );
        let input_data = InputLayerData::new(
            input_layer.id(),
            vec![input_shred_1_data, input_shred_2_data, input_shred_3_data],
            None,
        );

        let component_1 =
            DataparallelDistributedMultiplication::new(ctx, &input_shred_1, &input_shred_2);

        let component_2 =
            DiffTwoInputsBuilder::new(ctx, &component_1.get_output_sector(), &input_shred_3);

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            input_shred_1.into(),
            input_shred_2.into(),
            input_shred_3.into(),
        ];

        all_nodes.extend(component_1.yield_nodes());
        all_nodes.extend(component_2.yield_nodes());

        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
            vec![input_data],
        )
    });

    test_circuit(circuit, None)
}
