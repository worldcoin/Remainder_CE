use ark_std::test_rng;
use itertools::Itertools;
use remainder::{
    layouter::{
        compiling::LayouterCircuit,
        component::{Component, ComponentSet},
        nodes::{
            circuit_inputs::{InputLayerNode, InputLayerType},
            circuit_outputs::OutputNode,
            node_enum::NodeEnum,
            sector::Sector,
            CircuitNode, ClaimableNode, Context,
        },
    },
    mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
    prover::helpers::test_circuit,
};
use remainder_shared_types::{Field, Fr};
use utils::{get_dummy_random_mle, get_input_shred_from_vec};
pub mod utils;

pub struct DataParallelRecombinationInterleaveBuilder<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> DataParallelRecombinationInterleaveBuilder<F> {
    pub fn new(
        ctx: &Context,
        mle_1: &dyn ClaimableNode<F = F>,
        mle_2: &dyn ClaimableNode<F = F>,
        mle_3: &dyn ClaimableNode<F = F>,
        mle_4: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let combine_sector = Sector::new(
            ctx,
            &[mle_1, mle_2, mle_3, mle_4],
            |input_nodes| {
                assert_eq!(input_nodes.len(), 4);
                let mle_1_id = input_nodes[0];
                let mle_2_id = input_nodes[1];
                let mle_3_id = input_nodes[2];
                let mle_4_id = input_nodes[3];

                let lhs = mle_2_id.expr().concat_expr(mle_1_id.expr());
                let rhs = mle_4_id.expr().concat_expr(mle_3_id.expr());

                rhs.concat_expr(lhs)
            },
            |data| {
                let mle_1_data = data[0];
                let mle_2_data = data[1];
                let mle_3_data = data[2];
                let mle_4_data = data[3];

                let lhs_bt = mle_1_data
                    .get_evals_vector()
                    .iter()
                    .zip(mle_2_data.get_evals_vector())
                    .flat_map(|(elem_1, elem_2)| vec![elem_1, elem_2]);

                let rhs_bt = mle_3_data
                    .get_evals_vector()
                    .iter()
                    .zip(mle_4_data.get_evals_vector())
                    .flat_map(|(elem_1, elem_2)| vec![elem_1, elem_2]);

                let final_bt = lhs_bt
                    .zip(rhs_bt)
                    .flat_map(|(elem_1, elem_2)| vec![*elem_1, *elem_2])
                    .collect_vec();

                MultilinearExtension::new(final_bt)
            },
        );

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
        mle_1: &dyn ClaimableNode<F = F>,
        mle_2: &dyn ClaimableNode<F = F>,
        mle_3: &dyn ClaimableNode<F = F>,
        mle_4: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let combine_sector = Sector::new(
            ctx,
            &[mle_1, mle_2, mle_3, mle_4],
            |input_nodes| {
                assert_eq!(input_nodes.len(), 4);
                let mle_1_id = input_nodes[0];
                let mle_2_id = input_nodes[1];
                let mle_3_id = input_nodes[2];
                let mle_4_id = input_nodes[3];

                let lhs = mle_2_id.expr().concat_expr(mle_1_id.expr());
                let rhs = mle_4_id.expr().concat_expr(mle_3_id.expr());

                rhs.concat_expr(lhs)
            },
            |data| {
                let mle_1_data = data[0];
                let mle_2_data = data[1];
                let mle_3_data = data[2];
                let mle_4_data = data[3];

                let final_bt = mle_1_data
                    .get_evals_vector()
                    .clone()
                    .into_iter()
                    .chain(mle_2_data.get_evals_vector().clone().into_iter())
                    .chain(mle_3_data.get_evals_vector().clone().into_iter())
                    .chain(mle_4_data.get_evals_vector().clone().into_iter())
                    .collect_vec();
                MultilinearExtension::new(final_bt)
            },
        );

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
    pub output_sector: OutputNode<F>,
}

impl<F: Field> DiffTwoInputsBuilder<F> {
    pub fn new(
        ctx: &Context,
        mle_1: &dyn ClaimableNode<F = F>,
        mle_2: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let first_layer_sector = Sector::new(
            ctx,
            &[mle_1, mle_2],
            |input_nodes| {
                assert_eq!(input_nodes.len(), 2);
                let mle_1_id = input_nodes[0];
                let mle_2_id = input_nodes[1];

                mle_1_id.expr() - mle_2_id.expr()
            },
            |data| {
                let mle_1_data = data[0];
                MultilinearExtension::new_sized_zero(mle_1_data.num_vars())
            },
        );

        let output_node = OutputNode::new_zero(ctx, &first_layer_sector);

        Self {
            first_layer_sector,
            output_sector: output_node,
        }
    }

    pub fn get_output_sector(&self) -> &OutputNode<F> {
        &self.output_sector
    }
}

impl<F: Field, N> Component<N> for DiffTwoInputsBuilder<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into(), self.output_sector.into()]
    }
}

#[test]
fn test_dataparallel_recombination_newmainder() {
    const ITERATED_VARS: usize = 2;
    const DATAPARALLEL_VARS: usize = 2;
    let mut rng = test_rng();

    let (mles_vec, vecs_vec): (Vec<DenseMle<Fr>>, Vec<Vec<Fr>>) = (0..(1 << DATAPARALLEL_VARS))
        .map(|_| {
            let mle = get_dummy_random_mle(ITERATED_VARS, &mut rng);
            let mle_copy = mle.clone();
            let mle_vec = mle_copy.bookkeeping_table();
            (mle, mle_vec.to_vec())
        })
        .unzip();

    let combined_mle = DenseMle::batch_mles_lil(mles_vec); // This works
                                                           // let combined_mle = DenseMle::batch_mles(mles_vec); // This fails
    let combined_mle_vec = combined_mle.bookkeeping_table();

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let input_shred_1 = get_input_shred_from_vec(vecs_vec[0].clone(), ctx, &input_layer);
        let input_shred_2 = get_input_shred_from_vec(vecs_vec[1].clone(), ctx, &input_layer);
        let input_shred_3 = get_input_shred_from_vec(vecs_vec[2].clone(), ctx, &input_layer);
        let input_shred_4 = get_input_shred_from_vec(vecs_vec[3].clone(), ctx, &input_layer);
        let dataparallel_shred =
            get_input_shred_from_vec(combined_mle_vec.to_vec(), ctx, &input_layer);

        // Stack currently fails at layer 0, because expr and witgen for the first component is inconsistent.
        // But if you change from stack to interleave, then it fails at layer 1, because the subtraction of the dataparallel
        // mle from the output mle is not actually 0.
        let component_1 = DataParallelRecombinationInterleaveBuilder::new(
            ctx,
            &input_shred_1,
            &input_shred_2,
            &input_shred_3,
            &input_shred_4,
        );

        let component_2 =
            DiffTwoInputsBuilder::new(ctx, component_1.get_output_sector(), &dataparallel_shred);

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            input_shred_1.into(),
            input_shred_2.into(),
            input_shred_3.into(),
            input_shred_4.into(),
            dataparallel_shred.into(),
        ];

        all_nodes.extend(component_1.yield_nodes());
        all_nodes.extend(component_2.yield_nodes());

        ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes)
    });

    test_circuit(circuit, None)
}
