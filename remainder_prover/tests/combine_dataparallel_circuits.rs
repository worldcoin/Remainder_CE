use ark_std::test_rng;

use itertools::Itertools;
use remainder::{
    layouter::{
        compiling::LayouterCircuit,
        component::{Component, ComponentSet},
        nodes::{
            circuit_inputs::{InputLayerNode, InputLayerNodeData, InputLayerType},
            circuit_outputs::OutputNode,
            node_enum::NodeEnum,
            sector::Sector,
            CircuitNode, Context,
        },
    },
    mle::{dense::DenseMle, Mle},
    prover::helpers::test_circuit, utils::mle::get_random_mle,
};
use remainder_shared_types::{Field, Fr};
use utils::{
    ConstantScaledSumBuilderComponent, DifferenceBuilderComponent, ProductScaledBuilderComponent,
    ProductSumBuilderComponent,
};

use crate::utils::get_input_shred_and_data_from_vec;
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
    pub fn new(
        ctx: &Context,
        mle_1_input: &dyn CircuitNode,
        mle_2_input: &dyn CircuitNode,
    ) -> Self {
        let first_layer_component =
            ConstantScaledSumBuilderComponent::new(ctx, mle_1_input, mle_2_input);

        let second_layer_component = ProductScaledBuilderComponent::new(
            ctx,
            &first_layer_component.get_output_sector(),
            mle_1_input,
        );

        let output_component =
            DifferenceBuilderComponent::new(ctx, &second_layer_component.get_output_sector());

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
    pub fn new(
        ctx: &Context,
        mle_1_input: &dyn CircuitNode,
        mle_2_input: &dyn CircuitNode,
    ) -> Self {
        let first_layer_component = ProductSumBuilderComponent::new(ctx, mle_1_input, mle_2_input);

        let second_layer_component = ConstantScaledSumBuilderComponent::new(
            ctx,
            &first_layer_component.get_output_sector(),
            mle_1_input,
        );

        let output_component =
            DifferenceBuilderComponent::new(ctx, &second_layer_component.get_output_sector());

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
    pub fn new(
        ctx: &Context,
        mle_1_input: &dyn CircuitNode,
        mle_2_input: &dyn CircuitNode,
    ) -> Self {
        let first_layer_component =
            ProductScaledBuilderComponent::new(ctx, mle_1_input, mle_2_input);

        let second_layer_component = ProductSumBuilderComponent::new(
            ctx,
            &first_layer_component.get_output_sector(),
            mle_1_input,
        );

        let output_component =
            DifferenceBuilderComponent::new(ctx, &second_layer_component.get_output_sector());

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

#[test]
fn test_combined_dataparallel_circuit_alt_newmainder() {
    const NUM_DATAPARALLEL_BITS: usize = 1;
    const VARS_MLE_1_2: usize = 2;
    let mut rng = test_rng();

    // This is not strictly necessary; the setup of `DenseMle` -->
    // `batch_mles()` --> `bookkeeping_table` is just to emulate what
    // batching *would* look like
    let mle_1_vec = (0..1 << NUM_DATAPARALLEL_BITS)
        .map(|_| get_random_mle(VARS_MLE_1_2, &mut rng))
        .collect_vec();
    let mle_2_vec = (0..1 << NUM_DATAPARALLEL_BITS)
        .map(|_| get_random_mle(VARS_MLE_1_2, &mut rng))
        .collect_vec();

    let mle_1_vec_batched = DenseMle::batch_mles(mle_1_vec.clone());
    let mle_2_vec_batched = DenseMle::batch_mles(mle_2_vec.clone());
    let mle_1_vec_raw = mle_1_vec_batched.bookkeeping_table();
    let mle_2_vec_raw = mle_2_vec_batched.bookkeeping_table();

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
    assert_eq!(mle_1_vec.len(), 1 << NUM_DATAPARALLEL_BITS);
    // These checks can possibly be done with the newly designed batching bits/system

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (dataparallel_input_mle_1, dataparallel_input_mle_1_data) =
            get_input_shred_and_data_from_vec(mle_1_vec_raw.to_vec(), ctx, &input_layer);
        let (dataparallel_input_mle_2, dataparallel_input_mle_2_data) =
            get_input_shred_and_data_from_vec(mle_2_vec_raw.to_vec(), ctx, &input_layer);
        let input_data = InputLayerNodeData::new(
            input_layer.id(),
            vec![dataparallel_input_mle_1_data, dataparallel_input_mle_2_data],
            None,
        );

        let component_1 = DataParallelProductScaledSumCircuitAltComponent::new(
            ctx,
            &dataparallel_input_mle_1,
            &dataparallel_input_mle_2,
        );
        let component_2 = DataParallelSumConstantCircuitAltComponent::new(
            ctx,
            &dataparallel_input_mle_1,
            &dataparallel_input_mle_2,
        );
        let component_3 = DataParallelConstantScaledCircuitAltComponent::new(
            ctx,
            &dataparallel_input_mle_1,
            &dataparallel_input_mle_2,
        );

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            dataparallel_input_mle_1.into(),
            dataparallel_input_mle_2.into(),
        ];

        all_nodes.extend(component_1.yield_nodes());
        all_nodes.extend(component_2.yield_nodes());
        all_nodes.extend(component_3.yield_nodes());
        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
            vec![input_data],
        )
    });

    test_circuit(circuit, None)
}
