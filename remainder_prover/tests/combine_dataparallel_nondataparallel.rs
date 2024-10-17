use ark_std::test_rng;
use itertools::Itertools;

use remainder::{
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

pub mod utils;

use utils::{
    get_dummy_input_shred_and_data, get_input_shred_and_data_from_vec, DifferenceBuilderComponent,
    ProductScaledBuilderComponent, TripleNestedBuilderComponent,
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
    pub fn new(
        ctx: &Context,
        mle_1_input: &dyn CircuitNode,
        mle_2_input: &dyn CircuitNode,
    ) -> Self {
        let product_scaled_component =
            ProductScaledBuilderComponent::new(ctx, mle_1_input, mle_2_input);

        let product_scaled_meta_component = ProductScaledBuilderComponent::new(
            ctx,
            &product_scaled_component.get_output_sector(),
            &product_scaled_component.get_output_sector(),
        );
        let output_component = DifferenceBuilderComponent::new(
            ctx,
            &product_scaled_meta_component.get_output_sector(),
        );

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
        ctx: &Context,
        inner_inner_sel: &dyn CircuitNode,
        inner_sel: &dyn CircuitNode,
        outer_sel: &dyn CircuitNode,
    ) -> Self {
        let triple_nested_selector_component =
            TripleNestedBuilderComponent::new(ctx, inner_inner_sel, inner_sel, outer_sel);
        let output_component = DifferenceBuilderComponent::new(
            ctx,
            &triple_nested_selector_component.get_output_sector(),
        );

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
    /// * Layer 0: [ProductScaledBuilder] with the two inputs
    /// * Layer 1: [ZeroBuilder] with output of Layer 0 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1` - An MLE with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE with arbitrary bookkeeping table values, same size as `mle_1`.
    pub fn new(
        ctx: &Context,
        mle_1_input: &dyn CircuitNode,
        mle_2_input: &dyn CircuitNode,
    ) -> Self {
        let product_scaled_component =
            ProductScaledBuilderComponent::new(ctx, mle_1_input, mle_2_input);

        let output_component =
            DifferenceBuilderComponent::new(ctx, &product_scaled_component.get_output_sector());

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

/// A circuit which combines the [DataParallelCircuit], [TripleNestedSelectorCircuit],
/// and [ScaledProductCircuit].
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec`, `mle_2_vec` - inputs to [DataParallelCircuit] both arbitrary bookkeeping
/// table values, same size.
/// * `mle_4`, `mle_5`, `mle_6` - inputs to [TripleNestedSelectorCircuit], `mle_4` has the same
/// size as the mles in `mle_1_vec`, arbitrary bookkeeping table values. `mle_5` has one more
/// variable than `mle_4`, `mle_6` has one more variable than `mle_5`, both arbitrary bookkeeping
/// table values.
/// * `mle_3`, `mle_4` - inputs to [ScaledProductCircuit], both arbitrary bookkeeping table values,
/// same size.
/// * `num_dataparallel_bits` - The number of bits that represent which copy index the circuit is.

#[test]
fn test_combined_dataparallel_nondataparallel_circuit_newmainder() {
    const VARS_MLE_1_2: usize = 1;
    const VARS_MLE_3: usize = VARS_MLE_1_2 + 1;
    const VARS_MLE_4: usize = VARS_MLE_3 + 1;
    const NUM_DATAPARALLEL_BITS: usize = 1;
    let mut rng = test_rng();

    let mle_1_vec = (0..1 << NUM_DATAPARALLEL_BITS)
        .map(|_| get_dummy_random_mle(VARS_MLE_1_2, &mut rng))
        .collect_vec();
    let mle_2_vec = (0..1 << NUM_DATAPARALLEL_BITS)
        .map(|_| get_dummy_random_mle(VARS_MLE_1_2, &mut rng))
        .collect_vec();

    let mle_1_vec_batched = DenseMle::batch_mles(mle_1_vec);
    let mle_2_vec_batched = DenseMle::batch_mles(mle_2_vec);
    let mle_1_vec_iter = mle_1_vec_batched.iter();
    let mle_2_vec_iter = mle_2_vec_batched.iter();

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (input_shred_1, input_shred_1_data) =
            get_input_shred_and_data_from_vec(mle_1_vec_iter.clone().collect(), ctx, &input_layer);
        let (input_shred_2, input_shred_2_data) =
            get_input_shred_and_data_from_vec(mle_2_vec_iter.clone().collect(), ctx, &input_layer);
        let (input_shred_3, input_shred_3_data) =
            get_dummy_input_shred_and_data(VARS_MLE_1_2, &mut rng, ctx, &input_layer);
        let (input_shred_4, input_shred_4_data) =
            get_dummy_input_shred_and_data(VARS_MLE_1_2, &mut rng, ctx, &input_layer);
        let (input_shred_5, input_shred_5_data) =
            get_dummy_input_shred_and_data(VARS_MLE_3, &mut rng, ctx, &input_layer);
        let (input_shred_6, input_shred_6_data) =
            get_dummy_input_shred_and_data(VARS_MLE_4, &mut rng, ctx, &input_layer);
        let input_data = InputLayerData::new(
            input_layer.id(),
            vec![
                input_shred_1_data,
                input_shred_2_data,
                input_shred_3_data,
                input_shred_4_data,
                input_shred_5_data,
                input_shred_6_data,
            ],
            None,
        );

        let component_1 = DataParallelComponent::new(ctx, &input_shred_1, &input_shred_2);
        let component_2 =
            TripleNestedSelectorComponent::new(ctx, &input_shred_4, &input_shred_5, &input_shred_6);
        let component_3 = ScaledProductComponent::new(ctx, &input_shred_3, &input_shred_4);

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            input_shred_1.into(),
            input_shred_2.into(),
            input_shred_3.into(),
            input_shred_4.into(),
            input_shred_5.into(),
            input_shred_6.into(),
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
