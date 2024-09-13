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
    prover::helpers::test_circuit,
};
use remainder_shared_types::{Field, Fr};

use utils::{
    ConstantScaledSumBuilderComponent, DifferenceBuilderComponent, ProductScaledBuilderComponent,
    ProductSumBuilderComponent,
};

use crate::utils::get_dummy_input_shred;
pub mod utils;

struct ConstantScaledCircuitComponent<F: Field> {
    first_layer_component: ConstantScaledSumBuilderComponent<F>,
    second_layer_component: ProductScaledBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> ConstantScaledCircuitComponent<F> {
    /// A circuit which takes in two MLEs of the same size:
    /// * Layer 0: [ConstantScaledSumBuilderComponent] with the two inputs
    /// * Layer 1: [ProductScaledBuilderComponent] with the output of Layer 0 and `mle_1`
    /// * Layer 2: [DifferenceBuilderComponent] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1`.
    pub fn new(
        ctx: &Context,
        mle_1_input: &dyn ClaimableNode<F = F>,
        mle_2_input: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let first_layer_component =
            ConstantScaledSumBuilderComponent::new(ctx, mle_1_input, mle_2_input);

        let second_layer_component = ProductScaledBuilderComponent::new(
            ctx,
            first_layer_component.get_output_sector(),
            mle_1_input,
        );

        let output_component =
            DifferenceBuilderComponent::new(ctx, second_layer_component.get_output_sector());

        Self {
            first_layer_component,
            second_layer_component,
            output_component,
        }
    }
}

impl<F: Field, N> Component<N> for ConstantScaledCircuitComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
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

struct SumConstantCircuitComponent<F: Field> {
    first_layer_component: ProductSumBuilderComponent<F>,
    second_layer_component: ConstantScaledSumBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> SumConstantCircuitComponent<F> {
    /// A circuit which takes in two MLEs of the same size:
    /// * Layer 0: [ProductSumBuilderComponent] with the two inputs
    /// * Layer 1: [ConstantScaledSumBuilderComponent] with the output of Layer 0 and `mle_1`
    /// * Layer 2: [DifferenceBuilderComponent] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1` - An MLE vec with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1`.
    pub fn new(
        ctx: &Context,
        mle_1_input: &dyn ClaimableNode<F = F>,
        mle_2_input: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let first_layer_component = ProductSumBuilderComponent::new(ctx, mle_1_input, mle_2_input);

        let second_layer_component = ConstantScaledSumBuilderComponent::new(
            ctx,
            first_layer_component.get_output_sector(),
            mle_1_input,
        );

        let output_component =
            DifferenceBuilderComponent::new(ctx, second_layer_component.get_output_sector());

        Self {
            first_layer_component,
            second_layer_component,
            output_component,
        }
    }
}

impl<F: Field, N> Component<N> for SumConstantCircuitComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
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

struct ProductScaledSumCircuitComponent<F: Field> {
    first_layer_component: ProductScaledBuilderComponent<F>,
    second_layer_component: ProductSumBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> ProductScaledSumCircuitComponent<F> {
    /// A circuit which takes in two MLEs of the same size:
    /// * Layer 0: [ProductScaledBuilderComponent] with the two inputs
    /// * Layer 1: [ProductSumBuilderComponent] with the output of Layer 0 and `mle_1`
    /// * Layer 2: [DifferenceBuilderComponent] with output of Layer 1 and itself.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_1`  An MLE with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE with arbitrary bookkeeping table values, same size as `mle_1`.
    pub fn new(
        ctx: &Context,
        mle_1_input: &dyn ClaimableNode<F = F>,
        mle_2_input: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let first_layer_component =
            ProductScaledBuilderComponent::new(ctx, mle_1_input, mle_2_input);

        let second_layer_component = ProductSumBuilderComponent::new(
            ctx,
            first_layer_component.get_output_sector(),
            mle_1_input,
        );

        let output_component =
            DifferenceBuilderComponent::new(ctx, second_layer_component.get_output_sector());

        Self {
            first_layer_component,
            second_layer_component,
            output_component,
        }
    }
}

impl<F: Field, N> Component<N> for ProductScaledSumCircuitComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
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
fn test_combined_nondataparallel_circuit_newmainder() {
    const VARS_MLE_1_2: usize = 2;
    let mut rng = test_rng();

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let input_mle_1 = get_dummy_input_shred(VARS_MLE_1_2, &mut rng, ctx, &input_layer);
        let input_mle_2 = get_dummy_input_shred(VARS_MLE_1_2, &mut rng, ctx, &input_layer);

        let component_1 = ProductScaledSumCircuitComponent::new(ctx, &input_mle_1, &input_mle_2);
        let component_2 = SumConstantCircuitComponent::new(ctx, &input_mle_1, &input_mle_2);
        let component_3 = ConstantScaledCircuitComponent::new(ctx, &input_mle_1, &input_mle_2);

        let mut all_nodes: Vec<NodeEnum<Fr>> =
            vec![input_layer.into(), input_mle_1.into(), input_mle_2.into()];

        all_nodes.extend(component_1.yield_nodes());
        all_nodes.extend(component_2.yield_nodes());
        all_nodes.extend(component_3.yield_nodes());
        ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes)
    });

    test_circuit(circuit, None)
}
