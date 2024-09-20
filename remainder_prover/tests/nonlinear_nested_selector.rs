use ark_std::test_rng;

use remainder::{
    expression::abstract_expr::ExprBuilder,
    layouter::{
        compiling::LayouterCircuit,
        component::{Component, ComponentSet},
        nodes::{
            circuit_inputs::{InputLayerData, InputLayerNode, InputLayerType},
            node_enum::NodeEnum,
            sector::Sector,
            CircuitNode, Context,
        },
    },
    prover::helpers::test_circuit,
};
use remainder_shared_types::{Field, Fr};

use crate::utils::{get_dummy_input_shred_and_data, DifferenceBuilderComponent};
pub mod utils;

pub struct NonlinearNestedSelectorBuilderComponent<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> NonlinearNestedSelectorBuilderComponent<F> {
    /// A builder which returns the following expression:
    /// ```text
    /// sel(sel(left_inner_sel_mle, right_inner_sel_mle), right_outer_sel_mle)
    ///     + right_sum_mle_1 * right_sum_mle_2
    /// ```
    ///
    /// The idea is that this builder has two selector bits which are nonlinear.
    ///
    /// ## Arguments
    /// * `left_inner_sel_mle` - An MLE with arbitrary bookkeeping table values.
    /// * `right_inner_sel_mle` - An MLE with arbitrary bookkeeping table values, same size as `left_inner_sel_mle`.
    /// * `right_outer_sel_mle` - An MLE with arbitrary bookkeeping table values, one more variable
    /// than `right_inner_sel_mle`.
    /// * `right_sum_mle_1`, `right_sum_mle_2` - MLEs with arbitrary bookkeeping table values, same size,
    /// one more variable than `right_outer_sel_mle`.
    pub fn new(
        ctx: &Context,
        left_inner_sel_mle: &dyn CircuitNode,
        right_inner_sel_mle: &dyn CircuitNode,
        right_outer_sel_mle: &dyn CircuitNode,
        right_sum_mle_1: &dyn CircuitNode,
        right_sum_mle_2: &dyn CircuitNode,
    ) -> Self {
        let nonlinear_nested_selector_sector = Sector::new(
            ctx,
            &[
                left_inner_sel_mle,
                right_inner_sel_mle,
                right_outer_sel_mle,
                right_sum_mle_1,
                right_sum_mle_2,
            ],
            |nonlinear_nested_selector_nodes| {
                assert_eq!(nonlinear_nested_selector_nodes.len(), 5);

                let left_inner_sel_mle_id = nonlinear_nested_selector_nodes[0];
                let right_inner_sel_mle_id = nonlinear_nested_selector_nodes[1];
                let right_outer_sel_mle_id = nonlinear_nested_selector_nodes[2];
                let right_sum_mle_1_id = nonlinear_nested_selector_nodes[3];
                let right_sum_mle_2_id = nonlinear_nested_selector_nodes[4];

                let left_inner_sel_side = ExprBuilder::<F>::mle(left_inner_sel_mle_id);
                let right_inner_sel_side = ExprBuilder::<F>::mle(right_inner_sel_mle_id);
                let left_outer_sel_side = right_inner_sel_side.concat_expr(left_inner_sel_side);
                let left_sum_side =
                    ExprBuilder::<F>::mle(right_outer_sel_mle_id).concat_expr(left_outer_sel_side);
                let right_sum_side =
                    ExprBuilder::<F>::products(vec![right_sum_mle_1_id, right_sum_mle_2_id]);
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
    const VARS_PRODUCT_SIDE: usize = 5;
    const VARS_OUTER_SEL_SIDE: usize = VARS_PRODUCT_SIDE - 1;
    const VARS_INNER_SEL_SIDE: usize = VARS_OUTER_SEL_SIDE - 1;
    let mut rng = test_rng();

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);

        let (left_inner_sel_mle, left_inner_sel_mle_data) =
            get_dummy_input_shred_and_data(VARS_INNER_SEL_SIDE, &mut rng, ctx, &input_layer);
        let (right_inner_sel_mle, right_inner_sel_mle_data) =
            get_dummy_input_shred_and_data(VARS_INNER_SEL_SIDE, &mut rng, ctx, &input_layer);
        let (right_outer_sel_mle, right_outer_sel_mle_data) =
            get_dummy_input_shred_and_data(VARS_OUTER_SEL_SIDE, &mut rng, ctx, &input_layer);
        let (right_sum_mle_1, right_sum_mle_1_data) =
            get_dummy_input_shred_and_data(VARS_PRODUCT_SIDE, &mut rng, ctx, &input_layer);
        let (right_sum_mle_2, right_sum_mle_2_data) =
            get_dummy_input_shred_and_data(VARS_PRODUCT_SIDE, &mut rng, ctx, &input_layer);

        let component_1 = NonlinearNestedSelectorBuilderComponent::new(
            ctx,
            &left_inner_sel_mle,
            &right_inner_sel_mle,
            &right_outer_sel_mle,
            &right_sum_mle_1,
            &right_sum_mle_2,
        );
        let component_2 = DifferenceBuilderComponent::new(ctx, &component_1.get_output_sector());
        let input_data = InputLayerData::new(
            input_layer.id(),
            vec![
                left_inner_sel_mle_data,
                right_inner_sel_mle_data,
                right_outer_sel_mle_data,
                right_sum_mle_1_data,
                right_sum_mle_2_data,
            ],
            None,
        );

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            left_inner_sel_mle.into(),
            right_inner_sel_mle.into(),
            right_outer_sel_mle.into(),
            right_sum_mle_1.into(),
            right_sum_mle_2.into(),
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
