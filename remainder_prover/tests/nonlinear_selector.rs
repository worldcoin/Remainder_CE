use ark_std::test_rng;

use remainder::{
    expression::abstract_expr::ExprBuilder,
    layouter::{
        compiling::LayouterCircuit,
        component::{Component, ComponentSet},
        nodes::{
            circuit_inputs::{InputLayerNode, InputLayerType},
            node_enum::NodeEnum,
            sector::Sector,
            CircuitNode, ClaimableNode, Context,
        },
    },
    mle::evals::MultilinearExtension,
    prover::helpers::test_circuit,
};
use remainder_shared_types::{FieldExt, Fr};

use crate::utils::{get_dummy_input_shred, DifferenceBuilderComponent};
pub mod utils;

/// A builder which returns the following expression:
/// - sel(`left_sel_mle`, `right_sel_mle`)
///   + `right_sum_mle_1` * `right_sum_mle_2`
///
/// The idea is that this builder has one selector bit which is nonlinear.
///
/// ## Arguments
/// * `left_sel_mle` - An MLE with arbitrary bookkeeping table values.
/// * `right_sel_mle` - An MLE with arbitrary bookkeeping table values, same size as `left_sel_mle`.
/// * `right_sum_mle_1`, `right_sum_mle_2` - MLEs with arbitrary bookkeeping table values, same size,
/// one more variable than `right_sel_mle`.

pub struct NonlinearSelectorBuilderComponent<F: FieldExt> {
    pub first_layer_sector: Sector<F>,
}

impl<F: FieldExt> NonlinearSelectorBuilderComponent<F> {
    pub fn new(
        ctx: &Context,
        left_sel_mle: &dyn ClaimableNode<F = F>,
        right_sel_mle: &dyn ClaimableNode<F = F>,
        right_sum_mle_1: &dyn ClaimableNode<F = F>,
        right_sum_mle_2: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let nonlinear_selector_sector = Sector::new(
            ctx,
            &[
                left_sel_mle,
                right_sel_mle,
                right_sum_mle_1,
                right_sum_mle_2,
            ],
            |nonlinear_selector_nodes| {
                assert_eq!(nonlinear_selector_nodes.len(), 4);
                let left_sel_mle_id = nonlinear_selector_nodes[0];
                let right_sel_mle_id = nonlinear_selector_nodes[1];
                let right_sum_mle_1_id = nonlinear_selector_nodes[2];
                let right_sum_mle_2_id = nonlinear_selector_nodes[3];

                let left_sel_side = ExprBuilder::<F>::mle(left_sel_mle_id);
                let right_sel_side = ExprBuilder::<F>::mle(right_sel_mle_id);
                let left_sum_side = right_sel_side.concat_expr(left_sel_side);
                let right_sum_side =
                    ExprBuilder::<F>::products(vec![right_sum_mle_1_id, right_sum_mle_2_id]);
                left_sum_side + right_sum_side
            },
            |data| {
                assert_eq!(data.len(), 4);
                let left_sel_mle_data = data[0];
                let right_sel_mle_data = data[1];
                let right_sum_mle_1_data = data[2];
                let right_sum_mle_2_data = data[3];

                // --- Actual ---
                let right_side_product_bt: Vec<F> = right_sum_mle_1_data
                    .get_evals_vector()
                    .iter()
                    .zip(right_sum_mle_2_data.get_evals_vector().iter())
                    .map(|(elem_1, elem_2)| *elem_1 * elem_2)
                    .collect();
                let left_side_concat_bt: Vec<F> = left_sel_mle_data
                    .get_evals_vector()
                    .iter()
                    .zip(right_sel_mle_data.get_evals_vector().iter())
                    .flat_map(|(elem_1, elem_2)| vec![*elem_1, *elem_2])
                    .collect();
                let sum_bt: Vec<F> = left_side_concat_bt
                    .iter()
                    .zip(right_side_product_bt)
                    .map(|(left_sum, right_sum)| *left_sum + right_sum)
                    .collect();
                MultilinearExtension::new(sum_bt)
            },
        );

        Self {
            first_layer_sector: nonlinear_selector_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: FieldExt, N> Component<N> for NonlinearSelectorBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

/// A circuit which does the following:
/// * Layer 0: [NonlinearSelectorBuilderComponent] with all inputs.
/// * Layer 1: [DifferenceBuilderComponent] with output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// See [NonlinearSelectorBuilderComponent].
#[test]
fn test_nonlinear_sel_circuit_newmainder() {
    const VARS_PRODUCT_SIDE: usize = 3;
    const VARS_SEL_SIDE: usize = VARS_PRODUCT_SIDE - 1;
    let mut rng = &mut test_rng();

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let left_sel_mle = get_dummy_input_shred(VARS_SEL_SIDE, &mut rng, ctx, &input_layer);
        let right_sel_mle = get_dummy_input_shred(VARS_SEL_SIDE, &mut rng, ctx, &input_layer);
        let right_sum_mle_1 = get_dummy_input_shred(VARS_PRODUCT_SIDE, &mut rng, ctx, &input_layer);
        let right_sum_mle_2 = get_dummy_input_shred(VARS_PRODUCT_SIDE, &mut rng, ctx, &input_layer);

        let component_1 = NonlinearSelectorBuilderComponent::new(
            ctx,
            &left_sel_mle,
            &right_sel_mle,
            &right_sum_mle_1,
            &right_sum_mle_2,
        );
        let component_2 = DifferenceBuilderComponent::new(ctx, component_1.get_output_sector());

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            left_sel_mle.into(),
            right_sel_mle.into(),
            right_sum_mle_1.into(),
            right_sum_mle_2.into(),
        ];

        all_nodes.extend(component_1.yield_nodes());
        all_nodes.extend(component_2.yield_nodes());
        ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes)
    });

    test_circuit(circuit, None)
}
