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
use utils::{get_dummy_input_shred_and_data, DifferenceBuilderComponent};

pub mod utils;

/// A builder which returns the following expression:
/// - sel(`mle_1`, `mle_1`) + `mle_2` * `mle_2`
///
/// The idea is that the last bit in this expression is linear.
///
/// ## Arguments
/// * `sel_mle` - An MLE with arbitrary bookkeeping table values.
/// * `prod_mle` - An MLE with arbitrary bookkeeping table values; same size as `sel_mle`.

pub struct LastBitLinearBuilderComponent<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> LastBitLinearBuilderComponent<F> {
    pub fn new(ctx: &Context, sel_node: &dyn CircuitNode, prod_node: &dyn CircuitNode) -> Self {
        let last_bit_linear_sector = Sector::new(ctx, &[sel_node, prod_node], |input_nodes| {
            assert_eq!(input_nodes.len(), 2);
            let sel_mle = input_nodes[0];
            let prod_mle = input_nodes[1];

            let lhs_sum_expr = sel_mle.expr().concat_expr(sel_mle.expr());
            let rhs_sum_expr = ExprBuilder::<F>::products(vec![prod_mle, prod_mle]);
            lhs_sum_expr + rhs_sum_expr
        });

        Self {
            first_layer_sector: last_bit_linear_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: Field, N> Component<N> for LastBitLinearBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

/// A builder which returns the following expression:
/// - sel(`mle_1` * `mle_1`, `mle_1`)
///
/// The idea is that the first bit (selector bit) in this expression is linear.
///
/// ## Arguments
/// * `sel_mle` - An MLE with arbitrary bookkeeping table values.
pub struct FirstBitLinearBuilderComponent<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> FirstBitLinearBuilderComponent<F> {
    pub fn new(ctx: &Context, sel_node: &dyn CircuitNode) -> Self {
        let last_bit_linear_sector = Sector::new(ctx, &[sel_node], |input_nodes| {
            assert_eq!(input_nodes.len(), 1);
            let sel_mle = input_nodes[0];

            sel_mle
                .expr()
                .concat_expr(ExprBuilder::<F>::products(vec![sel_mle, sel_mle]))
        });

        Self {
            first_layer_sector: last_bit_linear_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: Field, N> Component<N> for FirstBitLinearBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

/// A circuit which does the following:
/// * Layer 0: [LastBitLinearBuilderComponent] with `sel_mle`, `prod_mle`
/// * Layer 1: [FirstBitLinearBuilderComponent] with `sel_mle`
/// * Layer 2: [DifferenceBuilderComponent] with the output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `sel_mle`, `prod_mle` both MLEs with arbitrary bookkeeping table values, same size.

#[test]
fn test_linear_and_nonlinear_bits_circuit_newmainder() {
    const VARS_SEL_SIDE: usize = 2;
    const VARS_PROD_SIDE: usize = VARS_SEL_SIDE;
    let mut rng = test_rng();

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (sel_input, sel_input_data) =
            get_dummy_input_shred_and_data(VARS_SEL_SIDE, &mut rng, ctx, &input_layer);
        let (prod_input, prod_input_data) =
            get_dummy_input_shred_and_data(VARS_PROD_SIDE, &mut rng, ctx, &input_layer);
        let input_data = InputLayerData::new(
            input_layer.id(),
            vec![sel_input_data, prod_input_data],
            None,
        );

        let component_1 = LastBitLinearBuilderComponent::new(ctx, &sel_input, &prod_input);
        let component_2 =
            FirstBitLinearBuilderComponent::new(ctx, &component_1.get_output_sector());
        let output_component =
            DifferenceBuilderComponent::new(ctx, &component_2.get_output_sector());

        let mut all_nodes: Vec<NodeEnum<Fr>> =
            vec![input_layer.into(), sel_input.into(), prod_input.into()];

        all_nodes.extend(component_1.yield_nodes());
        all_nodes.extend(component_2.yield_nodes());
        all_nodes.extend(output_component.yield_nodes());

        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
            vec![input_data],
        )
    });

    test_circuit(circuit, None)
}
