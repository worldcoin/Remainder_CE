use remainder::expression::abstract_expr::AbstractExpr;
use remainder::expression::generic_expr::Expression;
use remainder::layouter::compiling::LayouterCircuit;
use remainder::layouter::component::{Component, ComponentSet};
use remainder::layouter::nodes::circuit_inputs::{
    InputLayerNode, InputLayerNodeData,
};
use remainder::layouter::nodes::circuit_outputs::OutputNode;
use remainder::layouter::nodes::node_enum::NodeEnum;
use remainder::layouter::nodes::sector::Sector;
use remainder::layouter::nodes::{CircuitNode, Context};
use remainder::prover::helpers::test_circuit;
use remainder::utils::get_input_shred_and_data;
use remainder_shared_types::{Field, Fr};

pub struct ProductCheckerComponent<F: Field> {
    pub sector: Sector<F>,
}

impl<F: Field> ProductCheckerComponent<F> {
    /// Checks that factor1 * factor2 - expected_product == 0.
    pub fn new(
        ctx: &Context,
        factor1: &dyn CircuitNode,
        factor2: &dyn CircuitNode,
        expected_product: &dyn CircuitNode,
    ) -> Self {
        let sector = Sector::new(ctx, &[factor1, factor2, expected_product], |input_nodes| {
            assert_eq!(input_nodes.len(), 3);
            Expression::<F, AbstractExpr>::products(vec![input_nodes[0], input_nodes[1]])
                - input_nodes[2].expr()
        });
        Self { sector }
    }
}

impl<F: Field, N> Component<N> for ProductCheckerComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}

#[test]
fn test_product_checker() {
    let factor1 = vec![
        Fr::from(3u64),
        Fr::from(2u64),
        Fr::from(3u64),
        Fr::from(2u64),
    ];
    let factor2 = vec![
        Fr::from(5u64),
        Fr::from(6u64),
        Fr::from(5u64),
        Fr::from(6u64),
    ];
    let product = vec![
        Fr::from(15u64),
        Fr::from(12u64),
        Fr::from(15u64),
        Fr::from(12u64),
    ];
    // note that this test will pass if the MLEs have length only two!

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None);
        let (factor1_shred, factor1_shred_data) =
            get_input_shred_and_data(factor1.clone(), ctx, &input_layer);
        let (factor2_shred, factor2_shred_data) =
            get_input_shred_and_data(factor2.clone(), ctx, &input_layer);
        let (product_shred, product_shred_data) =
            get_input_shred_and_data(product.clone(), ctx, &input_layer);
        let input_data = InputLayerNodeData::new(
            input_layer.id(),
            vec![factor1_shred_data, factor2_shred_data, product_shred_data],
            None,
        );

        let checker =
            ProductCheckerComponent::new(ctx, &factor1_shred, &factor2_shred, &product_shred);

        let output = OutputNode::new_zero(ctx, &checker.sector);

        let all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            factor1_shred.into(),
            factor2_shred.into(),
            product_shred.into(),
            checker.sector.into(),
            output.into(),
        ];
        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
            vec![input_data],
        )
    });
    test_circuit(circuit, None);
}
