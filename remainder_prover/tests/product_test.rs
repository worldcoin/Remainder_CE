use remainder::expression::abstract_expr::AbstractExpr;
use remainder::expression::generic_expr::Expression;
use remainder::layouter::compiling::LayouterCircuit;
use remainder::layouter::component::{Component, ComponentSet};
use remainder::layouter::nodes::circuit_inputs::{InputLayerNode, InputLayerType};
use remainder::layouter::nodes::circuit_outputs::OutputNode;
use remainder::layouter::nodes::node_enum::NodeEnum;
use remainder::layouter::nodes::sector::Sector;
use remainder::layouter::nodes::{CircuitNode, ClaimableNode, Context};
use remainder::mle::evals::MultilinearExtension;
use remainder::prover::helpers::test_circuit;
use remainder::utils::get_input_shred_from_vec;
use remainder_shared_types::{FieldExt, Fr};

pub struct ProductCheckerComponent<F: FieldExt> {
    pub sector: Sector<F>,
}

impl<F: FieldExt> ProductCheckerComponent<F> {
    /// Checks that factor1 * factor2 - expected_product == 0.
    pub fn new(
        ctx: &Context,
        factor1: &dyn ClaimableNode<F>,
        factor2: &dyn ClaimableNode<F>,
        expected_product: &dyn ClaimableNode<F>,
    ) -> Self {
        let sector = Sector::new(
            ctx,
            &[factor1, factor2, expected_product],
            |input_nodes| {
                assert_eq!(input_nodes.len(), 3);
                Expression::<F, AbstractExpr>::products(vec![input_nodes[0], input_nodes[1]])
                    - input_nodes[2].expr()
            },
            |data| MultilinearExtension::new_sized_zero(data[0].num_vars()),
        );
        Self { sector }
    }
}

impl<F: FieldExt, N> Component<N> for ProductCheckerComponent<F>
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
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let factor1_shred = get_input_shred_from_vec(factor1.clone(), ctx, &input_layer);
        let factor2_shred = get_input_shred_from_vec(factor2.clone(), ctx, &input_layer);
        let product_shred = get_input_shred_from_vec(product.clone(), ctx, &input_layer);

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
        ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes)
    });
    test_circuit(circuit, None);
}
