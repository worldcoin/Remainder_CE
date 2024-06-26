use remainder::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layouter::{component::Component, nodes::ClaimableNode},
};
use remainder_shared_types::FieldExt;

use remainder::{
    layouter::nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode, Context},
    mle::evals::MultilinearExtension,
};

pub struct AttributeConsistencyComponent<F: FieldExt> {
    attr_cons_sector: Sector<F>,
}

impl<F: FieldExt> AttributeConsistencyComponent<F> {
    pub fn new(
        ctx: &Context,
        permuted_input: impl ClaimableNode<F = F>,
        decision_node_paths: impl ClaimableNode<F = F>,
    ) -> Self {
        let attr_cons_sector = Sector::new(
            ctx,
            &[&permuted_input, &decision_node_paths],
            |attr_cons_inputs| {
                assert_eq!(attr_cons_inputs.len(), 2);

                let permuted_input_data_mle_trees = attr_cons_inputs[0];
                let decision_node_paths_mle_trees = attr_cons_inputs[1];
                Expression::<F, AbstractExpr>::mle(permuted_input_data_mle_trees)
                    - Expression::<F, AbstractExpr>::mle(decision_node_paths_mle_trees)
            },
            |_data| MultilinearExtension::new_zero(),
        );

        Self { attr_cons_sector }
    }
}

impl<F: FieldExt, N> Component<N> for AttributeConsistencyComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.attr_cons_sector.into()]
    }
}
