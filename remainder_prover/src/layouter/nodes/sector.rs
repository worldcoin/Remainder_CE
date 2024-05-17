//! The basic building block of a regular gkr circuit. The Sector node

use remainder_shared_types::FieldExt;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    mle::evals::MultilinearExtension,
};

use super::{CircuitNode, ClaimableNode, Context, NodeId};

pub struct Sector<F: FieldExt> {
    id: NodeId,
    expr: Expression<F, AbstractExpr>,
    data: MultilinearExtension<F>,
}

impl<F: FieldExt> Sector<F> {
    pub fn new(
        ctx: &Context,
        inputs: &[&dyn ClaimableNode<F = F>],
        expr_builder: impl FnOnce(Vec<Expression<F, AbstractExpr>>) -> Expression<F, AbstractExpr>,
        data_builder: impl FnOnce(Vec<&MultilinearExtension<F>>) -> MultilinearExtension<F>,
    ) -> Self {
        let node_ids = inputs.iter().map(|node| node.get_expr()).collect();
        let expr = expr_builder(node_ids);
        let input_data = inputs.iter().map(|node| node.get_data()).collect();
        let data = data_builder(input_data);

        Self {
            id: ctx.get_new_id(),
            expr,
            data,
        }
    }
}

impl<F: FieldExt> CircuitNode for Sector<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        self.expr.get_sources()
    }
}

impl<F: FieldExt> ClaimableNode for Sector<F> {
    type F = F;

    fn get_data(&self) -> &MultilinearExtension<Self::F> {
        &self.data
    }

    fn get_expr(&self) -> Expression<Self::F, AbstractExpr> {
        self.expr.clone()
    }
}
