use remainder_shared_types::FieldExt;

use crate::layouter::{
    component::Component,
    nodes::{identity_gate::IdentityGateNode, CircuitNode, ClaimableNode, Context},
};

pub struct IdentityGateComponent<F: FieldExt> {
    pub identity_gate: IdentityGateNode<F>,
}

impl<F: FieldExt> IdentityGateComponent<F> {
    pub fn new(
        ctx: &Context,
        mle: &impl ClaimableNode<F = F>,
        wirings: Vec<(usize, usize)>,
    ) -> Self {
        let identity_gate = IdentityGateNode::new(ctx, mle, wirings);

        Self { identity_gate }
    }
}

impl<F: FieldExt, N> Component<N> for IdentityGateComponent<F>
where
    N: CircuitNode + From<IdentityGateNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.identity_gate.into()]
    }
}
