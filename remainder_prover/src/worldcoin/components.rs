use itertools::Itertools;
use remainder_shared_types::FieldExt;

use crate::{
    layouter::{
        component::Component,
        nodes::{
            sector::Sector, CircuitNode, ClaimableNode, Context,
        },
    },
    mle::evals::MultilinearExtension,
};

/// Calculates `a - b + c`, making the result available as self.sector.
pub struct Thresholder<F: FieldExt> {
    /// The sector that calculates a - b + c
    pub sector: Sector<F>,
}

impl<F: FieldExt> Thresholder<F> {
    /// Create a new [Thresholder] component.
    pub fn new(ctx: &Context, a: &dyn ClaimableNode<F = F>, b: &dyn ClaimableNode<F = F>, c: &dyn ClaimableNode<F = F>) -> Self {
        let sector = Sector::new(
            ctx,
            &[a, b, c],
            |nodes| {
                assert_eq!(nodes.len(), 3);
                nodes[0].expr() - nodes[1].expr() + nodes[2].expr()
            },
            |data| {
                assert_eq!(data.len(), 3);
                let result = data[0].get_evals_vector().iter().zip(data[1].get_evals_vector().iter()).zip(data[2].get_evals_vector().iter())
                    .map(|((a, b), c)| *a - *b + *c)
                    .collect_vec();
                MultilinearExtension::new(result)
            },
        );
        println!("{:?} = Thresholder sector", sector.id());
        Self { sector }
    }
}

impl<F: FieldExt, N> Component<N> for Thresholder<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}
