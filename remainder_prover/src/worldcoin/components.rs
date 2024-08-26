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

/// Calculates `matmult - thresholds + equality_allowed`, making the result available as self.sector.
/// It is assumed that `matmult` and `thresholds` have the same length, while `equality_allowed` has length 1.
pub struct Thresholder<F: FieldExt> {
    /// The sector that containing the result of the calculation.
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
                let (matmult, thresholds, equality_allowed) = (data[0], data[1], data[2]);
                assert_eq!(matmult.num_vars(), thresholds.num_vars());
                assert_eq!(equality_allowed.num_vars(), 0);
                let e = equality_allowed.get_evals_vector()[0];
                let result = matmult.get_evals_vector().iter().zip(thresholds.get_evals_vector().iter())
                    .map(|(m, t)| *m - *t + e)
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
