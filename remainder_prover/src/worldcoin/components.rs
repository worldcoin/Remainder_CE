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

/// Calculates `matmult - thresholds`, making the result available as self.sector.
/// It is assumed that `matmult` and `thresholds` have the same length.
pub struct Subtractor<F: FieldExt> {
    /// The sector that containing the result of the calculation.
    pub sector: Sector<F>,
}

impl<F: FieldExt> Subtractor<F> {
    /// Create a new [Thresholder] component.
    pub fn new(ctx: &Context, a: &dyn ClaimableNode<F = F>, b: &dyn ClaimableNode<F = F>) -> Self {
        let sector = Sector::new(
            ctx,
            &[a, b],
            |nodes| {
                assert_eq!(nodes.len(), 2);
                nodes[0].expr() - nodes[1].expr()
            },
            |data| {
                assert_eq!(data.len(), 2);
                let (matmult, thresholds) = (data[0], data[1]);
                assert_eq!(matmult.num_vars(), thresholds.num_vars());
                let result = matmult.get_evals_vector().iter().zip(thresholds.get_evals_vector().iter())
                    .map(|(m, t)| *m - *t)
                    .collect_vec();
                MultilinearExtension::new(result)
            },
        );
        println!("{:?} = Thresholder sector", sector.id());
        Self { sector }
    }
}

impl<F: FieldExt, N> Component<N> for Subtractor<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}
