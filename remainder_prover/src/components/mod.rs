//! Module for components that can be used to build a circuit.
use itertools::Itertools;
use remainder_shared_types::FieldExt;

use crate::{
    layouter::nodes::{
            sector::Sector, CircuitNode, ClaimableNode, Context,
        },
    mle::evals::MultilinearExtension,
};

/// Use this component to check if the values of two ClaimableNodes are equal, by adding self.sector
/// to the circuit as an output layer.
pub struct EqualityChecker<F: FieldExt> {
    /// To be added to the circuit as an output layer by the caller.
    pub sector: Sector<F>,
}

impl<F: FieldExt> EqualityChecker<F> {
    /// Create a new EqualityChecker.
    pub fn new(
        ctx: &Context,
        lhs: &dyn ClaimableNode<F = F>,
        rhs: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let sector = Sector::new(
            ctx,
            &[lhs, rhs],
            |nodes| {
                assert_eq!(nodes.len(), 2);
                nodes[0].expr() - nodes[1].expr()
            },
            |data| {
                assert_eq!(data.len(), 2);
                let result = data[0].get_evals_vector().iter().zip(data[1].get_evals_vector().iter())
                    .map(|(a, b)| *a - *b)
                    .collect_vec();
                MultilinearExtension::new(result)
            },
        );
        println!("{:?} = EqualityChecker sector", sector.id());
        Self { sector }
    }
}