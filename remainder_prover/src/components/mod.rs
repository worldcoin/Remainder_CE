//! Module for components that can be used to build a circuit.
use remainder_shared_types::Field;

use crate::layouter::nodes::{sector::Sector, CircuitNode, Context};

/// Use this component to check if the values of two ClaimableNodes are equal, by adding self.sector
/// to the circuit as an output layer.
pub struct EqualityChecker<F: Field> {
    /// To be added to the circuit as an output layer by the caller.
    pub sector: Sector<F>,
}

impl<F: Field> EqualityChecker<F> {
    /// Create a new EqualityChecker.
    pub fn new(ctx: &Context, lhs: &dyn CircuitNode, rhs: &dyn CircuitNode) -> Self {
        let sector = Sector::new(ctx, &[lhs, rhs], |nodes| {
            assert_eq!(nodes.len(), 2);
            nodes[0].expr() - nodes[1].expr()
        });
        println!("{:?} = EqualityChecker sector", sector.id());
        Self { sector }
    }
}
