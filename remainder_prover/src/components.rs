//! Module for components that can be used to build a circuit.

use remainder_shared_types::Field;

pub mod sha2;
use crate::layouter::builder::{CircuitBuilder, NodeRef};
/// Components relating to digital range checking, decomposition and recomposition
pub mod digits;
#[cfg(test)]
mod tests;

/// Use this component to check if the values of two ClaimableNodes are equal, by adding self.sector
/// to the circuit as an output layer.
pub struct EqualityChecker {
    /// To be added to the circuit as an output layer by the caller.
    pub sector: NodeRef,
}

impl EqualityChecker {
    /// Create a new EqualityChecker.
    pub fn new<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        lhs: &NodeRef,
        rhs: &NodeRef,
    ) -> Self {
        let sector = builder_ref.add_sector(lhs.expr() - rhs.expr());
        Self { sector }
    }

    /// Returns a weak pointer to the output sector of this component.
    pub fn get_output(&self) -> NodeRef {
        self.sector.clone()
    }
}
