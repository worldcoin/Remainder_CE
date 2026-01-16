//! Module for components that can be used to build a circuit.

use shared_types::Field;

use crate::layouter::builder::{CircuitBuilder, NodeRef};
/// Components relating to digital range checking, decomposition and recomposition
pub mod digits;

// Defines components for use in building binary circuits.
pub mod binary_operations;

pub mod sha2_gkr;

#[cfg(test)]
mod tests;

/// Components for building circuits
pub struct Components;

impl Components {
    /// Check if the values of two ClaimableNodes are equal, by adding self.sector
    /// to the circuit as an output layer.
    pub fn equality_check<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        lhs: &NodeRef<F>,
        rhs: &NodeRef<F>,
    ) -> NodeRef<F> {
        builder_ref.add_sector(lhs - rhs)
    }
}

/// Use this component to check if the values of two ClaimableNodes are equal, by adding self.sector
/// to the circuit as an output layer.
pub struct EqualityChecker<F: Field> {
    /// To be added to the circuit as an output layer by the caller.
    pub sector: NodeRef<F>,
}

impl<F: Field> EqualityChecker<F> {
    /// Create a new EqualityChecker.
    pub fn new(builder_ref: &mut CircuitBuilder<F>, lhs: &NodeRef<F>, rhs: &NodeRef<F>) -> Self {
        let sector = builder_ref.add_sector(lhs.expr() - rhs.expr());
        Self { sector }
    }

    /// Returns a weak pointer to the output sector of this component.
    pub fn get_output(&self) -> NodeRef<F> {
        self.sector.clone()
    }
}
