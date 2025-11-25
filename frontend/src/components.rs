//! Module for components that can be used to build a circuit.

use shared_types::Field;

use crate::layouter::builder::{CircuitBuilder, NodeRef};
/// Components relating to digital range checking, decomposition and recomposition
pub mod digits;
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
