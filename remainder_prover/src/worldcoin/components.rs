use remainder_shared_types::Field;

use crate::layouter::{
    component::Component,
    nodes::{sector::Sector, CircuitNode},
};

/// Calculates `matmult - thresholds`, making the result available as self.sector.
/// It is assumed that `matmult` and `thresholds` have the same length.
pub struct Subtractor<F: Field> {
    /// The sector that containing the result of the calculation.
    pub sector: Sector<F>,
}

impl<F: Field> Subtractor<F> {
    /// Create a new [Subtractor] component.
    pub fn new(a: &dyn CircuitNode, b: &dyn CircuitNode) -> Self {
        let sector = Sector::new(&[a, b], |nodes| {
            assert_eq!(nodes.len(), 2);
            nodes[0].expr() - nodes[1].expr()
        });
        println!("{:?} = Thresholder sector", sector.id());
        Self { sector }
    }
}

impl<F: Field, N> Component<N> for Subtractor<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}
