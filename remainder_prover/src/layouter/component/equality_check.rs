//! A Component for Asserting within a Circuit that 2 Sectors have equivilent data

use remainder_shared_types::FieldExt;

use crate::{
    layouter::nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode, Context},
    mle::evals::MultilinearExtension,
};

use super::Component;

/// A component for Asserting within a Circuit that 2 Sectors have equivilent data
///
/// Will create a Sector that subtracts the two sectors together and then attach
/// that sector to an output layer with a ZeroMle associated with it.
pub struct EqualityComponent<F: FieldExt> {
    subtraction_sector: Sector<F>,
    output: OutputNode<F>,
}

impl<F: FieldExt> EqualityComponent<F> {
    /// Creates a new EqualityComponent based on two input sectors
    pub fn new(ctx: &Context, inputs: [&Sector<F>; 2]) -> Self {
        let subtraction_sector = Sector::new(
            ctx,
            &[inputs[0], inputs[1]],
            |_ids| todo!(),
            |_data| MultilinearExtension::new_zero(),
        );
        let output = OutputNode::new(ctx, &subtraction_sector);

        Self {
            subtraction_sector,
            output,
        }
    }
}

impl<F: FieldExt, N> Component<N> for EqualityComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.subtraction_sector.into(), self.output.into()]
    }
}
