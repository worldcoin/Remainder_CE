use std::ops::Neg;

use remainder_shared_types::Field;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layouter::{
        component::Component,
        nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode, Context},
    },
    worldcoin_mpc::parameters::GR4_MODULO,
};

/// Calculates masked iris code from iris code and mask, making the result
/// available as self.sector.
/// It is assumed that `iris_code` and `mask` have the same length.
/// the expression is `mask - 2 * (iris_code * mask)`, a.k.a
/// (-2) * iris_code - mask
/// see notion page: Worldcoin specification: iris code versions and Hamming distance
pub struct MaskedIrisCoder<F: Field> {
    /// The sector that containing the result of the calculation.
    pub sector: Sector<F>,
}

impl<F: Field> MaskedIrisCoder<F> {
    /// Create a new [Thresholder] component.
    pub fn new(ctx: &Context, iris_code: &dyn CircuitNode, mask: &dyn CircuitNode) -> Self {
        let sector = Sector::new(ctx, &[iris_code, mask], |nodes| {
            assert_eq!(nodes.len(), 2);
            Expression::<F, AbstractExpr>::scaled(nodes[0].expr(), F::from(2)).neg()
                - nodes[1].expr()
        });

        Self { sector }
    }
}

impl<F: Field, N> Component<N> for MaskedIrisCoder<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}

/// Calculates `iris_code + slope_times_evaluation_point`, making the result available as self.sector.
/// It is assumed that `iris_code` and `slope_times_evaluation_point` have the same length.
pub struct Summer<F: Field> {
    /// The sector that containing the result of the calculation.
    pub sector: Sector<F>,
}

impl<F: Field> Summer<F> {
    /// Create a new [Summer] component.
    pub fn new(
        ctx: &Context,
        iris_code: &dyn CircuitNode,
        slope_times_evaluation_point: &dyn CircuitNode,
    ) -> Self {
        let sector = Sector::new(ctx, &[iris_code, slope_times_evaluation_point], |nodes| {
            assert_eq!(nodes.len(), 2);
            nodes[0].expr() + nodes[1].expr()
        });

        Self { sector }
    }
}

impl<F: Field, N> Component<N> for Summer<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}

/// Checks `computed_shares` and `shared_moduled` are the same modulo const `GR4_MODULO`.
/// Uses an auxilary `quotient` to store their difference divided by `GR4_MODULO`.
/// Makes the result available as self.sector. Also makes a zero output layer.
/// It is assumed that `computed_shares` and `shared_moduled` have the same length.
pub struct ModuloEquator<F: Field> {
    /// The sector that containing the result of the calculation.
    pub sector: Sector<F>,
    /// The output node that is zero.
    pub output_node: OutputNode,
}

impl<F: Field> ModuloEquator<F> {
    /// Create a new [ModuloEquator] component.
    pub fn new(
        ctx: &Context,
        quotient: &dyn CircuitNode,
        computed_shares: &dyn CircuitNode,
        shared_moduled: &dyn CircuitNode,
    ) -> Self {
        let sector = Sector::new(ctx, &[quotient, computed_shares, shared_moduled], |nodes| {
            assert_eq!(nodes.len(), 3);
            Expression::<F, AbstractExpr>::scaled(nodes[0].expr(), F::from(GR4_MODULO))
                + nodes[1].expr()
                - nodes[2].expr()
        });

        let output_node = OutputNode::new_zero(ctx, &sector);

        Self {
            sector,
            output_node,
        }
    }
}

impl<F: Field, N> Component<N> for ModuloEquator<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into(), self.output_node.into()]
    }
}
