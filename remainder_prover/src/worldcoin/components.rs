use itertools::Itertools;
use remainder_shared_types::Field;

use crate::{
    expression::{
        abstract_expr::{calculate_selector_values, AbstractExpr},
        generic_expr::Expression,
    },
    layouter::{
        component::Component,
        nodes::{sector::Sector, CircuitNode, Context},
    },
    mle::evals::MultilinearExtension,
};

/// Calculates `matmult - thresholds`, making the result available as self.sector.
/// It is assumed that `matmult` and `thresholds` have the same length.
pub struct Subtractor<F: Field> {
    /// The sector that containing the result of the calculation.
    pub sector: Sector<F>,
}

impl<F: Field> Subtractor<F> {
    /// Create a new [Thresholder] component.
    pub fn new(ctx: &Context, a: &dyn CircuitNode, b: &dyn CircuitNode) -> Self {
        let sector = Sector::new(ctx, &[a, b], |nodes| {
            assert_eq!(nodes.len(), 2);
            nodes[0].expr() - nodes[1].expr()
        });
        println!("{:?} = Thresholder sector", sector.id());
        Self { sector }
    }
}

impl<F: FieldExt, N> Component<N> for SubtractionComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}

/// A component that concatenates all the separate digit MLEs (there is one for each digital place)
/// into a single MLE using a selector tree.
/// (Necessary to interact with logup).
pub struct DigitsConcatenator<F: FieldExt> {
    /// The sector that concatenates the digits (to be constrained by the lookup)
    pub sector: Sector<F>,
}

impl<F: FieldExt> DigitsConcatenator<F> {
    /// Create a new DigitsConcatenator component.
    pub fn new(ctx: &Context, mles: &[&dyn CircuitNode]) -> Self {
        let sector = Sector::new(ctx, mles, |digital_places| {
            assert_eq!(digital_places.len(), NUM_DIGITS);
            Expression::<F, AbstractExpr>::selectors(
                digital_places.iter().map(|node| node.expr()).collect(),
            )
        });
        println!("DigitsConcatenator sector = {:?}", sector.id());
        Self { sector }
    }
}

impl<F: FieldExt, N> Component<N> for DigitsConcatenator<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}

/// Component performing digital recomposition, i.e. deriving the number from its digits.
pub struct DigitalRecompositionComponent<F: FieldExt> {
    /// The recomposed numbers
    pub sector: Sector<F>,
}

impl<F: FieldExt> DigitalRecompositionComponent<F> {
    /// Each of the Nodes in `mles` specifies the digits for a different "decimal place".  Most
    /// significant digit comes first.
    pub fn new(ctx: &Context, mles: &[&dyn CircuitNode], base: u64) -> Self {
        let num_digits = mles.len();
        let sector = Sector::new(ctx, mles, |input_nodes| {
            assert_eq!(input_nodes.len(), num_digits);

            // --- Let's just do a linear accumulator for now ---
            // TODO!(ryancao): Rewrite this expression but as a tree
            let b_s_initial_acc = Expression::<F, AbstractExpr>::constant(F::ZERO);

            input_nodes.into_iter().enumerate().fold(
                b_s_initial_acc,
                |acc_expr, (bit_idx, bin_decomp_mle)| {
                    let b_i_mle_expression_ptr = bin_decomp_mle.expr();
                    let power = F::from(base.pow((num_digits - (bit_idx + 1)) as u32));
                    let b_s_times_coeff_times_base =
                        Expression::<F, AbstractExpr>::scaled(b_i_mle_expression_ptr, power);
                    acc_expr + b_s_times_coeff_times_base
                },
            )
        });
        println!("DigitsRecompComponent sector = {:?}", sector.id());
        Self { sector }
    }
}

impl<F: FieldExt, N> Component<N> for DigitalRecompositionComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}

/// Ensures that each bit is either 0 or 1. Add self.sector to the circuit as an output layer to
/// enforce this constraint.
pub struct BitsAreBinary<F: FieldExt> {
    /// To be added to the circuit as an output layer by the caller.
    pub sector: Sector<F>,
}

impl<F: FieldExt> BitsAreBinary<F> {
    /// Creates a new BitsAreBinary component.
    pub fn new(ctx: &Context, values_node: &dyn CircuitNode) -> Self {
        let sector = Sector::new(ctx, &[values_node], |nodes| {
            assert_eq!(nodes.len(), 1);
            let values_mle_ref = nodes[0];
            Expression::<F, AbstractExpr>::products(vec![values_mle_ref, values_mle_ref])
                - values_mle_ref.expr()
        });
        println!("BitsAreBinary sector = {:?}", sector.id());
        Self { sector }
    }
}

impl<F: FieldExt, N> Component<N> for BitsAreBinary<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}
