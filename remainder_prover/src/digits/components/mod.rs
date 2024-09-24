use remainder_shared_types::Field;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layouter::{
        component::Component,
        nodes::{sector::Sector, CircuitNode, Context},
    },
};

#[cfg(test)]
mod tests;

/// Component performing digital recomposition of an unsigned integer, i.e. deriving the number from its digits.
pub struct UnsignedRecomposition<F: Field> {
    /// The recomposed numbers
    pub sector: Sector<F>,
}

impl<F: Field> UnsignedRecomposition<F> {
    /// Each of the Nodes in `mles` specifies the digits for a different "decimal place".  Most
    /// significant digit comes first.
    pub fn new(ctx: &Context, mles: &[&dyn CircuitNode], base: u64) -> Self {
        let num_digits = mles.len();
        let sector = Sector::new(ctx, mles, |input_nodes| {
            assert_eq!(input_nodes.len(), num_digits);
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
        println!("{:?} = UnsignedRecomposition sector", sector.id());
        Self { sector }
    }
}

impl<F: Field, N> Component<N> for UnsignedRecomposition<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}

/// Component that checks that the complementary decomposition of a signed integer.
/// To be used in conjunction with [UnsignedRecomposition].
/// See [crate::digits::complementary_decomposition] and [Notion](https://www.notion.so/Constraining-for-the-response-zero-case-using-the-complementary-representation-d77ddfe258a74a9ab949385cc6f7eda4).
/// Add self.sector to the circuit as an output layer to enforce this constraint.
pub struct ComplementaryRecompChecker<F: Field> {
    /// To be added to the circuit as an output layer by the caller.
    pub sector: Sector<F>,
}

impl<F: Field> ComplementaryRecompChecker<F> {
    /// Create a new ComplementaryDecompChecker.
    /// `values` are the original values.
    /// `bits` are the bits of the complementary decomposition.
    /// `unsigned_recomps` are the unsigned recompositions.
    pub fn new(
        ctx: &Context,
        values: &dyn CircuitNode,
        bits: &dyn CircuitNode,
        unsigned_recomps: &dyn CircuitNode,
        base: u64,
        num_digits: usize,
    ) -> Self {
        let mut pow = F::from(1_u64);
        for _ in 0..num_digits {
            pow *= F::from(base);
        }

        let sector = Sector::new(ctx, &[values, bits, unsigned_recomps], |input_nodes| {
            assert_eq!(input_nodes.len(), 3);
            let values_mle_ref = input_nodes[0];
            let bits_mle_ref = input_nodes[1];
            let unsigned_recomps_mle_ref = input_nodes[2];
            Expression::<F, AbstractExpr>::scaled(bits_mle_ref.expr(), pow)
                - unsigned_recomps_mle_ref.expr()
                - values_mle_ref.expr()
        });
        println!("{:?} = ComplementaryRecompChecker sector", sector.id());
        Self { sector }
    }
}

impl<F: Field, N> Component<N> for ComplementaryRecompChecker<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}

/// Ensures that each bit is either 0 or 1. Add self.sector to the circuit as an output layer to
/// enforce this constraint.
pub struct BitsAreBinary<F: Field> {
    /// To be added to the circuit as an output layer by the caller.
    pub sector: Sector<F>,
}

impl<F: Field> BitsAreBinary<F> {
    /// Creates a new BitsAreBinary component.
    pub fn new(ctx: &Context, values_node: &dyn CircuitNode) -> Self {
        let sector = Sector::new(ctx, &[values_node], |nodes| {
            assert_eq!(nodes.len(), 1);
            let values_mle_ref = nodes[0];
            Expression::<F, AbstractExpr>::products(vec![values_mle_ref, values_mle_ref])
                - values_mle_ref.expr()
        });
        println!("{:?} = BitsAreBinary sector", sector.id());
        Self { sector }
    }
}

impl<F: Field, N> Component<N> for BitsAreBinary<F>
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
pub struct DigitsConcatenator<F: Field> {
    /// The sector that concatenates the digits (to be constrained by the lookup)
    pub sector: Sector<F>,
}

impl<F: Field> DigitsConcatenator<F> {
    /// Create a new DigitsConcatenator component.
    pub fn new(ctx: &Context, mles: &[&dyn CircuitNode]) -> Self {
        let num_digits = mles.len();
        let sector = Sector::new(ctx, mles, |digital_places| {
            assert_eq!(digital_places.len(), num_digits);
            Expression::<F, AbstractExpr>::selectors(
                digital_places.iter().map(|node| node.expr()).collect(),
            )
        });
        println!("{:?} = DigitsConcatenator sector", sector.id());
        Self { sector }
    }
}

impl<F: Field, N> Component<N> for DigitsConcatenator<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}
