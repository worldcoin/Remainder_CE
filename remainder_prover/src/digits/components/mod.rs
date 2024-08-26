use itertools::{all, Itertools};
use remainder_shared_types::FieldExt;

use crate::{
    expression::{abstract_expr::{calculate_selector_values, AbstractExpr}, generic_expr::Expression},
    layouter::{
        component::Component,
        nodes::{
            sector::Sector, CircuitNode, ClaimableNode, Context,
        },
    },
    mle::evals::MultilinearExtension,
};

#[cfg(test)]
mod tests;

/// Component performing digital recomposition of an unsigned integer, i.e. deriving the number from its digits.
pub struct UnsignedRecomposition<F: FieldExt> {
    /// The recomposed numbers
    pub sector: Sector<F>,
}

impl<F: FieldExt> UnsignedRecomposition<F> {
    /// Each of the Nodes in `mles` specifies the digits for a different "decimal place".  Most
    /// significant digit comes first.
    pub fn new(ctx: &Context, mles: &[&dyn ClaimableNode<F = F>], base: u64) -> Self {
        let num_digits = mles.len();
        let sector = Sector::new(
            ctx,
            mles,
            |input_nodes| {
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
            },
            |digit_positions| {
                assert_eq!(digit_positions.len(), num_digits);
                let init_vec = vec![F::ZERO; digit_positions[0].get_evals_vector().len()];

                let result_iter =
                    digit_positions.into_iter()
                        .enumerate()
                        .fold(init_vec, |acc, (bit_idx, curr_bits)| {
                            let base_power = F::from(base.pow((num_digits - (bit_idx + 1)) as u32));
                            acc.into_iter()
                                .zip(curr_bits.get_evals_vector().into_iter())
                                .map(|(elem, curr_bit)| elem + base_power * curr_bit)
                                .collect_vec()
                        });
                MultilinearExtension::new(result_iter)
            },
        );
        println!("{:?} = UnsignedRecomposition sector", sector.id());
        Self { sector }
    }
}

impl<F: FieldExt, N> Component<N> for UnsignedRecomposition<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}

/// Component that checks that the complementary decomposition of a signed integer.
/// To be used in conjunction with [UnsignedRecomposition].
/// See [digits::complementary_decomposition] and (Notion)[https://www.notion.so/Constraining-for-the-response-zero-case-using-the-complementary-representation-d77ddfe258a74a9ab949385cc6f7eda4].
/// Add self.sector to the circuit as an output layer to enforce this constraint.
pub struct ComplementaryRecompChecker<F: FieldExt> {
    /// To be added to the circuit as an output layer by the caller.
    pub sector: Sector<F>,
}

impl<F: FieldExt> ComplementaryRecompChecker<F> {
    /// Create a new ComplementaryDecompChecker.
    /// `values` are the original values.
    /// `bits` are the bits of the complementary decomposition.
    /// `unsigned_recomps` are the unsigned recompositions.
    pub fn new(
        ctx: &Context,
        values: &dyn ClaimableNode<F = F>,
        bits: &dyn ClaimableNode<F = F>,
        unsigned_recomps: &dyn ClaimableNode<F = F>,
        base: u64,
        num_digits: usize,
    ) -> Self {
        let mut pow = F::from(1 as u64);
        for _ in 0..num_digits {
            pow = pow * F::from(base);
        }

        let sector = Sector::new(
            ctx,
            &[values, bits, unsigned_recomps],
            |input_nodes| {
                assert_eq!(input_nodes.len(), 3);
                let values_mle_ref = input_nodes[0];
                let bits_mle_ref = input_nodes[1];
                let unsigned_recomps_mle_ref = input_nodes[2];
                Expression::<F, AbstractExpr>::scaled(bits_mle_ref.expr(), pow) - unsigned_recomps_mle_ref.expr() - values_mle_ref.expr()
            },
            |data| {
                assert_eq!(data.len(), 3);

                let values = data[0]
                    .get_evals_vector()
                    .iter()
                    .zip(data[1].get_evals_vector())
                    .zip(data[2].get_evals_vector())
                    .map(|((val, bit), recomp)| {
                        *bit * pow - *recomp - *val
                    })
                    .collect_vec();
                assert!(all(values.into_iter(), |val| val == F::ZERO));

                MultilinearExtension::new_sized_zero(data[0].num_vars())
            },
        );
        println!("{:?} = ComplementaryRecompChecker sector", sector.id());
        Self { sector }
    }
}

impl<F: FieldExt, N> Component<N> for ComplementaryRecompChecker<F>
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
    pub fn new(
        ctx: &Context,
        values_node: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let sector = Sector::new(
            ctx,
            &[values_node],
            |nodes| {
                assert_eq!(nodes.len(), 1);
                let values_mle_ref = nodes[0];
                Expression::<F, AbstractExpr>::products(vec![
                    values_mle_ref,
                    values_mle_ref
                ]) - values_mle_ref.expr()
            },
            |data| {
                assert_eq!(data.len(), 1);
                let values = data[0].get_evals_vector();
                let result = values.iter().map(|val| *val * *val - *val).collect_vec();
                assert!(all(result.into_iter(), |val| val == F::ZERO));
                MultilinearExtension::new_sized_zero(data[0].num_vars())
            },
        );
        println!("{:?} = BitsAreBinary sector", sector.id());
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

/// A component that concatenates all the separate digit MLEs (there is one for each digital place)
/// into a single MLE using a selector tree.
/// (Necessary to interact with logup).
pub struct DigitsConcatenator<F: FieldExt> {
    /// The sector that concatenates the digits (to be constrained by the lookup)
    pub sector: Sector<F>,
}

impl<F: FieldExt> DigitsConcatenator<F> {
    /// Create a new DigitsConcatenator component.
    pub fn new(ctx: &Context, mles: &[&dyn ClaimableNode<F = F>]) -> Self {
        let num_digits = mles.len();
        let sector = Sector::new(
            ctx,
            mles,
            |digital_places| {
                assert_eq!(digital_places.len(), num_digits);
                Expression::<F, AbstractExpr>::selectors(
                    digital_places
                        .iter()
                        .map(|node| node.expr())
                        .collect(),
                )
            },
            |digits_at_places| {
                assert_eq!(digits_at_places.len(), num_digits);
                let all_digits = calculate_selector_values(
                    digits_at_places
                        .iter()
                        .map(|digits_at_place| {
                            digits_at_place.get_evals_vector().clone()
                        })
                        .collect(),
                );
                MultilinearExtension::new(all_digits)
            },
        );
        println!("{:?} = DigitsConcatenator sector", sector.id());
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