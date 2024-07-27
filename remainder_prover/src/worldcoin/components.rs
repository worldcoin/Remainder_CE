use itertools::{all, Itertools};
use remainder_shared_types::FieldExt;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layouter::{
        component::Component,
        nodes::{
            identity_gate::IdentityGateNode, sector::Sector, CircuitNode, ClaimableNode, Context,
        },
    },
    mle::evals::MultilinearExtension,
    worldcoin::digit_decomposition::NUM_DIGITS,
};

pub struct IdentityGateComponent<F: FieldExt> {
    pub identity_gate: IdentityGateNode<F>,
}

impl<F: FieldExt> IdentityGateComponent<F> {
    pub fn new(
        ctx: &Context,
        mle: &impl ClaimableNode<F = F>,
        wirings: Vec<(usize, usize)>,
    ) -> Self {
        let identity_gate = IdentityGateNode::new(ctx, mle, wirings);

        Self { identity_gate }
    }
}

impl<F: FieldExt, N> Component<N> for IdentityGateComponent<F>
where
    N: CircuitNode + From<IdentityGateNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.identity_gate.into()]
    }
}

pub struct DigitRecompComponent<F: FieldExt> {
    pub recomp_sector: Sector<F>,
}

impl<F: FieldExt> DigitRecompComponent<F> {
    pub fn new(ctx: &Context, mles: &[&dyn ClaimableNode<F = F>], base: u64) -> Self {
        let recomp_sector = Sector::new(
            ctx,
            mles,
            |input_nodes| {
                assert_eq!(input_nodes.len(), NUM_DIGITS);

                // --- Let's just do a linear accumulator for now ---
                // TODO!(ryancao): Rewrite this expression but as a tree
                let b_s_initial_acc = Expression::<F, AbstractExpr>::constant(F::ZERO);

                input_nodes.into_iter().enumerate().fold(
                    b_s_initial_acc,
                    |acc_expr, (bit_idx, bin_decomp_mle)| {
                        let b_i_mle_expression_ptr = bin_decomp_mle.expr();
                        let power = F::from(base.pow((NUM_DIGITS - (bit_idx + 1)) as u32));
                        let b_s_times_coeff_times_base =
                            Expression::<F, AbstractExpr>::scaled(b_i_mle_expression_ptr, power);
                        acc_expr + b_s_times_coeff_times_base
                    },
                )
            },
            |data| {
                assert_eq!(data.len(), NUM_DIGITS);
                let init_vec = vec![F::ZERO; data[0].get_evals_vector().len()];

                let result_iter =
                    data.into_iter()
                        .enumerate()
                        .fold(init_vec, |acc, (bit_idx, curr_bits)| {
                            let base_power = F::from(base.pow((NUM_DIGITS - (bit_idx + 1)) as u32));
                            acc.into_iter()
                                .zip(curr_bits.get_evals_vector().into_iter())
                                .map(|(elem, curr_bit)| elem + base_power * curr_bit)
                                .collect_vec()
                        });
                MultilinearExtension::new(result_iter)
            },
        );

        Self { recomp_sector }
    }
}

impl<F: FieldExt, N> Component<N> for DigitRecompComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.recomp_sector.into()]
    }
}

pub struct SignedRecompComponent<F: FieldExt> {
    pub signed_recomp_sector: Sector<F>,
}

impl<F: FieldExt> SignedRecompComponent<F> {
    pub fn new(
        ctx: &Context,
        values: &dyn ClaimableNode<F = F>,
        sign_bits: &dyn ClaimableNode<F = F>,
        abs_values: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let recomp_sector = Sector::new(
            ctx,
            &[values, sign_bits, abs_values],
            |input_nodes| {
                assert_eq!(input_nodes.len(), 3);

                let values_mle_ref = input_nodes[0];
                let sign_bits_mle_ref = input_nodes[1];
                let abs_values_mle_ref = input_nodes[2];

                // (values + abs_values) + -2 * sign_bits * abs_values
                let first_summand = abs_values_mle_ref.expr() + values_mle_ref.expr();
                let second_summand = Expression::<F, AbstractExpr>::scaled(
                    Expression::<F, AbstractExpr>::products(vec![
                        sign_bits_mle_ref,
                        abs_values_mle_ref,
                    ]),
                    F::from(2).neg(),
                );
                first_summand + second_summand
            },
            |data| {
                assert_eq!(data.len(), 3);

                let values = data[0]
                    .get_evals_vector()
                    .iter()
                    .zip(data[1].get_evals_vector())
                    .zip(data[2].get_evals_vector())
                    .map(|((values, signed), abs_val)| {
                        *values + *abs_val + F::from(2).neg() * signed * abs_val
                    })
                    .collect_vec();
                all(values.into_iter(), |val| val == F::ZERO);

                MultilinearExtension::new_sized_zero(data[0].num_vars())
            },
        );

        Self {
            signed_recomp_sector: recomp_sector,
        }
    }
}

impl<F: FieldExt, N> Component<N> for SignedRecompComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.signed_recomp_sector.into()]
    }
}
