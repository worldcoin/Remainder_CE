use itertools::{all, Itertools};
use num_traits::sign;
use remainder_shared_types::FieldExt;

use crate::{
    expression::{abstract_expr::{calculate_selector_values, AbstractExpr}, generic_expr::Expression},
    layouter::{
        component::Component,
        nodes::{
            identity_gate::IdentityGateNode, sector::Sector, CircuitNode, ClaimableNode, Context,
        },
    },
    mle::evals::MultilinearExtension,
    worldcoin::digit_decomposition::NUM_DIGITS,
};

// FIXME is this component even necessary?? not even being used, right?
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
        let sector = Sector::new(
            ctx,
            mles,
            |digital_places| {
                assert_eq!(digital_places.len(), NUM_DIGITS);
                Expression::<F, AbstractExpr>::selectors(
                    digital_places
                        .iter()
                        .map(|node| node.expr())
                        .collect(),
                )
            },
            |digits_at_places| {
                assert_eq!(digits_at_places.len(), NUM_DIGITS);
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

pub struct DigitRecompComponent<F: FieldExt> {
    pub sector: Sector<F>,
}

impl<F: FieldExt> DigitRecompComponent<F> {
    /// Each of the Nodes in `mles` specifies the digits for a different "decimal place".  Most
    /// significant digit comes first.
    pub fn new(ctx: &Context, mles: &[&dyn ClaimableNode<F = F>], base: u64) -> Self {
        let num_digits = mles.len();
        let sector = Sector::new(
            ctx,
            mles,
            |input_nodes| {
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
        println!("DigitsRecompComponent sector = {:?}", sector.id());
        Self { sector }
    }
}

impl<F: FieldExt, N> Component<N> for DigitRecompComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}

// FIXME remove
pub struct EqualityCheckerComponent<F: FieldExt> {
    pub sector: Sector<F>,
}
impl<F: FieldExt> EqualityCheckerComponent<F> {
    /// Checks if two MLEs are equal.
    pub fn new(
        ctx: &Context,
        lhs: &dyn ClaimableNode<F = F>,
        rhs: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let sector = Sector::new(
            ctx,
            &[lhs, rhs],
            |input_nodes| {
                assert_eq!(input_nodes.len(), 2);
                input_nodes[0].expr() - input_nodes[1].expr()
            },
            |data| {
                assert_eq!(data.len(), 2);
                let values = data[0].get_evals_vector()
                    .iter()
                    .zip(data[1].get_evals_vector())
                    .map(|((value_lhs, value_rhs))| {
                        *value_lhs - *value_rhs
                    })
                    .collect_vec();
                assert!(all(values.into_iter(), |val| val == F::ZERO));
                MultilinearExtension::new_sized_zero(data[0].num_vars())
            },
        );

        Self {
            sector,
        }
    }
}

impl<F: FieldExt, N> Component<N> for EqualityCheckerComponent<F>
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


/// Calculates (values + abs_values) + -2 * sign_bits * abs_values
/// (So sign bit of 0 indicates negative, 1 indicates positive).
/// Add self.sector to the circuit as an output layer to enforce this constraint.
pub struct SignCheckerComponent<F: FieldExt> {
    /// To be added to the circuit as an output layer by the caller.
    pub sector: Sector<F>,
}

impl<F: FieldExt> SignCheckerComponent<F> {
    pub fn new(
        ctx: &Context,
        values: &dyn ClaimableNode<F = F>,
        sign_bits: &dyn ClaimableNode<F = F>,
        abs_values: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let sector = Sector::new(
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
                        abs_values_mle_ref,
                        sign_bits_mle_ref,
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
                    .map(|((val, sign_bit), abs_val)| {
                        *val + *abs_val + F::from(2).neg() * sign_bit * abs_val
                    })
                    .collect_vec();
                assert!(all(values.into_iter(), |val| val == F::ZERO));

                MultilinearExtension::new_sized_zero(data[0].num_vars())
            },
        );
        println!("SignCheckerComponent sector = {:?}", sector.id());
        Self { sector }
    }
}

impl<F: FieldExt, N> Component<N> for SignCheckerComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.sector.into()]
    }
}
