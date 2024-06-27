use itertools::Itertools;
use remainder::{
    expression::abstract_expr::ExprBuilder,
    layouter::{component::Component, nodes::ClaimableNode},
};
use remainder_shared_types::FieldExt;

use remainder::{
    layouter::nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode, Context},
    mle::evals::MultilinearExtension,
};

/// compute the value of the positive recomp of the binary decompositions
/// to check that they are in fact the correct decompositions
pub struct PosBinaryRecompComponent<F: FieldExt> {
    bin_recomp_sector: Sector<F>,
}

impl<F: FieldExt> PosBinaryRecompComponent<F> {
    pub fn new(ctx: &Context, bin_decomp_wo_sign_bit: [&dyn ClaimableNode<F = F>; 15]) -> Self {
        let bin_recomp_sector = Sector::new(
            ctx,
            &bin_decomp_wo_sign_bit,
            |bin_decomp_mles| {
                // ignore the signed bit
                assert_eq!(bin_decomp_mles.len(), 15);

                // --- Let's just do a linear accumulator for now ---
                // TODO!(ryancao): Rewrite this expression but as a tree
                let b_s_initial_acc = ExprBuilder::<F>::constant(F::ZERO);

                bin_decomp_mles.into_iter().rev().enumerate().fold(
                    b_s_initial_acc,
                    |acc_expr, (bit_idx, bin_decomp_mle)| {
                        // --- Coeff MLE ref (i.e. b_i) ---
                        let b_i_mle_expression_ptr = bin_decomp_mle.expr();

                        // --- Compute (coeff) * 2^{14 - bit_idx} ---
                        let base = F::from(2_u64.pow(14 - bit_idx as u32));
                        let b_s_times_coeff_times_base =
                            ExprBuilder::<F>::scaled(b_i_mle_expression_ptr, base);

                        acc_expr + b_s_times_coeff_times_base
                    },
                )
            },
            |data| {
                let init_vec = vec![F::ZERO; data[0].get_evals_vector().len()];

                let result_iter =
                    data.into_iter()
                        .rev()
                        .enumerate()
                        .fold(init_vec, |acc, (bit_idx, cur_bit)| {
                            let base = F::from(2_u64.pow(14 - bit_idx as u32));
                            acc.into_iter()
                                .zip(cur_bit.get_evals_vector().into_iter())
                                .map(|(elem, elem_curr_bit)| elem + base * elem_curr_bit)
                                .collect_vec()
                        });

                MultilinearExtension::new(result_iter)
            },
        );

        Self { bin_recomp_sector }
    }
}

impl<F: FieldExt, N> Component<N> for PosBinaryRecompComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.bin_recomp_sector.into()]
    }
}

/// checks that two sectors are indeed the same
pub struct EqualityComponent<F: FieldExt> {
    equality_sector: Sector<F>,
}

impl<F: FieldExt> EqualityComponent<F> {
    pub fn new(ctx: &Context, inputs: [&dyn ClaimableNode<F = F>; 2]) -> Self {
        let equality_sector = Sector::new(
            ctx,
            &inputs,
            |equality_inputs| {
                assert_eq!(equality_inputs.len(), 2);

                // --- Let's just do a linear accumulator for now ---
                // TODO!(ryancao): Rewrite this expression but as a tree
                ExprBuilder::<F>::mle(equality_inputs[0])
                    - ExprBuilder::<F>::mle(equality_inputs[1])
            },
            |data| {
                assert_eq!(data.len(), 2);
                MultilinearExtension::new_zero()
            },
        );

        Self { equality_sector }
    }
}

impl<F: FieldExt, N> Component<N> for EqualityComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.equality_sector.into()]
    }
}

/// This builder computes the value
///     (1 - b_s)(`pos_recomp` - `diff`) + `b_s`(`pos_recomp` + `diff`)
/// Note that this is equivalent to
///     `pos_recomp` - `diff` + 2 * `sign_bit` * `diff`.
/// conceptually b_s is the sign bit of the binary decomposition, and
/// it checks that diff and bin_decomp match each other
pub struct BinRecompCheckerComponent<F: FieldExt> {
    bin_recomp_checker_sector: Sector<F>,
}

impl<F: FieldExt> BinRecompCheckerComponent<F> {
    pub fn new(
        ctx: &Context,
        positive_recomp: impl ClaimableNode<F = F>,
        signed_bit: impl ClaimableNode<F = F>,
        diff: impl ClaimableNode<F = F>,
    ) -> Self {
        let bin_recomp_checker_sector = Sector::new(
            ctx,
            &[&positive_recomp, &signed_bit, &diff],
            |recomp_checker_inputs| {
                assert_eq!(recomp_checker_inputs.len(), 3);

                let positive_recomp_mle = recomp_checker_inputs[0];
                let signed_bit_mle = recomp_checker_inputs[1];
                let diff_mle = recomp_checker_inputs[2];

                // --- LHS of addition ---
                let pos_recomp_minus_diff = positive_recomp_mle.expr() - diff_mle.expr();

                // --- RHS of addition ---
                let sign_bit_times_diff_ptr =
                    ExprBuilder::<F>::products(vec![signed_bit_mle, diff_mle]);
                let two_times_sign_bit_times_diff =
                    ExprBuilder::<F>::scaled(sign_bit_times_diff_ptr, F::from(2));

                pos_recomp_minus_diff + two_times_sign_bit_times_diff
            },
            |data| {
                assert_eq!(data.len(), 3);
                MultilinearExtension::new_zero()
            },
        );

        Self {
            bin_recomp_checker_sector,
        }
    }
}

impl<F: FieldExt, N> Component<N> for BinRecompCheckerComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.bin_recomp_checker_sector.into()]
    }
}
