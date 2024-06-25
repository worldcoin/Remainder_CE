use itertools::Itertools;
use remainder::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layouter::{component::Component, nodes::ClaimableNode},
    mle::evals::Evaluations,
};
use remainder_shared_types::FieldExt;

use remainder::{
    layouter::nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode, Context},
    mle::evals::MultilinearExtension,
};

pub struct PosBinaryRecompComponent<F: FieldExt> {
    bin_recomp_sector: Sector<F>,
}

impl<F: FieldExt> PosBinaryRecompComponent<F> {
    pub fn new(ctx: &Context, inputs: [&Sector<F>; 15]) -> Self {
        let inputs_as_claimable_nodes: Vec<&dyn ClaimableNode<F = F>> = inputs
            .iter()
            .map(|&sector| sector as &dyn ClaimableNode<F = F>)
            .collect();

        let bin_recomp_sector = Sector::new(
            ctx,
            &inputs_as_claimable_nodes,
            |bits_16_mles| {
                // ignore the signed bit
                assert_eq!(bits_16_mles.len(), 15);

                // --- Let's just do a linear accumulator for now ---
                // TODO!(ryancao): Rewrite this expression but as a tree
                let b_s_initial_acc = Expression::<F, AbstractExpr>::constant(F::ZERO);

                bits_16_mles.into_iter().rev().enumerate().fold(
                    b_s_initial_acc,
                    |acc_expr, (bit_idx, bin_decomp_mle)| {
                        // --- Coeff MLE ref (i.e. b_i) ---
                        let b_i_mle_expression_ptr =
                            Expression::<F, AbstractExpr>::mle(bin_decomp_mle);

                        // --- Compute (coeff) * 2^{14 - bit_idx} ---
                        let base = F::from(2_u64.pow(14 - bit_idx as u32));
                        let b_s_times_coeff_times_base =
                            Expression::<F, AbstractExpr>::scaled(b_i_mle_expression_ptr, base);

                        acc_expr + b_s_times_coeff_times_base
                    },
                )
            },
            |data| {
                let init_vec = vec![F::ZERO; data[0].get_evals_vector().len()];
                let bits_num_var = data[0].num_vars();

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

                MultilinearExtension::new(Evaluations::new(bits_num_var, result_iter))
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

pub struct EqualityComponent<F: FieldExt> {
    equality_sector: Sector<F>,
}

impl<F: FieldExt> EqualityComponent<F> {
    pub fn new(ctx: &Context, inputs: [&Sector<F>; 2]) -> Self {
        let inputs_as_claimable_nodes: Vec<&dyn ClaimableNode<F = F>> = inputs
            .iter()
            .map(|&sector| sector as &dyn ClaimableNode<F = F>)
            .collect();

        let equality_sector = Sector::new(
            ctx,
            &inputs_as_claimable_nodes,
            |equality_inputs| {
                assert_eq!(equality_inputs.len(), 2);

                // --- Let's just do a linear accumulator for now ---
                // TODO!(ryancao): Rewrite this expression but as a tree
                Expression::<F, AbstractExpr>::mle(equality_inputs[0])
                    - Expression::<F, AbstractExpr>::mle(equality_inputs[1])
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

/// This builder computes the value `pos_recomp` - `diff` + 2 * `sign_bit` * `diff`.
/// Note that this is equivalent to
/// (1 - b_s)(`pos_recomp` - `diff`) + `b_s`(`pos_recomp` + `diff`)
pub struct BinRecompCheckerComponent<F: FieldExt> {
    bin_recomp_checker_sector: Sector<F>,
}

impl<F: FieldExt> BinRecompCheckerComponent<F> {
    pub fn new(ctx: &Context, inputs: [&Sector<F>; 3]) -> Self {
        let inputs_as_claimable_nodes: Vec<&dyn ClaimableNode<F = F>> = inputs
            .iter()
            .map(|&sector| sector as &dyn ClaimableNode<F = F>)
            .collect();

        let bin_recomp_checker_sector = Sector::new(
            ctx,
            &inputs_as_claimable_nodes,
            |recomp_checker_inputs| {
                assert_eq!(recomp_checker_inputs.len(), 3);

                let positive_recomp_mle = recomp_checker_inputs[0];
                let signed_bit_mle = recomp_checker_inputs[1];
                let diff_mle = recomp_checker_inputs[2];

                // --- LHS of addition ---
                let pos_recomp_minus_diff = Expression::<F, AbstractExpr>::mle(positive_recomp_mle)
                    - Expression::<F, AbstractExpr>::mle(diff_mle);

                // --- RHS of addition ---
                let sign_bit_times_diff_ptr =
                    Expression::<F, AbstractExpr>::products(vec![signed_bit_mle, diff_mle]);
                let two_times_sign_bit_times_diff =
                    Expression::<F, AbstractExpr>::scaled(sign_bit_times_diff_ptr, F::from(2));

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
