use remainder::{
    expression::abstract_expr::ExprBuilder,
    layouter::{component::Component, nodes::ClaimableNode},
};
use remainder_shared_types::FieldExt;

use remainder::{
    layouter::nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode, Context},
    mle::evals::MultilinearExtension,
};

// TODO use [remainder::digits::components::ComplementaryRecompChecker] instead of BinRecompCheckerComponent.
// (we need to use the complementary decomposition for soundness.)

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
    /// # Arguments
    /// + `unsigned_recomp` is an instance of [remainder::digits::components::UnsignedRecomposition].
    pub fn new(
        ctx: &Context,
        unsigned_recomp: impl ClaimableNode<F = F>,
        signed_bit: impl ClaimableNode<F = F>,
        diff: impl ClaimableNode<F = F>,
    ) -> Self {
        let bin_recomp_checker_sector = Sector::new(
            ctx,
            &[&unsigned_recomp, &signed_bit, &diff],
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
