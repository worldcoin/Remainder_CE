use remainder_shared_types::Field;

use crate::{
    expression::generic_expr::Expression,
    layouter::builder::{CircuitBuilder, NodeRef},
};

/// Digits-related circuit components
pub struct DigitComponents;

impl DigitComponents {
    /// Digital recomposition of an unsigned integer, i.e. deriving the number from its digits.
    /// Each of the Nodes in `mles` specifies the digits for a different
    /// "decimal place".  Most significant digit comes first.
    pub fn unsigned_recomposition<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        mles: &[&NodeRef<F>],
        base: u64,
    ) -> NodeRef<F> {
        let num_digits = mles.len();

        let b_s_initial_acc = AbstractExpression::<F>::constant(F::ZERO);

        let sector_expr = mles.into_iter().enumerate().fold(
            b_s_initial_acc,
            |acc_expr, (bit_idx, bin_decomp_node)| {
                let power = F::from(base.pow((num_digits - (bit_idx + 1)) as u32));
                acc_expr + *bin_decomp_node * power
            },
        );
        let sector = builder_ref.add_sector(sector_expr);
        sector
    }

    /// Checks that the complementary decomposition of a signed
    /// integer. To be used in conjunction with [Self::unsigned_recomposition]. See
    /// [crate::digits::complementary_decomposition] and
    /// [Notion](https://www.notion.so/Constraining-for-the-response-zero-case-using-the-complementary-representation-d77ddfe258a74a9ab949385cc6f7eda4).
    /// Add self.sector to the circuit as an output layer to enforce this constraint.
    /// Create a new ComplementaryDecompChecker. `values` are the original
    /// values. `bits` are the bits of the complementary decomposition.
    /// `unsigned_recomps` are the unsigned recompositions.
    pub fn complementary_recomp_check<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        values: &NodeRef<F>,
        bits: &NodeRef<F>,
        unsigned_recomps: &NodeRef<F>,
        base: u64,
        num_digits: usize,
    ) -> NodeRef<F> {
        let mut pow = F::from(1_u64);
        for _ in 0..num_digits {
            pow *= F::from(base);
        }

        let sector = builder_ref.add_sector(bits * pow - unsigned_recomps - values);
        sector
    }

    /// Ensures that each bit is either 0 or 1. Add self.sector to the circuit as an
    /// output layer to enforce this constraint.
    pub fn bits_are_binary<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        values_node: &NodeRef<F>,
    ) -> NodeRef<F> {
        let sector = builder_ref.add_sector(values_node * values_node - values_node);
        sector
    }

    /// A component that concatenates all the separate digit MLEs (there is one for
    /// each digital place) into a single MLE using a selector tree. (Necessary to
    /// interact with logup).
    pub fn digits_concatenator<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        mles: &[&NodeRef<F>],
    ) -> NodeRef<F> {
        let sector = builder_ref.add_sector(Expression::<F, AbstractExpr>::binary_tree_selector(
            mles.to_vec(),
        ));
        sector
    }
}
