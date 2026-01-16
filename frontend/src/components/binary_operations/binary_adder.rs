//! Implements binary addition gates.

use std::cmp::max;

use shared_types::Field;

use crate::{
    abstract_expr::AbstractExpression,
    components::binary_operations::logical_shift::ShiftNode,
    layouter::builder::{CircuitBuilder, NodeRef},
};
// use remainder::expression::abstract_expr::ExprBuilder;

/// Performs binary addition between two nodes that represent binary values, given the vector of
/// carries as a witness. Works with bit-widths that are powers of 2 up to `2^30 = 1,073,741,824`
/// bits (this constraint is inherited from `ShiftNode`).
///
/// # Requires
/// All inputs are assumed to only contain binary digits (i.e. only values from the set
/// `{F::ZERO, F::ONE}` for a field `F`).
#[derive(Clone, Debug)]
pub struct BinaryAdder<F: Field> {
    adder_sector: NodeRef<F>,
}

impl<F: Field> BinaryAdder<F> {
    /// Generates a new [BinaryAdder] adding the values in nodes `lhs_bits` and `rhs_bits`, given
    /// the `carry_bits` as a witness.
    pub fn new(
        builder_ref: &mut CircuitBuilder<F>,
        lhs_bits: &NodeRef<F>,
        rhs_bits: &NodeRef<F>,
        carry_bits: &NodeRef<F>,
    ) -> Self {
        let num_vars = max(
            carry_bits.get_num_vars(),
            max(lhs_bits.get_num_vars(), rhs_bits.get_num_vars()),
        );

        // Shift the carry bits by one to the left to align then so that they can
        // be added along with the corresponding LHS and RHS bit using a full-adder circuit.
        let shifted_carries = ShiftNode::new(builder_ref, num_vars, -1, carry_bits);

        let b0 = lhs_bits;
        let b1 = rhs_bits;
        let c = shifted_carries.get_output();

        let b0_c = AbstractExpression::products(vec![b0.id(), c.id()]);
        let b1_c = AbstractExpression::products(vec![b1.id(), c.id()]);
        let b0_b1 = AbstractExpression::products(vec![b0.id(), b1.id()]);
        let b0_b1_c = AbstractExpression::products(vec![b0.id(), b1.id(), c.id()]);

        let two_b0_c = AbstractExpression::scaled(b0_c, F::from(2));
        let two_b1_c = AbstractExpression::scaled(b1_c, F::from(2));
        let two_b0_b1 = AbstractExpression::scaled(b0_b1, F::from(2));
        let four_b0_b1_c = AbstractExpression::scaled(b0_b1_c, F::from(4));

        let full_adder_result_sector = builder_ref.add_sector(
            // The following expression is equivalent to: `b0 XOR b1 XOR c`
            b0.expr() + b1.expr() + c.expr() - two_b0_c - two_b1_c - two_b0_b1 + four_b0_b1_c,
        );

        let b0 = lhs_bits;
        let b1 = rhs_bits;
        let c = shifted_carries.get_output();
        let expected_c = carry_bits;

        let carry_check_sector = builder_ref.add_sector(
            // The next carry is 1 iff at least 2 of the 3 input bits (`b0`, `b1` and `c`) are
            // 1. The following expression is the multilinear polynomial extending the boolean
            // function described in the previous sentence.
            b0.expr() * b1.expr() * c.expr()
                + b0.expr() * b1.expr() * -(c.expr() - F::ONE)
                + b0.expr() * -(b1.expr() - F::ONE) * c.expr()
                + -(b0.expr() - F::ONE) * b1.expr() * c.expr()
                - expected_c.expr(),
        );

        builder_ref.set_output(&carry_check_sector);

        Self {
            adder_sector: full_adder_result_sector,
        }
    }

    /// Returns a reference to the output of the adder circuit.
    pub fn get_output(&self) -> NodeRef<F> {
        self.adder_sector.clone()
    }
}
