//! Implements binary addition gates.

use std::cmp::max;

use itertools::Itertools;
use remainder_shared_types::Field;

use crate::{
    binary_operations::logical_shift::ShiftNode,
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layouter::{
        component::Component,
        nodes::{
            circuit_outputs::OutputNode, identity_gate::IdentityGateNode, sector::Sector,
            CircuitNode,
        },
    },
};

/// Performs binary addition between two nodes that represent binary values, given the vector of
/// carries as a witness. Works with bit-widths that are powers of 2 up to `2^30 = 1,073,741,824`
/// bits (this constraint is inherited from `ShiftNode`).
///
/// # Requires
/// All inputs are assumed to only contain binary digits (i.e. only values from the set
/// `{F::ZERO, F::ONE}` for a field `F`).
#[derive(Clone, Debug)]
pub struct BinaryAdder<F: Field> {
    adder_sector: Sector<F>,
    shifted_carries: ShiftNode,
    carry_check_sector: Sector<F>,
    carry_check_output: OutputNode,
}

impl<F: Field> BinaryAdder<F> {
    /// Generates a new [BinaryAdder] adding the values in nodes `lhs_bits` and `rhs_bits`, given
    /// the `carry_bits` as a witness.
    pub fn new(
        lhs_bits: &impl CircuitNode,
        rhs_bits: &impl CircuitNode,
        carry_bits: &impl CircuitNode,
    ) -> Self {
        let num_vars = max(
            carry_bits.get_num_vars(),
            max(lhs_bits.get_num_vars(), rhs_bits.get_num_vars()),
        );

        // Shift the carry bits by one to the left to align then so that they can
        // be added along with the corresponding LHS and RHS bit using a full-adder circuit.
        let shifted_carries = ShiftNode::new(num_vars, -1, carry_bits);

        let full_adder_result_sector = Sector::new(
            &[lhs_bits, rhs_bits, shifted_carries.get_output()],
            |nodes| {
                assert_eq!(nodes.len(), 3);

                let b0 = nodes[0];
                let b1 = nodes[1];
                let c = nodes[2];

                let b0_c = Expression::<F, AbstractExpr>::products(vec![b0, c]);
                let b1_c = Expression::<F, AbstractExpr>::products(vec![b1, c]);
                let b0_b1 = Expression::<F, AbstractExpr>::products(vec![b0, b1]);
                let b0_b1_c = Expression::<F, AbstractExpr>::products(vec![b0, b1, c]);

                let two_b0_c = Expression::<F, AbstractExpr>::scaled(b0_c, F::from(2));
                let two_b1_c = Expression::<F, AbstractExpr>::scaled(b1_c, F::from(2));
                let two_b0_b1 = Expression::<F, AbstractExpr>::scaled(b0_b1, F::from(2));
                let four_b0_b1_c = Expression::<F, AbstractExpr>::scaled(b0_b1_c, F::from(4));

                // The following expression is equivalent to: `b0 XOR b1 XOR c`
                b0.expr() + b1.expr() + c.expr() - two_b0_c - two_b1_c - two_b0_b1 + four_b0_b1_c
            },
        );

        // Verify that the carry bits witness provided is consistent.
        let carry_check_sector = Sector::new(
            &[lhs_bits, rhs_bits, shifted_carries.get_output(), carry_bits],
            |nodes| {
                assert_eq!(nodes.len(), 4);

                let b0 = nodes[0];
                let b1 = nodes[1];
                let c = nodes[2];
                let expected_c = nodes[3];

                let b0_b1 = Expression::<F, AbstractExpr>::products(vec![b0, b1]);
                let b0_c = Expression::<F, AbstractExpr>::products(vec![b0, c]);
                let b1_c = Expression::<F, AbstractExpr>::products(vec![b1, c]);
                let two_b0_b1_c = Expression::<F, AbstractExpr>::scaled(
                    Expression::<F, AbstractExpr>::products(vec![b0, b1, c]),
                    F::from(2),
                );

                // The next carry is 1 iff at least 2 of the 3 input bits (`b0`, `b1` and `c`) are
                // 1. The following expression is the multilinear polynomial extending the boolean
                // function described in the previous sentence.
                // TODO: Use something like this after merging Benny's PR:
                // b0 * b1 * c + b0 * b1 * (1 - c) + b0 * (1 - b1) * c + (1 - b0) * b1 * c - expected_c
                b0_b1 + b0_c + b1_c - two_b0_b1_c - expected_c.expr()
            },
        );
        let carry_check_output = OutputNode::new_zero(&carry_check_sector);

        Self {
            adder_sector: full_adder_result_sector,
            shifted_carries,
            carry_check_sector,
            carry_check_output,
        }
    }

    /// Returns a reference to the output of the adder circuit.
    pub fn get_output(&self) -> &Sector<F> {
        &self.adder_sector
    }
}

impl<F: Field, N> Component<N> for BinaryAdder<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode> + From<IdentityGateNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        self.shifted_carries
            .yield_nodes()
            .into_iter()
            .chain([
                self.adder_sector.into(),
                self.carry_check_sector.into(),
                self.carry_check_output.into(),
            ])
            .collect_vec()
    }
}
