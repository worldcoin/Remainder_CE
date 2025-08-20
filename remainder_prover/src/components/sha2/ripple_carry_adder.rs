use crate::expression::abstract_expr::ExprBuilder;
use crate::layouter::builder::{CircuitBuilder, NodeRef};
use itertools::Itertools;

use remainder_shared_types::Field;

#[derive(Clone, Debug)]
pub struct FullAdder {
    s: NodeRef,
    c: NodeRef,
}

impl FullAdder {
    pub fn new<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        x: &NodeRef,
        y: &NodeRef,
        carry: &NodeRef,
    ) -> Self {
        debug_assert!(x.get_num_vars() == 0);
        debug_assert!(y.get_num_vars() == 0);
        debug_assert!(carry.get_num_vars() == 0);

        // ab = a `and` b
        let ab = x.clone().expr() * y.clone().expr();

        // xor_ab1 = a `xor` b
        let xor_ab1 = x.clone().expr() + y.clone().expr()
            - ExprBuilder::constant(F::from(2)) * x.clone().expr() * y.clone().expr();

        // sum = a `xor` b `xor` carry = ab1 `xor` carry
        let bit_sum = builder_ref.add_sector(
            xor_ab1.clone() + carry.clone().expr()
                - ExprBuilder::constant(F::from(2)) * xor_ab1.clone() * carry.clone().expr(),
        );

        // cin_x_xor1  = carry & xor_ab1
        let cin_x_xor1 = carry.expr() * xor_ab1.clone();

        // Carry out = cin_x_xor1 | ab
        let carry_out = builder_ref.add_sector(ab.clone() + cin_x_xor1.clone() - ab * cin_x_xor1);

        Self {
            s: bit_sum,
            c: carry_out,
        }
    }

    pub fn get_output(&self) -> (/* Sum */ NodeRef, /* carry */ NodeRef) {
        (self.s.clone(), self.c.clone())
    }
}

// mode 2^BITWIDTH adder with no input carry and not output carry
pub struct AdderNoCarry<const BITWIDTH: usize> {
    sum_node: NodeRef,
}

impl<const BITWIDTH: usize> AdderNoCarry<BITWIDTH> {
    // x_word and y_word are assumed to be MSB-first decomposition of
    // of the input data.
    pub fn new<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        x_word: &NodeRef,
        y_word: &NodeRef,
    ) -> Self {
        debug_assert!(BITWIDTH.is_power_of_two());
        debug_assert!(x_word.get_num_vars() == BITWIDTH.ilog2() as usize);
        debug_assert!(y_word.get_num_vars() == BITWIDTH.ilog2() as usize);
        let num_vars = BITWIDTH.ilog2() as usize;
        let x_word_wires = builder_ref.add_split_node(x_word, num_vars);
        let y_word_wires = builder_ref.add_split_node(y_word, num_vars);
        let mut c_in = builder_ref.add_sector(ExprBuilder::constant(F::from(0)));

        let mut sum_expr = Vec::<NodeRef>::with_capacity(BITWIDTH);

        for (x, y) in x_word_wires.iter().zip(y_word_wires.iter()).rev() {
            // Wires are in MSB first, hence the .rev()
            let fa = FullAdder::new(builder_ref, x, y, &c_in);
            let (s, c) = fa.get_output();
            c_in = c;
            sum_expr.push(s);
        }

        // Swap to MSB first hence the rev()
        let sum_rewired = sum_expr.iter().rev().map(|n| n.expr()).collect_vec();

        let sum_node = builder_ref.add_sector(ExprBuilder::<F>::selectors(sum_rewired));

        Self { sum_node }
    }

    pub fn get_output(&self) -> NodeRef {
        self.sum_node.clone()
    }
}
