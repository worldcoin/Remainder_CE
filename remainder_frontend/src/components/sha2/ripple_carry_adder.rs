use super::AdderGateTrait;
use crate::expression::abstract_expr::ExprBuilder;
use crate::layouter::builder::{CircuitBuilder, InputLayerNodeRef, NodeRef};
use itertools::Itertools;
use remainder_shared_types::Field;
use std::marker::PhantomData;

/// A single bit full adder
#[derive(Clone, Debug)]
pub struct FullAdder {
    s: NodeRef,
    c: NodeRef,
}

impl FullAdder {
    /// Build a full adder circuit where `x` is the first input wire,
    /// `y` is the second input wire and `carry` is the input carry.
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

        // xor_ab1 = a `xor` b. `xor` output is normalized, i.e., its
        // output is guaranteed to be in {0,1} if the input is in {0,1}.
        // In the multiplicative basis where 0 -> 1, and 1 -> -1 (i.e.,
        // the homomorphism x --> (-1)^x ), xor becomes a single
        // multiplication, but such optimizations are outside the scope
        // of this code.
        let xor_ab1 = x.clone().expr() + y.clone().expr()
            - ExprBuilder::constant(F::from(2)) * x.clone().expr() * y.clone().expr();

        // sum = a `xor` b `xor` carry = ab1 `xor` carry
        let bit_sum = builder_ref.add_sector(
            xor_ab1.clone() + carry.clone().expr()
                - ExprBuilder::constant(F::from(2)) * xor_ab1.clone() * carry.clone().expr(),
        );

        // cin_x_xor1  = carry & xor_ab1. Output is naturally normalized
        let cin_x_xor1 = carry.expr() * xor_ab1.clone();

        // Carry out = cin_x_xor1 | ab. Output is normalized.
        let carry_out = builder_ref.add_sector(ab.clone() + cin_x_xor1.clone() - ab * cin_x_xor1);

        Self {
            s: bit_sum,
            c: carry_out,
        }
    }

    /// Return the output and carry of full adder
    pub fn get_output(&self) -> (/* sum */ NodeRef, /* carry */ NodeRef) {
        (self.s.clone(), self.c.clone())
    }
}

/// mod 2^BITWIDTH adder with no input carry and no output carry
#[derive(Clone)]
pub struct RippleCarryAdderMod2w<const BITWIDTH: usize, F> {
    sum_node: NodeRef,
    _phantom: PhantomData<F>,
}

impl<F: Field> AdderGateTrait<F> for RippleCarryAdderMod2w<32, F> {
    type IntegralType = u32;

    fn layout_adder_circuit(
        circuit_builder: &mut CircuitBuilder<F>, // Circuit builder
        x_node: &NodeRef,                        // reference to x in x + y
        y_node: &NodeRef,                        // reference to y in x + y
        _: Option<InputLayerNodeRef>,            // Carry Layer information
    ) -> Self {
        Self::new(circuit_builder, x_node, y_node)
    }

    /// Returns the output of AdderNoCarry
    fn get_output(&self) -> NodeRef {
        self.sum_node.clone()
    }

    fn perform_addition(
        &self,
        _circuit: &mut crate::layouter::builder::Circuit<F>,
        x: u32,
        y: u32,
    ) -> u32 {
        x.wrapping_add(y)
    }
}

impl<const BITWIDTH: usize, F> RippleCarryAdderMod2w<BITWIDTH, F>
where
    F: Field,
{
    /// Creates a BITWIDTH word Integer adder. For SHA-256/224 BITWIDTH is 32, for SHA-512/384, BITWIDTH is 64
    ///
    /// [x_word] and [y_word] are assumed to be MSB-first decomposition of
    /// of the data.
    pub fn new(builder_ref: &mut CircuitBuilder<F>, x_word: &NodeRef, y_word: &NodeRef) -> Self {
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

        // Make sure that c_out of the last round is either 0 or 1
        let c_out_expr = c_in.expr();
        let one_or_zero = c_out_expr.clone() * (ExprBuilder::constant(F::from(1)) - c_out_expr);
        let carry_sector = builder_ref.add_sector(one_or_zero);
        builder_ref.set_output(&carry_sector);

        // Swap to MSB first hence the rev()
        let sum_rewired = sum_expr.iter().rev().map(|n| n.expr()).collect_vec();

        let sum_node = builder_ref.add_sector(ExprBuilder::<F>::selectors(sum_rewired));

        Self {
            sum_node,
            _phantom: Default::default(),
        }
    }
}
