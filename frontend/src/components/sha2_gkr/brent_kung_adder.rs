//! Brent-Kung adder is a parallel prefix adder described here https://maths-people.anu.edu.au/~brent/pd/rpb060_IEEETC.pdf
//!
//! For a comparison of different adders, see also https://www.lirmm.fr/arith18/papers/patil-RobustEnergyEffcientAdder.pdf

use super::AdderGateTrait;
use crate::{
    abstract_expr::AbstractExpression,
    layouter::builder::{CircuitBuilder, InputLayerNodeRef, NodeRef},
};
use shared_types::Field;
use std::marker::PhantomData;

#[inline(always)]
fn mul<F: Field>(a: AbstractExpression<F>, b: AbstractExpression<F>) -> AbstractExpression<F> {
    a * b
}

#[inline(always)]
fn xor<F: Field>(a: AbstractExpression<F>, b: AbstractExpression<F>) -> AbstractExpression<F> {
    a.clone() + b.clone() - AbstractExpression::constant(F::from(2)) * a * b
}

// #[inline(always)]
// fn or<F: Field>(a: AbstractExpression<F>, b: AbstractExpression<F>) -> AbstractExpression<F> {
//     a.clone() + b.clone() - a * b
// }

// Input values in MSB format and compute the 4-bit parallel prefix
// adder
fn pp_adder_4_bit<F: Field>(
    builder_ref: &mut CircuitBuilder<F>,
    x_val: NodeRef<F>,
    y_val: NodeRef<F>,
    carry_in: Option<NodeRef<F>>,
) -> (NodeRef<F> /* sum */, NodeRef<F> /*carry  */)
where
    F: Field,
{
    debug_assert!(x_val.get_num_vars() == 2);
    debug_assert!(y_val.get_num_vars() == 2);
    debug_assert!(carry_in
        .as_ref()
        .map(|v| v.get_num_vars() == 0)
        .unwrap_or(true));

    let propagate = builder_ref.add_sector(
        x_val.expr() + y_val.expr()
            - AbstractExpression::constant(F::from(2)) * x_val.expr() * y_val.expr(),
    );
    let generate = builder_ref.add_sector(x_val.expr() * y_val.expr());
    let mut p = builder_ref.add_split_node(&propagate, 2);
    let mut g = builder_ref.add_split_node(&generate, 2);

    // Convert to LSB first mode
    p.reverse();
    g.reverse();

    assert!(p.len() == 4);
    assert!(g.len() == 4);

    // Step 2: Prefix computation
    let p1g0 = mul(p[1].expr(), g[0].expr());
    let p0p1 = mul(p[0].expr(), p[1].expr());
    let p2p3 = mul(p[2].expr(), p[3].expr());

    let g10 = xor(g[1].expr(), p1g0.clone());
    let g20 = mul(p[2].expr(), g10.clone());
    let g20 = xor(g[2].expr(), g20.clone());
    let g30 = mul(p[3].expr(), g20.clone());
    let g30 = xor(g[3].expr(), g30.clone());

    // Step 3: Calculate carries
    let c0 = carry_in
        .map(|v| v.expr())
        .unwrap_or(AbstractExpression::constant(F::ZERO));
    let tmp = mul(p[0].expr(), c0.clone());
    let c1 = xor(g[0].expr(), tmp);
    let tmp = mul(p0p1.clone(), c0.clone());
    let c2 = xor(g10.clone(), tmp);
    let tmp = mul(p[2].expr(), c0.clone());
    let tmp = mul(p0p1.clone(), tmp);
    let c3 = xor(g20, tmp);
    let tmp = mul(p0p1, p2p3);
    let tmp = mul(tmp, c0.clone());
    let c4 = xor(g30, tmp);

    // Reversed bit order
    let sum = vec![
        xor(p[3].expr(), c3),
        xor(p[2].expr(), c2),
        xor(p[1].expr(), c1),
        xor(p[0].expr(), c0),
    ];

    (
        builder_ref.add_sector(AbstractExpression::binary_tree_selector(sum)),
        builder_ref.add_sector(c4),
    )
}

/// Brent-Kung Adder
#[derive(Debug, Clone)]
pub struct BKAdder<const BITWIDTH: usize, F: Field> {
    sum_node: NodeRef<F>,
    _phantom: PhantomData<F>,
}

impl<F: Field> AdderGateTrait<F> for BKAdder<32, F> {
    type IntegralType = u32;

    fn layout_adder_circuit(
        circuit_builder: &mut CircuitBuilder<F>,   // Circuit builder
        x_node: &NodeRef<F>,                       // reference to x in x + y
        y_node: &NodeRef<F>,                       // reference to y in x + y
        carry_layer: Option<InputLayerNodeRef<F>>, // Carry Layer information
    ) -> Self {
        Self::new(circuit_builder, x_node, y_node, carry_layer)
    }

    /// Returns the output of AdderNoCarry
    fn get_output(&self) -> NodeRef<F> {
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

impl<const BITWIDTH: usize, F> BKAdder<BITWIDTH, F>
where
    F: Field,
{
    fn new(
        builder_ref: &mut CircuitBuilder<F>,
        x_word: &NodeRef<F>,
        y_word: &NodeRef<F>,
        _carry_layer: Option<InputLayerNodeRef<F>>,
    ) -> Self {
        assert!(
            BITWIDTH % 8 == 0,
            "Only bitwidths of multiple of 8 are supported"
        );

        assert!(
            x_word.get_num_vars() == BITWIDTH.ilog2() as usize,
            "The number of variables must match"
        );

        let chunks = BITWIDTH.ilog2().saturating_sub(2) as usize;

        assert!(chunks > 0);
        let x_4bit_chunks = builder_ref.add_split_node(x_word, chunks);
        let y_4bit_chunks = builder_ref.add_split_node(y_word, chunks);

        let mut sum_chunks = Vec::<NodeRef<F>>::new();
        let mut carry: Option<NodeRef<F>> = None;

        x_4bit_chunks
            .into_iter()
            .rev()
            .zip(y_4bit_chunks.into_iter().rev())
            .for_each(|(x, y)| {
                let (s, c) = pp_adder_4_bit(builder_ref, x, y, carry.clone());
                sum_chunks.push(s);
                carry = Some(c);
            });

        let final_sum = sum_chunks.into_iter().rev().map(|v| v.expr()).collect();

        let final_carry = carry.map(|c| c.expr()).unwrap();

        let final_carry_is_one_or_zero = builder_ref
            .add_sector(final_carry.clone() * (AbstractExpression::constant(F::ONE) - final_carry));

        builder_ref.set_output(&final_carry_is_one_or_zero);

        Self {
            sum_node: builder_ref.add_sector(AbstractExpression::binary_tree_selector(final_sum)),
            _phantom: Default::default(),
        }
    }
}
