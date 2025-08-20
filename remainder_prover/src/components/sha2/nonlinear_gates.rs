//! Implementation of Different non-linear Gates used in SHA-2 family of
//! circuits as described in NIST SP-180-4
//! https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf

#![allow(non_snake_case)]

use crate::binary_operations::{logical_shift::ShiftNode, rotate_bits::RotateNode};
use crate::expression::abstract_expr::ExprBuilder;
use crate::layouter::builder::{CircuitBuilder, NodeRef};
use std::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};

use remainder_shared_types::Field;

const fn sha_words_2_num_vars(value: usize) -> usize {
    if value == 32 {
        5
    } else if value == 64 {
        6
    } else {
        panic!("Invalid SHA wordsize")
    }
}

/// rotate right by count if count is +ve or by left if its -ne. Count
/// must be less than the number of bits in the size of T
#[inline]
fn rot<T>(value: T, count: i32) -> T
where
    T: Shr<i32, Output = T> + Shl<i32, Output = T> + BitOr<Output = T> + Copy,
{
    let bits = 8 * std::mem::size_of::<T>() as i32;

    debug_assert!(8 * std::mem::size_of::<T>() > count.abs().try_into().unwrap());
    if count > 0 {
        let delta = bits - count;
        (value >> count) | (value << delta)
    } else {
        let count: i32 = (-count).try_into().unwrap_or_default();
        let delta = bits - count;
        (value << count) | (value >> delta)
    }
}

#[derive(Clone, Debug)]
pub struct ChGate {
    ch_sector: NodeRef,
}

impl ChGate {
    pub fn new<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        x_vars: &NodeRef,
        y_vars: &NodeRef,
        z_vars: &NodeRef,
    ) -> Self {
        debug_assert!(x_vars.get_num_vars() == 5 || x_vars.get_num_vars() == 6);
        debug_assert!(y_vars.get_num_vars() == 5 || x_vars.get_num_vars() == 6);
        debug_assert!(z_vars.get_num_vars() == 5 || x_vars.get_num_vars() == 6);

        assert!(x_vars.get_num_vars() == y_vars.get_num_vars());
        assert!(x_vars.get_num_vars() == z_vars.get_num_vars());

        // Compute x `and` y
        let x_AND_y = ExprBuilder::products(vec![x_vars.clone().id(), y_vars.clone().id()]);

        // Compute NOT x = 1 - x
        let NOT_x = builder_ref.add_sector(ExprBuilder::constant(F::ONE) - x_vars.expr());

        // Compute (x `and` y) `xor` (NOT x `and` Z) Note that x only
        // selects one bit from either x or z, so the output is
        // guaranteed to be in {0,1}
        let ch_sector = builder_ref.add_sector(x_AND_y + (z_vars.expr() * NOT_x.expr()));

        Self { ch_sector }
    }

    pub fn get_output(&self) -> NodeRef {
        self.ch_sector.clone()
    }

    pub fn evaluate<T>(x: T, y: T, z: T) -> T
    where
        T: BitAnd<Output = T> + BitXor<Output = T> + Not<Output = T> + Copy,
    {
        (x & y) ^ (!x & z)
    }
}

#[derive(Clone, Debug)]
pub struct MajGate {
    maj_sector: NodeRef,
}

impl MajGate {
    pub fn new<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        x_vars: &NodeRef,
        y_vars: &NodeRef,
        z_vars: &NodeRef,
    ) -> Self {
        debug_assert!(x_vars.get_num_vars() == 5 || x_vars.get_num_vars() == 6);
        debug_assert!(y_vars.get_num_vars() == 5 || x_vars.get_num_vars() == 6);
        debug_assert!(z_vars.get_num_vars() == 5 || x_vars.get_num_vars() == 6);

        assert!(x_vars.get_num_vars() == y_vars.get_num_vars());
        assert!(x_vars.get_num_vars() == z_vars.get_num_vars());

        // We need the gates to produce normalize output (i.e., output
        // in {0,1} basis) therefore the arithmetization of
        //
        // maj(x,y,z) = x*y + y*z + x*z - 2*x*y*z*(x + y + z - 2*x*y*z)
        //
        // As long as inputs to this function are in {0,1} the output
        // will be in {0,1}

        let const_2 = ExprBuilder::constant(F::from(2));
        let xy = x_vars.expr() * y_vars.expr();
        let yz = y_vars.expr() * z_vars.expr();
        let xz = x_vars.expr() * z_vars.expr();
        let xyz = xy.clone() * z_vars.expr();
        let x_p_y_p_z =
            x_vars.expr() + y_vars.expr() + z_vars.expr() - const_2.clone() * xyz.clone();

        let maj_sector = builder_ref.add_sector(xy + yz + xz - const_2 * xyz * x_p_y_p_z);

        Self { maj_sector }
    }

    pub fn get_output(&self) -> NodeRef {
        self.maj_sector.clone()
    }

    pub fn evaluate<T>(x: T, y: T, z: T) -> T
    where
        T: BitAnd<Output = T> + BitXor<Output = T> + Copy,
    {
        (x & y) ^ (y & z) ^ (x & z)
    }
}

/// The Capital Sigma function described on Printed Page Number 10 in
/// NIST SP-180.4. The const parameters have following meaning ROTR1 :
///  Value of rotation in first ROTR ROTR2 : Value of rotation in second
///  ROTR ROTR3 : Value of rotation in third ROTR
///
/// NOTE: In this code the wires are assumed to be numbers in MSB first
/// (i.e., most significant bit is treated as wire 0.
#[derive(Clone, Debug)]
pub struct Sigma<const WORD_SIZE: usize, const ROTR1: i32, const ROTR2: i32, const ROTR3: i32> {
    sigma_sector: NodeRef,
}

impl<const WORD_SIZE: usize, const ROTR1: i32, const ROTR2: i32, const ROTR3: i32>
    Sigma<WORD_SIZE, ROTR1, ROTR2, ROTR3>
{
    pub fn new<F: Field>(builder_ref: &mut CircuitBuilder<F>, x_vars: &NodeRef) -> Self {
        let num_vars: usize = sha_words_2_num_vars(WORD_SIZE);
        let rotr1 = RotateNode::new(builder_ref, num_vars, ROTR1, x_vars);
        let rotr2 = RotateNode::new(builder_ref, num_vars, ROTR2, x_vars);
        let rotr3 = RotateNode::new(builder_ref, num_vars, ROTR3, x_vars);

        let r1_expr = rotr1.get_output().expr();
        let r2_expr = rotr2.get_output().expr();
        let r3_expr = rotr3.get_output().expr();
        let r1_xor_r2 = r1_expr.clone() + r2_expr.clone()
            - ExprBuilder::constant(F::from(2)) * r1_expr * r2_expr;

        let r1_xor_r2_xor_r3 = r1_xor_r2.clone() + r3_expr.clone()
            - ExprBuilder::constant(F::from(2)) * r1_xor_r2 * r3_expr;

        let sigma_sector = builder_ref.add_sector(r1_xor_r2_xor_r3);

        Self { sigma_sector }
    }

    pub fn get_output(&self) -> NodeRef {
        self.sigma_sector.clone()
    }

    pub fn evaluate<T>(x: T) -> T
    where
        T: BitAnd<Output = T>
            + BitOr<Output = T>
            + BitXor<Output = T>
            + Shr<i32, Output = T>
            + Shl<i32, Output = T>
            + Copy,
    {
        rot(x, ROTR1) ^ rot(x, ROTR2) ^ rot(x, ROTR3)
    }
}

/// The Small Sigma function described on Printed Page Number 10
/// in NIST SP-180.4. The const parameters have following meaning
///  ROTR1 : Value of rotation in first ROTR
///  ROTR2 : Value of rotation in second ROTR
///  SHR3 : Value of rotation in third SHR
///
/// NOTE: In this code the wires are assumed to be numbers in MSB first
/// (i.e., most significant bit is represented with wire 0)
#[derive(Clone, Debug)]
pub struct SmallSigma<const WORD_SIZE: usize, const ROTR1: i32, const ROTR2: i32, const SHR3: i32> {
    sigma_sector: NodeRef,
}

impl<const WORD_SIZE: usize, const ROTR1: i32, const ROTR2: i32, const SHR3: i32>
    SmallSigma<WORD_SIZE, ROTR1, ROTR2, SHR3>
{
    pub fn new<F: Field>(builder_ref: &mut CircuitBuilder<F>, x_vars: &NodeRef) -> Self {
        let num_vars: usize = sha_words_2_num_vars(WORD_SIZE);
        let rotr1 = RotateNode::new(builder_ref, num_vars, ROTR1, x_vars);
        let rotr2 = RotateNode::new(builder_ref, num_vars, ROTR2, x_vars);
        let shr3 = ShiftNode::new(builder_ref, num_vars, SHR3, x_vars);

        let r1_expr = rotr1.get_output().expr();
        let r2_expr = rotr2.get_output().expr();
        let s3_expr = shr3.get_output().expr();

        let r1_xor_r2 = r1_expr.clone() + r2_expr.clone()
            - ExprBuilder::constant(F::from(2)) * r1_expr * r2_expr;

        let r1_xor_r2_xor_s3 = r1_xor_r2.clone() + s3_expr.clone()
            - ExprBuilder::constant(F::from(2)) * r1_xor_r2 * s3_expr;

        let sigma_sector = builder_ref.add_sector(r1_xor_r2_xor_s3);

        Self { sigma_sector }
    }

    pub fn get_output(&self) -> NodeRef {
        self.sigma_sector.clone()
    }

    pub fn evaluate<T>(x: T) -> T
    where
        T: BitAnd<Output = T>
            + BitOr<Output = T>
            + BitXor<Output = T>
            + Shr<i32, Output = T>
            + Shl<i32, Output = T>
            + Copy,
    {
        rot(x, ROTR1) ^ rot(x, ROTR2) ^ (x >> SHR3)
    }
}
