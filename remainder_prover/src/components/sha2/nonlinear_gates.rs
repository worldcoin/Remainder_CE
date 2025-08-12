//! Implementation of Different non-linear Gates used in SHA-2 family of
//! circuits as described in NIST SP-180-4
//! https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf

#![allow(non_snake_case)]

use crate::binary_operations::{logical_shift::ShiftNode, rotate_bits::RotateNode};
use crate::expression::abstract_expr::ExprBuilder;
use crate::layouter::builder::{CircuitBuilder, NodeRef};

use remainder_shared_types::Field;

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

        // Compute (x `and` y) + (NOT x `and` Z) Note that x only
        // selects one bit from either x or z, so the output is
        // guaranteed to be in {0,1}
        let ch_sector = builder_ref.add_sector(x_AND_y + (z_vars.expr() * NOT_x.expr()));

        Self { ch_sector }
    }

    pub fn get_output(&self) -> NodeRef {
        self.ch_sector.clone()
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

        // Computes the normalized version of xor
        let maj_sector = builder_ref.add_sector(xy + yz + xz - const_2 * xyz * x_p_y_p_z);

        Self { maj_sector }
    }

    pub fn get_output(&self) -> NodeRef {
        self.maj_sector.clone()
    }
}

const fn sha_words_2_num_vars(value: usize) -> usize {
    if value == 32 {
        5
    } else if value == 64 {
        6
    } else {
        panic!("Invalid SHA wordsize")
    }
}

/// The Capital Sigma function described on Printed Page Number 10
/// in NIST SP-180.4. The const parameters have following meaning
///  ROTR1 : Value of rotation in first ROTR
///  ROTR2 : Value of rotation in second ROTR
///  ROTR3 : Value of rotation in third ROTR
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

        let sigma_sector = builder_ref.add_sector(
            rotr1.get_output().expr() + rotr2.get_output().expr() + rotr3.get_output().expr(),
        );

        Self { sigma_sector }
    }

    pub fn get_output(&self) -> NodeRef {
        self.sigma_sector.clone()
    }
}

/// The Small Sigma function described on Printed Page Number 10
/// in NIST SP-180.4. The const parameters have following meaning
///  ROTR1 : Value of rotation in first ROTR
///  ROTR2 : Value of rotation in second ROTR
///  SHR3 : Value of rotation in third ROTR
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

        let sigma_sector = builder_ref.add_sector(
            rotr1.get_output().expr() + rotr2.get_output().expr() + shr3.get_output().expr(),
        );

        Self { sigma_sector }
    }

    pub fn get_output(&self) -> NodeRef {
        self.sigma_sector.clone()
    }
}
