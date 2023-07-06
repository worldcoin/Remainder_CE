#![warn(missing_docs)]
#![feature(closure_lifetime_binder)]
//!Remainder: A fast GKR based library for building zkSNARKS for ML applications

use ark_bn254::Fr;
use ark_ff::PrimeField;

pub mod expression;
pub mod layer;
pub mod mle;
pub mod sumcheck;

///External definition of Field element trait, will remain an Alias for now
pub trait FieldExt: PrimeField {}

impl FieldExt for Fr {}
