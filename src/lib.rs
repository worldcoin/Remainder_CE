#![warn(missing_docs)]
//!Remainder: A fast GKR based library for building zkSNARKS for ML applications

use ark_ff::PrimeField;

pub mod expression;
pub mod layer;
pub mod mle;
pub mod sumcheck;

///External definition of Field element trait, will remain an Alias for now
pub trait FieldExt: PrimeField {}