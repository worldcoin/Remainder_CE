#![warn(missing_docs)]
#![feature(closure_lifetime_binder)]
//!Remainder: A fast GKR based library for building zkSNARKS for ML applications

#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

pub mod claims;
pub mod expression;
pub mod gate;
pub mod layer;
pub mod mle;
pub mod prover;
pub mod sumcheck;
pub mod utils;
pub use remainder_shared_types;
