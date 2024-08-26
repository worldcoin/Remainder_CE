#![warn(missing_docs)]
//!Remainder: A fast GKR based library for building zkSNARKS for ML
//! applications

#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

// module for tracking all the claims, aggregating them,
// and generating proofs of aggregation
pub mod claims;

// Module containing functions for deriving digital decompositions and building associated circuit
// components.
pub mod digits;

// module for defining the expressions (and relavant helper
// functions) for which the prover and the verifier interact with
pub mod expression;

// module with the Layer trait, which is the main trait for
// which we prove/verify over
pub mod layer;

// For the various output layers to the GKR circuit
pub mod output_layer;

// For the various input layers to the GKR circuit
pub mod input_layer;

// module for defining the MLE (multi-linear extension) data structure
// we keep track of prefix bits, the evaluations there
// it also includes implementations for manipulating the MLE
pub mod mle;

// module where the main prove/verify functions are defined
// it also includes various data structures, e.g. the Layers,
// Witness, InputLayerProof, GKRProof, etc. that's necessary
// for proving and verifying the GKR circuit
pub mod prover;

// module for defining the function compute_sumcheck_message_beta_cascade,
// which is the main function that the prover calls to compute the messages
// for the verifier to check
pub mod sumcheck;

// module for generating and manipulating mles, also includes a function to
// generate the description of circuits
pub mod utils;

// module for tools to help circuit designers build circuits
pub mod builders;

pub mod layouter;

pub use remainder_shared_types;

// module for worldcoin circuits
pub mod worldcoin;
