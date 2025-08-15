//! Contains files responsible for the MPC dealing of Worldcoin's iris code

/// The circuit itself
pub mod circuits;

/// Components that are particular to the worldcoin circuit
pub mod components;

/// Data loading and witness generation
pub mod data;

/// Parameters for the mpc circuit
pub mod parameters;

/// Tests
#[cfg(test)]
pub mod tests;

/// Helper functions for both
pub mod test_helpers;
