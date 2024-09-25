//! Contains the worldcoin circuits, and its supporting functions

/// The circuit itself
pub mod circuits;

/// Components that are particular to the worldcoin circuit
pub mod components;

/// Data loading and witness generation
pub mod data;

/// IO helpers
pub mod io;

/// Parameters defining the circuits
pub mod parameters;

/// Parameters for the v2 circuit
pub mod parameters_v2;

/// Parameters for the v3 circuit
pub mod parameters_v3;

/// Tests
#[cfg(test)]
pub mod tests;
