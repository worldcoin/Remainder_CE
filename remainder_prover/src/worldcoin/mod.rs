//! Contains the worldcoin circuits, and its supporting functions

/// The circuit itself
pub mod circuits;

/// Components that are particular to the worldcoin circuit
pub mod components;

/// Data loading and witness generation
pub mod data;

/// Data loading and witness generation for v3
pub mod data_v3;

/// IO helpers
pub mod io;

/// Parameters for the v2 circuit
pub mod parameters_v2;

/// Tests
pub mod tests;

