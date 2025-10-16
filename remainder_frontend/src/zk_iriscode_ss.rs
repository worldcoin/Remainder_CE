//! Contains the v3 iriscode circuit implemented using an RLC of image strips.

/// The circuit itself
pub mod circuits;

/// Components that are particular to the worldcoin circuit
pub mod components;

/// Data loading and witness generation
pub mod data;

/// IO helpers
pub mod io;

/// Decoding functions for the wirings and other constants
pub mod decode;

/// Parameters for the v3 RLC circuit
pub mod parameters;

/// Tests
#[cfg(test)]
pub mod tests;

/// Test helpers (also used in Hyrax tests)
pub mod test_helpers;

/// v3 data loaders
pub mod v3;
