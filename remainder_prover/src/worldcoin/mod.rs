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

/// Tests
pub mod tests;

const LOG_NUM_DIGITS: usize = 3;
/// The number of digits in the complementary decomposition of the thresholded responses.
pub const NUM_DIGITS: usize = (1 << LOG_NUM_DIGITS) as usize;
/// The base of the complementary decomposition of the thresholded responses.
pub const BASE: u16 = 256;
