//! Contains the worldcoin circuits, and its supporting functions
pub mod circuits;
pub mod components;
pub mod data;
pub mod data_v3;
// IO helpers
pub mod io;
pub mod tests;

const LOG_NUM_DIGITS: usize = 3;
/// The number of digits in the complementary decomposition of the thresholded responses.
pub const NUM_DIGITS: usize = (1 << LOG_NUM_DIGITS) as usize;
/// The base of the complementary decomposition of the thresholded responses.
pub const BASE: u16 = 256;
