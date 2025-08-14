//!
//! Implementation of SHA-256 circuit using bitwise decomposition
//!

use super::nonlinear_gates::*;

pub const WORD_SIZE: usize = 32;

// See https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf for details about constants
pub type Sigma0 = Sigma<WORD_SIZE, 2, 13, 22>;
pub type Sigma1 = Sigma<WORD_SIZE, 6, 11, 25>;
pub type SmallSigma0 = SmallSigma<WORD_SIZE, 7, 18, 3>;
pub type SmallSigma1 = Sigma<WORD_SIZE, 17, 19, 10>;
