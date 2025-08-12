//!
//! Implementation of SHA-256 circuit using bitwise decomposition
//!

use super::nonlinear_gates::*;

pub const SHA256_WORD_SIZE: usize = 32;

// See https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf for details about constants
pub type Sha256Sigma0 = Sigma<SHA256_WORD_SIZE, 2, 13, 22>;
pub type Sha256Sigma1 = Sigma<SHA256_WORD_SIZE, 6, 11, 25>;
pub type Sha256SmallSigma0 = SmallSigma<SHA256_WORD_SIZE, 7, 18, 3>;
pub type Sha256SmallSigma1 = Sigma<SHA256_WORD_SIZE, 17, 19, 10>;
