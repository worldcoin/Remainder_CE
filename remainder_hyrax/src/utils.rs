use rand::{rngs::OsRng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

use remainder_shared_types::Fr;
use thiserror::Error;

pub mod vandermonde;

/// Returns `true` if the parallel feature is on for the [remainder_hyrax]
/// crate.
pub fn is_parallel_feature_on() -> bool {
    #[cfg(feature = "parallel")]
    return true;

    #[cfg(not(feature = "parallel"))]
    return false;
}

/// Returns `true` if the parallel feature is on for the [remainder_prover]
/// crate.
pub fn is_remainder_parallel_feature_on() -> bool {
    remainder::utils::is_parallel_feature_on()
}

/// Creates a [ChaCha20Rng] instance, seeded by an [OsRng] source of entropy.
/// (note that `OsRng` calls `/dev/urandom` under the hood)
pub fn get_crypto_chacha20_prng() -> ChaCha20Rng {
    let mut seed = [0u8; 32];
    OsRng.fill_bytes(&mut seed);
    ChaCha20Rng::from_seed(seed)
}

#[derive(Debug, Error)]
pub enum ConversionError {
    #[error("Error: the field element is not within the specified range for conversion")]
    OutOfRangeError,
}

/// Converts a field element `x` (which is enforced to be in the range
/// [0, 2^{16} - 1]) into a `u16` element.
pub fn convert_fr_into_u16(x: Fr) -> Result<u16, ConversionError> {
    let u16_bound_as_field_elem = Fr::from(1 << 16);
    if x >= u16_bound_as_field_elem {
        return Err(ConversionError::OutOfRangeError);
    }
    let result: u16 = 0;

    // Only first two bytes are nonzero, and the bytes are in little-endian
    // (although bits are in big-endian).
    Ok(x.to_bytes()
        .iter()
        .take(2)
        .enumerate()
        .fold(result, |acc, (idx, byte)| {
            acc + 2_u16.pow(idx as u32) * (*byte as u16)
        }))
}
