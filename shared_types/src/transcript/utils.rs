use itertools::Itertools;
use sha2::{Digest, Sha256};

use crate::Field;

/// A heuristic number of hash iterations over SHA-256 which is assumed
/// to be incomputable by GKR circuits which are written within Remainder.
///
/// Note that this value should be increased for particularly high-degree or
/// deep circuits.
pub const NUM_SHA_256_ITERATIONS: usize = 1000;

/// This function computes the [NUM_SHA_256_ITERATIONS]-deep SHA-256 hash chain
/// with input `elems`.
pub(crate) fn sha256_hash_chain_on_field_elems<F: Field>(elems: &[F]) -> Vec<F> {
    // First, convert the `elems` which are passed in to a `Vec<u8>`.
    let elems_bytes = elems
        .iter()
        .flat_map(|elem| elem.to_bytes_le())
        .collect_vec();

    // Next, compute the SHA-256 digest of the input elems as bytes.
    let mut hasher = Sha256::new();
    hasher.update(elems_bytes);
    let sha_256_result = hasher.finalize();

    // Next, compute the [NUM_SHA_256_ITERATIONS] iterated SHA-256 digest.
    let final_iterated_bytes = (0..NUM_SHA_256_ITERATIONS).fold(sha_256_result, |acc, _| {
        let mut hasher = Sha256::new();
        hasher.update(acc);
        hasher.finalize()
    });

    // Since the output is 32 bytes and can be out of range,
    // we split instead into two chunks of 16 bytes each and
    // absorb two field elements.
    // TODO(ryancao): Update this by using `REPR_NUM_BYTES` after merging with the testing branch
    let mut hash_bytes_first_half = [0; 32];
    let mut hash_bytes_second_half = [0; 32];

    hash_bytes_first_half[..16].copy_from_slice(&final_iterated_bytes.to_vec()[..16]);
    hash_bytes_second_half[..16].copy_from_slice(&final_iterated_bytes.to_vec()[16..]);

    vec![
        F::from_bytes_le(hash_bytes_first_half.as_ref()),
        F::from_bytes_le(hash_bytes_second_half.as_ref()),
    ]
}
