//! Module for useful functions
/// Helpful arithmetic functions.
pub mod arithmetic;
/// Helpful functions for debugging.
pub mod debug;
/// Helpful functions for manipulating MLEs (e.g. padding).
pub mod mle;

#[cfg(test)]
/// Utilities that are only useful for tests
pub(crate) mod test_utils;

use std::fs;

/// Returns whether a particular file exists in the filesystem
pub fn file_exists(file_path: &String) -> bool {
    match fs::metadata(file_path) {
        Ok(file_metadata) => file_metadata.is_file(),
        Err(_) => false,
    }
}

/// Returns `true` if the parallel feature is on for the `remainder`
/// crate.
pub fn is_parallel_feature_on() -> bool {
    #[cfg(feature = "parallel")]
    return true;

    #[cfg(not(feature = "parallel"))]
    return false;
}
