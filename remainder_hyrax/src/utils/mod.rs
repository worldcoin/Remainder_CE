pub mod vandermonde;

/// Returns `true` if the parallel feature is on for the [remainder_hyrax]
/// crate.
pub fn is_parallel_feature_on() -> bool {
    #[cfg(feature = "parallel")]
    return true;

    #[cfg(not(feature = "parallel"))]
    return false;
}
