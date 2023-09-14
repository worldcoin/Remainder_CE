/// For cache-ing for testing
pub const CACHED_BATCHED_MLES_FILE: &str = "upshot_data/cached_batched_mles.json";
/// For a standardized string
pub fn get_cached_batched_mles_filename_with_exp_size(exp_size: usize) -> String {
    format!("upshot_data/cached_batched_upshot_mles_exp_batch_size_{}.json", exp_size)
}