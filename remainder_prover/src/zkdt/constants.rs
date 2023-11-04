use std::path::Path;

/// For cache-ing for testing
pub const CACHED_BATCHED_MLES_FILE: &str = "upshot_data/cached_batched_mles.json";
/// For a standardized string
pub fn get_cached_batched_mles_filepath_with_exp_size(
    exp_size: usize, 
    upshot_data_dir_path: &Path
) -> String {
    let upshot_data_filepath = upshot_data_dir_path.join(format!("cached_batched_upshot_mles_exp_batch_size_{}.json", exp_size));
    String::from(upshot_data_filepath.to_str().unwrap())
}
/// For tree commitment cache-ing
pub fn get_tree_commitment_filepath_for_tree_number(
    tree_number: usize, 
    tree_commitment_dir_path: &Path
) -> String {
    let tree_commitment_filepath = tree_commitment_dir_path.join(format!("tree_commitment_{}.json", tree_number));
    String::from(tree_commitment_filepath.to_str().unwrap())
}

/// For tree commitment cache-ing
pub fn get_tree_commitment_filepath_for_tree_batch(
    tree_batch_size: usize, 
    tree_batch_number: usize,
    tree_commitment_dir_path: &Path
) -> String {
    let tree_commitment_filepath = tree_commitment_dir_path.join(format!("tree_commitment_batch_num_{}_size_{}.json", tree_batch_number, tree_batch_size));
    String::from(tree_commitment_filepath.to_str().unwrap())
}

/// For sample minibatch cache-ing
pub fn get_sample_minibatch_commitment_filepath_for_batch_size(
    log_batch_size: usize, 
    batch_number: usize, 
    sample_minibatch_commitments_dir: &Path
) -> String {
    let get_sample_minibatch_commitment_filepath = sample_minibatch_commitments_dir.join(format!("sample_minibatch_logsize_{}_commitment_{}.json", log_batch_size, batch_number));
    String::from(get_sample_minibatch_commitment_filepath.to_str().unwrap())
}

/// For sample minibatch cache-ing
pub fn get_sample_minibatch_commitment_filepath_for_batch_size_tree_batch(
    log_batch_size: usize, 
    batch_number: usize, 
    sample_minibatch_commitments_dir: &Path,
    tree_batch_size: usize,
) -> String {
    let get_sample_minibatch_commitment_filepath = sample_minibatch_commitments_dir.join(format!("sample_minibatch_logsize_{}_commitment_{}_num_trees_{}.json", log_batch_size, batch_number, tree_batch_size));
    String::from(get_sample_minibatch_commitment_filepath.to_str().unwrap())
}