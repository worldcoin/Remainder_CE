//! For testing purposes only! Caches the generation of the Catboost

use std::{path::Path, fs};
use remainder_shared_types::FieldExt;
use serde_json::{to_writer, from_reader};
use crate::{zkdt::{constants::get_cached_batched_mles_filepath_with_exp_size, input_data_to_circuit_adapter::convert_zkdt_circuit_data_into_mles, data_pipeline::dt2zkdt::{RawTreesModel, load_raw_trees_model, load_raw_samples, RawSamples, TreesModel, CircuitizedTrees, to_samples, circuitize_samples, Samples}}, utils::file_exists};
use super::{input_data_to_circuit_adapter::{MinibatchData, load_upshot_data_single_tree_batch, ZKDTCircuitData, BatchedZKDTCircuitMles}, constants::CACHED_BATCHED_MLES_FILE};

/// Writes the results of the [`load_upshot_data_single_tree_batch`] function call
/// to a file for ease of reading (i.e. faster testing, mostly lol)
pub fn write_mles_batch_catboost_single_tree<F: FieldExt>() {

    // Minibatch size of 2^1, 0th minibatch
    let minibatch_data = MinibatchData {
        log_sample_minibatch_size: 1,
        sample_minibatch_number: 0,
    };

    let loaded_zkdt_circuit_data = load_upshot_data_single_tree_batch::<F>(
        Some(minibatch_data),
        0,
        Path::new("upshot_data/quantized-upshot-model.json"),
        Path::new("upshot_data/upshot-quantized-samples.npy")
    );
    let mut f = fs::File::create(CACHED_BATCHED_MLES_FILE).unwrap();
    to_writer(&mut f, &loaded_zkdt_circuit_data).unwrap();
}

/// Reads the cached results from [`load_upshot_data_single_tree_batch`] and returns them.
pub fn read_upshot_data_single_tree_branch_from_file<F: FieldExt>() -> (ZKDTCircuitData<F>, (usize, usize)) {
    let file = std::fs::File::open(CACHED_BATCHED_MLES_FILE).unwrap();
    from_reader(&file).unwrap()
}

/// Reads the cached results from `cached_file_path` and returns them
pub fn read_upshot_data_single_tree_branch_from_filepath<F: FieldExt>(cached_file_path: &str) -> (ZKDTCircuitData<F>, (usize, usize)) {
    let file = std::fs::File::open(cached_file_path).unwrap();
    from_reader(&file).unwrap()
}

/// Loads a result from [`generate_upshot_data_all_batch_sizes`].
pub fn read_upshot_data_single_tree_branch_from_file_with_batch_exp<F: FieldExt>(
    exp_batch_size: usize,
    upshot_data_dir_path: &Path
) -> (ZKDTCircuitData<F>, (usize, usize)) {

    // --- Sanitychecks ---
    debug_assert!(exp_batch_size >= 1);
    debug_assert!(exp_batch_size <= 12);

    // --- Load ---
    let file = std::fs::File::open(get_cached_batched_mles_filepath_with_exp_size(exp_batch_size, upshot_data_dir_path)).unwrap();
    from_reader(&file).unwrap()
}

/// Generates circuit data in batched form for a single Catboost tree
/// 
/// ## Arguments
/// * `exp_batch_size` - 2^{`exp_batch_size`} is the actual batch size that we want.
///     Note that this value must be between 1 and 12, inclusive!
pub fn generate_mles_batch_catboost_single_tree<F: FieldExt>(exp_batch_size: usize, upshot_data_dir_path: &Path) -> (BatchedZKDTCircuitMles<F>, (usize, usize)) {

    // --- Sanitychecks ---
    debug_assert!(exp_batch_size >= 1);
    debug_assert!(exp_batch_size <= 12);

    // --- Check to see if the cached file exists ---
    let cached_file_path = get_cached_batched_mles_filepath_with_exp_size(exp_batch_size, upshot_data_dir_path);

    // --- If no cached file exists, run the entire cache thingy ---
    if !file_exists(&cached_file_path) {
        generate_upshot_data_all_batch_sizes::<F>(None, upshot_data_dir_path);
    }

    // --- First generate the dummy data, then convert to MLE form factor ---
    let (zkdt_circuit_data, (tree_height, input_len)) = read_upshot_data_single_tree_branch_from_filepath::<F>(&cached_file_path);
    convert_zkdt_circuit_data_into_mles(zkdt_circuit_data, tree_height, input_len)
}

/// Generates all batched data of size 2^1, ..., 2^{12} and caches for testing
/// purposes. Note that this only generates cached data for the FIRST TREE!!!
/// 
/// ## Arguments
/// * `num_trees_if_multiple` - Currently unused!!!
/// 
/// ## Notes
/// Note that `raw_samples.values.len()` is currently 4573! This means we can go
/// up to 4096 in terms of batch sizes which are powers of 2
pub fn generate_upshot_data_all_batch_sizes<F: FieldExt>(
    _num_trees_if_multiple: Option<usize>,
    upshot_data_dir_path: &Path,
) {

    println!("Generating Upshot data (to be cached) for all batch sizes (2^1, ..., 2^{{12}})...\n");
    let raw_trees_model: RawTreesModel = load_raw_trees_model(Path::new("upshot_data/quantized-upshot-model.json"));
    let mut raw_samples: RawSamples = load_raw_samples(Path::new("upshot_data/upshot-quantized-samples.npy"));
    let orig_raw_samples = raw_samples.clone();
    let trees_model: TreesModel = (&raw_trees_model).into();
    let ctrees: CircuitizedTrees<F> = (&trees_model).into();

    // --- We create batches of size 2^1, ..., 2^{12} ---
    (1..12).for_each(|batch_size_exp| {

        dbg!(&upshot_data_dir_path);

        let cached_filepath = get_cached_batched_mles_filepath_with_exp_size(batch_size_exp, upshot_data_dir_path);
        if file_exists(&cached_filepath) {
            return;
        }

        let generation_str = format!("Generating for batch size (exp) {}...", batch_size_exp);
        println!("{}", generation_str);

        let true_input_batch_size = 2_usize.pow(batch_size_exp as u32);
        raw_samples.values = orig_raw_samples.values[0..true_input_batch_size].to_vec();

        let samples: Samples = to_samples(&raw_samples);
        let csamples = circuitize_samples::<F>(&samples, &trees_model);
    
        let tree_height = ctrees.depth;
        let input_len = csamples.samples[0].len();
    
        let combined_zkdt_circuit_data = (ZKDTCircuitData::new(
            csamples.samples,
            csamples.attributes_on_paths[0].clone(),
            csamples.decision_paths[0].clone(),
            csamples.path_ends[0].clone(),
            csamples.differences[0].clone(),
            csamples.node_multiplicities[0].clone(),
            ctrees.decision_nodes[0].clone(),
            ctrees.leaf_nodes[0].clone(),
            csamples.attribute_multiplicities[0].clone()
        ), (tree_height, input_len));

        // --- Write to file ---
        let mut f = fs::File::create(cached_filepath).unwrap();
        to_writer(&mut f, &combined_zkdt_circuit_data).unwrap();
    });
}

mod tests {
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use super::write_mles_batch_catboost_single_tree;

    /// Literally just calls the [`write_mles_batch_catboost_single_tree`] function
    /// to write the preprocessed stuff to file so we can load it in later
    #[test]
    fn test_write_mles_batch_catboost_single_tree() {
        write_mles_batch_catboost_single_tree::<Fr>();
    }
}