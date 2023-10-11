//! Executable which generates all Ligero tree commitments

use std::{path::{Path, PathBuf}, time::Instant, fs};
use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use remainder::{prover::{GKRError, GKRCircuit, input_layer::{combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer}}, zkdt::{data_pipeline::{dt2zkdt::{RawTreesModel, RawSamples, load_raw_trees_model, load_raw_samples, circuitize_samples, TreesModel, Samples, CircuitizedTrees}}, zkdt_circuit::ZKDTCircuit, constants::get_tree_commitment_filepath_for_tree_number, structs::{LeafNode, DecisionNode}}, layer::LayerId, mle::{Mle, dense::DenseMle}};
use clap::Parser;
use remainder_ligero::ligero_commit::remainder_ligero_commit_prove;
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};
use serde_json::{to_writer, from_reader};
use thiserror::Error;
use tracing::{debug, event, Level, debug_span, span};
use tracing_subscriber::{FmtSubscriber, fmt::format::FmtSpan};

#[derive(Error, Debug, Clone)]
/// Errors for running the tree commitment binary.
/// TODO!(ryancao): File-handling errors!
pub enum GenerateLigeroTreeCommBinaryError {

}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the directory in which to write all the tree commitments.
    #[arg(long)]
    tree_commit_dir: String,

    /// File path to the (Python) processed `model.json` file.
    /// 
    /// TODO!(ryancao): Make this only need to take in the 
    /// singular tree that we are supposed to be proving, not
    /// the contents of the entire forest.
    #[arg(long)]
    decision_forest_model_filepath: String,

    /// Whether we want debug tracing subscriber logs or not.
    /// By default, we use `DEBUG` as the subscriber level.
    /// 
    /// TODO!(ryancao): Figure out `structopt` so we can pass in
    /// different trace levels
    #[arg(long, default_value_t = false)]
    debug_tracing_subscriber: bool,
}

/// Generates Ligero commitments for all trees within a given 
/// processed decision forest model.
/// 
/// ## Arguments
/// * `decision_forest_model_filepath` - File path to the (Python) processed `model.json` file.
/// * `tree_commit_dir` - Path to the directory in which to write all the tree commitments.
pub fn generate_all_tree_ligero_commitments<F: FieldExt>(
    decision_forest_model_filepath: &Path,
    tree_commit_dir: &Path,
) {
    let raw_trees_model: RawTreesModel = load_raw_trees_model(decision_forest_model_filepath);

    let trees_model: TreesModel = (&raw_trees_model).into();
    let ctrees: CircuitizedTrees<F> = (&trees_model).into();

    // --- Generate Ligero commitments for trees ---
    debug_assert_eq!(ctrees.decision_nodes.len(), ctrees.leaf_nodes.len());
    ctrees.decision_nodes.into_iter().zip(
        ctrees.leaf_nodes.into_iter()).enumerate().for_each(|(tree_number, (tree_decision_nodes, tree_leaf_nodes))| {

            // --- TRACING: Ligero tree commitment span ---
            let _ligero_tree_commit_span = span!(Level::DEBUG, "ligero_tree_commit_span", tree_number).entered();

            // --- Check if file already exists ---
            let tree_commitment_filepath = get_tree_commitment_filepath_for_tree_number(tree_number, tree_commit_dir);
            match fs::metadata(tree_commitment_filepath.clone()) {

                // --- File already exists; no need to commit ---
                Ok(_) => {
                    debug!(tree_number, tree_commitment_filepath, "File already exists! Skipping.");
                },

                // --- No commitment exists yet; create commitment ---
                Err(_) => {
                    debug!(tree_number, tree_commitment_filepath, "File doesn't exist yet! Generating commitment...");

                    // --- Create MLEs from each tree decision + leaf node list ---
                    let mut tree_decision_nodes_mle = DenseMle::new_from_iter(tree_decision_nodes
                        .into_iter()
                        .map(DecisionNode::from), LayerId::Input(0), None);
                    let mut tree_leaf_nodes_mle = DenseMle::new_from_iter(tree_leaf_nodes
                        .into_iter()
                        .map(LeafNode::from), LayerId::Input(0), None);

                    // --- Combine them as if creating an input layer ---
                    let tree_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
                        Box::new(&mut tree_decision_nodes_mle),
                        Box::new(&mut tree_leaf_nodes_mle),
                    ];
                    let tree_input_layer_builder = InputLayerBuilder::new(tree_mles, None, LayerId::Input(0));
                    let tree_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> = tree_input_layer_builder.to_input_layer();

                    // --- Create commitment to the combined MLEs via the input layer ---
                    let rho_inv = 4;
                    let ligero_commitment = remainder_ligero_commit_prove(&tree_input_layer.mle.mle_ref().bookkeeping_table, rho_inv);

                    // --- Write to file ---
                    let mut f = fs::File::create(tree_commitment_filepath).unwrap();
                    to_writer(&mut f, &ligero_commitment).unwrap();
                }
            }
    });
}

/// This binary performs the following:
/// * Take in as input the files which the Python code generates.
/// * Output as files Ligero commitments to all the trees within the passed-in models.
fn main() -> Result<(), GenerateLigeroTreeCommBinaryError> {
    let args = Args::parse();

    // --- Tracing subscriber (i.e. outputs trace messages in stdout) if asked for ---
    if args.debug_tracing_subscriber {
        let subscriber = FmtSubscriber::builder()
            .with_line_number(true)
            .with_max_level(tracing::Level::DEBUG)
            .with_level(true)
            .with_span_events(FmtSpan::ACTIVE)
            .finish();
        let _default_guard = tracing::subscriber::set_global_default(subscriber);
    }

    // --- Log the args ---
    let args_as_string = format!("{:?}", args);
    debug!(args_as_string);

    // --- Compute the commitment ---
    generate_all_tree_ligero_commitments::<Fr>(
        Path::new(&args.decision_forest_model_filepath), 
        Path::new(&args.tree_commit_dir)
    );

    Ok(())
}
