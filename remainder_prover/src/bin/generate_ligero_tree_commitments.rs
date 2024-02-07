//! Executable which generates all Ligero tree commitments

use clap::Parser;
use remainder::{
    layer::LayerId,
    mle::{dense::DenseMle, Mle},
    prover::{
        input_layer::{
            combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer,
        },
        GKRCircuit, GKRError,
    },
    zkdt::{
        constants::{
            get_tree_commitment_filepath_for_tree_batch,
            get_tree_commitment_filepath_for_tree_number,
        },
        data_pipeline::dt2zkdt::{
            load_raw_samples, load_raw_trees_model, CircuitizedTrees, RawSamples, RawTreesModel,
            Samples, TreesModel,
        },
        structs::{DecisionNode, LeafNode},
        zkdt_circuit::ZKDTCircuit,
    },
};
use remainder_shared_types::Fr;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use remainder_ligero::ligero_commit::remainder_ligero_commit_prove;
use remainder_shared_types::{transcript::poseidon_transcript::PoseidonSponge, FieldExt};
use serde_json::{from_reader, to_writer};
use std::io::BufWriter;
use thiserror::Error;
use tracing::{debug, span, Level};
use tracing_subscriber::{fmt::format::FmtSpan, FmtSubscriber};

#[derive(Error, Debug, Clone)]
/// Errors for running the tree commitment binary.
/// TODO!(ryancao): File-handling errors!
pub enum GenerateLigeroTreeCommBinaryError {}

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

    #[arg(long, default_value_t = 4)]
    tree_batch_size: usize,
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
    ctrees
        .decision_nodes
        .into_iter()
        .zip(ctrees.leaf_nodes.into_iter())
        .enumerate()
        .for_each(|(tree_number, (tree_decision_nodes, tree_leaf_nodes))| {
            // --- TRACING: Ligero tree commitment span ---
            let _ligero_tree_commit_span =
                span!(Level::DEBUG, "ligero_tree_commit_span", tree_number).entered();

            // --- Check if file already exists ---
            let tree_commitment_filepath =
                get_tree_commitment_filepath_for_tree_number(tree_number, tree_commit_dir);
            match fs::metadata(tree_commitment_filepath.clone()) {
                // --- File already exists; no need to commit ---
                Ok(_) => {
                    debug!(
                        tree_number,
                        tree_commitment_filepath, "File already exists! Skipping."
                    );
                }

                // --- No commitment exists yet; create commitment ---
                Err(_) => {
                    debug!(
                        tree_number,
                        tree_commitment_filepath,
                        "File doesn't exist yet! Generating commitment..."
                    );

                    // --- Create MLEs from each tree decision + leaf node list ---
                    let mut tree_decision_nodes_mle = DenseMle::new_from_iter(
                        tree_decision_nodes.into_iter().map(DecisionNode::from),
                        LayerId::Input(0),
                        None,
                    );
                    let mut tree_leaf_nodes_mle = DenseMle::new_from_iter(
                        tree_leaf_nodes.into_iter().map(LeafNode::from),
                        LayerId::Input(0),
                        None,
                    );

                    // --- Combine them as if creating an input layer ---
                    let tree_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
                        Box::new(&mut tree_decision_nodes_mle),
                        Box::new(&mut tree_leaf_nodes_mle),
                    ];
                    let tree_input_layer_builder =
                        InputLayerBuilder::new(tree_mles, None, LayerId::Input(0));
                    let tree_input_layer: LigeroInputLayer<F, PoseidonSponge<F>> =
                        tree_input_layer_builder.to_input_layer();

                    // --- Create commitment to the combined MLEs via the input layer ---
                    let rho_inv = 4;
                    let ratio = 1_f64;
                    let ligero_commitment = remainder_ligero_commit_prove(
                        &tree_input_layer.mle.mle_ref().bookkeeping_table,
                        rho_inv,
                        ratio,
                    );

                    // --- Write to file ---
                    let mut file = fs::File::create(tree_commitment_filepath).unwrap();
                    let mut bw = BufWriter::new(file);
                    serde_json::to_writer(bw, &ligero_commitment).unwrap();
                }
            }
        });
}

pub fn generate_batch_tree_ligero_commitments<F: FieldExt>(
    decision_forest_model_filepath: &Path,
    tree_commit_dir: &Path,
    tree_batch_size: usize,
) {
    let raw_trees_model: RawTreesModel = load_raw_trees_model(decision_forest_model_filepath);

    let trees_model: TreesModel = (&raw_trees_model).into();
    let ctrees: CircuitizedTrees<F> = (&trees_model).into();

    // --- Generate Ligero commitments for trees ---
    debug_assert_eq!(ctrees.decision_nodes.len(), ctrees.leaf_nodes.len());

    let batch_numbers = 0..ctrees.decision_nodes.len() / tree_batch_size;

    (batch_numbers).for_each(|tree_batch_number| {
        let tree_range =
            (tree_batch_size * tree_batch_number)..(tree_batch_size * (tree_batch_number + 1));
        let tree_batch_commitment_filepath = get_tree_commitment_filepath_for_tree_batch(
            tree_batch_size,
            tree_batch_number,
            tree_commit_dir,
        );

        match fs::metadata(tree_batch_commitment_filepath.clone()) {
            Ok(_) => {
                debug!(
                    tree_batch_size,
                    tree_batch_number,
                    tree_batch_commitment_filepath,
                    "File already exists! Skipping."
                );
            }

            Err(_) => {
                debug!(
                    tree_batch_size,
                    tree_batch_number,
                    tree_batch_commitment_filepath,
                    "File doesn't exist yet! Generating commitment..."
                );

                let (tree_decision_nodes_mle_vec, tree_leaf_nodes_mle_vec): (Vec<_>, Vec<_>) =
                    ctrees.decision_nodes[tree_range.clone()]
                        .to_vec()
                        .into_iter()
                        .zip(ctrees.leaf_nodes[tree_range].to_vec().into_iter())
                        .map(|(tree_decision_nodes, tree_leaf_nodes)| {
                            let _ligero_tree_batch_commit_span = span!(
                                Level::DEBUG,
                                "ligero_batch_tree_commit_span",
                                tree_batch_number,
                                tree_batch_size
                            )
                            .entered();

                            // --- Create MLEs from each tree decision + leaf node list ---
                            let tree_decision_nodes_mle = DenseMle::new_from_iter(
                                tree_decision_nodes.into_iter().map(DecisionNode::from),
                                LayerId::Input(0),
                                None,
                            );

                            let tree_leaf_nodes_mle = DenseMle::new_from_iter(
                                tree_leaf_nodes.into_iter().map(LeafNode::from),
                                LayerId::Input(0),
                                None,
                            );

                            (tree_decision_nodes_mle, tree_leaf_nodes_mle)
                        })
                        .unzip();

                let mut tree_decision_nodes_mle_combined =
                    DenseMle::<F, DecisionNode<F>>::combine_mle_batch(tree_decision_nodes_mle_vec);
                let mut tree_leaf_nodes_mle_combined =
                    DenseMle::<F, LeafNode<F>>::combine_mle_batch(tree_leaf_nodes_mle_vec);

                tree_decision_nodes_mle_combined.layer_id = LayerId::Input(0);
                tree_leaf_nodes_mle_combined.layer_id = LayerId::Input(0);

                let tree_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
                    Box::new(&mut tree_decision_nodes_mle_combined),
                    Box::new(&mut tree_leaf_nodes_mle_combined),
                ];

                let tree_input_layer_builder =
                    InputLayerBuilder::new(tree_mles, None, LayerId::Input(0));
                let tree_input_layer: LigeroInputLayer<F, PoseidonSponge<F>> =
                    tree_input_layer_builder.to_input_layer();

                // --- Create commitment to the combined MLEs via the input layer ---
                let rho_inv = 4;
                let ratio = 1_f64;
                let ligero_commitment = remainder_ligero_commit_prove(
                    &tree_input_layer.mle.mle_ref().bookkeeping_table,
                    rho_inv,
                    ratio,
                );

                // --- Write to file ---
                let file = fs::File::create(tree_batch_commitment_filepath).unwrap();
                let bw = BufWriter::new(file);
                serde_json::to_writer(bw, &ligero_commitment).unwrap();
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
    generate_batch_tree_ligero_commitments::<Fr>(
        Path::new(&args.decision_forest_model_filepath),
        Path::new(&args.tree_commit_dir),
        args.tree_batch_size,
    );

    Ok(())
}
