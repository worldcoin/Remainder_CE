cargo build --release
cargo run --release --bin generate_ligero_tree_commitments -- \
    --tree-commit-dir remainder_prover/upshot_data/tree_ligero_commitments/ \
    --decision-forest-model-filepath remainder_prover/upshot_data/quantized-upshot-model.json \
    --tree-batch-size 16 \
    --debug-tracing-subscriber \