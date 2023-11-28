cargo build --release
cargo run --release --bin run_remainder_multitree -- \
    --tree-commit-dir remainder_prover/upshot_data/tree_ligero_commitments/ \
    --tree-batch-number 0 \
    --tree-batch-size 16 \
    --quantized-samples-filepath remainder_prover/upshot_data/upshot-quantized-samples.npy \
    --decision-forest-model-filepath remainder_prover/upshot_data/quantized-upshot-model.json \
    --gkr-proof-to-be-written-filepath remainder_prover/upshot_data/input_layer_experiments/two_layer_combine_32_tree_10_logsample_matrix_ratio_16_one_ligero_commit.json \
    --sample-minibatch-commit-dir remainder_prover/upshot_data/sample_minibatch_commitments/ \
    --log-sample-minibatch-size 2 \
    --sample-minibatch-number 0 \
    --verify-proof \
    --matrix-ratio 16 \
    --rho-inv 4 \

# 1 tree took 12.5 seconds instead of 7.5-ish
# 16 trees took 517 seconds instead of 136-ish