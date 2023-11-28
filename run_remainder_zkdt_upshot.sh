cargo build --release
for i in {0..7}
do
    cargo run --release --bin run_remainder_zkdt -- \
        --tree-commit-dir remainder_prover/upshot_data/tree_ligero_commitments/ \
        --tree-number ${i} \
        --quantized-samples-filepath remainder_prover/upshot_data/upshot-quantized-samples.npy \
        --decision-forest-model-filepath remainder_prover/upshot_data/quantized-upshot-model.json \
        --sample-minibatch-commit-dir remainder_prover/upshot_data/sample_minibatch_commitments/ \
        --log-sample-minibatch-size 10 \
        --sample-minibatch-number 0 \
        --gkr-proof-to-be-written-filepath zkdt_proof_tree_${i}_benchmark.json
done