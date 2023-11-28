cargo build --release
cargo run --release --bin generate_sample_minibatch_commitments -- \
    --raw-samples-path remainder_prover/upshot_data/upshot-quantized-samples.npy \
    --sample-minibatch-commitments-dir remainder_prover/upshot_data/sample_minibatch_commitments/ \
    --log-sample-minibatch-commitment-size 10 \
    --debug-tracing-subscriber \