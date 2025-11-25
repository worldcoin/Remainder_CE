# --- Create folder if not already exists ---
mkdir -p iriscode_pcp_example/proving_stuff

# --- Clear out the directory for a clean run ---
rm -f iriscode_pcp_example/proving_stuff/*

# --- Create circuit descriptions ---
cargo run --release --bin world_gen_iriscode_secret_share_circuit_descriptions -- \
    --circuit iriscode_pcp_example/proving_stuff/iriscode_v3_and_secret_share_circuit.bin

# --- Generate all 7 proofs (4 iriscode + 3 MPC) ---
cargo run --release --bin world_prove -- \
    --circuit iriscode_pcp_example/proving_stuff/iriscode_v3_and_secret_share_circuit.bin \
    --input iriscode_pcp_example \
    --output-dir iriscode_pcp_example/proving_stuff

# --- Verify just the iriscode proof plus commitments ---
cargo run --release --bin world_upgrade_verify -- \
    --circuit iriscode_pcp_example/proving_stuff/iriscode_v3_and_secret_share_circuit.bin \
    --hashes iriscode_pcp_example/hashes.json \
    --v3-proof iriscode_pcp_example/proving_stuff/world_v3.zkp \
    --commitments iriscode_pcp_example/proving_stuff/commitments.zkp

# --- Verify each AMPC party's proof ---
cargo run --release --bin world_verify_ampc_party -- \
    --circuit iriscode_pcp_example/proving_stuff/iriscode_v3_and_secret_share_circuit.bin \
    --hashes iriscode_pcp_example/hashes.json \
    --secret-share-proof iriscode_pcp_example/proving_stuff/world_mpc_party_0.zkp \
    --commitments iriscode_pcp_example/proving_stuff/commitments.zkp \
    --ampc-party-index 0 \
    --secret-share-bytes-dir iriscode_pcp_example/proving_stuff
cargo run --release --bin world_verify_ampc_party -- \
    --circuit iriscode_pcp_example/proving_stuff/iriscode_v3_and_secret_share_circuit.bin \
    --hashes iriscode_pcp_example/hashes.json \
    --secret-share-proof iriscode_pcp_example/proving_stuff/world_mpc_party_1.zkp \
    --commitments iriscode_pcp_example/proving_stuff/commitments.zkp \
    --ampc-party-index 1 \
    --secret-share-bytes-dir iriscode_pcp_example/proving_stuff
cargo run --release --bin world_verify_ampc_party -- \
    --circuit iriscode_pcp_example/proving_stuff/iriscode_v3_and_secret_share_circuit.bin \
    --hashes iriscode_pcp_example/hashes.json \
    --secret-share-proof iriscode_pcp_example/proving_stuff/world_mpc_party_2.zkp \
    --commitments iriscode_pcp_example/proving_stuff/commitments.zkp \
    --ampc-party-index 2 \
    --secret-share-bytes-dir iriscode_pcp_example/proving_stuff

# --- Finally, sanitycheck/validate the secret shares ---
cargo run --release --bin sanitycheck_secret_shares -- \
    --secret-share-bytes-path iriscode_pcp_example/proving_stuff/secret_shares_party_0_left_masked_iriscode.bin
cargo run --release --bin sanitycheck_secret_shares -- \
    --secret-share-bytes-path iriscode_pcp_example/proving_stuff/secret_shares_party_0_right_masked_iriscode.bin
cargo run --release --bin sanitycheck_secret_shares -- \
    --secret-share-bytes-path iriscode_pcp_example/proving_stuff/secret_shares_party_1_left_masked_iriscode.bin
cargo run --release --bin sanitycheck_secret_shares -- \
    --secret-share-bytes-path iriscode_pcp_example/proving_stuff/secret_shares_party_1_right_masked_iriscode.bin
cargo run --release --bin sanitycheck_secret_shares -- \
    --secret-share-bytes-path iriscode_pcp_example/proving_stuff/secret_shares_party_2_left_masked_iriscode.bin
cargo run --release --bin sanitycheck_secret_shares -- \
    --secret-share-bytes-path iriscode_pcp_example/proving_stuff/secret_shares_party_2_right_masked_iriscode.bin