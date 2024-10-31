use std::collections::HashMap;

use remainder_hyrax::hyrax_worldcoin::orb::{load_image_commitment, SerializedImageCommitment};

fn print_features_status() {
    const STATUS_STR: [&str; 2] = ["OFF", "ON"];

    println!("=== FEATURES ===");
    println!(
        "Parallel feature for remainder_prover: {}",
        STATUS_STR[remainder::utils::is_parallel_feature_on() as usize]
    );
    println!(
        "Parallel feature for remainder_hyrax: {}",
        STATUS_STR[remainder_hyrax::utils::is_parallel_feature_on() as usize]
    );
    println!(
        "Lazy beta evaluation: {}",
        STATUS_STR[remainder::layer::gate::LAZY_BETA_EVALUATION as usize]
    );
    println!(
        "BitPackedVector: {}",
        STATUS_STR[remainder::mle::evals::bit_packed_vector::ENABLE_BIT_PACKING as usize]
    );
    println!(
        "Claim aggregation constant column optimization for remainder_prover: {}",
        STATUS_STR
            [remainder::claims::claim_aggregation::CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION
                as usize]
    );
    println!("================\n");
}

fn main() {
    print_features_status();

    let mut commitments: HashMap<(u8, bool, bool), SerializedImageCommitment> = HashMap::new();
    for version in 2..=3 {
        for mask in [false, true] {
            for left_eye in [false, true] {
                let serialized_commitment = load_image_commitment(version, mask, left_eye);
                commitments.insert((version, mask, left_eye), serialized_commitment.clone());
            }
        }
    }
    let _proofs =
        remainder_hyrax::hyrax_worldcoin::upgrade::prove_upgrade_v2_to_v3(&commitments.clone());
}
