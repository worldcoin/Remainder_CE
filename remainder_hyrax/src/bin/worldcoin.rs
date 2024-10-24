use std::collections::HashMap;

use remainder_hyrax::hyrax_worldcoin::orb::{load_image_commitment, SerializedImageCommitment};

fn main() {
    let mut commitments: HashMap<(u8, bool, bool), SerializedImageCommitment> = HashMap::new();
    for version in 2..=3 {
        for mask in [false, true] {
            for left_eye in [false, true] {
                let serialized_commitment = load_image_commitment(version, mask, left_eye);
                commitments.insert((version, mask, left_eye), serialized_commitment.clone());

                println!(
                    "({}, {}, {}, data) = {}",
                    version,
                    mask,
                    left_eye,
                    serialized_commitment.image.len()
                );
                println!(
                    "({}, {}, {}, commitment) = {}",
                    version,
                    mask,
                    left_eye,
                    serialized_commitment.commitment_bytes.len()
                );
                println!(
                    "({}, {}, {}, blinding) = {}",
                    version,
                    mask,
                    left_eye,
                    serialized_commitment.blinding_factors_bytes.len()
                );
            }
        }
    }
    let _proofs =
        remainder_hyrax::hyrax_worldcoin::upgrade::prove_upgrade_v2_to_v3(&commitments.clone());
}
