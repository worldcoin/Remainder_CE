use std::collections::HashMap;

use remainder_hyrax::hyrax_worldcoin::orb::{load_image_commitment, SerializedImageCommitment};

fn main() {
    let mut commitments: HashMap<(bool, bool), SerializedImageCommitment> = HashMap::new();
    let version = 3;
    for mask in [false, true] {
        for left_eye in [false, true] {
            let serialized_commitment = load_image_commitment(version, mask, left_eye);
            commitments.insert((mask, left_eye), serialized_commitment.clone());
        }
    }
    let _proofs = remainder_hyrax::hyrax_worldcoin::v3::prove_v3(&commitments.clone());
}
