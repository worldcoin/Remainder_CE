use std::collections::HashMap;

use remainder_hyrax::{
    hyrax_gkr::HyraxProof,
    hyrax_worldcoin::orb::{load_image_commitment, SerializedImageCommitment},
};
use remainder_shared_types::Bn256Point;

use sha256::digest as sha256_digest;

use remainder::worldcoin::{
    parameters_v2::IRISCODE_LEN as V2_IRISCODE_LEN, parameters_v3::IRISCODE_LEN as V3_IRISCODE_LEN,
};

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
    let proofs =
        remainder_hyrax::hyrax_worldcoin::upgrade::prove_upgrade_v2_to_v3(&commitments.clone());
}
