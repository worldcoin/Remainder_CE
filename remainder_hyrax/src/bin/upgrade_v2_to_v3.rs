use std::collections::HashMap;
use remainder_hyrax::hyrax_worldcoin::orb::load_image_commitment;

fn main() {
    // Load the data for both versions, both eyes and both iris and mask.
    let mut data: HashMap<(u8, bool, bool), (Vec<u8>, Vec<u8>, Vec<u8>)> = HashMap::new();
    for version in 2..=3 {
        for mask in [false, true] {
            for left_eye in [false, true] {
                data.insert((version, mask, left_eye), load_image_commitment(version, mask, left_eye));
            }
        }
    }
    println!("Proving");
    let proofs = remainder_hyrax::hyrax_worldcoin::upgrade::prove_upgrade_v2_to_v3(&data);
    println!("Verifying");
    let codes = remainder_hyrax::hyrax_worldcoin::upgrade::verify_upgrade_v2_to_v3(&proofs).unwrap();
    dbg!(codes);
}
