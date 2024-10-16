use std::collections::HashMap;
use remainder::worldcoin::io::read_bytes_from_file;

fn main() {
    // Load the data for both versions, both eyes and both iris and mask.
    let mut data: HashMap<(u8, bool, bool), (Vec<u8>, Vec<u8>, Vec<u8>)> = HashMap::new();
    for version in 2..=3 {
        let sizing_suffix = if version == 2 { "_resized" } else { "" };
        for mask in [false, true] {
            let image_or_mask = if mask { "mask" } else { "image" };
            for left_eye in [false, true] {
                let eye = if left_eye { "left" } else { "right" };
                let image_fn = format!("pcp_example/{eye}_normalized_{image_or_mask}{sizing_suffix}.bin");
                dbg!(&image_fn);
                let image = read_bytes_from_file(&image_fn);
                let blinding_factors_fn = format!("pcp_example/{eye}_normalized_{image_or_mask}_blinding_factors{sizing_suffix}.bin");
                dbg!(&blinding_factors_fn);
                let blinding_factors = read_bytes_from_file(&blinding_factors_fn);
                let commitment_fn = format!("pcp_example/{eye}_normalized_{image_or_mask}_commitment{sizing_suffix}.bin");
                dbg!(&commitment_fn);
                let commitment = read_bytes_from_file(&commitment_fn);
                data.insert((version, mask, left_eye), (image, blinding_factors, commitment));
            }
        }
    }
    println!("Proving");
    let proofs = remainder_hyrax::hyrax_worldcoin::upgrade::prove_upgrade_v2_to_v3(&data);
    println!("Verifying");
    let codes = remainder_hyrax::hyrax_worldcoin::upgrade::verify_upgrade_v2_to_v3(&proofs).unwrap();
    dbg!(codes);
}
