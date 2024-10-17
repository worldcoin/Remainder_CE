use itertools::Itertools;
use remainder::worldcoin::io::read_bytes_from_file;
use remainder_shared_types::curves::PrimeOrderCurve;

// log of the number of columns for the Hyrax pre-commitment to the image.
pub const IMAGE_COMMIT_LOG_NUM_COLS: usize = 9;
// public string used to derive the generators (arbitrary constant)
pub const PUBLIC_STRING: &str = "Modulus <3 Worldcoin: ZKML Self-Custody Edition";

/// Helper functions for deserializing commitments/blinding factors from byte array
pub fn deserialize_commitment_from_bytes_compressed<C: PrimeOrderCurve>(bytes: &[u8]) -> Vec<C> {
    let commitment = bytes
        .chunks(C::COMPRESSED_CURVE_POINT_BYTEWIDTH)
        .map(|byte_repr| C::from_bytes_compressed(byte_repr))
        .collect_vec();
    commitment
}

pub fn deserialize_blinding_factors_from_bytes_compressed<C: PrimeOrderCurve>(
    bytes: &[u8],
) -> Vec<C::Scalar> {
    use remainder_shared_types::HasByteRepresentation;
    let blinding_factors: Vec<<C as PrimeOrderCurve>::Scalar> = bytes
        .chunks(C::SCALAR_ELEM_BYTEWIDTH)
        .map(|byte_repr| C::Scalar::from_bytes_le(byte_repr))
        .collect_vec();
    blinding_factors
}

/// Returns the image, commitment, and blinding factors for the given version, mask, and eye (uses dummy data).
pub fn load_image_commitment(version: u8, mask: bool, left_eye: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let sizing_suffix = if version == 2 { "_resized" } else { "" };
    let image_or_mask = if mask { "mask" } else { "image" };
    let eye = if left_eye { "left" } else { "right" };
    let image_fn = format!("iriscode_pcp_example/{eye}_normalized_{image_or_mask}{sizing_suffix}.bin");
    dbg!(&image_fn);
    let image = read_bytes_from_file(&image_fn);
    let commitment_fn = format!("iriscode_pcp_example/{eye}_normalized_{image_or_mask}_commitment{sizing_suffix}.bin");
    dbg!(&commitment_fn);
    let commitment = read_bytes_from_file(&commitment_fn);
    let blinding_factors_fn = format!("iriscode_pcp_example/{eye}_normalized_{image_or_mask}_blinding_factors{sizing_suffix}.bin");
    dbg!(&blinding_factors_fn);
    let blinding_factors = read_bytes_from_file(&blinding_factors_fn);
    (image, commitment, blinding_factors)
}