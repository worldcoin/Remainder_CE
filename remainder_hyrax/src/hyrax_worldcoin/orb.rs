use remainder::utils::mle::pad_with;
use remainder::worldcoin::io::read_bytes_from_file;
use itertools::Itertools;
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::HasByteRepresentation;
use crate::hyrax_gkr::hyrax_input_layer::HyraxProverInputCommitment;
use crate::hyrax_pcs::MleCoefficientsVector;

// log of the number of columns for the Hyrax pre-commitment to the image.
pub const IMAGE_COMMIT_LOG_NUM_COLS: usize = 9;
// public string used to derive the generators (arbitrary constant)
pub const PUBLIC_STRING: &str = "Modulus <3 Worldcoin: ZKML Self-Custody Edition";

// FIXME(Ben) do we need this?
// //use base64::{engine::general_purpose::STANDARD, Engine};
// use serde_json::Value;
// /// Helper function to get iris code specifically by deserializing base64 and the speciifc key in json
// pub fn read_iris_code_from_file_with_key(filename: &str, key: &str) -> Vec<bool> {
//     let mut file = std::fs::File::open(filename).unwrap();
//     let initial_buffer_size = file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
//     let mut bufreader = Vec::with_capacity(initial_buffer_size);
//     file.read_to_end(&mut bufreader).unwrap();
//     let v: Value = serde_json::de::from_slice(&bufreader[..]).unwrap();
//     let base64_string = v[key].as_str().unwrap();

//     let bits: Vec<bool> = STANDARD
//         .decode(base64_string)
//         .unwrap()
//         .iter()
//         .flat_map(|byte| (0..8).rev().map(move |i| byte & (1 << i) != 0))
//         .collect();
//     bits
// }

/// Helper function to read a Vec of bytes as a serialized Hyrax commitment, in a manner compatible
/// with the Orb's serialization functions.
pub fn deserialize_commitment_from_bytes<C: PrimeOrderCurve>(bytes: &[u8]) -> Vec<C> {
    let commitment: Vec<C> = bytes
        .chunks(C::COMPRESSED_CURVE_POINT_BYTEWIDTH)
        .map(|chunk| C::from_bytes_compressed(chunk))
        .collect_vec();
    commitment
}

/// Helper function to read a Vec of bytes as serialized blinding factors in a manner compatiable
/// with the Orb's serialization functions.
pub fn deserialize_blinding_factors_from_bytes<C: PrimeOrderCurve>(bytes: &[u8]) -> Vec<C::Scalar> {
    let blinding_factors: Vec<C::Scalar> = bytes
        .chunks(C::SCALAR_ELEM_BYTEWIDTH)
        .map(|chunk| C::Scalar::from_bytes_le(&chunk.to_vec()))
        .collect_vec();
    blinding_factors
}

/// An image, its serialized commitment, and its serialized blinding factors (serialized as per the
/// Orb's serialization functions).
/// (This is useful since the Vec<u8> representation of the attributes are convenient for receiving
/// data from the phone app.)
#[derive(Debug, Clone)]
pub struct SerializedImageCommitment {
    /// The image, unpadded.
    pub image: Vec<u8>,
    /// The commitment to the image, serialized using the curve's compressed representation.
    pub commitment_bytes: Vec<u8>,
    /// The blinding factors for the image, serialized in little-endian byte order.
    pub blinding_factors_bytes: Vec<u8>,
}

impl<C: PrimeOrderCurve> From<SerializedImageCommitment> for HyraxProverInputCommitment<C> {
    /// Deserializes a [SerializedImageCommitment] into a [HyraxProverInputCommitment].
    fn from(serialization: SerializedImageCommitment) -> Self {
        let commitment = deserialize_commitment_from_bytes(&serialization.commitment_bytes);
        let blinding_factors_matrix = deserialize_blinding_factors_from_bytes::<C>(&serialization.blinding_factors_bytes);
        HyraxProverInputCommitment {
            mle: MleCoefficientsVector::<C>::U8Vector(pad_with(0, &serialization.image)),
            commitment: commitment,
            blinding_factors_matrix: blinding_factors_matrix,
        }
    }
}

/// Returns a [SerializedImageCommitment] containing image, serialized commitment, and serialized
/// blinding factors for the given version, mask, and eye (uses dummy data).
pub fn load_image_commitment(version: u8, mask: bool, left_eye: bool) -> SerializedImageCommitment {
    let sizing_suffix = if version == 2 { "_resized" } else { "" };
    let image_or_mask = if mask { "mask" } else { "image" };
    let eye = if left_eye { "left" } else { "right" };
    let image_fn = format!("iriscode_pcp_example/{eye}_normalized_{image_or_mask}{sizing_suffix}.bin");
    let image = read_bytes_from_file(&image_fn);
    let commitment_fn = format!("iriscode_pcp_example/{eye}_normalized_{image_or_mask}_commitment{sizing_suffix}.bin");
    let commitment_bytes = read_bytes_from_file(&commitment_fn);
    let blinding_factors_fn = format!("iriscode_pcp_example/{eye}_normalized_{image_or_mask}_blinding_factors{sizing_suffix}.bin");
    let blinding_factors_bytes = read_bytes_from_file(&blinding_factors_fn);
    SerializedImageCommitment{ image, commitment_bytes, blinding_factors_bytes }
}