use std::path::Path;

use crate::zk_iriscode_ss::io::read_bytes_from_file;
use hyrax::gkr::input_layer::HyraxProverInputCommitment;
use itertools::Itertools;
use remainder::mle::evals::MultilinearExtension;
use remainder::utils::mle::pad_with;
use shared_types::curves::PrimeOrderCurve;
use shared_types::ff_field;
use shared_types::HasByteRepresentation;
use zeroize::Zeroize;

// log of the number of columns for the Hyrax pre-commitment to the image.
pub const IMAGE_COMMIT_LOG_NUM_COLS: usize = 9;
// public string used to derive the generators (arbitrary constant)
pub const PUBLIC_STRING: &str = "Modulus <3 Worldcoin: ZKML Self-Custody Edition";

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
        .map(C::Scalar::from_bytes_le)
        .collect_vec();
    blinding_factors
}

/// An image, its serialized commitment, and its serialized blinding factors (serialized as per the
/// Orb's serialization functions).
/// (This is useful since the `Vec<u8>` representation of the attributes are convenient for receiving
/// data from the phone app.)
#[derive(Debug, Clone, Zeroize)]
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
        let image_as_field_elements = serialization
            .image
            .iter()
            .map(|elem| C::Scalar::from(*elem as u64))
            .collect_vec();
        let commitment = deserialize_commitment_from_bytes(&serialization.commitment_bytes);
        let blinding_factors_matrix =
            deserialize_blinding_factors_from_bytes::<C>(&serialization.blinding_factors_bytes);
        HyraxProverInputCommitment {
            mle: MultilinearExtension::new(pad_with(C::Scalar::ZERO, &image_as_field_elements)),
            commitment,
            blinding_factors_matrix,
        }
    }
}

/// Returns a [SerializedImageCommitment] containing image, serialized commitment, and serialized
/// blinding factors for the given version, mask, and eye (uses dummy data).
pub fn load_image_commitment(
    base_path: &Path,
    version: u8,
    mask: bool,
    left_eye: bool,
) -> SerializedImageCommitment {
    let sizing_suffix = if version == 2 { "_resized" } else { "" };
    let image_or_mask = if mask { "mask" } else { "image" };
    let eye = if left_eye { "left" } else { "right" };
    let image_fn = base_path.join(format!(
        "{eye}_normalized_{image_or_mask}{sizing_suffix}.bin"
    ));
    let image = read_bytes_from_file(image_fn.as_os_str().to_str().unwrap());
    let commitment_fn = base_path.join(format!(
        "{eye}_normalized_{image_or_mask}_commitment{sizing_suffix}.bin"
    ));
    let commitment_bytes = read_bytes_from_file(commitment_fn.as_os_str().to_str().unwrap());
    let blinding_factors_fn = base_path.join(format!(
        "{eye}_normalized_{image_or_mask}_blinding_factors{sizing_suffix}.bin"
    ));
    let blinding_factors_bytes =
        read_bytes_from_file(blinding_factors_fn.as_os_str().to_str().unwrap());
    SerializedImageCommitment {
        image,
        commitment_bytes,
        blinding_factors_bytes,
    }
}
