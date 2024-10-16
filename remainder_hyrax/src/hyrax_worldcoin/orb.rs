use itertools::Itertools;
use remainder_shared_types::curves::PrimeOrderCurve;

// log of the number of columns in the re-arrangement of the image as a matrix
pub const LOG_NUM_COLS: usize = 9;
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