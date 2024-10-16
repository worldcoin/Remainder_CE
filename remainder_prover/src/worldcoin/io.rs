use ark_serialize::Read;
use base64::{engine::general_purpose::STANDARD, Engine};
use itertools::Itertools;
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::HasByteRepresentation;
use serde_json::Value;

/// Helper function to read bytes from a file, preallocating the required space.
pub fn read_bytes_from_file(filename: &str) -> Vec<u8> {
    let mut file = std::fs::File::open(filename).unwrap();
    let initial_buffer_size = file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
    let mut bufreader = Vec::with_capacity(initial_buffer_size);
    file.read_to_end(&mut bufreader).unwrap();
    bufreader
}

/// Helper function to get iris code specifically by deserializing base64 and the speciifc key in json
pub fn read_iris_code_from_file_with_key(filename: &str, key: &str) -> Vec<bool> {
    let mut file = std::fs::File::open(filename).unwrap();
    let initial_buffer_size = file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
    let mut bufreader = Vec::with_capacity(initial_buffer_size);
    file.read_to_end(&mut bufreader).unwrap();
    let v: Value = serde_json::de::from_slice(&bufreader[..]).unwrap();
    let base64_string = v[key].as_str().unwrap();

    let bits: Vec<bool> = STANDARD
        .decode(base64_string)
        .unwrap()
        .iter()
        .flat_map(|byte| (0..8).rev().map(move |i| byte & (1 << i) != 0))
        .collect();
    bits
}

/// helper function to read a stream of bytes as a hyrax commitment
pub fn deserialize_commitment_from_bytes<C: PrimeOrderCurve>(bytes: &[u8]) -> Vec<C> {
    let commitment: Vec<C> = bytes
        .chunks(C::COMPRESSED_CURVE_POINT_BYTEWIDTH)
        .map(|chunk| C::from_bytes_compressed(chunk))
        .collect_vec();
    commitment
}

/// helper function to read a stream of bytes as blinding factors
pub fn deserialize_blinding_factors_from_bytes<C: PrimeOrderCurve>(bytes: &[u8]) -> Vec<C::Scalar> {
    let blinding_factors: Vec<C::Scalar> = bytes
        .chunks(C::SCALAR_ELEM_BYTEWIDTH)
        .map(|chunk| C::Scalar::from_bytes_le(&chunk.to_vec()))
        .collect_vec();
    blinding_factors
}
