pub mod curves;
pub mod ec;
pub mod transcript;

use std::hash::Hash;

use halo2curves::ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use serde::{Deserialize, Serialize};

pub use halo2curves;
pub use halo2curves::bn256::Fr;
pub use poseidon::Poseidon;

///External definition of Field element trait, will remain an Alias for now
pub trait FieldExt:
    PrimeField
    + Field
    + FromUniformBytes<64>
    + WithSmallOrderMulGroup<3>
    + Hash
    + Ord
    + Serialize
    + for<'de> Deserialize<'de>
    + HasByteRepresentation
{
}

impl<
        F: PrimeField
            + Field
            + FromUniformBytes<64>
            + WithSmallOrderMulGroup<3>
            + Hash
            + Ord
            + Serialize
            + for<'de> Deserialize<'de>
            + HasByteRepresentation,
    > FieldExt for F
{
}

/// Simple trait which allows us to convert to and from
/// a little-endian byte representation.
pub trait HasByteRepresentation {
    fn from_bytes_le(bytes: Vec<u8>) -> Self;
    fn to_bytes_le(&self) -> Vec<u8>;
}
