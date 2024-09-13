pub mod curves;
pub mod ec;
pub mod transcript;

use std::hash::Hash;

use halo2curves::ff::{FromUniformBytes, WithSmallOrderMulGroup};
use serde::{Deserialize, Serialize};

pub use halo2curves::ff::Field as ff_field;

pub use halo2curves;
pub use halo2curves::bn256::Fr;
pub use poseidon::Poseidon;

///External definition of Field element trait, will remain an Alias for now
pub trait Field:
ff_field
    + FromUniformBytes<64> // only need this bc of Poseidon transcript,
                              // see func `next_field_element_without_rejection`

    + WithSmallOrderMulGroup<3> // only need this bc of halo2_fft,
                                   // EvaluationDomain<F> uses ZETA to compute the
                                   // `EvaluationDomain`

                                   // These two traits will ideally be removed, bc
                                   // they actually are a sub trait of `PrimeField`
                                   // which sumcheck does not need
    + Hash
    + Ord
    + Serialize
    + for<'de> Deserialize<'de>
    + HasByteRepresentation
{
}

impl<
        F: ff_field
            + FromUniformBytes<64>
            + WithSmallOrderMulGroup<3>
            + Hash
            + Ord
            + Serialize
            + for<'de> Deserialize<'de>
            + HasByteRepresentation,
    > Field for F
{
}

/// Simple trait which allows us to convert to and from
/// a little-endian byte representation.
pub trait HasByteRepresentation {
    fn from_bytes_le(bytes: Vec<u8>) -> Self;
    fn to_bytes_le(&self) -> Vec<u8>;
}
