pub mod curves;
pub mod ec;
pub mod transcript;

use std::hash::Hash;

use halo2curves::ff::{FromUniformBytes, WithSmallOrderMulGroup};
use num::{BigUint, One};
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

    fn to_u64s_le(&self) -> Vec<u64> {
        let bytes = self.to_bytes_le();

        let fold_bytes = |acc, x: &u8| (acc << 8) + (*x as u64);

        vec![
            bytes[0..8].iter().rev().fold(0, fold_bytes),
            bytes[8..16].iter().rev().fold(0, fold_bytes),
            bytes[16..24].iter().rev().fold(0, fold_bytes),
            bytes[24..32].iter().rev().fold(0, fold_bytes),
        ]
    }

    fn from_u64s_le(words: Vec<u64>) -> Self
    where
        Self: Sized,
    {
        let mask_8bit = (1_u64 << 8) - 1;

        Self::from_bytes_le(vec![
            (words[0] & mask_8bit) as u8,
            ((words[0] & (mask_8bit << 8)) >> 8) as u8,
            ((words[0] & (mask_8bit << 16)) >> 16) as u8,
            ((words[0] & (mask_8bit << 24)) >> 24) as u8,
            ((words[0] & (mask_8bit << 32)) >> 32) as u8,
            ((words[0] & (mask_8bit << 40)) >> 40) as u8,
            ((words[0] & (mask_8bit << 48)) >> 48) as u8,
            ((words[0] & (mask_8bit << 56)) >> 56) as u8,
            (words[1] & mask_8bit) as u8,
            ((words[1] & (mask_8bit << 8)) >> 8) as u8,
            ((words[1] & (mask_8bit << 16)) >> 16) as u8,
            ((words[1] & (mask_8bit << 24)) >> 24) as u8,
            ((words[1] & (mask_8bit << 32)) >> 32) as u8,
            ((words[1] & (mask_8bit << 40)) >> 40) as u8,
            ((words[1] & (mask_8bit << 48)) >> 48) as u8,
            ((words[1] & (mask_8bit << 56)) >> 56) as u8,
            (words[2] & mask_8bit) as u8,
            ((words[2] & (mask_8bit << 8)) >> 8) as u8,
            ((words[2] & (mask_8bit << 16)) >> 16) as u8,
            ((words[2] & (mask_8bit << 24)) >> 24) as u8,
            ((words[2] & (mask_8bit << 32)) >> 32) as u8,
            ((words[2] & (mask_8bit << 40)) >> 40) as u8,
            ((words[2] & (mask_8bit << 48)) >> 48) as u8,
            ((words[2] & (mask_8bit << 56)) >> 56) as u8,
            (words[3] & mask_8bit) as u8,
            ((words[3] & (mask_8bit << 8)) >> 8) as u8,
            ((words[3] & (mask_8bit << 16)) >> 16) as u8,
            ((words[3] & (mask_8bit << 24)) >> 24) as u8,
            ((words[3] & (mask_8bit << 32)) >> 32) as u8,
            ((words[3] & (mask_8bit << 40)) >> 40) as u8,
            ((words[3] & (mask_8bit << 48)) >> 48) as u8,
            ((words[3] & (mask_8bit << 56)) >> 56) as u8,
        ])
    }

    fn to_big_uint(&self) -> BigUint {
        self.to_bytes_le()
            .iter()
            .rev()
            .fold(BigUint::ZERO, |acc, byte| {
                acc * (BigUint::one() << 8) + *byte
            })
    }
}
