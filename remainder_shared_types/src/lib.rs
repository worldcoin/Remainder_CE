pub mod transcript;

use std::hash::Hash;

use halo2curves::FieldExt as H2FieldExt;
use poseidon_circuit::Hashable;
use serde::{Deserialize, Serialize};

pub use halo2curves::bn256::Fr;
pub use halo2curves;
pub use poseidon::Poseidon;

///External definition of Field element trait, will remain an Alias for now
pub trait FieldExt: H2FieldExt + Serialize + for<'de> Deserialize<'de> + std::cmp::Ord + Hash + Hashable {
    // fn zero() -> Self {
    //     Self::ZERO
    // }

    // fn one() -> Self {
    //     Self::ONE
    // }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.to_repr().as_ref().to_vec()
    }

    fn from_bytes_le(bytes: &[u8]) -> Self {
        let mut repr = Self::Repr::default();
        repr.as_mut()[..bytes.len()].copy_from_slice(bytes);
        Self::from_repr(repr).unwrap()
    }

    // fn get_lower_128(&self) -> u128 {
    //     let bytes = self.to_bytes_le();
    //     let mut lower_64 = 0u128;
    //     for (i, byte) in bytes.into_iter().enumerate().take(8) {
    //         lower_64 |= (byte as u128) << (i * 8);
    //     }
    //     lower_64
    // }
}

impl<F: H2FieldExt + Serialize + for<'de> Deserialize<'de> + std::cmp::Ord + Hash + Hashable> FieldExt for F {}
