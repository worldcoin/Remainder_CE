pub mod claims;
pub mod curves;
pub mod ec;
pub mod input_layer;
pub mod layer;
pub mod pedersen;
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
            + for<'de> Deserialize<'de>,
    > FieldExt for F
{
}
