pub mod curves;
pub mod pedersen;
pub mod transcript;
pub mod utils;

use std::hash::Hash;

use halo2curves::ff::{FromUniformBytes, WithSmallOrderMulGroup};
use serde::{Deserialize, Serialize};

pub use halo2curves::ff::Field as ff_field;

pub use halo2curves;
pub use halo2curves::bn256::{Fq, Fr};
pub use poseidon::Poseidon;

use halo2curves::CurveExt;
pub use halo2curves::{bn256::G1 as Bn256Point, group::Group};
pub type Scalar = <Bn256Point as Group>::Scalar;
pub type Base = <Bn256Point as CurveExt>::Base;

/// The primary finite field used within a GKR circuit, as well as within
/// sumcheck. Note that the field's size should be large enough such that
/// d / |F| bits of computational soundness is considered secure!
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
    + Zeroizable
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
            + HasByteRepresentation
            + Zeroizable,
    > Field for F
{
}

/// Simple trait which allows us to convert to and from
/// a little-endian byte representation.
pub trait HasByteRepresentation {
    /// Number of bytes within the element's representation.
    const REPR_NUM_BYTES: usize;
    /// Constructor which creates an instance of the element from a vec of
    /// less than or equal to length `REPR_NUM_BYTES`.
    /// If length less than `REPR_NUM_BYTES`, pads the most significant
    /// bits with 0s until it is of equal length to `REPR_NUM_BYTES`.
    fn from_bytes_le(bytes: &[u8]) -> Self;
    /// Function which creates an equivalent representation of the element
    /// in a byte array of length `REPR_NUM_BYTES`.
    fn to_bytes_le(&self) -> Vec<u8>;

    /// Similar to `to_bytes_le` but returns chunks of `u64`s.
    fn to_u64s_le(&self) -> Vec<u64>;

    /// Similar to `from_bytes_le` but takes chunks of `u64`s.
    fn from_u64s_le(words: Vec<u64>) -> Self
    where
        Self: Sized;

    /// Creates a Vec of elements from an arbitrary string
    /// of bytes.
    fn vec_from_bytes_le(bytes: &[u8]) -> Vec<Self>
    where
        Self: Sized;
}

/// A trait which allows zeroizing of Field elements.
pub trait Zeroizable {
    fn zeroize(&mut self);
}

/// Simple trait which allows for ease of converting e.g. a `Vec<u64>`
/// into a `Vec<F>`.
pub trait IntoVecF<F: Field> {
    fn into_vec_f(self) -> Vec<F>;
}

impl<F: Field, T> IntoVecF<F> for Vec<T>
where
    F: From<T>,
{
    fn into_vec_f(self) -> Vec<F> {
        self.into_iter().map(F::from).collect()
    }
}
