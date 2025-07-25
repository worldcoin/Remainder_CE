// can optimize `vector_commit` to use a MSM (pippenger's) instead of the for loop

use std::cmp::max;
use std::ops::{Add, Mul};

use crate::curves::PrimeOrderCurve;
use crate::curves::Sha3XofReaderWrapper;
use crate::ff_field;
use crate::utils::pippengers::scalar_mult_pippenger;
use ark_std::log2;
use itertools::Itertools;
use num::traits::ToBytes;
use num::PrimInt;
use num::Unsigned;
use num::Zero;
use serde::{Deserialize, Serialize};
use sha3::digest::ExtendableOutput;
use sha3::digest::Update;
use sha3::Shake256;

#[cfg(test)]
/// The tests for pedersen commitments.
pub mod tests;

/// For committing to vectors of integers and scalars using the Pedersen commitment scheme.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct PedersenCommitter<C: PrimeOrderCurve> {
    /// vector of "g" generators, i.e. the generators that are exponentiated by the message elements themselves (length > 0)
    /// The first N generators are used to commit to a vector of N values. The last generator is used for scalar commitments.
    pub generators: Vec<C>,
    /// the "h" generator which is exponentiated by the blinding factor
    pub blinding_generator: C,
    /// The bitwidth of the absolute values of the integers that can be committed to using integral vector commit methods (has no bearing on scalar_commit and vector_commit).
    pub int_abs_val_bitwidth: usize,
    generator_doublings: Vec<Vec<C>>,
}

impl<C: PrimeOrderCurve> PedersenCommitter<C> {
    const DEFAULT_INT_ABS_VAL_BITWIDTH: usize = 8;
    /// Creates a new PedersenCommitter with random generators.  See also [PedersenCommitter].
    /// Generators are sampled using the public string and the Shake256 hash function.
    /// Post: self.generators.len() == num_generators
    pub fn new(
        num_generators: usize,
        public_string: &str,
        int_abs_val_bitwidth: Option<usize>,
    ) -> Self {
        let all_generators = Self::sample_generators(num_generators + 1, public_string);
        let blinding_generator_h = all_generators[0];
        let generators_g_i = all_generators[1..].to_vec();
        let int_abs_val_bitwidth =
            int_abs_val_bitwidth.unwrap_or(Self::DEFAULT_INT_ABS_VAL_BITWIDTH);

        let generator_doublings: Vec<Vec<C>> = generators_g_i
            .clone()
            .into_iter()
            .map(|gen| precompute_doublings(gen, int_abs_val_bitwidth))
            .collect();

        Self {
            generators: generators_g_i,
            blinding_generator: blinding_generator_h,
            int_abs_val_bitwidth,
            generator_doublings,
        }
    }

    /// Return the generator used for scalar commitments.
    pub fn scalar_commit_generator(&self) -> C {
        self.generators[self.generators.len() - 1]
    }

    /// Sample generators using the public string and the Shake256 hash function.
    /// Pre: public_string.len() >= 32
    /// Post: result.len() == num_generators
    /// TODO: make seekable (block cipher)
    fn sample_generators(num_generators: usize, public_string: &str) -> Vec<C> {
        assert!(public_string.len() >= 32);
        let mut public_string_array: [u8; 32] = [0; 32];
        public_string_array.copy_from_slice(&public_string.as_bytes()[..32]);
        let mut shake = Shake256::default();
        shake.update(&public_string_array);
        let reader = shake.finalize_xof();
        let mut reader_wrapper = Sha3XofReaderWrapper::new(reader);
        let generators: Vec<_> = (0..num_generators)
            .map(|_| C::random(&mut reader_wrapper))
            .collect();
        generators
    }

    /// Create a new PedersenCommitter with the provided generators.
    /// See [PedersenCommitter].
    /// DEFAULT_INT_ABS_VAL_BITWIDTH is used for `int_abs_val_bitwidth` if None is provided.
    pub fn new_with_generators(
        generators: Vec<C>,
        blinding_generator: C,
        int_abs_val_bitwidth: Option<usize>,
    ) -> Self {
        let int_abs_val_bitwidth =
            int_abs_val_bitwidth.unwrap_or(Self::DEFAULT_INT_ABS_VAL_BITWIDTH);
        let generator_doublings: Vec<Vec<C>> = generators
            .clone()
            .into_iter()
            .map(|gen| precompute_doublings(gen, int_abs_val_bitwidth))
            .collect();
        Self {
            generators,
            blinding_generator,
            int_abs_val_bitwidth,
            generator_doublings,
        }
    }

    /// An optimized version of Pedersen vector commit when the message is
    /// comprised of values that fit within 128 bits.
    pub fn unsigned_integer_vector_commit<T: Unsigned + Zero + ToBytes>(
        &self,
        message: &[T],
        blinding: &C::Scalar,
    ) -> C {
        assert!(message.len() <= self.generators.len());
        let unblinded_commit = self
            .generators
            .iter()
            .zip(message.iter())
            .fold(C::zero(), |acc, (gen, input)| {
                acc + gen.scalar_mult_unsigned_integer(input)
            });
        unblinded_commit + self.blinding_generator * *blinding
    }

    /// Commits to the vector of u8s using the specified blinding factor.
    /// Uses the precomputed generator powers and the binary decomposition.
    /// Convient wrapper of integer_vector_commit.
    /// Pre: self.int_abs_val_bitwidth >= 8.
    /// Post: same result as vector_commit, assuming uints are smaller than scalar field order.
    pub fn u8_vector_commit(&self, message: &[u8], blinding: &C::Scalar) -> C {
        debug_assert!(self.int_abs_val_bitwidth >= 8);
        let message_is_negative_bits = vec![false; message.len()];
        self.integer_vector_commit(message, &message_is_negative_bits, blinding)
    }

    /// Commits to the vector of i8s using the specified blinding factor.
    /// Uses the precomputed generator powers and the binary decomposition.
    /// Convient wrapper of integer_vector_commit.
    /// Pre: self.int_abs_val_bitwidth >= 8.
    /// Post: same result as vector_commit, assuming ints are smaller than scalar field order.
    pub fn i8_vector_commit(&self, message: &[i8], blinding: &C::Scalar) -> C {
        debug_assert!(self.int_abs_val_bitwidth >= 8);
        let message_is_negative_bits = message.iter().map(|x| *x < 0i8).collect_vec();
        let message: Vec<u8> = message
            .iter()
            .map(|x| (*x as i16).unsigned_abs() as u8)
            .collect(); // convert i8 to i16 first so that .abs() doesn't fail for i8::MIN
        self.integer_vector_commit(&message, &message_is_negative_bits, blinding)
    }

    /// Commits to the vector of integers using the specified blinding factor.
    /// Integers are provided as a vector of UNSIGNED ints and a vector of bits indicating whether the integer is negative.
    /// Pre: values in message are non-negative.
    /// Pre: values have unsigned binary expressions using at most (self.highest_generator_power + 1) bits.
    /// Pre: message.len() <= self.message_generators.len()
    pub fn integer_vector_commit<T: PrimInt>(
        &self,
        message: &[T],
        message_is_negative_bits: &[bool],
        blinding: &C::Scalar,
    ) -> C {
        assert!(message.len() <= self.generators.len());
        let unblinded_commit = message
            .iter()
            .zip(self.generator_doublings.iter())
            .map(|(input, generator_doublings)| {
                debug_assert!(*input >= T::zero());
                let bits = binary_decomposition_le(*input);
                let mut acc = C::zero();
                bits.into_iter().enumerate().for_each(|(i, bit)| {
                    if bit {
                        debug_assert!(i < self.int_abs_val_bitwidth); // ensure bit decomp is not longer than our precomputed generator powers
                        acc += generator_doublings[i];
                    }
                });
                acc
            })
            .zip(message_is_negative_bits.iter())
            .map(
                |(gen_power, is_negative)| {
                    if *is_negative {
                        -gen_power
                    } else {
                        gen_power
                    }
                },
            )
            .fold(C::zero(), |acc, value| acc + value);

        unblinded_commit + self.blinding_generator * *blinding
    }

    /// Commit to the provided vector using the specified blinding factor.
    /// The first message.len() generators are used to commit to the message.
    /// Note that self.int_abs_val_bitwidth is not relevant here.
    /// Pre: message.len() <= self.message_generators.len()
    pub fn vector_commit(&self, message: &[C::Scalar], blinding: &C::Scalar) -> C {
        assert!(message.len() <= self.generators.len());
        let bucket_size = max(1, log2(message.len()));
        let unblinded_commit = scalar_mult_pippenger(
            bucket_size as usize,
            &self.generators[0..message.len()],
            message,
        );
        unblinded_commit + self.blinding_generator * *blinding
    }

    /// Convenience wrapper of [self.vector_commit] that returns the commitment wrapped in a
    /// CommittedVector.
    pub fn committed_vector(
        &self,
        message: &[C::Scalar],
        blinding: &C::Scalar,
    ) -> CommittedVector<C> {
        CommittedVector {
            value: message.to_vec(),
            blinding: *blinding,
            commitment: self.vector_commit(message, blinding),
        }
    }

    /// Commit to the provided scalar using the specified blinding factor.
    /// Note that self.int_abs_val_bitwidth is not relevant here.
    /// Pre: self.message_generators.len() >= 1
    pub fn scalar_commit(&self, message: &C::Scalar, blinding: &C::Scalar) -> C {
        self.generators[self.generators.len() - 1] * *message + self.blinding_generator * *blinding
    }

    /// Convenience wrapper of [self.scalar_commit] that returns the commitment wrapped in a
    /// CommittedScalar.
    pub fn committed_scalar(
        &self,
        message: &C::Scalar,
        blinding: &C::Scalar,
    ) -> CommittedScalar<C> {
        CommittedScalar {
            value: *message,
            blinding: *blinding,
            commitment: self.scalar_commit(message, blinding),
        }
    }
}

// Compute the little endian binary decomposition of the provided integer value.
// Pre: value is non-negative.
// Post: result.len() is std::mem::size_of::<T>() * 8;
fn binary_decomposition_le<T: PrimInt>(value: T) -> Vec<bool> {
    debug_assert!(value >= T::zero());
    let bit_size = std::mem::size_of::<T>() * 8;
    (0..bit_size)
        .map(|i| value & (T::one() << i) != T::zero())
        .collect()
}

// Returns the vector [2^i * base for i in 0..bitwidth]
// Post: powers.len() == bitwidth
fn precompute_doublings<G: PrimeOrderCurve>(base: G, bitwidth: usize) -> Vec<G> {
    let mut powers = vec![];
    let mut last = base;
    for _exponent in 0..bitwidth {
        powers.push(last);
        last = last.double();
    }
    powers
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
#[serde(bound = "C: PrimeOrderCurve")]
/// The committer's view of a scalar commitment, i.e. not just the commitment itself but also the
/// underlying value and the blinding factor.
pub struct CommittedScalar<C: PrimeOrderCurve> {
    /// The value committed to
    pub value: C::Scalar,
    /// The blinding factor
    pub blinding: C::Scalar,
    /// The commitment
    pub commitment: C,
}

impl<C: PrimeOrderCurve> CommittedScalar<C> {
    pub fn zero() -> Self {
        Self {
            value: C::Scalar::ZERO,
            blinding: C::Scalar::ZERO,
            commitment: C::zero(),
        }
    }

    pub fn one() -> Self {
        Self {
            value: C::Scalar::ONE,
            blinding: C::Scalar::ZERO,
            commitment: C::generator(),
        }
    }

    /// Panic if the commitment does not match the value and blinding factor when using the provided
    /// PedersenCommitter.
    pub fn verify(&self, committer: &PedersenCommitter<C>) {
        let recomputed_commitment = committer.scalar_commit(&self.value, &self.blinding);
        assert!(recomputed_commitment == self.commitment)
    }
}

impl<C: PrimeOrderCurve> Mul<C::Scalar> for CommittedScalar<C> {
    type Output = CommittedScalar<C>;
    /// Multiply the committed scalar by a scalar.
    fn mul(self, scalar: C::Scalar) -> Self::Output {
        Self::Output {
            value: self.value * scalar,
            blinding: self.blinding * scalar,
            commitment: self.commitment * scalar,
        }
    }
}

impl<C: PrimeOrderCurve> Mul<C::Scalar> for &CommittedScalar<C> {
    type Output = CommittedScalar<C>;
    /// Multiply the committed scalar by a scalar.
    fn mul(self, scalar: C::Scalar) -> Self::Output {
        Self::Output {
            value: self.value * scalar,
            blinding: self.blinding * scalar,
            commitment: self.commitment * scalar,
        }
    }
}

impl<C: PrimeOrderCurve> Add<CommittedScalar<C>> for CommittedScalar<C> {
    type Output = CommittedScalar<C>;
    /// Add two committed scalars.
    fn add(self, other: CommittedScalar<C>) -> Self::Output {
        Self::Output {
            value: self.value + other.value,
            blinding: self.blinding + other.blinding,
            commitment: self.commitment + other.commitment,
        }
    }
}

#[derive(Clone, Debug)]
/// The committer's view of a vector commitment, i.e. not just the commitment itself but also the
/// underlying value and the blinding factor.
pub struct CommittedVector<C: PrimeOrderCurve> {
    /// The vector of values committed to
    pub value: Vec<C::Scalar>,
    /// The blinding factor
    pub blinding: C::Scalar,
    /// The commitment
    pub commitment: C,
}

impl<C: PrimeOrderCurve> CommittedVector<C> {
    pub fn zero(length: usize) -> Self {
        Self {
            value: (0..length).map(|_| C::Scalar::ZERO).collect(),
            blinding: C::Scalar::ZERO,
            commitment: C::zero(),
        }
    }
}

impl<C: PrimeOrderCurve> Mul<C::Scalar> for CommittedVector<C> {
    type Output = CommittedVector<C>;
    /// Multiply the committed vector by a scalar.
    fn mul(self, scalar: C::Scalar) -> Self::Output {
        Self::Output {
            value: self.value.iter().map(|val| *val * scalar).collect(),
            blinding: self.blinding * scalar,
            commitment: self.commitment * scalar,
        }
    }
}

impl<C: PrimeOrderCurve> Add<CommittedVector<C>> for CommittedVector<C> {
    type Output = CommittedVector<C>;
    /// Add two committed vectors.
    fn add(self, other: CommittedVector<C>) -> Self::Output {
        Self::Output {
            value: self
                .value
                .iter()
                .zip(other.value.iter())
                .map(|(a, b)| *a + *b)
                .collect(),
            blinding: self.blinding + other.blinding,
            commitment: self.commitment + other.commitment,
        }
    }
}
