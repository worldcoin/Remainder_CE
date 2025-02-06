//! Implements [BitPackedVector], a version of an immutable vector optimized for
//! storing field elements compactly.
#![allow(clippy::needless_lifetimes)]
use ::serde::{Deserialize, Serialize};
use ark_std::cfg_into_iter;
use itertools::Itertools;
use remainder_shared_types::{config::global_config::global_prover_enable_bit_packing, Field};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use itertools::FoldWhile::{Continue, Done};
use zeroize::Zeroize;

// -------------- Helper Functions -----------------

/// Returns the minimum numbers of bits required to represent prime field
/// elements in the range `[0, n]`. This is equivalent to computing
/// `ceil(log_2(n+1))`.
///
/// # Complexity
/// Constant in the size of the representation of `n`.
///
/// # Example
/// ```
///     use remainder_shared_types::Fr;
///     use remainder::mle::evals::bit_packed_vector::num_bits;
///
///     assert_eq!(num_bits(Fr::from(0)), 0);
///     assert_eq!(num_bits(Fr::from(31)), 5);
///     assert_eq!(num_bits(Fr::from(32)), 6);
/// ```
pub fn num_bits<F: Field>(n: F) -> usize {
    let u64_chunks = n.to_u64s_le();
    debug_assert_eq!(u64_chunks.len(), 4);
    u64_chunks
        .iter()
        .rev()
        .fold_while(192_usize, |acc, chunk| {
            if *chunk == 0 {
                if acc == 0 {
                    Done(0)
                } else {
                    Continue(acc - 64)
                }
            } else {
                Done(acc + chunk.ilog2() as usize + 1)
            }
        })
        .into_inner()
}

// ---------------------------------------------------------

/// A space-efficient representation of an immutable vector of prime field
/// elements. Particularly useful when all elements have values close to each
/// other. It provides an interface similar to that of a `Vec`.
///
/// # Encoding method
///
/// This struct interpretes elements of the prime field `F_p` as integers in the
/// range `[0, p-1]` and tries to encode each with fewer bits than the default
/// of `sizeof::<F>() * 8` bits.
///
/// In particular, when a new bitpacked vector is created, the
/// [BitPackedVector::new] method computes the smallest interval `[a, b]`, with
/// `a < b`, such that all elements `v[i]` in the input vector (interpreted as
/// integers) belong to `[a, b]`, and then instead of storing `v[i]`, it stores
/// the value `(v[i] - a) \in [0, b - a]`. If `b - a` is a small integer,
/// representing `v[i] - a` can be done using `ceil(log_2(b - a + 1))` bits.
///
/// It then stores the encoded values compactly by packing together the
/// representation of many consecutive elements into a single machine word (when
/// possible). This encoding can store `n` elements using a total size of `(n *
/// ceil(log_2(b - a + 1)) * word_width) bits`.
///
/// # Notes
/// 1. Currently the implementation uses more storage than the theoretically
///    optimal mentioned above. This is because:
///    a. If `ceil(log_2(b - a + 1)) > 64`, we resort to the standard
///       representation of using `sizeof::<F>()` bytes. This is because for our
///       use-case, there are not many instances of vectors needing `c \in [65,
///       256]` bits to encode each value.
///    b. We round `ceil(log_2(b - a + 1))` up to the nearest divisor of 64.
///       This is to simplify the implementation by avoiding the situation where
///       the encoding of an element spans multiple words.
/// 2. For optimal performance, the buffer used to store the encoded values
///    should be using machine words (e.g. `buf: Vec<usize>`) instead of always
///    defaulting to 64-bit entries (`buf: Vec<u64>`) Here we always assume a
///    64-bit architecture for the simplicity of the implementation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub(in crate::mle::evals) struct BitPackedVector<F: Field> {
    /// The buffer for storing the bit-packed representation.
    /// As noted above, for optimal performance, the type of each element should
    /// be the machine's word size.
    /// For now, we're keeping it always to `u64` to make it easier to
    /// work with `F` chunks.
    ///
    /// *Invariant*: For every instance of a [BitPackedVector], either
    /// [BitPackedVector::buf] or [BitPackedVector::naive_buf] is populated but
    /// NEVER both.
    buf: Vec<u64>,

    /// If during initialization it is deduced that the number of bits
    /// needed per element is more than 64, we revert back to a standard
    /// representation. In that case, `Self::buf` is never used but instead
    /// `Self::naive_buf` is populated.
    ///
    /// *Invariant*: For every instance of a [BitPackedVector], either
    /// [BitPackedVector::buf] or [BitPackedVector::naive_buf] is populated but
    /// NEVER both.
    naive_buf: Vec<F>,

    /// The number of field elements stored in this vector.
    /// This is generally different from `self.buf.len()`.
    num_elements: usize,

    /// The value of the smallest element in the original vector.
    /// This is the value of `a` such that all elements of the original
    /// vector belong to the interval `[a, b]` as described above.
    offset: F,

    /// The number of bits required to represent each element optimally.
    /// This is equal to `ceil(log_2(b - a + 1))` as described above.
    bits_per_element: usize,
}

impl<F: Field> Zeroize for BitPackedVector<F> {
    fn zeroize(&mut self) {
        self.buf.iter_mut().for_each(|x| x.zeroize());
        self.naive_buf.iter_mut().for_each(|x| x.zeroize());
        self.num_elements.zeroize();
        self.offset.zeroize();
        self.bits_per_element.zeroize();
    }
}

impl<F: Field> BitPackedVector<F> {
    /// Generates a bit-packed vector initialized with `data`.
    pub fn new(data: &[F]) -> Self {
        // TODO(ryancao): Distinguish between prover and verifier here
        if !global_prover_enable_bit_packing() {
            return Self {
                buf: vec![],
                naive_buf: data.to_vec(),
                num_elements: data.len(),
                offset: F::ZERO,
                bits_per_element: 4 * (u64::BITS as usize),
            };
        }

        // Handle empty vectors separately.
        if data.is_empty() {
            return Self {
                buf: vec![],
                naive_buf: vec![],
                num_elements: 0,
                offset: F::ZERO,
                bits_per_element: 0,
            };
        }

        let num_elements = data.len();

        let min_val = *cfg_into_iter!(data).min().unwrap();
        let max_val = *cfg_into_iter!(data).max().unwrap();

        let range = max_val - min_val;

        // Handle constant values separately.
        if min_val == max_val {
            return Self {
                buf: vec![],
                naive_buf: vec![],
                num_elements,
                offset: min_val,
                bits_per_element: 0,
            };
        }

        // Number of bits required to encode each element in the range.
        let bits_per_element = num_bits(range);

        // Bits available per buffer entry.
        let entry_width = u64::BITS as usize;
        // println!("Buffer entry width: {entry_width}");

        // To simplify the implementation, for now we only support bit-packing
        // of values whose bit-width evenly divides the available bits per
        // buffer entry, or their bit-width equals 4*64 = 256 bits.  Any other
        // case is reduced to one of the two by rounding up `bits_per_element`
        // accordingly.
        let bits_per_element = if bits_per_element > entry_width {
            // Resort to storing the raw representation of the field element.
            4 * entry_width
        } else {
            // Round up to next power of two to make sure it evenly divides the
            // `entry_width`. This assumes that `entry_width` is always a power
            // of two.
            assert!(entry_width.is_power_of_two());
            bits_per_element.next_power_of_two()
        };

        assert!(
            bits_per_element == 4 * entry_width
                || (bits_per_element <= entry_width && entry_width % bits_per_element == 0)
        );

        if bits_per_element > entry_width {
            let naive_buf = data.to_vec();

            Self {
                buf: vec![],
                naive_buf,
                num_elements,
                offset: F::ZERO,
                bits_per_element,
            }
        } else {
            // Compute an upper bound to the number of buffer entries needed.
            let buf_len = (bits_per_element * num_elements).div_ceil(entry_width);

            let mut buf = vec![0_u64; buf_len];

            for (i, x) in data.iter().enumerate() {
                let encoded_x = *(*x - min_val).to_u64s_le().first().unwrap();
                // println!("Encoded value of {:?}: {encoded_x}", x);

                let buffer_idx = i * bits_per_element / entry_width;
                assert!(buffer_idx < buf_len);

                let word_idx = i * bits_per_element % entry_width;
                assert!(word_idx < entry_width);

                // println!(
                //     "Placing {i}-th element into buffer_idx: {buffer_idx}, and word_idx: {word_idx}"
                // );

                let prev_entry = &mut buf[buffer_idx];

                // Set new entry.
                *prev_entry |= encoded_x << word_idx;
            }

            Self {
                buf,
                naive_buf: vec![],
                num_elements,
                offset: min_val,
                bits_per_element,
            }
        }
    }

    /// Return the `index`-th element stored in the array,
    /// or `None` if `index` is out of bounds.
    pub fn get(&self, index: usize) -> Option<F> {
        // Check for index-out-of-bounds.
        if index >= self.num_elements {
            return None;
        }

        if self.bits_per_element == 0 {
            return Some(self.offset);
        }

        // Bits per buffer entry.
        let entry_width = u64::BITS as usize;

        if self.bits_per_element > entry_width {
            Some(self.naive_buf[index])
        } else {
            let buffer_idx = index * self.bits_per_element / entry_width;
            assert!(buffer_idx < self.buf.len());

            let word_idx = index * self.bits_per_element % entry_width;
            // println!("Getting buffer idx: {buffer_idx}, word_idx: {word_idx}");
            assert!(word_idx < entry_width);

            let entry = &self.buf[buffer_idx];
            let mask: u64 = if self.bits_per_element == 64 {
                !0x0
            } else {
                ((1_u64 << self.bits_per_element) - 1) << word_idx
            };
            // println!("Mask: {:#x}", mask);

            let encoded_value = (entry & mask) >> word_idx;
            let value = self.offset + F::from(encoded_value);

            Some(value)
        }
    }

    pub fn len(&self) -> usize {
        self.num_elements
    }

    /// Returns the number of bits used to encode each element.
    #[allow(unused)]
    pub fn get_bits_per_element(&self) -> usize {
        self.bits_per_element
    }

    pub fn iter(&self) -> BitPackedIterator<F> {
        BitPackedIterator {
            vec: self,
            current_index: 0,
        }
    }

    #[cfg(test)]
    pub fn to_vec(&self) -> Vec<F> {
        self.iter().collect()
    }
}

/// Iterator for a `BitPackedVector`. See [BitPackedVector::iter] for generating
/// one.
pub struct BitPackedIterator<'a, F: Field> {
    vec: &'a BitPackedVector<F>,
    current_index: usize,
}

impl<'a, F: Field> Iterator for BitPackedIterator<'a, F> {
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.vec.len() {
            let val = self.vec.get(self.current_index).unwrap();
            self.current_index += 1;

            Some(val)
        } else {
            None
        }
    }
}
