#[cfg(test)]
mod tests;

use ark_std::{cfg_into_iter, log2};
use itertools::{EitherOrBoth::*, Itertools};
use ndarray::{Dimension, IxDyn};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use shared_types::Field;
use thiserror::Error;

pub mod bit_packed_vector;

use bit_packed_vector::BitPackedVector;
use zeroize::Zeroize;

use crate::utils::arithmetic::i64_to_field;

use anyhow::{anyhow, Result};

#[derive(Error, Debug, Clone)]
/// the errors associated with the dimension of the MLE.
pub enum DimensionError {
    #[error("Dimensions: {0} do not match with number of axes: {1} as indicated by their names.")]
    /// The dimensions of the MLE do not match the number of axes.
    DimensionMismatchError(usize, usize),
    #[error("Dimensions: {0} do not match with the numvar: {1} of the mle.")]
    /// The dimensions of the MLE do not match the number of variables.
    DimensionNumVarError(usize, usize),
    #[error("Trying to get the underlying mle as an ndarray, but there is no dimension info.")]
    /// No dimension info of the MLE.
    NoDimensionInfoError(),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
/// the dimension information of the MLE. contains the dim: [type@IxDyn], see ndarray
/// for more detailed documentation and the names of the axes.
pub struct DimInfo {
    dims: IxDyn,
    axes_names: Vec<String>,
}

impl DimInfo {
    /// Creates a new DimInfo from the dimensions and the axes names.
    pub fn new(dims: IxDyn, axes_names: Vec<String>) -> Result<Self> {
        if dims.ndim() != axes_names.len() {
            return Err(anyhow!(DimensionError::DimensionMismatchError(
                dims.ndim(),
                axes_names.len(),
            )));
        }
        Ok(Self { dims, axes_names })
    }
}

impl std::fmt::Debug for DimInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DimensionInfo")
            .field("dim sizes", &self.dims.slice())
            .field("axes names", &self.axes_names)
            .finish()
    }
}

// -------------- Various Helper Functions -----------------

/// Mirrors the `num_bits` LSBs of `value`.
/// # Example
/// ```ignore
///     assert_eq!(mirror_bits(4, 0b1110), 0b0111);
///     assert_eq!(mirror_bits(3, 0b1110), 0b1011);
///     assert_eq!(mirror_bits(2, 0b1110), 0b1101);
///     assert_eq!(mirror_bits(1, 0b1110), 0b1110);
///     assert_eq!(mirror_bits(0, 0b1110), 0b1110);
/// ```
fn mirror_bits(num_bits: usize, mut value: usize) -> usize {
    let mut result: usize = 0;

    for _ in 0..num_bits {
        result = (result << 1) | (value & 1);
        value >>= 1;
    }

    // Add back the remaining bits.
    result | (value << num_bits)
}

/// Stores a boolean function `f: {0, 1}^n -> F` represented as a list of up to
/// `2^n` evaluations of `f` on the boolean hypercube. The `n` variables are
/// indexed from `0` to `n-1` throughout the lifetime of the object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct Evaluations<F: Field> {
    /// To understand how evaluations are stored, let's index `f`'s input bits
    /// as follows: `f(b_0, b_1, ..., b_{n-1})`. Evaluations are ordered using
    /// the bit-string `b_0b_1...b_{n-2}b_{n-1}` as key. This ordering is
    /// sometimes referred to as "big-endian" due to its resemblance to
    /// big-endian byte ordering. A suffix of contiguous evaluations all equal
    /// to `F::ZERO` may be omitted in this internal representation but this
    /// struct is not responsible for maintaining this property at all times.
    /// # Example
    /// * The evaluations of a 2-dimensional function are stored in the
    ///   following order: `[ f(0, 0), f(0, 1), f(1, 0), f(1, 1) ]`.
    /// * The evaluation table `[ 1, 0, 5, 0 ]` may be stored as `[1, 0, 5]` by
    ///   omitting the trailing zero. Note that both representations are valid.
    evals: BitPackedVector<F>,

    /// Number of input variables to `f`. Invariant: `0 <= evals.len() <=
    /// 2^num_vars`. The length can be less than `2^num_vars` due to suffix
    /// omission.
    num_vars: usize,

    /// TODO(Makis): Is there a better way to handle this?? When accessing an
    /// element of the bookkeping table, we return a reference to a field
    /// element. In case the element is stored implicitly as a missing entry, we
    /// need someone to own the "zero" of the field. If I make this a const, I'm
    /// not sure how to initialize it.
    zero: F,
}

impl<F: Field> Zeroize for Evaluations<F> {
    fn zeroize(&mut self) {
        self.evals.zeroize();
        self.num_vars.zeroize();
        self.zero.zeroize();
    }
}

impl<F: Field> Evaluations<F> {
    /// Returns a representation of the constant function on zero variables
    /// equal to `F::ZERO`.
    pub fn new_zero() -> Self {
        Self::new(0, vec![])
    }

    /// Builds an evaluation representation for a function `f: {0, 1}^num_vars
    /// -> F` from a vector of evaluations in big-endian order (see
    /// documentation comment for [Self::evals] for explanation).
    ///
    /// # Example
    /// For a function `f: {0, 1}^2 -> F`, an evaluations table may be built as:
    /// `Evaluations::new(2, vec![ f(0, 0), f(0, 1), f(1, 0), f(1, 1) ])`.
    ///
    /// An example of suffix omission is when `f(1, 0) == f(1, 1) == F::ZERO`.
    /// In that case those zero values may be omitted and the following
    /// statement generates an equivalent representation: `Evaluations::new(2,
    /// vec![ f(0, 0), f(0, 1) ])`.
    pub fn new(num_vars: usize, evals: Vec<F>) -> Self {
        debug_assert!(evals.len() <= (1 << num_vars));

        // debug_evals(&evals);

        Evaluations::<F> {
            evals: BitPackedVector::new(&evals),
            num_vars,
            zero: F::ZERO,
        }
    }

    /// Builds an evaluation representation for a function `f: {0, 1}^num_vars
    /// -> F` from a vector of evaluations in _little-endian_ order (see
    /// documentation comment for [Self::evals] for explanation).
    ///
    /// # Example
    /// For a function `f: {0, 1}^2 -> F`, an evaluations table may be built as:
    /// `Evaluations::new_from_little_endian(2, vec![ f(0, 0), f(1, 0), f(0, 1),
    /// f(1, 1) ])`.
    pub fn new_from_little_endian(num_vars: usize, evals: &[F]) -> Self {
        debug_assert!(evals.len() <= (1 << num_vars));

        println!("New MLE (big-endian) on {} entries.", evals.len());

        Self {
            evals: BitPackedVector::new(&Self::flip_endianess(num_vars, evals)),
            num_vars,
            zero: F::ZERO,
        }
    }

    /// Returns the number of variables of the current `Evalations`.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Returns true if the boolean function has not free variables. Equivalent
    /// to checking whether that [Self::num_vars] is equal to zero.
    pub fn is_fully_bound(&self) -> bool {
        self.num_vars == 0
    }

    /// Returns the first element of the bookkeeping table. This operation
    /// should always be successful because even in the case that
    /// [Self::num_vars] is zero, there is a non-zero number of vertices on the
    /// boolean hypercube and hence there's at least one evaluation stored in
    /// the bookkeeping table, either explicitly as a value inside
    /// [Self::evals], or implicitly if it's a `F::ZERO` that has been pruned as
    /// part of a zero suffix.
    pub fn first(&self) -> F {
        self.evals.get(0).unwrap_or(F::ZERO)
    }

    /// If `self` represents a fully-bound boolean function (i.e.
    /// [Self::num_vars] is zero), it returns its value. Otherwise panics.
    pub fn value(&self) -> F {
        assert!(self.is_fully_bound());
        self.first()
    }

    /// Returns an iterator that traverses the evaluations in "big-endian"
    /// order.
    pub fn iter(&self) -> EvaluationsIterator<'_, F> {
        EvaluationsIterator::<F> {
            evals: self,
            current_index: 0,
        }
    }

    /// Temporary function returning the length of the internal representation.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.evals.len()
    }

    /// Temporary function for accessing a the `idx`-th element in the internal
    /// representation.
    pub fn get(&self, idx: usize) -> Option<F> {
        self.evals.get(idx)
    }

    // --------  Helper Functions --------

    /// Checks whether its arguments correspond to equivalent representations of
    /// the same list of evaluations. Two representations are equivalent if
    /// omitting the longest contiguous suffix of `F::ZERO`s from each results
    /// in the same vectors.
    #[allow(dead_code)]
    fn equiv_repr(evals1: &BitPackedVector<F>, evals2: &BitPackedVector<F>) -> bool {
        evals1
            .iter()
            .zip_longest(evals2.iter())
            .all(|pair| match pair {
                Both(l, r) => l == r,
                Left(l) => l == F::ZERO,
                Right(r) => r == F::ZERO,
            })
    }

    /// Sorts the elements of `values` by their 0-based index transformed by
    /// mirroring the `num_bits` LSBs. This operation effectively flips the
    /// "endianess" of the index ordering. If `values.len() < 2^num_bits`, the
    /// missing values are assumed to be zeros. The resulting vector is always
    /// of size `2^num_bits`.
    /// # Example
    /// ```
    /// use remainder::mle::evals::Evaluations;
    /// use shared_types::Fr;
    /// assert_eq!(Evaluations::flip_endianess(
    ///     2,
    ///     &[Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]),
    ///     vec![ Fr::from(1), Fr::from(3), Fr::from(2), Fr::from(4) ]
    /// );
    /// assert_eq!(Evaluations::flip_endianess(
    ///     2,
    ///     &[ Fr::from(1), Fr::from(2) ]),
    ///     vec![ Fr::from(1), Fr::from(0), Fr::from(2), Fr::from(0) ]
    /// );
    /// ```
    pub fn flip_endianess(num_bits: usize, values: &[F]) -> Vec<F> {
        let num_evals = values.len();

        let result: Vec<F> = cfg_into_iter!(0..(1 << num_bits))
            .map(|idx| {
                let mirrored_idx = mirror_bits(num_bits, idx);
                if mirrored_idx >= num_evals {
                    F::ZERO
                } else {
                    values[mirrored_idx]
                }
            })
            .collect();

        result
    }
}

/// An iterator over evaluations in a "big-endian" order.
pub struct EvaluationsIterator<'a, F: Field> {
    /// Reference to the original `Evaluations` struct.
    evals: &'a Evaluations<F>,

    /// Index of the next evaluation to be retrieved.
    current_index: usize,
}

impl<F: Field> Iterator for EvaluationsIterator<'_, F> {
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.evals.len() {
            let val = self.evals.get(self.current_index).unwrap();
            self.current_index += 1;

            Some(val)
        } else {
            None
        }
    }
}

impl<F: Field> Clone for EvaluationsIterator<'_, F> {
    fn clone(&self) -> Self {
        Self {
            evals: self.evals,
            current_index: self.current_index,
        }
    }
}

/// An iterator over evaluations indexed by vertices of a projection of the
/// boolean hypercube on `num_vars - 1` dimensions. See documentation for
/// `Evaluations::project` for more information.
#[allow(dead_code)]
pub struct EvaluationsPairIterator<'a, F: Field> {
    /// Reference to original bookkeeping table.
    evals: &'a Evaluations<F>,

    /// A mask for isolating the `k` LSBs of the `current_eval_index` where `k`
    /// is the dimension on which the original hypercube is projected on.
    lsb_mask: usize,

    /// 0-base index of the next element to be returned. Invariant:
    /// `current_eval_index \in [0, 2^(evals.num_vars() - 1)]`. If equal to
    /// `2^(evals.num_vars() - 1)`, the iterator has reached the end.
    current_pair_index: usize,
}

impl<F: Field> Iterator for EvaluationsPairIterator<'_, F> {
    type Item = (F, F);

    fn next(&mut self) -> Option<Self::Item> {
        let num_vars = self.evals.num_vars();
        let num_pairs = 1_usize << (num_vars - 1);

        if self.current_pair_index < num_pairs {
            // Compute the two indices by inserting a `0` and a `1` respectively
            // in the appropriate position of `current_pair_index`. For example,
            // if this is an Iterator projecting on `fix_variable_index == 2`
            // for an Evaluations table of `num_vars == 5`, then `lsb_mask ==
            // 0b00011` (the `fix_variable_index` LSBs are on). When, for
            // example `current_pair_index == 0b1010`, it is split into a "right
            // part": `lsb_idx == 0b00 0 10`, and a "shifted left part":
            // `msb_idx == 0b10 0 00`.  The two parts are then combined with the
            // middle bit on and off respectively: `idx1 == 0b10 0 10`, `idx2 ==
            // 0b10 1 10`.
            let lsb_idx = self.current_pair_index & self.lsb_mask;
            let msb_idx = (self.current_pair_index & (!self.lsb_mask)) << 1;
            let mid_idx = self.lsb_mask + 1;

            let idx1 = lsb_idx | msb_idx;
            let idx2 = lsb_idx | mid_idx | msb_idx;

            self.current_pair_index += 1;

            let val1 = self.evals.get(idx1).unwrap();
            let val2 = self.evals.get(idx2).unwrap();

            Some((val1, val2))
        } else {
            None
        }
    }
}

/// Stores a function `\tilde{f}: F^n -> F`, the unique Multilinear Extension
/// (MLE) of a given function `f: {0, 1}^n -> F`:
/// ```text
///     \tilde{f}(x_0, ..., x_{n-1})
///         = \sum_{b_0, ..., b_{n-1} \in {0, 1}^n}
///             \tilde{beta}(x_0, ..., x_{n-1}, b_0, ..., b_{n-1})
///             * f(b_0, ..., b_{n-1}).
/// ```
/// where `\tilde{beta}` is the MLE of the equality function:
/// ```text
///     \tilde{beta}(x_0, ..., x_{n-1}, b_0, ..., b_{n-1})
///         = \prod_{i  = 0}^{n-1} ( x_i * b_i + (1 - x_i) * (1 - b_i) )
/// ```
/// Internally, `f` is represented as a list of evaluations of `f` on the
/// boolean hypercube. The `n` variables are indexed from `0` to `n-1`
/// throughout the lifetime of the object even if `n` is modified by fixing a
/// variable to a constant value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct MultilinearExtension<F: Field> {
    /// The bookkeeping table with the evaluations of `f` on the hypercube.
    pub f: Evaluations<F>,
}

impl<F: Field> Zeroize for MultilinearExtension<F> {
    fn zeroize(&mut self) {
        self.f.zeroize();
    }
}

impl<F: Field> From<Vec<bool>> for MultilinearExtension<F> {
    fn from(bools: Vec<bool>) -> Self {
        let evals = bools
            .into_iter()
            .map(|b| if b { F::ONE } else { F::ZERO })
            .collect();
        MultilinearExtension::new(evals)
    }
}

impl<F: Field> From<Vec<u32>> for MultilinearExtension<F> {
    fn from(uints: Vec<u32>) -> Self {
        let evals = uints.into_iter().map(|v| F::from(v as u64)).collect();
        MultilinearExtension::new(evals)
    }
}

impl<F: Field> From<Vec<u64>> for MultilinearExtension<F> {
    fn from(uints: Vec<u64>) -> Self {
        let evals = uints.into_iter().map(F::from).collect();
        MultilinearExtension::new(evals)
    }
}

impl<F: Field> From<Vec<i32>> for MultilinearExtension<F> {
    fn from(ints: Vec<i32>) -> Self {
        let evals = ints.into_iter().map(|v| i64_to_field(v as i64)).collect();
        MultilinearExtension::new(evals)
    }
}

impl<F: Field> From<Vec<i64>> for MultilinearExtension<F> {
    fn from(ints: Vec<i64>) -> Self {
        let evals = ints.into_iter().map(i64_to_field).collect();
        MultilinearExtension::new(evals)
    }
}

impl<F: Field> MultilinearExtension<F> {
    /// Create a new MultilinearExtension from a [`Vec<F>`] of evaluations.
    pub fn new(evals_vec: Vec<F>) -> Self {
        let num_vars = log2(evals_vec.len()) as usize;
        let evals = Evaluations::new(num_vars, evals_vec);
        MultilinearExtension::new_from_evals(evals)
    }

    /// Generate a new MultilinearExtension from a representation `evals` of a
    /// function `f`.
    pub fn new_from_evals(evals: Evaluations<F>) -> Self {
        Self { f: evals }
    }

    /// Creates a new mle which is all zeroes of a specific num_vars. In this
    /// case the size of the evals and the num_vars will not match up
    pub fn new_sized_zero(num_vars: usize) -> Self {
        Self {
            f: Evaluations {
                evals: BitPackedVector::new(&[]),
                num_vars,
                zero: F::ZERO,
            },
        }
    }

    /// Returns an iterator accessing the evaluations defining this MLE in
    /// "big-endian" order.
    pub fn iter(&self) -> EvaluationsIterator<'_, F> {
        self.f.iter()
    }

    /// Generate a Vector of the evaluations of `f` over the hypercube.
    pub fn to_vec(&self) -> Vec<F> {
        self.f.iter().collect()
    }

    /// Returns true if the MLE has not free variables. Equivalent to checking
    /// whether that [Self::num_vars] is equal to zero.
    pub fn is_fully_bound(&self) -> bool {
        self.f.is_fully_bound()
    }

    /// Returns the first element of the bookkeeping table of this MLE,
    /// corresponding to the value of the MLE when all varables are set to zero.
    /// This operation never fails (see [Evaluations::first]).
    pub fn first(&self) -> F {
        self.f.first()
    }

    /// If `self` represents a fully-bound MLE (i.e. on zero variables), it
    /// returns its value. Otherwise panics.
    pub fn value(&self) -> F {
        self.f.value()
    }

    /// Generates a representation for the MLE of the zero function on zero
    /// variables.
    pub fn new_zero() -> Self {
        let zero_evals = Evaluations::new_zero();
        Self::new_from_evals(zero_evals)
    }

    /// Returns `n`, the number of arguments `\tilde{f}` takes.
    pub fn num_vars(&self) -> usize {
        self.f.num_vars()
    }

    /// Returns the `idx`-th element, if `idx` is in the range `[0,
    /// 2^self.num_vars)`.
    pub fn get(&self, idx: usize) -> Option<F> {
        if idx >= (1 << self.num_vars()) {
            // `idx` is out of range.
            None
        } else if idx >= self.f.len() {
            // `idx` is within range, but value is implicitly assumed to be
            // zero.
            Some(F::ZERO)
        } else {
            // `idx`-th position is stored explicitly in `self.f`
            self.f.get(idx)
        }
    }

    /// Evaluate `\tilde{f}` at `point \in F^n`.
    /// # Panics
    /// If `point` does not contain exactly `self.num_vars()` elements.
    pub fn evaluate_at_point(&self, point: &[F]) -> F {
        let n = self.num_vars();
        assert_eq!(n, point.len());

        // TODO: Provide better access mechanism.
        self.f
            .evals
            .clone()
            .iter() // was into_iter()
            .enumerate()
            .fold(F::ZERO, |acc, (idx, v)| {
                let beta = (0..n).fold(F::ONE, |acc, i| {
                    let bit_i = idx & (1 << (n - 1 - i));
                    if bit_i > 0 {
                        acc * point[i]
                    } else {
                        acc * (F::ONE - point[i])
                    }
                });
                acc + v * beta
            })
    }

    /// Returns the length of the evaluations vector.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.f.len()
    }

    /// Fix the 0-based `var_index`-th bit of `\tilde{f}` to an arbitrary field
    /// element `point \in F` by destructively modifying `self`.
    /// # Params
    /// * `var_index`: A 0-based index of the input variable to be fixed.
    /// * `point`: The field element to set `x_{var_index}` equal to.
    /// # Example
    /// If `self` represents a function `\tilde{f}: F^3 -> F`,
    /// `self.fix_variable_at_index(1, r)` fixes the middle variable to `r \in
    /// F`. After the invocation, `self` represents a function `\tilde{g}: F^2
    /// -> F` defined as the multilinear extension of the following function:
    /// `g(b_0, b_1) = \tilde{f}(b_0, r, b_1)`.
    /// # Panics
    /// if `var_index` is outside the interval `[0, self.num_vars())`.
    pub fn fix_variable_at_index(&mut self, var_index: usize, point: F) {
        let num_vars = self.num_vars();
        let lsb_mask = (1_usize << (num_vars - 1 - var_index)) - 1;

        let num_pairs = 1_usize << (num_vars - 1);

        let new_evals: Vec<F> = cfg_into_iter!(0..num_pairs)
            .map(|idx| {
                // This iteration computes the value of
                // `f'(idx[0], ..., idx[var_index-1], idx[var_index+1], ..., idx[num_vars - 1])`
                // where `f'` is the resulting function after fixing the
                // the `var_index`-th variable.
                // To do this, we must combine the values of:
                // `f(idx1) = f(idx[0], ..., idx[var_index-1], 0, idx[var_index+1], ..., idx[num_vars-1])`
                // and
                // `f(idx2) = f(idx[0], ..., idx[var_index-1], 1, idx[var_index+1], ..., idx[num_vars-1])`
                // Below we compute `idx1` and `idx2` corresponding to the two
                // indices above.

                // Compute the two indices by inserting a `0` and a `1`
                // respectively in the appropriate position of `idx`. For
                // example, if `var_index == 2` and `self.num_vars == 5`, then
                // `lsb_mask == 0b0011` (the `num_var - 1 - var_index` LSBs are
                // on). When, for example `idx == 0b1010`, it is split into a
                // "right part": `lsb_idx == 0b00 0 10`, and a "shifted left
                // part": `msb_idx == 0b10 0 00`.  The two parts are then
                // combined with the middle bit on and off respectively: `idx1
                // == 0b10 0 10`, `idx2 == 0b10 1 10`.
                let lsb_idx = idx & lsb_mask;
                let msb_idx = (idx & (!lsb_mask)) << 1;
                let mid_idx = lsb_mask + 1;

                let idx1 = lsb_idx | msb_idx;
                let idx2 = lsb_idx | mid_idx | msb_idx;

                let val1 = self.get(idx1).unwrap_or(F::ZERO);
                let val2 = self.get(idx2).unwrap_or(F::ZERO);

                val1 + (val2 - val1) * point
            })
            .collect();

        debug_assert_eq!(new_evals.len(), 1 << (num_vars - 1));
        self.f = Evaluations::new(num_vars - 1, new_evals);
    }

    /// Optimized version of `fix_variable_at_index` for `var_index == 0`.
    /// # Panics
    /// If `self.num_vars() == 0`.
    pub fn fix_variable(&mut self, point: F) {
        self.fix_variable_at_index(0, point);
    }

    /// Stacks the MLEs into a single MLE, assuming they are stored in a "big
    /// endian" fashion.
    pub fn stack_mles(mles: Vec<MultilinearExtension<F>>) -> MultilinearExtension<F> {
        let first_len = mles[0].len();

        if !mles.iter().all(|v| v.len() == first_len) {
            panic!("All mles's underlying bookkeeping table must have the same length");
        }

        let out = mles.iter().flat_map(|mle| mle.to_vec()).collect();
        Self::new(out)
    }

    /// Convert a [MultilinearExtension] into a vector of u8s.
    /// Every element is padded to contain 8 bits.
    pub fn convert_into_u8_vec(&self) -> Vec<u8> {
        self.f
            .iter()
            .map(|field_element| {
                let field_element_le_bytes = field_element.to_bytes_le();
                let mut padded_u8 = [0u8; 1];
                padded_u8.copy_from_slice(&field_element_le_bytes[..1]);
                u8::from_le_bytes(padded_u8)
            })
            .collect_vec()
    }

    /// Convert a [MultilinearExtension] into a vector of u16s.
    /// Every element is padded to contain 16 bits.
    pub fn convert_into_u16_vec(&self) -> Vec<u16> {
        self.f
            .iter()
            .map(|field_element| {
                let field_element_le_bytes = field_element.to_bytes_le();
                let mut padded_u16 = [0u8; 2];
                padded_u16.copy_from_slice(&field_element_le_bytes[..2]);
                u16::from_le_bytes(padded_u16)
            })
            .collect_vec()
    }

    /// Convert a [MultilinearExtension] into a vector of u32s.
    /// Every element is padded to contain 32 bits.
    pub fn convert_into_u32_vec(&self) -> Vec<u32> {
        self.f
            .iter()
            .map(|field_element| {
                let field_element_le_bytes = field_element.to_bytes_le();
                let mut padded_u32 = [0u8; 4];
                padded_u32.copy_from_slice(&field_element_le_bytes[..4]);
                u32::from_le_bytes(padded_u32)
            })
            .collect_vec()
    }

    /// Convert a [MultilinearExtension] into a vector of u64s.
    /// Every element is padded to contain 64 bits.
    pub fn convert_into_u64_vec(&self) -> Vec<u64> {
        self.f
            .iter()
            .map(|field_element| {
                let field_element_le_bytes = field_element.to_bytes_le();
                let mut padded_u64 = [0u8; 8];
                padded_u64.copy_from_slice(&field_element_le_bytes[..8]);
                u64::from_le_bytes(padded_u64)
            })
            .collect_vec()
    }

    /// Convert a [MultilinearExtension] into a vector of u128s.
    /// Every element is padded to contain 128 bits.
    pub fn convert_into_u128_vec(&self) -> Vec<u128> {
        self.f
            .iter()
            .map(|field_element| {
                let field_element_le_bytes = field_element.to_bytes_le();
                let mut padded_u128 = [0u8; 16];
                padded_u128.copy_from_slice(&field_element_le_bytes[..16]);
                u128::from_le_bytes(padded_u128)
            })
            .collect_vec()
    }
}
