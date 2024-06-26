#[cfg(test)]
mod tests;

use std::{error::Error, ops::Index};

use ark_std::{cfg_into_iter, log2};
use itertools::{EitherOrBoth::*, Itertools};
use ndarray::{Array, ArrayView, Dimension, IxDyn};
use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

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
/// the dimension information of the MLE.
/// contains the dim: [IxDyn], see ndarray for more detailed documentation
/// and the names of the axes.
pub struct DimInfo {
    dims: IxDyn,
    axes_names: Vec<String>,
}

impl DimInfo {
    /// Creates a new DimInfo from the dimensions and the axes names.
    pub fn new(dims: IxDyn, axes_names: Vec<String>) -> Result<Self, DimensionError> {
        if dims.ndim() != axes_names.len() {
            return Err(DimensionError::DimensionMismatchError(
                dims.ndim(),
                axes_names.len(),
            ));
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

// ---------------------------------------------------------

/// Stores a boolean function `f: {0, 1}^n -> F` represented as a list of up to
/// `2^n` evaluations of `f` on the boolean hypercube.
/// The `n` variables are indexed from `0` to `n-1` throughout the lifetime of
/// the object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evaluations<F> {
    /// To understand how evaluations are stored, let's index `f`'s input bits
    /// as follows: `f(b_0, b_1, ..., b_{n-1})`. Evaluations are ordered using
    /// the bit-string `b_{n-1}b_{n-2}...b_1b_0` as key, hence the bit `b_{n-1}`
    /// will be referred to as the Most Significant Bit (MSB) and bit `b_0` as
    /// Least Significant Bit (LSB). This ordering is sometimes referred to as
    /// "little-endian" due to its resemblance to little-endian byte ordering.
    /// A suffix of contiguous evaluations all equal to `F::ZERO` may be
    /// omitted in this internal representation but this struct is not
    /// responsible for maintaining this property at all times.
    /// # Example
    /// * The evaluations of a 2-dimensional function are stored in the
    ///   following order: `[ f(0, 0), f(1, 0), f(0, 1), f(1, 1) ]`.
    /// * The evaluation table `[ 1, 0, 5, 0 ]` may be stored as `[1, 0, 5]` by
    ///   omitting the trailing zero. Note that both representations are valid.
    evals: Vec<F>,

    /// Number of input variables to `f`.
    /// Invariant: `0 <= evals.len() <= 2^num_vars`.
    /// The length can be zero due to suffix omission.
    num_vars: usize,

    /// TODO(Makis): Is there a better way to handle this??
    /// When accessing an element of the bookkeping table, we return a reference
    /// to a field element. In case the element is stored implicitly as a
    /// missing entry, we need someone to own the "zero" of the field.
    /// If I make this a const, I'm not sure how to initialize it.
    zero: F,
}

impl<F: FieldExt> Evaluations<F> {
    /// Returns a representation of the constant function on zero variables
    /// equal to `F::ZERO`.
    pub fn new_zero() -> Self {
        Self::new(0, vec![])
    }

    /// Builds an evaluation representation for a function `f: {0, 1}^num_vars
    /// -> F` from a vector of evaluations which are ordered by MSB first (see
    /// documentation comment for `self.evals` for explanation).
    /// # Example
    /// For a function `f: {0, 1}^2 -> F`, an evaluations table may be built as:
    /// `Evaluations::new(2, vec![ f(0, 0), f(1, 0), f(0, 1), f(1, 1) ])`.  In
    /// case, for example,`f(0, 1) == f(1, 1) == F::ZERO`, those values may be
    /// omitted and the following will generate an equivalent representation:
    /// `Evaluations::new(2, vec![ f(0, 0), f(1, 0) ])`.
    pub fn new(num_vars: usize, evals: Vec<F>) -> Self {
        debug_assert!(evals.len() <= (1 << num_vars));

        Evaluations::<F> {
            evals,
            num_vars,
            zero: F::ZERO,
        }
    }

    /// Builds an evaluation representation for a function `f: {0, 1}^num_vars
    /// -> F` from a vector of evaluations which are ordered by LSB first, in
    /// contrast to `new` which assumes an MSB-first representation.
    /// # Example
    /// For a function `f: {0, 1}^2 -> F`, an evaluations table may be built as:
    /// `Evaluations::new(2, &[ f(0, 0), f(0, 1), f(1, 0), f(1, 1) ])`.  In
    /// case, for example,`f(1, 0) == f(1, 1) == F::ZERO`, those values may be
    /// omitted and the following will generate an equivalent representation:
    /// `Evaluations::new(2, &[ f(0, 0), f(0, 1) ])`.
    pub fn new_from_big_endian(num_vars: usize, evals: &[F]) -> Self {
        debug_assert!(evals.len() <= (1 << num_vars));

        Self {
            evals: Self::flip_endianess(num_vars, evals),
            num_vars,
            zero: F::ZERO,
        }
    }

    /// returns the number of variables of the current Evalations.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Returns a iterator over the projection of the hypercube on `num_vars -
    /// 1` dimensions by pairing up evaluations on the dimension
    /// `fixed_variable_index`. For example, if `fix_variable_index == i`, the
    /// returned iterator returns elements of the form:
    /// `( f(x0, x_1, ..., x_i = 0, ..., x_{n-1}), f(x0, x1, ..., x_i = 1, ...,
    /// x_{n-1}) )`.
    /// The pairs are returned in little-endian order. For example:
    /// ```ignore
    /// [
    ///     ( f(0, 0, ..., 0, ..., 0), f(0, 0, ..., 1, ..., 0) ),
    ///     ( f(1, 0, ..., 0, ..., 0), f(1, 0, ..., 1, ..., 0) ),
    ///     ( f(0, 1, ..., 0, ..., 0), f(0, 1, ..., 1, ..., 0) ),
    ///      ....
    ///     ( f(1, 1, ..., 0, ..., 1), f(1, 1, ..., 1, ..., 1) ),
    /// ]
    /// ```
    pub fn project(&self, fixed_variable_index: usize) -> EvaluationsPairIterator<F> {
        let lsb_mask = (1_usize << fixed_variable_index) - 1;

        EvaluationsPairIterator::<F> {
            evals: self,
            lsb_mask,
            current_pair_index: 0,
        }
    }

    /// Temporary function returning the length of the internal representation.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.evals.len()
    }

    /// Temporary function returning a reference to the internal representation.
    pub fn repr(&self) -> &[F] {
        &self.evals
    }

    /// Temporary function returning a clone of the internal representation
    /// vector.
    pub fn to_vec(&self) -> Vec<F> {
        self.evals.clone()
    }

    /// Temporary function for accessing a the `idx`-th element in the internal
    /// representation.
    pub fn get(&self, idx: usize) -> Option<&F> {
        self.evals.get(idx)
    }

    // --------  Helper Functions --------

    /// Checks whether its arguments correspond to equivalent representations of
    /// the same list of evaluations. Two representations are equivalent if
    /// omitting the longest contiguous suffix of `F::ZERO`s from each results
    /// in the same vectors.
    #[allow(dead_code)]
    fn equiv_repr(evals1: &[F], evals2: &[F]) -> bool {
        evals1
            .iter()
            .zip_longest(evals2.iter())
            .map(|pair| match pair {
                Both(l, r) => *l == *r,
                Left(l) => *l == F::ZERO,
                Right(r) => *r == F::ZERO,
            })
            .all(|x| x)
    }

    /// Sorts the elements of `values` by their 0-based index transformed by
    /// mirroring the `num_bits` LSBs. This operation effectively flips the
    /// "endianess" of the index ordering. If `values.len() < 2^num_bits`, the
    /// missing values are assumed to be zeros. The resulting vector is always
    /// of size `2^num_bits`.
    /// # Example
    /// ```ignore
    ///     assert_eq!(flip_endianess(2, &[1, 2, 3, 4], vec![ 1, 3, 2, 4 ]);
    ///     assert_eq!(flip_endianess(2, &[ 1, 2 ]), vec![ 1, 0, 2, 0 ]);
    /// ```
    /// TODO(Makis): Benchmark and provide alternative implementations.
    fn flip_endianess(num_bits: usize, values: &[F]) -> Vec<F> {
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

/// Provides a vector-like interface to evaluations; useful during refactoring
/// but also for implementing `EvaluationsIterator`.
impl<F: FieldExt> Index<usize> for Evaluations<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        self.evals.get(index).unwrap_or(&self.zero)
    }
}

/// An iterator over evaluations indexed by vertices of a projection of the
/// boolean hypercube on `num_vars - 1` dimensions. See documentation for
/// `Evaluations::iter` for more information.
pub struct EvaluationsPairIterator<'a, F: FieldExt> {
    /// Reference to original bookkeeping table.
    evals: &'a Evaluations<F>,

    /// A mask for isolating the `k` LSBs of the `current_eval_index` where `k`
    /// is the dimension on which the original hypercube is projected on.
    lsb_mask: usize,

    /// 0-base index of the next element to be returned.
    /// Invariant: `current_eval_index \in [0, 2^(evals.num_vars() - 1)]`.
    /// If equal to `2^(evals.num_vars() - 1)`, the iterator has reached the
    /// end.
    current_pair_index: usize,
}

impl<'a, F: FieldExt> Iterator for EvaluationsPairIterator<'a, F> {
    type Item = (F, F);

    fn next(&mut self) -> Option<Self::Item> {
        let num_vars = self.evals.num_vars();
        let num_pairs = 1_usize << (num_vars - 1);

        if self.current_pair_index < num_pairs {
            // Compute the two indices by inserting a `0` and a `1` respectively
            // in the appropriate position of `current_pair_index`.
            // For example, if this is an Iterator projecting on
            // `fix_variable_index == 2` for an Evaluations table of `num_vars
            // == 5`, then `lsb_mask == 0b00011` (the `fix_variable_index` LSBs
            // are on). When, for example `current_pair_index == 0b1010`, it is
            // split into a "right part": `lsb_idx == 0b00 0 10`, and a "shifted
            // left part": `msb_idx == 0b10 0 00`.  The two parts are then
            // combined with the middle bit on and off respectively: `idx1 ==
            // 0b10 0 10`, `idx2 == 0b10 1 10`.
            let lsb_idx = self.current_pair_index & self.lsb_mask;
            let msb_idx = (self.current_pair_index & (!self.lsb_mask)) << 1;
            let mid_idx = self.lsb_mask + 1;

            let idx1 = lsb_idx | msb_idx;
            let idx2 = lsb_idx | mid_idx | msb_idx;

            self.current_pair_index += 1;

            let val1 = self.evals[idx1];
            let val2 = self.evals[idx2];

            Some((val1, val2))
        } else {
            None
        }
    }
}

/// Stores a function `\tilde{f}: F^n -> F`, the unique Multilinear
/// Extension (MLE) of a given function `f: {0, 1}^n -> F`:
/// ```ignore
///     \tilde{f}(x_0, ..., x_{n-1})
///         = \sum_{b_0, ..., b_{n-1} \in {0, 1}^n}
///             \tilde{beta}(x_0, ..., x_{n-1}, b_0, ..., b_{n-1})
///             * f(b_0, ..., b_{n-1}).
/// ```
/// where `\tilde{beta}` is the MLE of the equality function:
/// ```ignore
///     \tilde{beta}(x_0, ..., x_{n-1}, b_0, ..., b_{n-1})
///         = \prod_{i  = 0}^{n-1} ( x_i * b_i + (1 - x_i) * (1 - b_i) )
/// ```
/// Internally, `f` is represented as a list of evaluations of `f` on the
/// boolean hypercube.
/// The `n` variables are indexed from `0` to `n-1` throughout the lifetime of
/// the object even if `n` is modified by fixing a variable to a constant value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultilinearExtension<F> {
    /// The bookkeeping table with the evaluations of `f` on the hypercube.
    pub f: Evaluations<F>,
    dim_info: Option<DimInfo>,
}

impl<F: FieldExt> MultilinearExtension<F> {
    /// create a new MultilinearExtension from a [Vec<F>] of evaluations.
    pub fn new(vec: Vec<F>) -> Self {
        let num_vars = log2(vec.len()) as usize;
        let evals = Evaluations::new(num_vars, vec);
        MultilinearExtension::new_from_evals(evals)
    }

    /// Generate a new MultilinearExtension from a representation `evals` of a
    /// function `f`.
    pub fn new_from_evals(evals: Evaluations<F>) -> Self {
        Self {
            f: evals,
            dim_info: None,
        }
    }

    /// Generate a new MultilinearExtension from `evals` and `dim_info`.
    pub fn new_with_dim_info(evals: Evaluations<F>, dim_info: DimInfo) -> Self {
        let mut mle = Self::new_from_evals(evals);
        mle.set_dim_info(dim_info).unwrap();
        mle
    }

    /// Generate a new MultilinearExtension from a representation of `ndarray`
    pub fn new_from_ndarray(
        ndarray: Array<F, IxDyn>,
        axes_names: Vec<String>,
    ) -> Result<Self, Box<dyn Error>> {
        let dim_info = DimInfo::new(ndarray.raw_dim(), axes_names)?;
        let evals_vec = ndarray.into_raw_vec();
        let evals = Evaluations::new(log2(evals_vec.len()) as usize, evals_vec);
        let mle = Self::new_with_dim_info(evals, dim_info);
        Ok(mle)
    }

    /// Set the dimension information for the MLE.
    pub fn set_dim_info(&mut self, dim_info: DimInfo) -> Result<(), DimensionError> {
        let num_var_from_dim: u32 = dim_info
            .dims
            .slice()
            .iter()
            .map(|dim| log2(*dim as usize))
            .sum();
        if num_var_from_dim as usize != self.num_vars() {
            return Err(DimensionError::DimensionNumVarError(
                num_var_from_dim as usize,
                self.num_vars(),
            ));
        }

        self.dim_info = Some(dim_info);
        Ok(())
    }

    /// Get the dimension information for the MLE.
    pub fn dim_info(&self) -> &Option<DimInfo> {
        &self.dim_info
    }

    /// Set the MLE as an ndarray.
    pub fn get_mle_as_ndarray(&mut self) -> Result<ArrayView<F, IxDyn>, Box<dyn Error>> {
        if let Some(dim_info) = self.dim_info() {
            let ndarray: ArrayView<F, IxDyn> =
                ArrayView::from_shape(dim_info.dims.clone(), self.get_evals_vector())?;
            Ok(ndarray)
        } else {
            return Err(DimensionError::NoDimensionInfoError().into());
        }
    }

    /// Get the names of the axes of the MLE (multi-dimensional).
    pub fn get_axes_names(&mut self) -> Option<Vec<String>> {
        self.dim_info()
            .as_ref()
            .map(|dim_info| dim_info.axes_names.clone())
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

    /// Temporary function for accessing the bookkeping table.
    pub fn get_evals(&self) -> &Evaluations<F> {
        &self.f
    }

    /// Temporary function for accessing the bookkeping table.
    pub fn get_evals_vector(&self) -> &Vec<F> {
        &self.f.evals
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
                    let bit_i = idx & (1 << i);
                    if bit_i > 0 {
                        acc * point[i]
                    } else {
                        acc * (F::ONE - point[i])
                    }
                });
                acc + *v * beta
            })
    }

    /// For constant-function MLEs, returns its value.
    /// # Panics
    /// If `self.num_vars()` is non-zero.
    pub fn value(&self) -> F {
        assert_eq!(self.num_vars(), 0);

        self.f[0]
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
        // NEWEST IMPLEMENTATION: manually parallelize.
        let lsb_mask = (1_usize << var_index) - 1;

        let num_vars = self.num_vars();
        let num_pairs = 1_usize << (num_vars - 1);

        let new_evals: Vec<F> = (0..num_pairs)
            .into_par_iter()
            .map(|idx| {
                // Compute the two indices by inserting a `0` and a `1` respectively
                // in the appropriate position of `current_pair_index`.
                // For example, if this is an Iterator projecting on
                // `fix_variable_index == 2` for an Evaluations table of `num_vars
                // == 5`, then `lsb_mask == 0b00011` (the `fix_variable_index` LSBs
                // are on). When, for example `current_pair_index == 0b1010`, it is
                // split into a "right part": `lsb_idx == 0b00 0 10`, and a "shifted
                // left part": `msb_idx == 0b10 0 00`.  The two parts are then
                // combined with the middle bit on and off respectively: `idx1 ==
                // 0b10 0 10`, `idx2 == 0b10 1 10`.
                let lsb_idx = idx & lsb_mask;
                let msb_idx = (idx & (!lsb_mask)) << 1;
                let mid_idx = lsb_mask + 1;

                let idx1 = lsb_idx | msb_idx;
                let idx2 = lsb_idx | mid_idx | msb_idx;

                let val1 = self[idx1];
                let val2 = self[idx2];

                val1 + (val2 - val1) * point
            })
            .collect();

        debug_assert_eq!(new_evals.len(), 1 << (num_vars - 1));
        self.f = Evaluations::new(num_vars - 1, new_evals);

        // OLD IMPLEMENTATION: By accessing the bookkeeping table directly and
        // using parallel iterators.
        /*
        // Switch to 1-based indices.
        let var_index = var_index + 1;
        assert!(1 <= var_index && var_index <= self.num_vars());

        let chunk_size: usize = 1 << var_index;

        let outer_transform = |chunk: &[F]| {
            // This produces the wrong result when `self.evals.len()` is not an
            // exact power of 2.
            /*
            let window_size: usize = (1 << (var_index - 1)) + 1;

            let inner_transform = |window: &[F]| {
                let zero = F::ZERO;
                let first = window[0];
                let second = *window.get(window_size - 1).unwrap_or(&zero);

                // (1 - r) * V(i) + r * V(i + 1)
                first + (second - first) * point
            };

            #[cfg(feature = "parallel")]
            let new = chunk.par_windows(window_size).map(inner_transform);

            #[cfg(not(feature = "parallel"))]
            let new = chunk.windows(window_size).map(inner_transform);

            let inner_bookkeeping_table: Vec<F> = new.collect();

            inner_bookkeeping_table
            */
            let window_len = 1_usize << (var_index - 1);

            let inner_transform = |i: usize| {
                let first = *chunk.get(i).unwrap_or(&F::ZERO);
                let second = *chunk.get(i + window_len).unwrap_or(&F::ZERO);

                first + (second - first) * point
            };

            #[cfg(feature = "parallel")]
            let evals: Vec<F> = cfg_into_iter!(0..window_len).map(inner_transform).collect();

            #[cfg(not(feature = "parallel"))]
            let evals: Vec<F> = (0..window_len).into_iter().map(inner_transform).collect();

            evals
        };

        // --- So this goes through and applies the formula from [Tha13], bottom ---
        // --- of page 23 ---
        #[cfg(feature = "parallel")]
        let evals: Vec<F> = self
            .f
            .evals
            .par_chunks(chunk_size)
            .map(outer_transform)
            .flatten()
            .collect();

        #[cfg(not(feature = "parallel"))]
        let evals: Vec<F> = self
            .f
            .evals
            .chunks(chunk_size)
            .map(outer_transform)
            .flatten()
            .collect();

        // --- Note that MLE is destructively modified into the new bookkeeping table
        // here ---
        self.f = Evaluations::<F>::new(self.num_vars() - 1, evals);
        */

        // NEW IMPLEMENTATION: using projection iterators for succinct
        // description.
        /*
        let n = self.num_vars();
        let new_evals: Vec<F> = self
            .f
            .project(var_index)
            .map(|(v1, v2)| v1 + (v2 - v1) * point)
            .collect();
        debug_assert_eq!(new_evals.len(), 1 << (n - 1));

        self.f = Evaluations::new(n - 1, new_evals);
        */
    }

    /// Optimized version of `fix_variable_at_index` for `var_index == 0`.
    /// # Panics
    /// If `self.num_vars() == 0`.
    pub fn fix_variable(&mut self, point: F) {
        // OLD IMPLEMENTATION: Using direct access mechanism.
        assert!(self.num_vars() > 0);

        let transform = |chunk: &[F]| {
            let zero = F::ZERO;
            let first = chunk[0];
            let second = chunk.get(1).unwrap_or(&zero);

            // (1 - r) * V(i) + r * V(i + 1)
            first + (*second - first) * point
        };

        // --- So this goes through and applies the formula from [Tha13], bottom ---
        // --- of page 23 ---
        #[cfg(feature = "parallel")]
        let new = self.f.evals.par_chunks(2).map(transform);

        #[cfg(not(feature = "parallel"))]
        let new = self.f.evals.chunks(2).map(transform);

        // --- Note that MLE is destructively modified into the new bookkeeping
        // table here ---
        self.f = Evaluations::<F>::new(self.num_vars() - 1, new.collect());
    }
}

/// Provides a vector-like interface to MultilinearExtensions;
/// useful during refactoring.
impl<F: FieldExt> Index<usize> for MultilinearExtension<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.f[index]
    }
}
