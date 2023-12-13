use ark_std::{cfg_into_iter, log2};
use itertools::Itertools;
use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use remainder_shared_types::FieldExt;

/// Stores a boolean function `f: {0, 1}^n -> F` represented as a list of up to
/// `2^n` evaluations of `f` on the boolean hypercube.
/// This struct additionally supports operations related to the unique
/// Multi-linear Extension (MLE) `\tilde{f}` of `f` over the field `F`.
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluations<F: FieldExt> {
    /// To understand how evaluations are stored, let's index `f`'s input bits
    /// as follows: `f(b_0, b_1, ..., b_{n-1})`. Evaluations are ordered using
    /// the bit-string `b_{n-1}b_{n-2}...b_1b_0` as key, hence the bit `b_{n-1}`
    /// will be referred to as the Most Significant Bit (MSB) and bit `b_0` as
    /// Least Significant Bit (LSB). This ordering is sometimes referred to as
    /// "little-endian" due to its resemblance to little-endian byte ordering.
    /// A suffix of contiguous evaluations all equal to `F::zero()` may be
    /// omitted.
    /// # Example
    /// * The evaluations of a 2-dimensional function are stored in the
    ///   following order: `[ f(0, 0), f(1, 0), f(0, 1), f(1, 1) ]`.
    /// * The evaluation table `[ 1, 0, 5, 0 ]` may be stored as `[1, 0, 5]` by
    ///   omitting the trailing zero.
    evals: Vec<F>,

    /// Number of input variables to `f`.
    /// Invariant: `0 < evals.len() <= 2^num_vars`.
    num_vars: usize,
}

impl<F: FieldExt> Evaluations<F> {
    /// Builds an evaluation representation for a function `f: {0, 1}^num_vars
    /// -> F` from a vector of evaluations which are ordered by MSB first (see
    /// documentation comment for `self.evals` for explanation).
    /// # Example
    /// For a function `f: {0, 1}^2 -> F`, an evaluations table may be built as:
    /// `Evaluations::new(2, vec![ f(0, 0), f(1, 0), f(0, 1), f(1, 1) ])`.  In
    /// case, for example,`f(0, 1) = f(1, 1) = F::zero()`, those values may be
    /// omitted and the following will generate an equivalent representation:
    /// `Evaluations::new(2, vec![ f(0, 0), f(1, 0) ])`.
    /// # Panics
    /// If `evals.is_empty()` or if `eval` contains more than `2^num_vars`
    /// evaluations.
    pub fn new(num_vars: usize, evals: Vec<F>) -> Self {
        assert!(0 < evals.len() && evals.len() <= (1 << num_vars));
        Evaluations::<F> { evals, num_vars }
    }

    /// Builds an evaluation representation for a function `f: {0, 1}^num_vars
    /// -> F` from a vector of evaluations which are ordered by LSB first, in
    /// contrast to `new` which assumes an MSB-first representation.
    /// # Example
    /// For a function `f: {0, 1}^2 -> F`, an evaluations table may be built as:
    /// `Evaluations::new(2, &[ f(0, 0), f(0, 1), f(1, 0), f(1, 1) ])`.  In
    /// case, for example,`f(1, 0) = f(1, 1) = F::zero()`, those values may be
    /// omitted and the following will generate an equivalent representation:
    /// `Evaluations::new(2, &[ f(0, 0), f(0, 1) ])`.
    /// # Panics
    /// If `evals.is_empty()` or if `eval` contains more than `2^num_vars`
    /// evaluations.
    pub fn new_from_big_endian(num_vars: usize, evals: &[F]) -> Self {
        assert!(0 < evals.len() && evals.len() <= (1 << num_vars));

        Self {
            evals: Self::flip_endianess(num_vars, evals),
            num_vars,
        }
    }

    /// Fix the `var_index`-th bit of the Multi-linear Extension `\tilde{f}(x_0,
    /// ..., x_{n-1})` of `f` to an arbitrary field element `point \in F` by
    /// destructively modifying `self`.
    /// # Params
    /// * `var_index`: A 0-based index of the input variable to be fixed.
    /// * `point`: The field element to set `x_{var_index}` equal to.
    /// # Example
    /// If `self` represents a function `f: {0, 1}^3 -> F`,
    /// `self.fix_variable_at_index(1, r)` fixes the middle variable to `r \in
    /// F`. After the invocation, `self` represents a function `g: {0, 1}^2 ->
    /// F` defined as follows:
    /// ```
    ///     g(b_0, b_1) = \tilde{f}(b_0, r, b_1),
    /// ```
    /// where `\tilde{f}` is the unique MLE of `f` which in general for
    /// `num_vars == n` is defined as:
    /// ```
    ///     \tilde{f}(x_0, ..., x_{n-1})
    ///         = \sum_{b_0, ..., b_{n-1} \in {0, 1}^n}
    ///             \tilde{beta}(x_0, ..., x_{n-1}, b_0, ..., b_{n-1})
    ///             * f(b_0, ..., b_{n-1}).
    /// ```
    /// # Panics
    /// if `var_index` is outside the interval [0, num_vars).
    fn fix_variable_at_index(&mut self, var_index: usize, point: F) {
        // Switch to 1-based indices.
        let var_index = var_index + 1;
        assert!(1 <= var_index && var_index <= self.num_vars);

        let chunk_size: usize = 1 << var_index;

        let outer_transform = |chunk: &[F]| {
            // This produces the wrong result when `self.evals.len()` is not an
            // exact power of 2.
            /*
            let window_size: usize = (1 << (var_index - 1)) + 1;

            let inner_transform = |window: &[F]| {
                let zero = F::zero();
                let first = window[0];
                let second = *window.get(window_size - 1).unwrap_or(&zero);

                // (1 - r) * V(i) + r * V(i + 1)
                first + (second - first) * point
            };

            // TODO(Makis): Consider using a custom iterator here instead of windows.
            #[cfg(feature = "parallel")]
            let new = chunk.par_windows(window_size).map(inner_transform);

            #[cfg(not(feature = "parallel"))]
            let new = chunk.windows(window_size).map(inner_transform);

            let inner_bookkeeping_table: Vec<F> = new.collect();

            inner_bookkeeping_table
            */
            let window_len = 1_usize << (var_index - 1);

            let inner_transform = |i: usize| {
                let first = *chunk.get(i).unwrap_or(&F::zero());
                let second = *chunk.get(i + window_len).unwrap_or(&F::zero());

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
            .evals
            .par_chunks(chunk_size)
            .map(outer_transform)
            .flatten()
            .collect();

        #[cfg(not(feature = "parallel"))]
        let evals: Vec<F> = self
            .evals
            .chunks(chunk_size)
            .map(outer_transform)
            .flatten()
            .collect();

        // --- Note that MLE is destructively modified into the new bookkeeping table here ---
        self.evals = evals;
        self.num_vars -= 1;
    }

    /// Optimized version of `fix_variable_at_index` for `var_index == 0`.
    /// # Panics
    /// If `self.num_vars == 0`.
    fn fix_variable(&mut self, point: F) {
        assert!(self.num_vars > 0);

        let transform = |chunk: &[F]| {
            let zero = F::zero();
            let first = chunk[0];
            let second = chunk.get(1).unwrap_or(&zero);

            // (1 - r) * V(i) + r * V(i + 1)
            first + (*second - first) * point
        };

        // --- So this goes through and applies the formula from [Tha13], bottom ---
        // --- of page 23 ---
        #[cfg(feature = "parallel")]
        let new = self.evals.par_chunks(2).map(transform);

        #[cfg(not(feature = "parallel"))]
        let new = self.evals.chunks(2).map(transform);

        // --- Note that MLE is destructively modified into the new bookkeeping table here ---
        self.evals = new.collect();
        self.num_vars -= 1;
    }

    // --------  Helper Functions --------

    /// Returns `true` when `n > 0` is a power of two.
    fn is_power_of_two(n: usize) -> bool {
        n > 0 && n & (n - 1) == 0
    }

    /// Mirrors the `num_bits` LSBs of `value`.
    /// # Example
    /// ```
    ///     use remainder::mle::evals::Evaluations;
    ///     use remainder_shared_types::Fr;
    ///
    ///     assert_eq!(Evaluations::<Fr>::mirror_bits(4, 0b1110), 0b0111);
    ///     assert_eq!(Evaluations::<Fr>::mirror_bits(3, 0b1110), 0b1011);
    ///     assert_eq!(Evaluations::<Fr>::mirror_bits(2, 0b1110), 0b1101);
    ///     assert_eq!(Evaluations::<Fr>::mirror_bits(1, 0b1110), 0b1110);
    ///     assert_eq!(Evaluations::<Fr>::mirror_bits(0, 0b1110), 0b1110);
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

    /// Sorts the elements of `values` by their 0-based index transformed by
    /// mirroring the `num_bits` LSBs. This operation effectively flips the
    /// "endianess" of the index ordering. If `values.len() < 2^num_bits`, the
    /// missing values are assumed to be zeros. The resulting vector is always
    /// of size `2^num_bits`.
    /// # Example
    /// ```
    ///     assert_eq!(flip_endianess(2, &[ 1, 2, 3, 4 ]), vec![ 1, 3, 2, 4 ]);
    ///     assert_eq!(flip_endianess(2, &[ 1, 2 ]), vec![ 1, 0, 2, 0 ]);
    /// ```
    /// TODO(Makis): Benchmark and provide alternative implementations.
    fn flip_endianess(num_bits: usize, values: &[F]) -> Vec<F> {
        let num_evals = values.len();

        cfg_into_iter!(0..(1 << num_bits))
            .map(|idx| {
                let mirrored_idx = Self::mirror_bits(num_bits, idx);
                if mirrored_idx >= num_evals {
                    F::zero()
                } else {
                    values[mirrored_idx]
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use quickcheck::{Arbitrary, TestResult};
    use remainder_shared_types::{halo2curves::group::ff::Field, Fr};

    use super::*;

    // Quickcheck wrapper for a Field Fr.
    #[derive(Debug, Clone, PartialEq)]
    struct Qfr(Fr);

    impl Arbitrary for Qfr {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            // Qfr(Fr::random(g))
            Qfr(Fr::from_raw([
                u64::arbitrary(g),
                u64::arbitrary(g),
                u64::arbitrary(g),
                u64::arbitrary(g),
            ]))
        }
    }

    impl Into<Fr> for Qfr {
        fn into(self) -> Fr {
            self.0
        }
    }

    #[test]
    fn is_power_of_two_or_zero_test() {
        assert!(!Evaluations::<Fr>::is_power_of_two(0));
        assert!(Evaluations::<Fr>::is_power_of_two(1));
        assert!(Evaluations::<Fr>::is_power_of_two(2));
        assert!(!Evaluations::<Fr>::is_power_of_two(3));
        assert!(Evaluations::<Fr>::is_power_of_two(4));
        assert!(!Evaluations::<Fr>::is_power_of_two(5));
        assert!(!Evaluations::<Fr>::is_power_of_two(6));
        assert!(!Evaluations::<Fr>::is_power_of_two(7));
        assert!(Evaluations::<Fr>::is_power_of_two(8));
        assert!(!Evaluations::<Fr>::is_power_of_two(10));
    }

    #[test]
    #[should_panic]
    fn evals_new_empty() {
        let _f: Evaluations<Fr> = Evaluations::new(0, vec![]);
    }

    #[test]
    fn evals_new_1_var() {
        let _f = Evaluations::new(0, vec![Fr::one()]);
    }

    #[test]
    fn evals_new_2_vars() {
        let _f = Evaluations::new(2, vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()]);
    }

    #[test]
    fn evals_new_from_big_endian_2_vars() {
        let evals: Vec<Fr> = [1, 2, 3, 4].into_iter().map(|v| Fr::from(v)).collect();
        let expected_evals: Vec<Fr> = [1, 3, 2, 4].into_iter().map(|v| Fr::from(v)).collect();

        let f = Evaluations::new_from_big_endian(2, &evals);

        assert_eq!(f.evals, expected_evals);
    }

    #[test]
    fn test_mirror_bits() {
        assert_eq!(Evaluations::<Fr>::mirror_bits(4, 0b1110), 0b0111);
        assert_eq!(Evaluations::<Fr>::mirror_bits(3, 0b1110), 0b1011);
        assert_eq!(Evaluations::<Fr>::mirror_bits(2, 0b1110), 0b1101);
        assert_eq!(Evaluations::<Fr>::mirror_bits(1, 0b1110), 0b1110);
        assert_eq!(Evaluations::<Fr>::mirror_bits(0, 0b1110), 0b1110);
    }

    /// Property: mirror_bits(n, mirror_bits(n, value)) == value.
    #[quickcheck]
    fn test_mirror_bit_prop(num_bits: usize, value: usize) -> TestResult {
        if num_bits > std::mem::size_of::<usize>() * 8 {
            return TestResult::discard();
        }

        TestResult::from_bool(
            value
                == Evaluations::<Fr>::mirror_bits(
                    num_bits,
                    Evaluations::<Fr>::mirror_bits(num_bits, value),
                ),
        )
    }

    /// Extends `vals` to length `2^n` by appending zeros if necessary.
    fn pad_with_zeros(n: usize, vals: &[Fr]) -> Vec<Fr> {
        debug_assert!(vals.len() <= (1 << n));
        let mut vals: Vec<Fr> = vals.to_vec();
        vals.resize(1 << n, Fr::zero());
        vals
    }

    /// Property: flip_endianess(n, flip_endianess(n, vals)) == pad_with_zeros(vals).
    #[quickcheck]
    fn flip_endianess_cancellation_property(vals: Vec<Qfr>) -> TestResult {
        if vals.len() == 0 {
            return TestResult::discard();
        }
        let n = log2(vals.len()) as usize;

        let vals: Vec<Fr> = vals.into_iter().map(|x| x.0).collect();

        let padded_vals = pad_with_zeros(n, &vals);

        TestResult::from_bool(
            Evaluations::<Fr>::flip_endianess(n, &Evaluations::<Fr>::flip_endianess(n, &vals))
                == padded_vals,
        )
    }

    #[test]
    fn test_flip_endianess_0() {
        let input = [Fr::from(0)];
        let expected_output = vec![Fr::from(0)];

        assert_eq!(
            Evaluations::<Fr>::flip_endianess(0, &input),
            expected_output,
        );
    }

    #[test]
    fn test_flip_endianess_1() {
        let input = [Fr::from(0), Fr::from(1)];
        let expected_output = vec![Fr::from(0), Fr::from(1)];

        assert_eq!(
            Evaluations::<Fr>::flip_endianess(1, &input),
            expected_output,
        );
    }

    #[test]
    fn test_flip_endianess_2() {
        let input: Vec<Fr> = [0, 1, 2, 3].into_iter().map(Fr::from).collect();
        let expected_output: Vec<Fr> = [0, 2, 1, 3].into_iter().map(Fr::from).collect();

        assert_eq!(
            Evaluations::<Fr>::flip_endianess(2, &input),
            expected_output
        );
    }

    #[test]
    fn test_flip_endianess_3() {
        let input: Vec<Fr> = [0, 1, 2, 3, 4, 5, 6, 7].into_iter().map(Fr::from).collect();
        let expected_output: Vec<Fr> = [0, 4, 2, 6, 1, 5, 3, 7].into_iter().map(Fr::from).collect();

        assert_eq!(
            Evaluations::<Fr>::flip_endianess(3, &input),
            expected_output,
        );
    }

    /// Helper function for testing purposes.
    /// Evaluates the MLE `\tilde{f}` of `f` at a point `point \in F^n`
    /// using the definition of an MLE.
    /// # Complexity
    /// `O(n * 2^n)`
    fn evaluate_mle_at_point(f: &Evaluations<Fr>, point: Vec<Fr>) -> Fr {
        assert_eq!(f.num_vars, point.len());
        let n = f.num_vars;

        (*f).evals
            .clone()
            .into_iter()
            .enumerate()
            .fold(Fr::zero(), |acc, (idx, v)| {
                let beta = (0..n).into_iter().fold(Fr::one(), |acc, i| {
                    let bit_i = idx & (1 << i);
                    if bit_i > 0 {
                        acc * point[i]
                    } else {
                        acc * (Fr::one() - point[i])
                    }
                });
                acc + v * beta
            })
    }

    #[quickcheck]
    fn test_fix_variable_evaluation(mut evals: Vec<Qfr>, mut point: Vec<Qfr>) -> TestResult {
        if evals.is_empty() {
            return TestResult::discard();
        }

        let n = log2(evals.len()) as usize;

        // Discard or shrink `point` as necessary
        if point.len() < n {
            return TestResult::discard();
        }
        point.resize(n, Qfr(Fr::zero()));

        // Unwrap `evals` and `point`.
        let evals: Vec<Fr> = evals.into_iter().map(|x| x.0).collect();
        let point: Vec<Fr> = point.into_iter().map(|x| x.0).collect();

        let mut f1 = Evaluations::<Fr>::new(n, evals.clone());
        let f2 = Evaluations::<Fr>::new(n, evals);

        for i in 0..n {
            f1.fix_variable(point[i]);
        }
        assert!(f1.evals.len() == 1);
        assert_eq!(f1.num_vars, 0);

        let res1 = f1.evals[0];
        let res2 = evaluate_mle_at_point(&f2, point);

        TestResult::from_bool(res1 == res2)
    }

    /// Property: f.fix_variable(r) == f.fix_variable_at_index(0, r)
    #[quickcheck]
    fn fix_variable_at_index_equivalence(evals: Vec<Qfr>, r: Qfr) -> TestResult {
        if evals.len() <= 1 {
            return TestResult::discard();
        }
        let n = log2(evals.len()) as usize;
        debug_assert!(n >= 1);

        // Unwrap `evals` and `r`
        let evals: Vec<Fr> = evals.into_iter().map(|x| x.0).collect();
        let r = r.0;

        let mut f1 = Evaluations::new(n, evals.clone());
        let mut f2 = Evaluations::new(n, evals);

        f1.fix_variable(r);
        f2.fix_variable_at_index(0, r);

        TestResult::from_bool(f1 == f2)
    }

    #[test]
    fn evaluate_mle_at_point_2_vars() {
        // f(x, y) = 5(1 - x)(1-y) + 2x(1-y) + (1-x)y + 3xy
        let input: Vec<Fr> = [5, 2, 1, 3].into_iter().map(Fr::from).collect();
        let f = Evaluations::<Fr>::new(2, input);

        // Ensure f(2, 3) = 17.
        assert_eq!(
            evaluate_mle_at_point(&f, vec![Fr::from(2), Fr::from(3)]),
            Fr::from(17)
        );
    }

    #[test]
    fn fix_variable_2_vars() {
        // f(x, y) = 5(1 - x)(1-y) + 2x(1-y) + (1-x)y + 3xy
        let input: Vec<Fr> = [5, 2, 1, 3].into_iter().map(Fr::from).collect();
        let mut f = Evaluations::<Fr>::new(2, input.clone());

        // Fix 1st variable to 2:
        // f(2, y) = ... = -(1-y) + 5y
        f.fix_variable(Fr::from(2));
        let expected_output: Vec<Fr> = vec![Fr::from(1).neg(), Fr::from(5)];
        assert_eq!(f.evals, expected_output);

        // Now fix y to 3.
        // f(2, 3) = ... = 17.
        f.fix_variable(Fr::from(3));
        let expected_output: Vec<Fr> = vec![Fr::from(17)];
        assert_eq!(f.evals, expected_output);
    }

    #[test]
    fn fix_variable_at_index_two_vars_fix_first() {
        // f(x, y) = 5(1 - x)(1-y) + 2x(1-y) + (1-x)y + 3xy
        let input: Vec<Fr> = [5, 2, 1, 3].into_iter().map(Fr::from).collect();
        let mut f = Evaluations::<Fr>::new(2, input);

        // Fix 1st variable to 2.
        f.fix_variable_at_index(0, Fr::from(2));
        let expected_output: Vec<Fr> = vec![Fr::from(1).neg(), Fr::from(5)];

        assert_eq!(f.evals, expected_output);
    }

    #[test]
    fn fix_variable_at_index_two_vars_fix_second() {
        // f(x, y) = 5(1 - x)(1-y) + 2x(1-y) + (1-x)y + 3xy
        let input: Vec<Fr> = [5, 2, 1, 3].into_iter().map(Fr::from).collect();
        let mut f = Evaluations::<Fr>::new(2, input);

        // Fix 2nd variable to 2.
        // f(x, 2) = ... = -3(1-x) + 4x
        f.fix_variable_at_index(1, Fr::from(2));
        let expected_output: Vec<Fr> = vec![Fr::from(3).neg(), Fr::from(4)];

        assert_eq!(f.evals, expected_output);
    }

    #[test]
    fn fix_variable_at_index_3_vars_fix_first() {
        // f(x, y, z) = 2x(1-y)(1-z) + 2xy(1-z) + 3x(1-y)z + (1-x)yz + 4xyz
        let evals = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mut f = Evaluations::<Fr>::new(3, evals);

        // Fix x = 3:
        // f(3, y, z) = ... = 6(1-y)(1-z) + 6y(1-z) + 9(1-y)z + 10yz.
        f.fix_variable_at_index(0, Fr::from(3));
        let expected_output = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];

        assert_eq!(f.evals, expected_output);
    }

    #[test]
    fn fix_variable_at_index_3_vars_fix_second() {
        // f(x, y, z) = 2x(1-y)(1-z) + 2xy(1-z) + 3x(1-y)z + (1-x)yz + 4xyz
        let evals = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mut f = Evaluations::<Fr>::new(3, evals);

        // Fix y = 4:
        // f(x, 4, z) = ... = 2x(1-z) + 4(1-x)z + 7xz.
        f.fix_variable_at_index(1, Fr::from(4));
        let expected_output = vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(7)];

        assert_eq!(f.evals, expected_output);
    }

    #[test]
    fn fix_variable_at_index_3_vars_fix_third() {
        // f(x, y, z) = 2x(1-y)(1-z) + 2xy(1-z) + 3x(1-y)z + (1-x)yz + 4xyz
        let evals = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mut f = Evaluations::<Fr>::new(3, evals);

        // Fix z = 5:
        // f(x, y, 5) = 7x(1-y) + 5(1-x)y + 12xy.
        f.fix_variable_at_index(2, Fr::from(5));
        let expected_output = vec![Fr::from(0), Fr::from(7), Fr::from(5), Fr::from(12)];

        assert_eq!(f.evals, expected_output);
    }
}
