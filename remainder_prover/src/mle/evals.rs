use ark_std::{cfg_into_iter, log2};
use itertools::Itertools;
use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use remainder_shared_types::FieldExt;

/// Stores a boolean function `f: {0, 1}^n -> F` represented as a list of all
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
    /// # Invariant
    /// `evals.len()` is a power of two.
    /// # Example
    /// The evaluations of a 2-dimensional function are stored as follows:
    /// ```
    ///   [ f(0, 0), f(1, 0), f(0, 1), f(1, 1) ]
    /// ```
    evals: Vec<F>,

    /// Number of inputs to `f`.
    /// Invariant: `evals.len() == 2^num_vars`.
    num_vars: usize,
}

impl<F: FieldExt> Evaluations<F> {
    /// Build an evaluation representation from a vector of evaluations ordered
    /// by MSB first (see documentation comment for `evals` for explanation).
    /// # Example
    /// ```
    ///   Evaluations::new(vec![ f(0, 0), f(1, 0), f(0, 1), f(1, 1) ]);
    /// ```
    /// # Panics
    /// If the length of the input is not a power of two.
    pub fn new(evals: Vec<F>) -> Self {
        let num_evals = evals.len();

        assert!(Self::is_power_of_two(num_evals));
        let num_vars = log2(num_evals) as usize;

        Evaluations::<F> { evals, num_vars }
    }

    /// Builds an evaluation representation from a given list of evaluations
    /// that is ordered by LSB first (or big-endian).
    /// # Example
    /// For a 2-dimensional function `f: {0, 1}^2 -> F`:
    /// ```
    ///   Evaluations::new_from_big_endian(&[ f(0, 0), f(0, 1), f(1, 0), f(1, 1) ]);
    /// ```
    /// # Panics
    /// If the length of the input is not a power of two.
    pub fn new_from_big_endian(big_endian_evals: &[F]) -> Self {
        let num_evals = big_endian_evals.len();

        assert!(Self::is_power_of_two(num_evals));
        let num_vars = log2(num_evals) as usize;

        Self {
            evals: Self::big_endian_to_little_endian(big_endian_evals),
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

        // TODO(Makis): Explain how this function works.

        let chunk_size: usize = 1 << var_index;

        let outer_transform = |chunk: &[F]| {
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
        };

        // --- So this goes through and applies the formula from [Tha13], bottom ---
        // --- of page 23 ---
        #[cfg(feature = "parallel")]
        let new = self
            .evals
            .par_chunks(chunk_size)
            .map(outer_transform)
            .flatten();

        #[cfg(not(feature = "parallel"))]
        let new = self.evals.chunks(chunk_size).map(outer_transform).flatten();

        // --- Note that MLE is destructively modified into the new bookkeeping table here ---
        self.evals = new.collect();
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

    /// Mirrors the `num_bits` least significant bits of `value`.
    /// # Example
    /// ```
    ///     use remainder::mle::evals::Evaluations;
    ///     use remainder_shared_types::Fr;
    ///
    ///     assert_eq!(Evaluations::<Fr>::mirror_bits(0b1110, 4), 0b0111);
    ///     assert_eq!(Evaluations::<Fr>::mirror_bits(0b1110, 3), 0b1011);
    ///     assert_eq!(Evaluations::<Fr>::mirror_bits(0b1110, 2), 0b1101);
    ///     assert_eq!(Evaluations::<Fr>::mirror_bits(0b1110, 1), 0b1110);
    ///     assert_eq!(Evaluations::<Fr>::mirror_bits(0b1110, 0), 0b1110);
    /// ```
    /// TODO(Makis): Fix Doctest compile error due to the method being private.
    fn mirror_bits(mut value: usize, num_bits: usize) -> usize {
        let mut result: usize = 0;

        for _ in 0..num_bits {
            result = (result << 1) | (value & 1);
            value >>= 1;
        }

        // Add back the remaining bits.
        result | (value << num_bits)
    }

    /// Returns the elements of `big_endian_evals` ordered by their
    /// "little-endian" index.
    /// # Example
    /// ```
    /// big_endian_to_little_endian(&[ a_0b00, a_0b01, a_0b10, a_0b11 ])
    ///     == vec![a_0b00, a_0b10, a_0b01, a_0b11 ]
    /// ```
    ///
    /// TODO(Makis): Benchmark and provide alternative implementations.
    /// TODO(Makis): Change function name. The terms "big-endian" and
    /// "little-endian" traditionally refer to byte ordering, not bit ordering,
    /// and thus they can be confusing.
    fn big_endian_to_little_endian(big_endian_evals: &[F]) -> Vec<F> {
        let num_evals = big_endian_evals.len();
        let n = log2(num_evals) as usize;

        cfg_into_iter!(0..num_evals)
            .map(|idx| {
                let mirrored_index = Self::mirror_bits(idx, n);
                big_endian_evals[mirrored_index]
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use quickcheck::{Arbitrary, TestResult};
    use remainder_shared_types::Fr;

    use super::*;

    // Quickcheck wrapper for a Field Fr.
    #[derive(Debug, Clone, PartialEq)]
    struct Qfr(Fr);

    impl Arbitrary for Qfr {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
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
        let _f: Evaluations<Fr> = Evaluations::new(vec![]);
    }

    #[test]
    fn evals_new_one_element() {
        let _f = Evaluations::new(vec![Fr::one()]);
    }

    #[test]
    fn evals_new_four_elements() {
        let _f = Evaluations::new(vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()]);
    }

    #[test]
    fn evals_new_from_big_endian_four_elements() {
        let _f = Evaluations::new_from_big_endian(&[Fr::one(), Fr::one(), Fr::one(), Fr::one()]);
    }

    #[test]
    fn test_mirror_bits() {
        assert_eq!(Evaluations::<Fr>::mirror_bits(0b1110, 4), 0b0111);
        assert_eq!(Evaluations::<Fr>::mirror_bits(0b1110, 3), 0b1011);
        assert_eq!(Evaluations::<Fr>::mirror_bits(0b1110, 2), 0b1101);
        assert_eq!(Evaluations::<Fr>::mirror_bits(0b1110, 1), 0b1110);
        assert_eq!(Evaluations::<Fr>::mirror_bits(0b1110, 0), 0b1110);
    }

    #[quickcheck]
    fn test_mirror_bit_prop((value, num_bits): (usize, usize)) -> TestResult {
        if num_bits > std::mem::size_of::<usize>() * 8 {
            return TestResult::discard();
        }

        TestResult::from_bool(
            value
                == Evaluations::<Fr>::mirror_bits(
                    Evaluations::<Fr>::mirror_bits(value, num_bits),
                    num_bits,
                ),
        )
    }

    #[quickcheck]
    fn big_endian_to_little_endian_cancelation_property(mut evals: Vec<Qfr>) -> TestResult {
        // Shrink `evals` to the nearest power of two.
        if evals.len() < 2 {
            return TestResult::discard();
        }
        let n = log2(evals.len());
        let num_evals = 1 << n;
        evals.resize(num_evals, Qfr(Fr::zero()));

        let evals: Vec<Fr> = evals.into_iter().map(|x| x.0).collect();

        TestResult::from_bool(
            Evaluations::<Fr>::big_endian_to_little_endian(
                &Evaluations::<Fr>::big_endian_to_little_endian(&evals),
            ) == evals,
        )
    }

    #[test]
    fn test_big_endian_to_little_endian() {
        let zero = Fr::zero();
        let one = Fr::one();

        let input_1 = [zero];
        let expected_output_1 = vec![zero];

        assert_eq!(
            Evaluations::<Fr>::big_endian_to_little_endian(&input_1),
            expected_output_1,
        );

        let input_2 = [zero, one];
        let expected_output_2 = vec![zero, one];

        assert_eq!(
            Evaluations::<Fr>::big_endian_to_little_endian(&input_2),
            expected_output_2,
        );

        let input_4: Vec<Fr> = [0, 1, 2, 3].into_iter().map(Fr::from).collect();
        let expected_output_4: Vec<Fr> = [0, 2, 1, 3].into_iter().map(Fr::from).collect();

        assert_eq!(
            Evaluations::<Fr>::big_endian_to_little_endian(&input_4),
            expected_output_4
        );

        let input_8: Vec<Fr> = [0, 1, 2, 3, 4, 5, 6, 7].into_iter().map(Fr::from).collect();
        let expected_output_8: Vec<Fr> =
            [0, 4, 2, 6, 1, 5, 3, 7].into_iter().map(Fr::from).collect();

        assert_eq!(
            Evaluations::<Fr>::big_endian_to_little_endian(&input_8),
            expected_output_8,
        );
    }

    /// Helper function for testing purposes.
    /// Evaluates the MLE `\tilde{f}` of `f` at a point `point \in F^n`
    /// using the definition of an MLE.
    /// # Complexity
    /// `O(n * 2^n)`
    fn evaluate_MLE_at_point(f: &Evaluations<Fr>, point: Vec<Fr>) -> Fr {
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

    // ======== `fix_variable_at_index` tests ========

    #[quickcheck]
    fn test_fix_variable_evaluation((mut evals, mut point): (Vec<Qfr>, Vec<Qfr>)) -> TestResult {
        // Shrink evals to nearest power of two.
        if evals.len() < 2 {
            return TestResult::discard();
        }
        let n = (log2(evals.len()) - 1) as usize;
        let num_evals = 1 << n;
        evals.resize(num_evals, Qfr(Fr::zero()));
        assert!(Evaluations::<Fr>::is_power_of_two(evals.len()));

        // Shrink `point` if it contains at least `n` points.
        if point.len() < n {
            return TestResult::discard();
        }
        point.resize(n, Qfr(Fr::zero()));

        // Unwrap `evals` and `point`.
        let evals: Vec<Fr> = evals.into_iter().map(|x| x.0).collect();
        let point: Vec<Fr> = point.into_iter().map(|x| x.0).collect();

        let mut f1 = Evaluations::<Fr>::new(evals.clone());
        let f2 = Evaluations::<Fr>::new(evals);

        for i in 0..n {
            f1.fix_variable(point[i]);
        }
        assert_eq!(f1.evals.len(), 1);
        assert_eq!(f1.num_vars, 0);

        let res1 = f1.evals[0];
        let res2 = evaluate_MLE_at_point(&f2, point);

        TestResult::from_bool(res1 == res2)
    }

    /// Property:
    /// ```
    ///     f.fix_variable(r) == f.fix_variable_at_index(0, r)
    /// ```
    #[quickcheck]
    fn fix_variable_at_index_equivalence((mut evals, r): (Vec<Qfr>, Qfr)) -> TestResult {
        // Shrink evals to nearest power of two.
        if evals.len() < 4 {
            return TestResult::discard();
        }
        let n = (log2(evals.len()) - 1) as usize;
        let num_evals = 1 << n;
        evals.resize(num_evals, Qfr(Fr::zero()));
        assert!(Evaluations::<Fr>::is_power_of_two(evals.len()));

        // Unwrap `evals` and `r`
        let evals: Vec<Fr> = evals.into_iter().map(|x| x.0).collect();
        let r = r.0;

        let mut f1 = Evaluations::new(evals.clone());
        let mut f2 = Evaluations::new(evals);

        f1.fix_variable(r);
        f2.fix_variable_at_index(0, r);

        TestResult::from_bool(f1 == f2)
    }

    #[test]
    fn fix_variable_two_vars() {
        let input: Vec<Fr> = [5, 2, 1, 3].into_iter().map(Fr::from).collect();
        let mut f = Evaluations::<Fr>::new(input.clone());

        // Fix 1st variable to 2.
        f.fix_variable_at_index(0, Fr::from(2));
        let expected_output: Vec<Fr> = vec![Fr::from(1).neg(), Fr::from(5)];

        assert_eq!(f.evals, expected_output);

        f.fix_variable_at_index(0, Fr::from(3));
        let expected_output: Vec<Fr> = vec![Fr::from(17)];

        assert_eq!(f.evals, expected_output);

        assert_eq!(
            evaluate_MLE_at_point(
                &Evaluations::<Fr>::new(input),
                vec![Fr::from(2), Fr::from(3)]
            ),
            Fr::from(17)
        );
    }

    #[test]
    fn fix_variable_at_index_two_vars_fix_first() {
        let input: Vec<Fr> = [5, 2, 1, 3].into_iter().map(Fr::from).collect();
        let mut f = Evaluations::<Fr>::new(input);

        // Fix 1st variable to 2.
        f.fix_variable_at_index(0, Fr::from(2));
        let expected_output: Vec<Fr> = vec![Fr::from(1).neg(), Fr::from(5)];

        assert_eq!(f.evals, expected_output);
    }

    #[test]
    fn fix_variable_at_index_two_vars_fix_second() {
        let input: Vec<Fr> = [5, 2, 1, 3].into_iter().map(Fr::from).collect();
        let mut f = Evaluations::<Fr>::new(input);

        // Fix 2nd variable to 2.
        f.fix_variable_at_index(1, Fr::from(2));
        let expected_output: Vec<Fr> = vec![Fr::from(3).neg(), Fr::from(4)];

        assert_eq!(f.evals, expected_output);
    }

    #[test]
    ///test fixing variables in an mle with three variables
    fn fix_variable_at_index_three_vars_fix_first() {
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
        let expected_output = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];

        let mut f = Evaluations::<Fr>::new(evals);
        // Fix 1st variable to 3.
        f.fix_variable_at_index(0, Fr::from(3));

        assert_eq!(f.evals, expected_output);
    }

    #[test]
    ///test fixing variables in an mle with three variables
    fn fix_variable_at_index_three_vars_fix_second() {
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
        let expected_output = vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(7)];

        let mut f = Evaluations::<Fr>::new(evals);
        // Fix 2nd variable to 4.
        f.fix_variable_at_index(1, Fr::from(4));

        assert_eq!(f.evals, expected_output);
    }

    #[test]
    ///test fixing variables in an mle with three variables
    fn fix_variable_at_index_three_vars_fix_third() {
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
        let expected_output = vec![Fr::from(0), Fr::from(7), Fr::from(5), Fr::from(12)];

        let mut f = Evaluations::<Fr>::new(evals);
        // Fix 3rd variable to 5.
        f.fix_variable_at_index(2, Fr::from(5));

        assert_eq!(f.evals, expected_output);
    }
}
