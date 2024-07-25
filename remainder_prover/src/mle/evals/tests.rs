use ark_std::log2;
use quickcheck::{Arbitrary, TestResult};
use remainder_shared_types::Fr;

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

impl From<Qfr> for Fr {
    fn from(val: Qfr) -> Self {
        val.0
    }
}

#[test]
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
    let evals: Vec<Fr> = [1, 2, 3, 4].into_iter().map(Fr::from).collect();
    let expected_evals: Vec<Fr> = [1, 3, 2, 4].into_iter().map(Fr::from).collect();

    let f = Evaluations::new_from_big_endian(2, &evals);

    assert_eq!(f.evals, expected_evals);
}

#[test]
fn test_mirror_bits() {
    assert_eq!(mirror_bits(4, 0b1110), 0b0111);
    assert_eq!(mirror_bits(3, 0b1110), 0b1011);
    assert_eq!(mirror_bits(2, 0b1110), 0b1101);
    assert_eq!(mirror_bits(1, 0b1110), 0b1110);
    assert_eq!(mirror_bits(0, 0b1110), 0b1110);
}

/// Property: mirror_bits(n, mirror_bits(n, value)) == value.
#[quickcheck]
fn test_mirror_bit_prop(num_bits: usize, value: usize) -> TestResult {
    if num_bits > std::mem::size_of::<usize>() * 8 {
        return TestResult::discard();
    }

    TestResult::from_bool(value == mirror_bits(num_bits, mirror_bits(num_bits, value)))
}

/// Extends `vals` to length `2^n` by appending zeros if necessary.
fn pad_with_zeros(n: usize, vals: &[Fr]) -> Vec<Fr> {
    debug_assert!(vals.len() <= (1 << n));
    let mut vals: Vec<Fr> = vals.to_vec();
    vals.resize(1 << n, Fr::zero());
    vals
}

/// Property: flip_endianess(n, flip_endianess(n, vals)) ==
/// pad_with_zeros(vals).
#[quickcheck]
fn flip_endianess_cancellation_property(vals: Vec<Qfr>) -> TestResult {
    if vals.is_empty() {
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

#[test]
fn test_eval_iterator_1() {
    let evals: Vec<Fr> = [0, 1, 2, 3, 4, 5, 6, 7].into_iter().map(Fr::from).collect();
    let f = Evaluations::<Fr>::new(3, evals);

    let mut it = f.project(0);

    assert_eq!(it.next().unwrap(), (Fr::from(0), Fr::from(1)));
    assert_eq!(it.next().unwrap(), (Fr::from(2), Fr::from(3)));
    assert_eq!(it.next().unwrap(), (Fr::from(4), Fr::from(5)));
    assert_eq!(it.next().unwrap(), (Fr::from(6), Fr::from(7)));
    assert_eq!(it.next(), None);
}

#[test]
fn test_eval_iterator_2() {
    let evals: Vec<Fr> = [0, 1, 2, 3, 4, 5, 6, 7].into_iter().map(Fr::from).collect();
    let f = Evaluations::<Fr>::new(3, evals);

    let mut it = f.project(1);

    assert_eq!(it.next().unwrap(), (Fr::from(0), Fr::from(2)));
    assert_eq!(it.next().unwrap(), (Fr::from(1), Fr::from(3)));
    assert_eq!(it.next().unwrap(), (Fr::from(4), Fr::from(6)));
    assert_eq!(it.next().unwrap(), (Fr::from(5), Fr::from(7)));
    assert_eq!(it.next(), None);
}

#[test]
fn test_eval_iterator_3() {
    let evals: Vec<Fr> = [0, 1, 2, 3, 4, 5, 6, 7].into_iter().map(Fr::from).collect();
    let f = Evaluations::<Fr>::new(3, evals);

    let mut it = f.project(2);

    assert_eq!(it.next().unwrap(), (Fr::from(0), Fr::from(4)));
    assert_eq!(it.next().unwrap(), (Fr::from(1), Fr::from(5)));
    assert_eq!(it.next().unwrap(), (Fr::from(2), Fr::from(6)));
    assert_eq!(it.next().unwrap(), (Fr::from(3), Fr::from(7)));
    assert_eq!(it.next(), None);
}

#[test]
fn test_equiv_repr() {
    let evals1: Vec<Fr> = [1, 2, 3].into_iter().map(Fr::from).collect();
    let evals2: Vec<Fr> = [1, 2, 3, 0, 0].into_iter().map(Fr::from).collect();

    assert!(Evaluations::equiv_repr(&evals1, &evals1));
    assert!(Evaluations::equiv_repr(&evals2, &evals2));
    assert!(Evaluations::equiv_repr(&evals1, &evals2));
    assert!(Evaluations::equiv_repr(&evals2, &evals1));
}

#[quickcheck]
fn test_fix_variable_evaluation(evals: Vec<Qfr>, mut point: Vec<Qfr>) -> TestResult {
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

    let f = Evaluations::<Fr>::new(n, evals);

    let mut mle1 = MultilinearExtension::<Fr>::new_from_evals(f.clone());
    let mle2 = MultilinearExtension::<Fr>::new_from_evals(f);

    for i in 0..n {
        mle1.fix_variable(point[i]);
    }
    assert_eq!(mle1.f.evals.len(), 1);
    assert_eq!(mle1.num_vars(), 0);

    let res1 = mle1.f[0];
    let res2 = mle2.evaluate_at_point(&point);

    TestResult::from_bool(res1 == res2)
}

/// Property: mle.fix_variable(r) == mle.fix_variable_at_index(0, r)
#[quickcheck]
fn fix_variable_at_index_equivalence(evals: Vec<Qfr>, r: Qfr) -> TestResult {
    if evals.len() <= 1 {
        return TestResult::discard();
    }
    let n = log2(evals.len()) as usize;
    assert!(n >= 1);

    // Unwrap `evals` and `r`
    let evals: Vec<Fr> = evals.into_iter().map(|x| x.0).collect();
    let r = r.0;

    let f = Evaluations::new(n, evals);

    let mut f_tilde_1 = MultilinearExtension::new_from_evals(f.clone());
    let mut f_tilde_2 = MultilinearExtension::new_from_evals(f);

    f_tilde_1.fix_variable(r);
    f_tilde_2.fix_variable_at_index(0, r);

    TestResult::from_bool(
        Evaluations::<Fr>::equiv_repr(&f_tilde_1.f.evals, &f_tilde_2.f.evals)
            && f_tilde_1.num_vars() == f_tilde_2.num_vars(),
    )
}

#[test]
fn evaluate_mle_at_point_2_vars() {
    // f(x, y) = 5(1 - x)(1-y) + 2x(1-y) + (1-x)y + 3xy
    let input: Vec<Fr> = [5, 2, 1, 3].into_iter().map(Fr::from).collect();
    let f = Evaluations::new(2, input);
    let f_tilde = MultilinearExtension::new_from_evals(f);

    // Ensure f(2, 3) = 17.
    assert_eq!(
        f_tilde.evaluate_at_point(&[Fr::from(2), Fr::from(3)]),
        Fr::from(17)
    );
}

#[test]
fn fix_variable_2_vars() {
    // f(x, y) = 5(1 - x)(1-y) + 2x(1-y) + (1-x)y + 3xy
    let input: Vec<Fr> = [5, 2, 1, 3].into_iter().map(Fr::from).collect();
    let f = Evaluations::new(2, input);
    let mut f_tilde = MultilinearExtension::new_from_evals(f);

    // Fix 1st variable to 2:
    // f(2, y) = ... = -(1-y) + 5y
    f_tilde.fix_variable(Fr::from(2));
    let expected_output: Vec<Fr> = vec![Fr::from(1).neg(), Fr::from(5)];
    assert_eq!(f_tilde.f.evals, expected_output);

    // Now fix y to 3.
    // f(2, 3) = ... = 17.
    f_tilde.fix_variable(Fr::from(3));
    let expected_output: Vec<Fr> = vec![Fr::from(17)];
    assert_eq!(f_tilde.f.evals, expected_output);
}

#[test]
fn fix_variable_at_index_two_vars_fix_first() {
    // f(x, y) = 5(1 - x)(1-y) + 2x(1-y) + (1-x)y + 3xy
    let input: Vec<Fr> = [5, 2, 1, 3].into_iter().map(Fr::from).collect();
    let f = Evaluations::new(2, input);
    let mut f_tilde = MultilinearExtension::new_from_evals(f);

    // Fix 1st variable to 2.
    f_tilde.fix_variable_at_index(0, Fr::from(2));
    let expected_output: Vec<Fr> = vec![Fr::from(1).neg(), Fr::from(5)];

    assert_eq!(f_tilde.f.evals, expected_output);
}

#[test]
fn fix_variable_at_index_two_vars_fix_second() {
    // f(x, y) = 5(1 - x)(1-y) + 2x(1-y) + (1-x)y + 3xy
    let input: Vec<Fr> = [5, 2, 1, 3].into_iter().map(Fr::from).collect();
    let f = Evaluations::new(2, input);
    let mut f_tilde = MultilinearExtension::new_from_evals(f);

    // Fix 2nd variable to 2.
    // f(x, 2) = ... = -3(1-x) + 4x
    f_tilde.fix_variable_at_index(1, Fr::from(2));
    let expected_output: Vec<Fr> = vec![Fr::from(3).neg(), Fr::from(4)];

    assert_eq!(f_tilde.f.evals, expected_output);
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
    let f = Evaluations::new(3, evals);
    let mut f_tilde = MultilinearExtension::new_from_evals(f);

    // Fix x = 3:
    // f(3, y, z) = ... = 6(1-y)(1-z) + 6y(1-z) + 9(1-y)z + 10yz.
    f_tilde.fix_variable_at_index(0, Fr::from(3));
    let expected_output = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];

    assert_eq!(f_tilde.f.evals, expected_output);
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
    let f = Evaluations::new(3, evals);
    let mut f_tilde = MultilinearExtension::new_from_evals(f);

    // Fix y = 4:
    // f(x, 4, z) = ... = 2x(1-z) + 4(1-x)z + 7xz.
    f_tilde.fix_variable_at_index(1, Fr::from(4));
    let expected_output = vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(7)];

    assert_eq!(f_tilde.f.evals, expected_output);
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
    let f = Evaluations::new(3, evals);
    let mut f_tilde = MultilinearExtension::new_from_evals(f);

    // Fix z = 5:
    // f(x, y, 5) = 7x(1-y) + 5(1-x)y + 12xy.
    f_tilde.fix_variable_at_index(2, Fr::from(5));
    let expected_output = vec![Fr::from(0), Fr::from(7), Fr::from(5), Fr::from(12)];

    assert_eq!(f_tilde.f.evals, expected_output);
}

#[test]
fn test_interlace_mles() {
    let mles = vec![
        vec![Fr::from(0), Fr::from(4), Fr::from(8), Fr::from(2)],
        vec![Fr::from(1), Fr::from(5), Fr::from(9), Fr::from(3)],
        vec![Fr::from(2), Fr::from(6), Fr::from(0), Fr::from(4)],
        vec![Fr::from(3), Fr::from(7), Fr::from(1), Fr::from(5)],
    ]
    .into_iter()
    .map(|data| MultilinearExtension::new(data))
    .collect::<Vec<_>>();

    let interlaced_mle = MultilinearExtension::interlace_mles(mles);

    assert_eq!(
        *interlaced_mle.get_evals_vector(),
        vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
            Fr::from(7),
            Fr::from(8),
            Fr::from(9),
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5)
        ]
    );
}
