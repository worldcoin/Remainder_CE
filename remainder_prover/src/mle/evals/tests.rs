use ark_std::log2;
use quickcheck::{Arbitrary, TestResult};
use remainder_shared_types::{halo2curves::ff::Field, Fr, HasByteRepresentation};

use super::*;

/// Wrapper for a Field element to be used for Quickcheck.
/// This is needed because Rust doesn't allow implementing the foreign trait
/// [Arbitrary] for the foreign type [Fr].
/// See
/// [https://stackoverflow.com/questions/25413201/how-do-i-implement-a-trait-i-dont-own-for-a-type-i-dont-own].
#[derive(Debug, Clone, PartialEq)]
pub struct Qfr(Fr);

impl Arbitrary for Qfr {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        // Generate an arbitrary byte-representation of a field element.
        // To ensure this is a valid representation, we zero-out the higher-order byte.
        // The resulting element is then upper bounded by
        // `0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff`
        // which is smaller that the `r` defining `Fr`.
        let bytes: Vec<u8> = [0; 31]
            .iter()
            .map(|_| u8::arbitrary(g))
            .chain([0_u8])
            .collect();
        Qfr(Fr::from_bytes_le(&bytes))

        // I think this is technically wrong although it never panicked.
        // Qfr(Fr::from_raw([
        //     u64::arbitrary(g),
        //     u64::arbitrary(g),
        //     u64::arbitrary(g),
        //     u64::arbitrary(g),
        // ]))
    }
}

impl From<Qfr> for Fr {
    fn from(val: Qfr) -> Self {
        val.0
    }
}

#[test]
fn test_bit_packed_vector_get_empty() {
    let data: Vec<Fr> = vec![];
    let bpv = BitPackedVector::new(&data);

    assert!(bpv.get(0).is_none());
    assert!(bpv.get(1).is_none());
}

#[test]
fn test_bit_packed_vector_get_constant() {
    let data: Vec<Fr> = vec![Fr::from(42), Fr::from(42), Fr::from(42)];
    let bpv = BitPackedVector::new(&data);

    assert_eq!(bpv.get(0).unwrap(), Fr::from(42));
    assert_eq!(bpv.get(1).unwrap(), Fr::from(42));
    assert_eq!(bpv.get(2).unwrap(), Fr::from(42));
    assert!(bpv.get(3).is_none());
}

#[test]
fn test_bit_packed_vector_get_small() {
    // Test bit-width: 2.

    let data: Vec<Fr> = [10, 11, 12, 13].into_iter().map(Fr::from).collect();
    let bpv = BitPackedVector::new(&data);

    assert_eq!(bpv.get(0).unwrap(), Fr::from(10));
    assert_eq!(bpv.get(1).unwrap(), Fr::from(11));
    assert_eq!(bpv.get(2).unwrap(), Fr::from(12));
    assert_eq!(bpv.get(3).unwrap(), Fr::from(13));
    assert!(bpv.get(4).is_none());
}

#[test]
fn test_bit_packed_vector_get_small_with_negative() {
    // Test bit-width: 2.
    let data = vec![Fr::from(0), Fr::from(1), Fr::from(1).neg(), Fr::from(2)];

    let bpv = BitPackedVector::new(&data);

    assert_eq!(bpv.get(0).unwrap(), data[0]);
    assert_eq!(bpv.get(1).unwrap(), data[1]);
    assert_eq!(bpv.get(2).unwrap(), data[2]);
    assert_eq!(bpv.get(3).unwrap(), data[3]);
    assert!(bpv.get(4).is_none());
}

#[test]
fn test_bit_packed_vector_get_large_1() {
    // Test bit-wdth: 7.

    let n: usize = 128; // = 2^7.
    let offset: u64 = 100;

    let data: Vec<Fr> = (0..n).map(|x| Fr::from(offset + x as u64)).collect();
    let bpv = BitPackedVector::new(&data);

    for i in 0..n {
        assert_eq!(bpv.get(i).unwrap(), Fr::from(100 + i as u64));
    }

    assert!(bpv.get(n).is_none());
}

#[test]
fn test_bit_packed_vector_get_large_2() {
    // Test bit-width = 8.

    let n: usize = 256; // = 2^8.
    let offset: u64 = 100;

    let data: Vec<Fr> = (0..n).map(|x| Fr::from(offset + x as u64)).collect();
    let bpv = BitPackedVector::new(&data);

    for i in 0..n {
        assert_eq!(bpv.get(i).unwrap(), Fr::from(100 + i as u64));
    }

    assert!(bpv.get(n).is_none());
}

#[test]
fn test_bit_packed_vector_get_large_3() {
    // Test bit-width: 100.

    let n = 100;
    let small_val = Fr::from(0);
    // 2^100 - 1
    let large_val = Fr::from_bytes_le(&[
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
    ]);
    let offset = Fr::from(1_234); // 0x4D2

    let data: Vec<Fr> = (0..n)
        .map(|i| offset + if i % 2 == 0 { small_val } else { large_val })
        .collect();
    let bpv = BitPackedVector::new(&data);

    for i in 0..n {
        assert_eq!(
            bpv.get(i).unwrap(),
            offset + if i % 2 == 0 { small_val } else { large_val }
        );
    }

    assert!(bpv.get(n).is_none());
}

#[quickcheck]
fn test_bit_packed_vector_get_extensive(data: Vec<Qfr>) -> TestResult {
    let data = data.iter().map(|e| e.0).collect::<Vec<_>>();

    let n = data.len();
    let bpv = BitPackedVector::new(&data);

    for i in 0..n {
        assert_eq!(bpv.get(i).unwrap(), data[i]);
    }

    assert!(bpv.get(n).is_none());

    TestResult::from_bool(true)
}

/// Tail-recursive implementation of the exponentiation function for field
/// elements performing `O(log(exp))` field element multiplications.
fn fr_pow(base: Fr, exp: u64) -> Fr {
    // helper(acc, base, exp) == acc * base^exp.
    fn helper(acc: Fr, base: Fr, exp: u64) -> Fr {
        if exp == 0 {
            acc
        } else if exp % 2 == 0 {
            helper(acc, base * base, exp / 2)
        } else {
            helper(acc * base, base * base, exp / 2)
        }
    }

    helper(Fr::ONE, base, exp)
}

#[test]
fn test_fr_pow() {
    assert_eq!(fr_pow(Fr::from(2), 0), Fr::ONE);
    assert_eq!(fr_pow(Fr::from(2), 1), Fr::from(2));
    assert_eq!(fr_pow(Fr::from(2), 5), Fr::from(32));
    assert_eq!(fr_pow(Fr::from(2), 6), Fr::from(64));
    assert_eq!(fr_pow(Fr::from(5), 2), Fr::from(25));
    assert_eq!(fr_pow(Fr::from(5), 3), Fr::from(125));
}

#[quickcheck]
fn test_bit_packed_vector_get_all_bits(offset: Qfr) -> bool {
    let offset = Fr::from(offset);
    for num_bits in 1..256 {
        let a = offset;
        let b = a + fr_pow(Fr::from(2), num_bits) - Fr::ONE;

        let num_elements = 10;
        let data: Vec<Fr> = [a, b].repeat(num_elements);
        let bpv = BitPackedVector::new(&data);

        for i in 0..num_elements {
            let x = bpv.get(i);
            if x.is_none() || x.unwrap() != data[i] {
                println!(
                    "Num bits: {num_bits}\nIdx: {i}\nExp: {:?}\nGot: {:?}",
                    data[i],
                    x.unwrap()
                );
                return false;
            }
        }
    }

    true
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
    let expected_evals =
        BitPackedVector::new(&[1, 3, 2, 4].into_iter().map(Fr::from).collect::<Vec<Fr>>());

    let f = Evaluations::new_from_big_endian(2, &evals);

    assert_eq!(f.evals, expected_evals);
}

#[test]
fn evals_first() {
    let f = Evaluations::new(2, vec![Fr::from(42), Fr::ZERO, Fr::ZERO, Fr::ZERO]);
    assert_eq!(f.first(), Fr::from(42));

    let f = Evaluations::<Fr>::new(2, vec![]);
    assert_eq!(f.first(), Fr::ZERO);

    let f = Evaluations::new(0, vec![Fr::from(42)]);
    assert_eq!(f.first(), Fr::from(42));
}

#[test]
fn evals_value_successful() {
    let f = Evaluations::new(0, vec![Fr::from(42)]);
    assert_eq!(f.value(), Fr::from(42));

    let f = Evaluations::<Fr>::new(0, vec![]);
    assert_eq!(f.value(), Fr::ZERO);
}

#[test]
#[should_panic]
fn evals_value_failing_1() {
    let f = Evaluations::new(2, vec![Fr::from(42), Fr::ZERO, Fr::ZERO, Fr::ZERO]);
    let _val = f.value();
}

#[test]
#[should_panic]
fn evals_value_failing_2() {
    let f = Evaluations::<Fr>::new(2, vec![]);
    let _val = f.value();
}

#[test]
fn evals_fully_bound() {
    let f = Evaluations::new(2, vec![Fr::from(42), Fr::ZERO, Fr::ZERO, Fr::ZERO]);
    assert!(!f.is_fully_bound());

    let f = Evaluations::<Fr>::new(2, vec![]);
    assert!(!f.is_fully_bound());

    let f = Evaluations::new(0, vec![Fr::from(42)]);
    assert!(f.is_fully_bound());

    let f = Evaluations::<Fr>::new(0, vec![]);
    assert!(f.is_fully_bound());
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
fn test_eval_projection_iterator_1() {
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
fn test_eval_projection_iterator_2() {
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
fn test_eval_projection_iterator_3() {
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
fn test_eval_iterator() {
    let evals: Vec<Fr> = [0, 1, 2, 3, 4, 5, 6, 7].into_iter().map(Fr::from).collect();
    let f = Evaluations::<Fr>::new(3, evals);

    let mut it = f.iter();

    assert_eq!(it.next().unwrap(), Fr::from(0));
    assert_eq!(it.next().unwrap(), Fr::from(1));
    assert_eq!(it.next().unwrap(), Fr::from(2));
    assert_eq!(it.next().unwrap(), Fr::from(3));
    assert_eq!(it.next().unwrap(), Fr::from(4));
    assert_eq!(it.next().unwrap(), Fr::from(5));
    assert_eq!(it.next().unwrap(), Fr::from(6));
    assert_eq!(it.next().unwrap(), Fr::from(7));
    assert_eq!(it.next(), None);
}

#[test]
fn test_equiv_repr() {
    let evals1 = BitPackedVector::new(&[1, 2, 3].into_iter().map(Fr::from).collect_vec());
    let evals2 = BitPackedVector::new(&[1, 2, 3, 0, 0].into_iter().map(Fr::from).collect_vec());

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

    let res1 = mle1.f.first();
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
    assert_eq!(f_tilde.f.evals.to_vec(), expected_output);

    // Now fix y to 3.
    // f(2, 3) = ... = 17.
    f_tilde.fix_variable(Fr::from(3));
    let expected_output: Vec<Fr> = vec![Fr::from(17)];
    assert_eq!(f_tilde.f.evals.to_vec(), expected_output);
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

    assert_eq!(f_tilde.f.evals.to_vec(), expected_output);
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

    assert_eq!(f_tilde.f.evals.to_vec(), expected_output);
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

    assert_eq!(f_tilde.f.evals.to_vec(), expected_output);
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

    assert_eq!(f_tilde.f.evals.to_vec(), expected_output);
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

    assert_eq!(f_tilde.f.evals.to_vec(), expected_output);
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
    .map(MultilinearExtension::new)
    .collect::<Vec<_>>();

    let interlaced_mle = MultilinearExtension::interlace_mles(mles);

    assert_eq!(
        *interlaced_mle.iter().collect::<Vec<_>>(),
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
