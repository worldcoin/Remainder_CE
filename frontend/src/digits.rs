//! Module containing functions for deriving digital decompositions and building associated circuit
//! components.
use itertools::Itertools;
use shared_types::Field;

use remainder::utils::arithmetic::log2_ceil;

use remainder::mle::evals::MultilinearExtension;

/// Decompose a number into N digits in a given base, MSB first.
/// Returns None iff the number is too large to fit in N digits.
/// # Requires:
///   + `log2(BASE) * N <= 64`
///   + `BASE <= (1 << 16)`
/// # Example
/// ```
/// # use frontend::digits::unsigned_decomposition;
/// assert_eq!(unsigned_decomposition::<10, 3>(987), Some([9, 8, 7]));
/// assert_eq!(unsigned_decomposition::<10, 3>(9870), None);
/// ```
pub fn unsigned_decomposition<const BASE: u64, const N: usize>(value: u64) -> Option<[u16; N]> {
    debug_assert!(BASE <= (1 << 16));
    debug_assert!(log2_ceil(BASE) * (N as u32) <= 64, "BASE * N must be <= 64");
    let mut value = value;
    let mut result = [0; N];
    for i in (0..N).rev() {
        result[i] = (value % BASE) as u16;
        value /= BASE;
    }
    if value != 0 {
        None
    } else {
        Some(result)
    }
}

/// Decompose a number into N digits in a given BASE, MSB first, in the complementary representation, i.e.
///   `value = b * BASE^N - (d[0] * BASE^(N-1) + d[1] * BASE^(N-2) + ... + d[N-1] * BASE^0)`
/// where `(d, b)` is the result.
/// Returns None iff value is out of range.
/// # Requires:
///   + `log2(BASE) * N <= 64`
///   + `BASE <= (1 << 16)`
/// # Example:
/// ```
/// # use frontend::digits::complementary_decomposition;
/// let decomp = complementary_decomposition::<2, 3>(3);
/// assert_eq!(decomp, Some(([1, 0, 1], true)));
/// ```
pub fn complementary_decomposition<const BASE: u64, const N: usize>(
    value: i64,
) -> Option<([u16; N], bool)> {
    debug_assert!(BASE <= (1 << 16));
    debug_assert!(log2_ceil(BASE) * (N as u32) <= 64, "BASE * N must be <= 64");
    let pow = (BASE as u128).pow(N as u32);
    if (value >= 0 && (value as u128) > pow) || (value < 0 && (-value as u128) > pow - 1) {
        return None;
    }
    let (val_to_decomp, bit) = if value > 0 {
        ((pow - (value as u128)), true)
    } else {
        ((-value) as u128, false)
    };
    // unwrap() guaranteed given the checks on already made
    Some((
        unsigned_decomposition::<BASE, N>(val_to_decomp as u64).unwrap(),
        bit,
    ))
}

/// Convert a slice of digits to a slice of field elements.
/// # Example:
/// ```
/// use shared_types::Fr;
/// use frontend::digits::digits_to_field;
/// assert_eq!(digits_to_field::<Fr, 3>(&[1, 2, 3]), [Fr::from(1), Fr::from(2), Fr::from(3)]);
/// ```
pub fn digits_to_field<F: Field, const N: usize>(digits: &[u16; N]) -> [F; N] {
    digits
        .iter()
        .map(|x| F::from(*x as u64))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Helper function that converts a `Vec<[F; N]>` into a `[MultilinearExtension<F>; N]`, i.e.
/// that changes the order of the enumeration by iterating first over the inner
/// index, then the outer index.
/// # Example:
/// ```
/// use shared_types::Fr;
/// use frontend::digits::to_slice_of_mles;
/// let inputs = vec![
///     [Fr::from(1), Fr::from(2), Fr::from(3)],
///     [Fr::from(4), Fr::from(5), Fr::from(6)],
/// ];
/// let expected_evals = [
///     vec![Fr::from(1), Fr::from(4)],
///     vec![Fr::from(2), Fr::from(5)],
///     vec![Fr::from(3), Fr::from(6)],
/// ];
/// let actual_evals: Vec<Vec<Fr>> = to_slice_of_mles(inputs).into_iter().map(|mle| mle.to_vec()).collect();
/// assert_eq!(actual_evals, expected_evals);
/// ```
pub fn to_slice_of_mles<F: Field, const N: usize>(
    inputs: Vec<[F; N]>,
) -> [MultilinearExtension<F>; N] {
    let mut result_evals: [Vec<F>; N] = vec![Vec::new(); N].try_into().unwrap();
    for subarray in inputs {
        for (i, element) in subarray.iter().enumerate() {
            result_evals[i].push(*element);
        }
    }
    let result_mles: [MultilinearExtension<F>; N] = result_evals
        .into_iter()
        .map(|evals| MultilinearExtension::new(evals))
        .collect_vec()
        .try_into()
        .unwrap();
    result_mles
}

#[test]
fn test_unsigned_decomposition() {
    assert_eq!(unsigned_decomposition::<10, 3>(987), Some([9, 8, 7]));
    assert_eq!(unsigned_decomposition::<10, 3>(0), Some([0, 0, 0]));
    assert_eq!(unsigned_decomposition::<2, 3>(1), Some([0, 0, 1]));
    assert_eq!(unsigned_decomposition::<2, 3>(4), Some([1, 0, 0]));
    assert_eq!(unsigned_decomposition::<2, 3>(8), None);
}

#[test]
fn test_complementary_decomposition() {
    // base 2
    assert_eq!(
        complementary_decomposition::<2, 3>(1),
        Some(([1, 1, 1], true))
    );
    assert_eq!(
        complementary_decomposition::<2, 3>(3),
        Some(([1, 0, 1], true))
    );
    assert_eq!(
        complementary_decomposition::<2, 3>(8),
        Some(([0, 0, 0], true))
    );
    assert_eq!(
        complementary_decomposition::<2, 3>(0),
        Some(([0, 0, 0], false))
    );
    assert_eq!(
        complementary_decomposition::<2, 3>(-1),
        Some(([0, 0, 1], false))
    );
    assert_eq!(
        complementary_decomposition::<2, 3>(-3),
        Some(([0, 1, 1], false))
    );
    assert_eq!(complementary_decomposition::<2, 3>(-8), None);
    assert_eq!(complementary_decomposition::<2, 3>(9), None);
    // base 10
    assert_eq!(
        complementary_decomposition::<10, 3>(-987),
        Some(([9, 8, 7], false))
    );
    assert_eq!(
        complementary_decomposition::<10, 3>(987),
        Some(([0, 1, 3], true))
    );
    assert_eq!(
        complementary_decomposition::<10, 3>(0),
        Some(([0, 0, 0], false))
    );
    assert_eq!(complementary_decomposition::<10, 3>(-1000), None);
    assert_eq!(complementary_decomposition::<10, 3>(1001), None);
}
