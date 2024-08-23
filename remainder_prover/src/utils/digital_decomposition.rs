/// Decompose a number into N digits in a given base, MSB first.
/// Returns None iff the number is too large to fit in N digits.
pub fn unsigned_decomposition<const BASE: u16, const N: usize>(value: u64) -> Option<[u16; N]> {
    let base = BASE as u64;
    let mut value = value;
    let mut result = [0; N];
    for i in (0..N).rev() {
        result[i] = (value % base) as u16;
        value /= base;
    }
    if value > 0 {
        return None;
    }
    Some(result)
}

/// Decompose a number into N digits in a given BASE, MSB first, in the complementary representation, i.e.
///   value = b * BASE^N - (d[0] * BASE^(N-1) + d[1] * BASE^(N-2) + ... + d[N-1] * BASE^0)
/// where (d, b) is the result.
/// Returns None iff value is out of range.
/// # Requires:
///   `log2(BASE) * N <= 64`
pub fn complementary_decomposition<const BASE: u16, const N: usize>(value: i64) -> Option<([u16; N], bool)> {
    debug_assert!(BASE.ilog2() * (N as u32) <= 64, "BASE * N must be <= 64");
    let pow = (BASE as u128).pow(N as u32) as u128;
    if (value >= 0 && (value as u128) > pow) || (value < 0 && (-value as u128) > pow - 1) {
        return None;
    }
    let (val_to_decomp, bit) = if value > 0 {
        ((pow - (value as u128)), true)
    } else {
        ((-value) as u128, false)
    };
    // unwrap() guaranteed given the checks on already made
    Some((unsigned_decomposition::<BASE, N>(val_to_decomp as u64).unwrap(), bit))
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
    assert_eq!(complementary_decomposition::<2, 3>(1), Some(([1, 1, 1], true)));
    assert_eq!(complementary_decomposition::<2, 3>(3), Some(([1, 0, 1], true)));
    assert_eq!(complementary_decomposition::<2, 3>(8), Some(([0, 0, 0], true)));
    assert_eq!(complementary_decomposition::<2, 3>(0), Some(([0, 0, 0], false)));
    assert_eq!(complementary_decomposition::<2, 3>(-1), Some(([0, 0, 1], false)));
    assert_eq!(complementary_decomposition::<2, 3>(-3), Some(([0, 1, 1], false)));
    assert_eq!(complementary_decomposition::<2, 3>(-8), None);
    assert_eq!(complementary_decomposition::<2, 3>(9), None);
    // base 10
    assert_eq!(complementary_decomposition::<10, 3>(-987), Some(([9, 8, 7], false)));
    assert_eq!(complementary_decomposition::<10, 3>(987), Some(([0, 1, 3], true)));
    assert_eq!(complementary_decomposition::<10, 3>(0), Some(([0, 0, 0], false)));
    assert_eq!(complementary_decomposition::<10, 3>(-1000), None);
    assert_eq!(complementary_decomposition::<10, 3>(1001), None);
}