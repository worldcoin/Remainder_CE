/// Decompose a number into N digits in a given base, MSB first.
/// Returns None iff the number is too large to fit in N digits.
/// # Requires:
/// + base is at most 2 ** 16.
pub fn unsigned_decomposition<const N: usize>(value: u64, base: u64) -> Option<[u16; N]> {
    debug_assert!(base <= 1 << 16, "Base {} too large", base);
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

/// Decompose a number into N digits in a given base, MSB first, in the complementary representation, i.e.
///   value = b * base^N - (d[0] * base^(N-1) + d[1] * base^(N-2) + ... + d[N-1] * base^0)
/// where (d, b) is the result.
/// Returns None iff value is out of range.
/// # Requires:
/// + base is at most 2 ** 16.
pub fn complementary_decomposition<const N: usize>(value: i64, base: u64) -> Option<([u16; N], bool)> {
    debug_assert!(base <= 1 << 16, "Base {} too large", base);
    let pow = base.pow(N as u32) as i64;
    if value > pow || value < -pow + 1 {
        return None;
    }
    let (val_to_decomp, bit) = if value > 0 {
        ((pow - value) as u64, true)
    } else {
        ((-value) as u64, false)
    };
    match unsigned_decomposition(val_to_decomp, base) {
        Some(decomp) => Some((decomp, bit)),
        None => None,
    }
}

#[test]
fn test_unsigned_decomposition() {
    assert_eq!(unsigned_decomposition::<3>(987, 10), Some([9, 8, 7]));
    assert_eq!(unsigned_decomposition::<3>(0, 10), Some([0, 0, 0]));
    assert_eq!(unsigned_decomposition::<3>(1, 2), Some([0, 0, 1]));
    assert_eq!(unsigned_decomposition::<3>(4, 2), Some([1, 0, 0]));
    assert_eq!(unsigned_decomposition::<3>(8, 2), None);
}