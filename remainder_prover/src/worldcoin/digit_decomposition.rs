use crate::mle::{
    dense::{get_padded_evaluations_for_list, DenseMle},
    Mle, MleIndex,
};
use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};

pub const LOG_NUM_DIGITS: usize = 3;
pub const NUM_DIGITS: usize = (1 << LOG_NUM_DIGITS) as usize;
pub const BASE: u32 = 256;

/// Decompose a number into its digits in a given base, MSB first.
/// Pre: base is at most 2 ** 16.
/// Post: The result has no leading zeros.
fn digital_decomposition_any_base(value: u64, base: u32) -> Vec<u16> {
    debug_assert!(base <= 1 << 16, "Base {} too large", base);
    let mut ret = vec![];
    let mut val = value;
    let base = base as u64;
    while val > 0 {
        ret.push((val % base) as u16);
        val /= base;
    }
    ret.into_iter().rev().collect()
}

#[test]
fn test_digital_decomposition_any_base() {
    assert_eq!(
        digital_decomposition_any_base(123456789, 10),
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    assert_eq!(digital_decomposition_any_base(0, 10).len(), 0);
    assert_eq!(digital_decomposition_any_base(4, 2), vec![1, 0, 0]);
}

/// Decompose a number into a NUM_DIGITS digit number in base BASE, MSB first.
/// Panics if the number is too large to fit in NUM_DIGITS digits.
pub fn digital_decomposition(value: u64) -> [u16; NUM_DIGITS] {
    let unpadded = digital_decomposition_any_base(value, BASE);
    assert!(
        unpadded.len() <= NUM_DIGITS,
        "Value {} too large to fit in NUM_DIGITS digits in base BASE",
        value
    );
    let padding_length = NUM_DIGITS - unpadded.len();
    repeat_n(0, padding_length)
        .chain(unpadded.into_iter())
        .collect_vec()
        .try_into()
        .unwrap()
}

#[test]
fn test_digital_decomposition() {
    assert!(3 < BASE);
    assert!(NUM_DIGITS >= 3);
    let val = 2u64 * (BASE as u64) + 3u64;
    assert_eq!(digital_decomposition(val)[NUM_DIGITS - 1], 3);
    assert_eq!(digital_decomposition(val)[NUM_DIGITS - 2], 2);
    assert_eq!(digital_decomposition(val)[0], 0);
    assert_eq!(digital_decomposition(val).len(), NUM_DIGITS);
}
