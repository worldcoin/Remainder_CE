use funty::Unsigned;
use num::Zero;
use shared_types::Field;

/// Helper function for conversion to field elements, handling negative values.
pub fn i64_to_field<F: Field>(value: i64) -> F {
    if value >= 0 {
        F::from(value as u64)
    } else {
        F::from(value.unsigned_abs()).neg()
    }
}

/// Take the ceil(log2(value)) for a u64 value.
pub fn log2_ceil<T: Unsigned + Zero>(value: T) -> u32 {
    if value == T::zero() {
        0
    } else if value.is_power_of_two() {
        1u64.leading_zeros() - value.leading_zeros()
    } else {
        0u64.leading_zeros() - value.leading_zeros()
    }
}
