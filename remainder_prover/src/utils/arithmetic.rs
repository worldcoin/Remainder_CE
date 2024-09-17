use remainder_shared_types::Field;

/// Helper function for conversion to field elements, handling negative values.
pub fn i64_to_field<F: Field>(value: i64) -> F {
    if value >= 0 {
        F::from(value as u64)
    } else {
        F::from(value.unsigned_abs()).neg()
    }
}