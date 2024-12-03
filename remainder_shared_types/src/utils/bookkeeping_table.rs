use crate::Field;

/// Initializes with every iterated combination of the (binary)
/// variables in `challenge_coord`.
///
/// In other words, if `challenge_coord` is [r_1, r_2, r_3] then
/// `initialize_tensor` should output the following:
///
/// (1 - r_1) * (1 - r_2) * (1 - r_3),
/// (1 - r_1) * (1 - r_2) * (r_3),
/// (1 - r_1) * (r_2) * (1 - r_3),
/// (1 - r_1) * (r_2) * (r_3),
/// (r_1) * (1 - r_2) * (1 - r_3),
/// (r_1) * (1 - r_2) * (r_3),
/// (r_1) * (r_2) * (1 - r_3),
/// (r_1) * (r_2) * (r_3),
///
/// ## Arguments
/// * `challenge_coord` - Challenge point to be expanded in big-endian.
pub fn initialize_tensor<F: Field>(challenge_coord: &[F]) -> Vec<F> {
    let mut cur_table = Vec::with_capacity(1 << challenge_coord.len());
    cur_table.push(F::ONE);
    if !challenge_coord.is_empty() {
        // Dynamic programming algorithm in Tha13 for computing these
        // equality values and returning them as a vector.

        // Iterate through remaining challenge coordinates in reverse,
        // starting with the least significant variable.
        for challenge in challenge_coord.iter().rev() {
            let (one_minus_r, r) = (F::ONE - challenge, *challenge);

            // Double the size of `cur_table` to hold new values.
            let len = cur_table.len();
            cur_table.resize(len * 2, F::ZERO);

            for i in 0..len {
                cur_table[i + len] = cur_table[i] * r;
                cur_table[i] *= one_minus_r;
            }
        }
    }
    cur_table
}
