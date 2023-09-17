use ark_std::log2;
use itertools::Itertools;
use remainder_shared_types::FieldExt;

/// Initializes with every iterated combination of the bits in `challenge_coord`.
/// TODO!(ryancao): Is this going in the correct endian-ness order?
fn initialize_tensor<F: FieldExt>(challenge_coord: &[F]) -> Vec<F> {
    // --- For each of the challenge coordinates ---
    challenge_coord
        .iter()
        .fold(vec![F::one()], |current_tensor, challenge| {
            // --- Take first coordinate and double current tensor ---
            current_tensor
                .clone()
                .into_iter()
                .map(|tensor_val| tensor_val * (F::one() - challenge))
                .chain(
                    current_tensor
                        .into_iter()
                        .map(|tensor_val| tensor_val * challenge),
                )
                .collect_vec()
        })
}

/// Returns `b^T` and `a` tensors for multilinear evaluation
/// such that b^T M a is the evaluation
///
/// ---
///
/// ## Example
///
/// M:
/// [a_{00}, a_{01}, a_{02}, a_{03}]
/// [a_{04}, a_{05}, a_{06}, a_{07}]
///
/// b:
/// [(1 - x_2), x_2]
///
/// a:
/// [(1 - x_0)(1 - x_1), x_0(1 - x_1), (1 - x_0)x_1, x_0x_1]
pub fn get_ml_inner_outer_tensors<F: FieldExt>(
    challenge_coord: &Vec<F>,
    num_rows: usize,
    orig_num_cols: usize,
) -> (Vec<F>, Vec<F>) {
    // --- Okay we need to actually think about this one ---
    // --- First assert that num_rows and orig_num_cols are both powers of 2 ---
    assert!(num_rows.is_power_of_two());
    assert!(orig_num_cols.is_power_of_two());

    // --- The number of rows + number of columns needs to equal 2^{total number of variables} ---
    assert_eq!(
        num_rows * orig_num_cols,
        2_usize.pow(challenge_coord.len() as u32)
    );

    // "a" tensor
    let orig_num_cols_num_vars = log2(orig_num_cols) as usize;
    let inner_tensor = initialize_tensor(&challenge_coord[0..orig_num_cols_num_vars]);
    assert_eq!(inner_tensor.len(), orig_num_cols);

    // "b" tensor
    let num_rows_num_vars = log2(num_rows) as usize;
    let outer_tensor = initialize_tensor(
        &challenge_coord[orig_num_cols_num_vars..orig_num_cols_num_vars + num_rows_num_vars],
    );
    assert_eq!(outer_tensor.len(), num_rows);

    (inner_tensor, outer_tensor)
}

/// Simply evaluates an MLE (specified via coefficients, i.e. evaluations over the
/// boolean hypercube), over the given challenge point.
/// TODO!(ryancao): Do we need to account for endian-ness here?
pub fn naive_eval_mle_at_challenge_point<F: FieldExt>(
    mle_coeffs: &Vec<F>,
    challenge_coord: &Vec<F>,
) -> F {
    assert!(mle_coeffs.len().is_power_of_two());
    assert_eq!(log2(mle_coeffs.len()), challenge_coord.len() as u32);

    let one = F::one();
    let reduced_bookkeeping_table =
        challenge_coord
            .iter()
            .fold(mle_coeffs.clone(), |bookkeeping_table, new_challenge| {
                // --- Grab every pair of elements and use the formula ---
                bookkeeping_table
                    .chunks(2)
                    .map(|elem_tuple| {
                        elem_tuple[0] * (one - new_challenge) + elem_tuple[1] * new_challenge
                    })
                    .collect_vec()
            });

    assert_eq!(reduced_bookkeeping_table.len(), 1);
    reduced_bookkeeping_table[0]
}

#[test]
fn test_initialize_tensor() {
    use ark_std::{test_rng, One};
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use rand::Rng;

    let mut rng = test_rng();

    // --- TODO!(ryancao): Change the random generation to how it should be!
    let first = Fr::from(rng.gen::<u64>());
    let second = Fr::from(rng.gen::<u64>());
    let third = Fr::from(rng.gen::<u64>());
    let challenge_coord = vec![first, second, third];

    let one = Fr::one();

    // NOTE that this is little-endian!!!
    let expected_tensor: Vec<Fr> = vec![
        (one - first) * (one - second) * (one - third),
        (first) * (one - second) * (one - third),
        (one - first) * (second) * (one - third),
        (first) * (second) * (one - third),
        (one - first) * (one - second) * (third),
        (first) * (one - second) * (third),
        (one - first) * (second) * (third),
        (first) * (second) * (third),
    ];

    let result_tensor = initialize_tensor(&challenge_coord);
    assert_eq!(expected_tensor, result_tensor);
}

#[test]
fn test_split_tensor() {
    use ark_std::{test_rng, One};
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use rand::Rng;

    let mut rng = test_rng();

    let first = Fr::from(rng.gen::<u64>());
    let second = Fr::from(rng.gen::<u64>());
    let third = Fr::from(rng.gen::<u64>());
    let fourth = Fr::from(rng.gen::<u64>());
    let fifth = Fr::from(rng.gen::<u64>());
    let challenge_coord = vec![first, second, third, fourth, fifth];

    let one = Fr::one();

    // TODO!(ryancao): Double check that this is right lol
    // NOTE that this is little-endian!!!
    let expected_inner_tensor: Vec<Fr> = vec![
        (one - first) * (one - second) * (one - third),
        (first) * (one - second) * (one - third),
        (one - first) * (second) * (one - third),
        (first) * (second) * (one - third),
        (one - first) * (one - second) * (third),
        (first) * (one - second) * (third),
        (one - first) * (second) * (third),
        (first) * (second) * (third),
    ];

    let expected_outer_tensor: Vec<Fr> = vec![
        (one - fourth) * (one - fifth),
        (fourth) * (one - fifth),
        (one - fourth) * (fifth),
        (fourth) * (fifth),
    ];

    let inner_tensor_num_vars = 3;
    let outer_tensor_num_vars = 2;
    let (result_inner_tensor, result_outer_tensor) = get_ml_inner_outer_tensors(
        &challenge_coord,
        2_usize.pow(outer_tensor_num_vars),
        2_usize.pow(inner_tensor_num_vars),
    );
    assert_eq!(expected_inner_tensor, result_inner_tensor);
    assert_eq!(expected_outer_tensor, result_outer_tensor);
}
