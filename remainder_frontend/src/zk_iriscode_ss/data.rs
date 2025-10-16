use crate::digits::{complementary_decomposition, digits_to_field, to_slice_of_mles};
use itertools::Itertools;
use ndarray::{Array, Array2};
use remainder::mle::evals::MultilinearExtension;
use remainder::utils::arithmetic::i64_to_field;
use remainder::utils::mle::pad_with;
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

/// Input data for the Worldcoin iriscode circuit.
#[derive(Debug, Clone)]
pub struct IriscodeCircuitInputData<F: Field> {
    /// The values to be re-routed to form the LH multiplicand of the matrix multiplication.
    /// Length is a power of two.
    pub to_reroute: MultilinearExtension<F>,
    /// The digits of the complementary digital decompositions (base BASE) of matmult minus `to_sub_from_matmult`.
    /// Length of each MLE is `1 << (MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS)`.
    pub digits: Vec<MultilinearExtension<F>>,
    /// The bits of the complementary digital decompositions of the values
    ///     matmult - to_sub_from_matmult.
    /// (This is the iris code (if processing the iris image) or the mask code (if processing the mask).)
    /// Length is `1 << (MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS)`.
    pub sign_bits: MultilinearExtension<F>,
    /// The number of times each digit 0 .. BASE - 1 occurs in the complementary digital decompositions of
    /// response - threshold.
    /// Length is `BASE`.
    pub digit_multiplicities: MultilinearExtension<F>,
}

/// Auxiliary input data for the Worldcoin iriscode circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct IriscodeCircuitAuxData<F: Field> {
    /// The MLE of the RH multiplicand of the matrix multiplication.
    /// Length is `1 << (MATMULT_INTERNAL_DIM_NUM_VARS + MATMULT_COLS_NUM_VARS)`.
    pub rh_matmult_multiplicand: MultilinearExtension<F>,

    /// Values to be subtracted from the result of the matrix multiplication.
    /// Length is `1 << (MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS)`.
    pub to_sub_from_matmult: MultilinearExtension<F>,
}

/// Wirings are a Vec of 4-tuples of u16s; each tuple maps a coordinate of the source matrix to a coordinate of
/// the destination matrix. This function returns the corresponding Vec of 2-tuples of
/// usize, which are the re-routings of the 1d MLEs.
/// Input order is `(src_row_idx, src_col_idx, dest_row_idx, dest_col_idx)`.
/// Output order is `(dest_idx, src_idx)` (to match [crate::layer::identity_gate::IdentityGate]).
pub fn wirings_to_reroutings(
    wirings: &[(u16, u16, u16, u16)],
    src_arr_num_cols: usize,
    dest_arr_num_cols: usize,
) -> Vec<(u32, u32)> {
    wirings
        .iter()
        .map(|row| {
            let (src_row_idx, src_col_idx, dest_row_idx, dest_col_idx) = (
                row.0 as usize,
                row.1 as usize,
                row.2 as usize,
                row.3 as usize,
            );
            let src_idx = src_row_idx * src_arr_num_cols + src_col_idx;
            let dest_idx = dest_row_idx * dest_arr_num_cols + dest_col_idx;
            (dest_idx as u32, src_idx as u32)
        })
        .collect_vec()
}

pub fn build_iriscode_circuit_auxiliary_data<
    F: Field,
    const MATMULT_COLS_NUM_VARS: usize,
    const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
    const NUM_STRIPS: usize,
    const MAT_CHUNK_SIZE: usize,
>(
    rh_multiplicand: &[i32],
    thresholds_matrix: &[i64],
) -> IriscodeCircuitAuxData<F> {
    // Build the RH multiplicand for the matmult.
    let rh_multiplicand = Array2::from_shape_vec(
        (
            1 << MATMULT_INTERNAL_DIM_NUM_VARS,
            1 << MATMULT_COLS_NUM_VARS,
        ),
        rh_multiplicand.iter().map(|&x| x as i64).collect_vec(),
    )
    .unwrap();

    // Flatten the kernel values, convert to field.  (Already padded)
    let rh_matmult_multiplicand: Vec<F> =
        rh_multiplicand.into_iter().map(i64_to_field).collect_vec();

    // Build the thresholds matrix from the 1d serialization.
    let thresholds_matrix = Array2::from_shape_vec(
        (NUM_STRIPS * MAT_CHUNK_SIZE, 1 << MATMULT_COLS_NUM_VARS),
        thresholds_matrix.to_vec(),
    )
    .unwrap();

    // Flatten the thresholds matrix, convert to field and pad.
    let thresholds_matrix: Vec<F> = pad_with(
        F::ZERO,
        &thresholds_matrix
            .into_iter()
            .map(i64_to_field)
            .collect_vec(),
    );

    IriscodeCircuitAuxData {
        rh_matmult_multiplicand: MultilinearExtension::new(rh_matmult_multiplicand),
        to_sub_from_matmult: MultilinearExtension::new(thresholds_matrix),
    }
}

/// Build an instance of [IriscodeCircuitData] from the given image, RH multiplicand, thresholds and
/// wiring data, by deriving the iris code.
pub fn build_iriscode_circuit_data<
    F: Field,
    const IM_STRIP_ROWS: usize,
    const IM_STRIP_COLS: usize,
    const MATMULT_ROWS_NUM_VARS: usize,
    const MATMULT_COLS_NUM_VARS: usize,
    const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
    const BASE: u64,
    const NUM_DIGITS: usize,
>(
    image: Array2<u8>,
    rh_multiplicand: &[i32],
    thresholds_matrix: &[i64],
    image_strip_wirings: Vec<Vec<(u16, u16, u16, u16)>>,
    lh_matrix_wirings: &[(u16, u16, u16, u16)],
) -> IriscodeCircuitInputData<F> {
    assert!(BASE.is_power_of_two());
    assert!(NUM_DIGITS.is_power_of_two());
    let num_strips = image_strip_wirings.len();

    // Calculate the left-hand side of the matrix multiplication
    let mat_chunk_size = 1 << MATMULT_ROWS_NUM_VARS;
    let mut rerouted_matrix: Array2<i64> = Array::zeros((
        num_strips * mat_chunk_size,
        (1 << MATMULT_INTERNAL_DIM_NUM_VARS),
    ));
    image_strip_wirings
        .iter()
        .enumerate()
        .for_each(|(strip_idx, wirings)| {
            // Build the image strip
            let mut image_strip: Array2<i64> = Array::zeros((IM_STRIP_ROWS, IM_STRIP_COLS));
            wirings.iter().for_each(|row| {
                let (im_row, im_col, im_strip_row, im_strip_col) = (
                    row.0 as usize,
                    row.1 as usize,
                    row.2 as usize,
                    row.3 as usize,
                );
                image_strip[[im_strip_row, im_strip_col]] = image[[im_row, im_col]] as i64;
            });
            // Route the image strip into the (un RLC'd) LH matrix
            lh_matrix_wirings.iter().for_each(|row| {
                let (im_strip_row, im_strip_col, mat_row, mat_col) = (
                    row.0 as usize,
                    row.1 as usize,
                    row.2 as usize,
                    row.3 as usize,
                );
                rerouted_matrix[[strip_idx * mat_chunk_size + mat_row, mat_col]] =
                    image_strip[[im_strip_row, im_strip_col]];
            });
        });

    // Build the RH multiplicand for the matmult.
    let rh_multiplicand = Array2::from_shape_vec(
        (
            1 << MATMULT_INTERNAL_DIM_NUM_VARS,
            1 << MATMULT_COLS_NUM_VARS,
        ),
        rh_multiplicand.iter().map(|&x| x as i64).collect_vec(),
    )
    .unwrap();

    // Build the thresholds matrix from the 1d serialization.
    let thresholds_matrix = Array2::from_shape_vec(
        (num_strips * mat_chunk_size, 1 << MATMULT_COLS_NUM_VARS),
        thresholds_matrix.to_vec(),
    )
    .unwrap();

    // Calculate the matrix product. Has dimensions (1 << MATMULT_ROWS_NUM_VARS, 1 << MATMULT_COLS_NUM_VARS).
    let responses = rerouted_matrix.dot(&rh_multiplicand);

    // Calculate the thresholded responses, which are the responses minus the thresholds. We pad
    // the thresholded responses to the nearest power of two, since logup expects the number of
    // constrained values (which will be the digits of the decomps of the threshold responses)
    // to be a power of two.
    let thres_resp = pad_with(
        0,
        &(responses - &thresholds_matrix).into_iter().collect_vec(),
    );

    // Calculate the complementary digital decompositions of the thresholded responses.
    // Both vectors have the same length as thres_resp.
    let (digits, code): (Vec<_>, Vec<_>) = thres_resp
        .into_iter()
        .map(|value| complementary_decomposition::<BASE, NUM_DIGITS>(value).unwrap())
        .unzip();

    // Count the number of times each digit occurs.
    let mut digit_multiplicities: Vec<usize> = vec![0; BASE as usize];
    digits.iter().for_each(|decomp| {
        decomp.iter().for_each(|&digit| {
            digit_multiplicities[digit as usize] += 1;
        })
    });

    // Derive the padded image MLE.
    // Note that this padding has nothing to do with the padding of the thresholded responses.
    let image_matrix_mle: Vec<F> = pad_with(0, &image.into_iter().collect_vec())
        .into_iter()
        .map(|v| F::from(v as u64))
        .collect_vec();

    // Convert the iris code to field elements (this is already padded by construction).
    let code: Vec<F> = code
        .into_iter()
        .map(|elem| F::from(elem as u64))
        .collect_vec();

    // Convert the digit multiplicities to field elements.
    let digit_multiplicities = digit_multiplicities
        .into_iter()
        .map(|x| F::from(x as u64))
        .collect_vec();
    let digits = to_slice_of_mles(digits.iter().map(digits_to_field).collect_vec()).to_vec();

    IriscodeCircuitInputData {
        to_reroute: MultilinearExtension::new(image_matrix_mle),
        digits,
        sign_bits: MultilinearExtension::new(code),
        digit_multiplicities: MultilinearExtension::new(digit_multiplicities),
    }
}
