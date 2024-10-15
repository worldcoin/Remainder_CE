use itertools::Itertools;
use ndarray::{Array, Array2};
use remainder_shared_types::Field;
use super::parameters::{decode_i64_array, decode_wirings};
use crate::digits::{complementary_decomposition, digits_to_field};
use crate::mle::bundled_input_mle::to_slice_of_vectors;
use crate::mle::evals::MultilinearExtension;
use crate::utils::arithmetic::i64_to_field;
use crate::utils::mle::pad_with;
use crate::worldcoin::parameters::decode_i32_array;

/// Input data for the Worldcoin iriscode circuit.
#[derive(Debug, Clone)]
pub struct IriscodeCircuitData<F: Field> {
    /// The values to be re-routed to form the LH multiplicand of the matrix multiplication.
    /// Length is a power of two.
    pub to_reroute: MultilinearExtension<F>,
    /// The MLE of the RH multiplicand of the matrix multiplication.
    /// Length is `1 << (MATMULT_INTERNAL_DIM_NUM_VARS + MATMULT_COLS_NUM_VARS)`.
    pub rh_matmult_multiplicand: MultilinearExtension<F>,
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
    /// Values to be subtracted from the result of the matrix multiplication.
    /// Length is `1 << (MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS)`.
    pub to_sub_from_matmult: MultilinearExtension<F>,
}

/// Wirings are a Vec of 4-tuples of u16s; each tuple maps a coordinate of image to a coordinate of
/// the LH multiplicand of the matmult. This function returns the corresponding Vec of 2-tuples of
/// usize, which are the re-routings of the 1d MLEs.
/// Input order is `(src_row_idx, src_col_idx, dest_row_idx, dest_col_idx)`.
/// Output order is `(dest_idx, src_idx)` (to match IdentityGate).
pub fn wirings_to_reroutings(
    wirings: &[(u16, u16, u16, u16)],
    src_arr_num_cols: usize,
    dest_arr_num_cols: usize,
) -> Vec<(usize, usize)> {
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
            (dest_idx, src_idx)
        })
        .collect_vec()
}

/// Build an instance of [IriscodeCircuitData] from the given image, RH multiplicand, thresholds and
/// wiring data, by deriving the iris code.
pub fn build_iriscode_circuit_data<
    F: Field,
    const MATMULT_ROWS_NUM_VARS: usize,
    const MATMULT_COLS_NUM_VARS: usize,
    const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
    const BASE: u64,
    const NUM_DIGITS: usize,
>(
    image: Array2<u8>,
    rh_multiplicand: &[i32],
    thresholds_matrix: &[i64],
    wirings: &[(u16, u16, u16, u16)],
) -> IriscodeCircuitData<F> {
    assert!(BASE.is_power_of_two());
    assert!(NUM_DIGITS.is_power_of_two());

    // Derive the re-routings from the wirings (this is what is needed for identity gate)
    // And calculate the left-hand side of the matrix multiplication
    let mut rerouted_matrix: Array2<i64> = Array::zeros((
        (1 << MATMULT_ROWS_NUM_VARS),
        (1 << MATMULT_INTERNAL_DIM_NUM_VARS),
    ));
    wirings.iter().for_each(|row| {
        let (im_row, im_col, a_row, a_col) = (
            row.0 as usize,
            row.1 as usize,
            row.2 as usize,
            row.3 as usize,
        );
        rerouted_matrix[[a_row, a_col]] = image[[im_row, im_col]] as i64;
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
        (1 << MATMULT_ROWS_NUM_VARS, 1 << MATMULT_COLS_NUM_VARS),
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

    // Flatten the kernel values, convert to field.  (Already padded)
    let rh_matmult_multiplicand: Vec<F> =
        rh_multiplicand.into_iter().map(i64_to_field).collect_vec();

    // Flatten the thresholds matrix, convert to field and pad.
    let thresholds_matrix: Vec<F> = pad_with(
        F::ZERO,
        &thresholds_matrix
            .into_iter()
            .map(i64_to_field)
            .collect_vec(),
    );

    // Convert the digit multiplicities to field elements.
    let digit_multiplicities = digit_multiplicities
        .into_iter()
        .map(|x| F::from(x as u64))
        .collect_vec();

    // FIXME(Ben) perform this in a more succinct way
    let digits: Vec<MultilinearExtension<F>> =
        to_slice_of_vectors(digits.iter().map(digits_to_field).collect_vec())
            .iter()
            .map(|digit_values| MultilinearExtension::new(digit_values.clone()))
            .collect_vec();

    IriscodeCircuitData {
        to_reroute: MultilinearExtension::new(image_matrix_mle),
        rh_matmult_multiplicand: MultilinearExtension::new(rh_matmult_multiplicand),
        digits,
        sign_bits: MultilinearExtension::new(code),
        digit_multiplicities: MultilinearExtension::new(digit_multiplicities),
        to_sub_from_matmult: MultilinearExtension::new(thresholds_matrix),
    }
}

/// Loads circuit structure data and witnesses for a run of the iris code circuit from disk for
/// either the iris or mask case.
///
/// # Arguments:
/// * `image_bytes` gives the bytes of an image (could be the iris or the mask).
/// * `is_mask` indicates whether to load the files for the mask or the iris.
pub fn load_worldcoin_data_v2<
    F: Field,
    const MATMULT_ROWS_NUM_VARS: usize,
    const MATMULT_COLS_NUM_VARS: usize,
    const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
    const BASE: u64,
    const NUM_DIGITS: usize,
    const IM_NUM_ROWS: usize,
    const IM_NUM_COLS: usize,
>(
    image_bytes: Vec<u8>,
    is_mask: bool,
) -> IriscodeCircuitData<F> {
    let image: Array2<u8> = Array2::from_shape_vec(
        (IM_NUM_ROWS, IM_NUM_COLS),
        image_bytes
    ).unwrap();

    use super::parameters_v2::{
        IRIS_RH_MULTIPLICAND, IRIS_THRESHOLDS, MASK_RH_MULTIPLICAND, MASK_THRESHOLDS, WIRINGS,
    };
    if is_mask {
        build_iriscode_circuit_data::<
            F,
            MATMULT_ROWS_NUM_VARS,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            BASE,
            NUM_DIGITS,
        >(
            image,
            &decode_i32_array(MASK_RH_MULTIPLICAND),
            &decode_i64_array(MASK_THRESHOLDS),
            &decode_wirings(WIRINGS),
        )
    } else {
        build_iriscode_circuit_data::<
            F,
            MATMULT_ROWS_NUM_VARS,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            BASE,
            NUM_DIGITS,
        >(
            image,
            &decode_i32_array(IRIS_RH_MULTIPLICAND),
            &decode_i64_array(IRIS_THRESHOLDS),
            &decode_wirings(WIRINGS),
        )
    }
}

/// Loads circuit structure data and witnesses for a run of the iris code circuit from disk for
/// either the iris or mask case.
///
/// # Arguments:
/// * `image_bytes` gives the bytes of an image (could be the iris or the mask).
/// * `is_mask` indicates whether to load the files for the mask or the iris.
pub fn load_worldcoin_data_v3<
    F: Field,
    const MATMULT_ROWS_NUM_VARS: usize,
    const MATMULT_COLS_NUM_VARS: usize,
    const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
    const BASE: u64,
    const NUM_DIGITS: usize,
    const IM_NUM_ROWS: usize,
    const IM_NUM_COLS: usize,
>(
    image_bytes: Vec<u8>,
    is_mask: bool,
) -> IriscodeCircuitData<F> {
    let image: Array2<u8> = Array2::from_shape_vec(
        (IM_NUM_ROWS, IM_NUM_COLS),
        image_bytes
    ).unwrap();
    use super::parameters_v3::{
        IRIS_RH_MULTIPLICAND, IRIS_THRESHOLDS, MASK_RH_MULTIPLICAND, MASK_THRESHOLDS, WIRINGS,
    };
    if is_mask {
        build_iriscode_circuit_data::<
            F,
            MATMULT_ROWS_NUM_VARS,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            BASE,
            NUM_DIGITS,
        >(
            image,
            &decode_i32_array(MASK_RH_MULTIPLICAND),
            &decode_i64_array(MASK_THRESHOLDS),
            &decode_wirings(WIRINGS),
        )
    } else {
        build_iriscode_circuit_data::<
            F,
            MATMULT_ROWS_NUM_VARS,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            BASE,
            NUM_DIGITS,
        >(
            image,
            &decode_i32_array(IRIS_RH_MULTIPLICAND),
            &decode_i64_array(IRIS_THRESHOLDS),
            &decode_wirings(WIRINGS),
        )
    }
}
