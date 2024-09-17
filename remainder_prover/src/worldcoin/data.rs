use itertools::Itertools;
use ndarray::{Array, Array2, Array3};
use ndarray_npy::read_npy;
use remainder_shared_types::Field;
use std::path::PathBuf;

use crate::digits::{complementary_decomposition, digits_to_field};
use crate::layer::LayerId;
use crate::mle::bundled_input_mle::BundledInputMle;
use crate::mle::bundled_input_mle::{to_slice_of_vectors, FlatMles};
use crate::mle::Mle;
use crate::utils::arithmetic::i64_to_field;
use crate::utils::array::pad_with_rows;
use crate::utils::mle::pad_with;

/// Generate toy data for the worldcoin circuit.
/// Image is 2x2, and there are two 2x1 kernels (1, 0).T and (6, -1).T
/// The rewirings are trivial: the image _is_ the LH multiplicand of matmult.
pub fn trivial_wiring_2x2_circuit_data<F: Field>() -> CircuitData<F, 1, 1, 1, 16, 1> {
    CircuitData::build_worldcoin_circuit_data(
        Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap(),
        Array3::from_shape_vec((2, 2, 1), vec![1, 0, 6, -1]).unwrap(),
        Array2::from_shape_vec((2, 2), vec![1, 0, 1, 0]).unwrap(),
        // rewirings for the 2x2 identity matrix
        Array2::from_shape_vec((4, 4), vec![0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1])
            .unwrap(),
    )
}

/// Generate toy data for the worldcoin circuit.
/// Image is 2x2, and there are two 3x1 kernels.
/// The rewirings are trivial: the image _is_ the LH multiplicand of matmult.
pub fn trivial_wiring_2x2_odd_kernel_dims_circuit_data<F: Field>() -> CircuitData<F, 1, 1, 2, 16, 1>
{
    CircuitData::build_worldcoin_circuit_data(
        Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap(),
        Array3::from_shape_vec((2, 3, 1), vec![1, 0, -4, 6, -1, 3]).unwrap(),
        Array2::from_shape_vec((2, 2), vec![1, 0, 1, 0]).unwrap(),
        Array2::from_shape_vec((4, 4), vec![0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1])
            .unwrap(),
    )
}

#[derive(Debug, Clone)]
/// Used for instantiating the circuit.
/// + `BASE` and `NUM_DIGITS` are powers of two.
/// + `MATMULT_ROWS_NUM_VARS` is the log2 number of rows of the result of matmult.
/// + `MATMULT_COLS_NUM_VARS` is the log2 number of columns of the result of matmult.
/// + `MATMULT_INTERNAL_DIM_NUM_VARS` is the log2 of the internal dimension size of matmult = number of cols of LH multiplicand = number of rows of RH multiplicand.
pub struct CircuitData<
    F: Field,
    const MATMULT_ROWS_NUM_VARS: usize,
    const MATMULT_COLS_NUM_VARS: usize,
    const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
    const BASE: u64,
    const NUM_DIGITS: usize,
> {
    /// The values to be re-routed to form the LH multiplicand of the matrix multiplication.
    /// Length is a power of two.
    pub to_reroute: Vec<F>,
    /// The reroutings from `to_reroute` to the MLE representing the LH multiplicand of the matrix
    /// multiplication, as pairs of gate labels.
    pub reroutings: Vec<(usize, usize)>,
    /// The MLE of the RH multiplicand of the matrix multiplication.
    /// Length is `1 << (MATMULT_INTERNAL_DIM_NUM_VARS + MATMULT_COLS_NUM_VARS)`.
    pub rh_matmult_multiplicand: Vec<F>,
    /// The digits of the complementary digital decompositions (base BASE) of matmult minus `to_sub_from_matmult`.
    /// Length of each MLE is `1 << (MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS)`.
    pub digits: FlatMles<F, NUM_DIGITS>,
    /// The bits of the complementary digital decompositions of the values
    ///     matmult - to_sub_from_matmult.
    /// (This is the iris code (if processing the iris image) or the mask code (if processing the mask).)
    /// Length is `1 << (MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS)`.
    pub sign_bits: Vec<F>,
    /// The number of times each digit 0 .. BASE - 1 occurs in the complementary digital decompositions of
    /// response - threshold.
    /// Length is `BASE`.
    pub digit_multiplicities: Vec<F>,
    /// Values to be subtracted from the result of the matrix multiplication.
    /// Length is `1 << (MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS)`.
    pub to_sub_from_matmult: Vec<F>,
}

impl<
        F: Field,
        const MATMULT_ROWS_NUM_VARS: usize,
        const MATMULT_COLS_NUM_VARS: usize,
        const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
        const BASE: u64,
        const NUM_DIGITS: usize,
    >
    CircuitData<
        F,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >
{
    /// Create a new instance.
    ///
    /// # Arguments:
    /// + `image` is the input image as a 2d array of u8s (unpadded).
    /// + `kernel_values` is the 3d i16 array of kernel values arranged to have shape `(num_kernels,
    ///   kernel_num_rows, kernel_num_cols)` (kernel_num_rows, kernel_num_cols can be any values)
    /// + `thresholds_matrix` is the 2d array specifying the threshold to use for each kernel and
    ///   kernel placement; these will be all zeroes for the iris code, but non-zero values for the
    ///   mask.  Unpadded.
    /// + `wirings` is a 2d array of u16s with 4 columns; each row maps a coordinate of image to a
    ///   coordinate of the LH multiplicand of the matmult.
    ///
    /// # Requires:
    /// + `thresholds_matrix.dim().1 == kernel_values.dim().0`
    /// + Generics should be constrained as per [CircuitData].
    pub fn build_worldcoin_circuit_data(
        image: Array2<u8>,
        kernel_values: Array3<i32>,
        thresholds_matrix: Array2<i64>,
        wirings: Array2<u16>,
    ) -> Self {
        assert!(BASE.is_power_of_two());
        assert!(NUM_DIGITS.is_power_of_two());
        let (_, im_num_cols) = image.dim();
        let (num_kernels, kernel_num_rows, kernel_num_cols) = kernel_values.dim();
        assert_eq!(num_kernels, (1 << MATMULT_COLS_NUM_VARS));
        assert!(kernel_num_rows * kernel_num_cols <= (1 << MATMULT_INTERNAL_DIM_NUM_VARS));
        assert!(thresholds_matrix.dim().0 <= (1 << MATMULT_ROWS_NUM_VARS));
        assert_eq!(thresholds_matrix.dim().1, (1 << MATMULT_COLS_NUM_VARS));
        assert_eq!(wirings.dim().1, 4);

        // Derive the re-routings from the wirings (this is what is needed for identity gate)
        // And calculate the left-hand side of the matrix multiplication
        let mut reroutings = Vec::new();
        let mut rerouted_matrix: Array2<i64> = Array::zeros((
            (1 << MATMULT_ROWS_NUM_VARS),
            (1 << MATMULT_INTERNAL_DIM_NUM_VARS),
        ));
        wirings.outer_iter().for_each(|row| {
            let (im_row, im_col, a_row, a_col) = (
                row[0] as usize,
                row[1] as usize,
                row[2] as usize,
                row[3] as usize,
            );
            let a_gate_label = a_row * (1 << MATMULT_INTERNAL_DIM_NUM_VARS) + a_col;
            let im_gate_label = im_row * im_num_cols + im_col;
            reroutings.push((a_gate_label, im_gate_label));
            rerouted_matrix[[a_row, a_col]] = image[[im_row, im_col]] as i64;
        });

        // Reshape and pad kernel values to have dimensions (MATMULT_INTERNAL_DIM_NUM_VARS, MATMULT_COLS_NUM_VARS).
        // This is the RH multiplicand of the matrix multiplication.
        let rh_multiplicand: Array2<i64> = pad_with_rows(
            kernel_values
                .into_shape((
                    (1 << MATMULT_COLS_NUM_VARS),
                    kernel_num_rows * kernel_num_cols,
                ))
                .unwrap()
                .t()
                .to_owned()
                .mapv(|elem| elem as i64),
            1 << MATMULT_INTERNAL_DIM_NUM_VARS,
        );

        // Pad the thresholds matrix with extra rows of zeros so that it has dimensions
        // (MATMULT_ROWS_NUM_VARS, MATMULT_COLS_NUM_VARS).
        let thresholds_matrix = pad_with_rows(thresholds_matrix, 1 << MATMULT_ROWS_NUM_VARS);

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
        let kernel_values: Vec<F> = rh_multiplicand.into_iter().map(i64_to_field).collect_vec();

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

        // FlatMles for the digits
        let digits: FlatMles<F, NUM_DIGITS> = FlatMles::new_from_raw(
            to_slice_of_vectors(digits.iter().map(digits_to_field).collect_vec()),
            LayerId::Input(0),
        );

        CircuitData {
            to_reroute: image_matrix_mle,
            reroutings,
            rh_matmult_multiplicand: kernel_values,
            digits,
            sign_bits: code,
            digit_multiplicities,
            to_sub_from_matmult: thresholds_matrix,
        }
    }

    /// Enforce the claims on the attributes of self made by CircuitData.
    pub fn ensure_guarantees(&self) {
        assert!(BASE.is_power_of_two());
        assert!(NUM_DIGITS.is_power_of_two());
        assert!(self.to_reroute.len().is_power_of_two());
        assert_eq!(
            self.rh_matmult_multiplicand.len(),
            (1 << MATMULT_INTERNAL_DIM_NUM_VARS) * (1 << MATMULT_COLS_NUM_VARS)
        );
        self.digits.get_mle_refs().iter().for_each(|mle| {
            assert_eq!(
                mle.original_num_vars(),
                MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS
            );
        });
        assert_eq!(
            self.sign_bits.len(),
            (1 << MATMULT_ROWS_NUM_VARS) * (1 << MATMULT_COLS_NUM_VARS)
        );
        assert_eq!(self.digit_multiplicities.len(), BASE as usize);
        assert_eq!(
            self.to_sub_from_matmult.len(),
            (1 << MATMULT_ROWS_NUM_VARS) * (1 << MATMULT_COLS_NUM_VARS)
        );
    }
}

/// Loads circuit structure data and witnesses for a run of the iris code circuit from disk for either the iris or mask case.
/// Works for both v2 and v3 of the iriscode circuit.
///
/// # Arguments:
///   `constant_data_path` is the path to the root folder (containing the wirings, and subfolders).
///   `image_path` is the path to an image file (could be the iris or the mask).
///   `is_mask` indicates whether to load the files for the mask or the iris.
pub fn load_worldcoin_data<
    F: Field,
    const MATMULT_ROWS_NUM_VARS: usize,
    const MATMULT_COLS_NUM_VARS: usize,
    const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
    const BASE: u64,
    const NUM_DIGITS: usize,
>(
    constant_data_folder: PathBuf,
    image_path: PathBuf,
    is_mask: bool,
) -> CircuitData<
    F,
    MATMULT_ROWS_NUM_VARS,
    MATMULT_COLS_NUM_VARS,
    MATMULT_INTERNAL_DIM_NUM_VARS,
    BASE,
    NUM_DIGITS,
> {
    dbg!(&constant_data_folder);
    println!("Current directory: {:?}", std::env::current_dir().unwrap());
    let wirings: Array2<u16> = read_npy(constant_data_folder.join("wirings.npy")).unwrap();
    assert_eq!(wirings.dim().1, 4);

    let data_folder = constant_data_folder.join(if is_mask { "mask" } else { "iris" });
    let image: Array2<u8> = read_npy(image_path).unwrap();

    let kernel_values: Array3<i32> = read_npy(data_folder.join("kernel_values.npy")).unwrap();

    let thresholds: Array2<i64> = read_npy(data_folder.join("thresholds.npy")).unwrap();

    CircuitData::build_worldcoin_circuit_data(image, kernel_values, thresholds, wirings)
}
