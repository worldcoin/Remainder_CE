use std::path::{Path, PathBuf};

use ark_std::log2;
use itertools::Itertools;
use ndarray::{Array, Array1, Array2, Array3};
use ndarray_npy::read_npy;
use remainder_shared_types::{FieldExt, Fr};

use crate::layer::LayerId;
use crate::mle::circuit_mle::{to_slice_of_vectors, CircuitMle, FlatMles};
use crate::digits::{complementary_decomposition, digits_to_field};
use crate::mle::Mle;
use crate::utils::arithmetic::i64_to_field;
use crate::utils::mle::pad_with;
use crate::worldcoin::parameters_v2::{MATMULT_INTERNAL_DIM, MATMULT_NUM_COLS};

pub fn trivial_wiring_1x1_circuit_data<F: FieldExt>() -> CircuitData<F, 1, 1, 1, 16, 1> {
    CircuitData::build_worldcoin_circuit_data(
        Array2::from_shape_vec((1, 1), vec![1]).unwrap(),
        Array3::from_shape_vec((1, 1, 1), vec![1]).unwrap(),
        Array2::from_shape_vec((1, 1), vec![0]).unwrap(),
        Array2::from_shape_vec((1, 4), vec![0, 0, 0, 0]).unwrap(),
    )
} 

/// Generate toy data for the worldcoin circuit.
/// Image is 2x2, and there are two 2x1 kernels (1, 0).T and (6, -1).T
/// The rewirings are trivial: the image _is_ the LH multiplicand of matmult.
pub fn trivial_wiring_2x2_circuit_data<F: FieldExt>() -> CircuitData<F, 2, 2, 2, 16, 1> {
    CircuitData::build_worldcoin_circuit_data(
        Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap(),
        Array3::from_shape_vec((2, 2, 1), vec![1, 0, 6, -1]).unwrap(),
        Array2::from_shape_vec((2, 2), vec![1, 0, 1, 0]).unwrap(),
        // rewirings for the 2x2 identity matrix
        Array2::from_shape_vec((4, 4), vec![0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]).unwrap(),
    )
} 

/// Generate toy data for the worldcoin circuit.
/// Image is 2x2, and there are two 3x1 kernels.
/// The rewirings are trivial: the image _is_ the LH multiplicand of matmult.
pub fn trivial_wiring_2x2_odd_kernel_dims_circuit_data<F: FieldExt>() -> CircuitData<F, 2, 2, 4, 16, 1> {
    CircuitData::build_worldcoin_circuit_data(
        Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap(),
        Array3::from_shape_vec((2, 3, 1), vec![1, 0, -4, 6, -1, 3]).unwrap(),
        Array2::from_shape_vec((2, 2), vec![1, 0, 1, 0]).unwrap(),
        Array2::from_shape_vec((4, 4), vec![0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]).unwrap(),
    )
} 

// /// Generate toy data for the worldcoin circuit.
// /// Image is 2x2, and there are two placements of two 2x1 kernels (1, 2).T and (3, 4).T
// pub fn tiny_worldcoin_data<F: FieldExt>() -> WorldcoinCircuitData<F, 16u64, 2> {
//     let image_shape = (2, 2);
//     let kernel_shape = (2, 2, 1);
//     let response_shape = (2, 2);
//     WorldcoinCircuitData::new(
//         Array2::from_shape_vec(image_shape, vec![3, 1, 4, 9]).unwrap(),
//         Array3::from_shape_vec(kernel_shape, vec![1, 2, 3, 4]).unwrap(),
//         &vec![0],
//         &vec![0, 1],
//         Array2::from_shape_vec(response_shape, vec![0, 0, 0, 0]).unwrap(),
//     )
// }

// /// Generate toy data for the worldcoin circuit in which the number of responses is not a power of
// /// two (this is the case for the true v2 data, but that test case takes 90 seconds).
// /// Image is 2x2, and there are three placements of one 2x1 kernel (1, 2).T
// pub fn tiny_worldcoin_data_non_power_of_two<F: FieldExt>() -> WorldcoinCircuitData<F, 16u64, 2> {
//     let image_shape = (2, 2);
//     let kernel_shape = (1, 2, 1);
//     let response_shape = (3, 1);
//     WorldcoinCircuitData::new(
//         Array2::from_shape_vec(image_shape, vec![3, 1, 4, 9]).unwrap(),
//         Array3::from_shape_vec(kernel_shape, vec![1, 2]).unwrap(),
//         &vec![0],
//         &vec![0, 1, 1],
//         Array2::from_shape_vec(response_shape, vec![0, 0, 0]).unwrap(),
//     )
// }

// /// Generate toy data for the worldcoin circuit.
// /// Image is 3x3:
// ///  3 1 4
// ///  1 5 9
// ///  2 6 5
// /// There are four 2x2 kernels
// ///  1 2    2 7    2 3    3 -3
// ///  3 4    1 8    5 7    -2 0
// /// There are four placements of the kernels.
// /// Threshold values are non-trivial.
// pub fn medium_worldcoin_data<F: FieldExt>() -> WorldcoinCircuitData<F, 16u64, 2> {
//     let image_shape = (3, 3);
//     let kernel_shape = (4, 2, 2);
//     let response_shape = (4, 4);
//     WorldcoinCircuitData::new(
//         Array2::from_shape_vec(image_shape, vec![3, 1, 4, 1, 5, 9, 2, 6, 5]).unwrap(),
//         Array3::from_shape_vec(
//             kernel_shape,
//             vec![1, 2, 3, 4, 2, 7, 1, 8, 2, 3, 5, 7, 3, -3, -2, 0],
//         )
//         .unwrap(),
//         &vec![0, 2],
//         &vec![0, 2],
//         Array2::from_shape_vec(response_shape, vec![
//             5, 5, 5, 5,
//             5, 5, 5, 5,
//             -5, -5, -5, -5,
//             -5, -5, -5, -5,
//             ]).unwrap(),
//     )
// }

#[derive(Debug, Clone)]
/// Used for instantiating the circuit.
/// + `BASE` and `NUM_DIGITS` are powers of two.
/// + `MATMULT_NUM_ROWS` is number of rows of the result of matmult.  A power of two.
/// + `MATMULT_NUM_COLS` is number of columns of the result of matmult.  A power of two.
/// + `MATMULT_INTERNAL_DIM` is the internal dimension size of matmult = number of cols of LH multiplicand = number of rows of RH multiplicand.  A power of two.
pub struct CircuitData<F: FieldExt, const MATMULT_NUM_ROWS: usize, const MATMULT_NUM_COLS: usize, const MATMULT_INTERNAL_DIM: usize, const BASE: u64, const NUM_DIGITS: usize> {
    /// The values to be re-routed to form the LH multiplicand of the matrix multiplication.
    /// Length is a power of two.
    pub to_reroute: Vec<F>,
    /// The reroutings from `to_reroute` to the MLE representing the LH multiplicand of the matrix
    /// multiplication, as pairs of gate labels.
    pub reroutings: Vec<(usize, usize)>,
    /// The MLE of the RH multiplicand of the matrix multiplication.
    /// Length is `MATMULT_INTERNAL_DIM * MATMULT_NUM_COLS`.
    pub rh_matmult_multiplicand: Vec<F>,
    /// The digits of the complementary digital decompositions (base BASE) of matmult minus `to_sub_from_matmult`.
    /// Length of each MLE is `MATMULT_NUM_ROWS * MATMULT_NUM_COLS`.
    pub digits: FlatMles<F, NUM_DIGITS>,
    /// The bits of the complementary digital decompositions of the values
    ///     matmult - to_sub_from_matmult.
    /// (This is the iris code (if processing the iris image) or the mask code (if processing the mask).)
    /// Length is `MATMULT_NUM_ROWS * MATMULT_NUM_COLS`.
    pub sign_bits: Vec<F>,
    /// The number of times each digit 0 .. BASE - 1 occurs in the complementary digital decompositions of
    /// response - threshold.
    /// Length is `BASE`.
    pub digit_multiplicities: Vec<F>,
    /// Values to be subtracted from the result of the matrix multiplication.
    /// Length is `MATMULT_NUM_ROWS * MATMULT_NUM_COLS`.
    pub to_sub_from_matmult: Vec<F>,
}

impl<F: FieldExt, const MATMULT_NUM_ROWS: usize, const MATMULT_NUM_COLS: usize, const MATMULT_INTERNAL_DIM: usize, const BASE: u64, const NUM_DIGITS: usize> CircuitData<F, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS> {
    /// Create a new instance.
    ///
    /// # Arguments:
    /// + `image` is the input image as a 2d array of u8s (unpadded). 
    /// + `kernel_values` is the 3d i16 array of kernel values arranged to have shape `(num_kernels,
    ///   kernel_num_rows, kernel_num_cols)` (kernel_num_rows, kernel_num_cols can be any values)
    /// + `thresholds_matrix` is the 2d array specifying the threshold to use for each kernel and
    ///   kernel placement; these will be all zeroes for the iris code, but non-zero values for the
    ///   mask.  Unpadded.
    /// + `wirings` is a 2d array of u32s with 4 columns; each row maps a coordinate of image to a
    ///   coordinate of the LH multiplicand of the matmult.
    ///
    /// # Requires:
    /// + `thresholds_matrix.dim().1 == kernel_values.dim().0`
    /// + Generics should be constrained as per [CircuitData].
    pub fn build_worldcoin_circuit_data(
        image: Array2<u8>,
        kernel_values: Array3<i32>,
        thresholds_matrix: Array2<i64>,
        wirings: Array2<u32>,
    ) -> Self {
        assert!(BASE.is_power_of_two());
        assert!(NUM_DIGITS.is_power_of_two());
        assert!(MATMULT_NUM_ROWS.is_power_of_two());
        assert!(MATMULT_NUM_COLS.is_power_of_two());
        assert!(MATMULT_INTERNAL_DIM.is_power_of_two());
        let (_, im_num_cols) = image.dim();
        let (num_kernels, kernel_num_rows, kernel_num_cols) = kernel_values.dim();
        // FIXME do we need num_kernel_values?
        let num_kernel_values = kernel_num_rows * kernel_num_cols;
        assert_eq!(num_kernels, MATMULT_NUM_COLS);
        assert!(num_kernel_values <= MATMULT_INTERNAL_DIM);
        assert_eq!(thresholds_matrix.dim(), (MATMULT_NUM_ROWS, MATMULT_NUM_COLS));
        assert_eq!(wirings.dim().1, 4);

        // Derive the re-routings from the wirings (this is what is needed for identity gate)
        // And calculate the left-hand side of the matrix multiplication
        let mut reroutings = Vec::new();
        let mut rerouted_matrix: Array2<i64> =
            Array::zeros((MATMULT_NUM_ROWS, MATMULT_INTERNAL_DIM));
        wirings.outer_iter()
            .for_each(|row| {
                let (im_row, im_col, a_row, a_col) = (row[0] as usize, row[1] as usize, row[2] as usize, row[3] as usize);
                let a_gate_label = a_row * MATMULT_INTERNAL_DIM + a_col;
                let im_gate_label = im_row * im_num_cols + im_col;
                reroutings.push((a_gate_label, im_gate_label));
                rerouted_matrix[[a_row, a_col]] = image[[im_row, im_col]] as i64;
            });


        // FIXME tidy this up
        // Reshape and pad kernel values to have dimensions (MATMULT_INTERNAL_DIM, MATMULT_NUM_COLS).
        // This is the RH multiplicand of the matrix multiplication.
        let rh_multiplicand: Array2<i64> = Array::from_shape_vec((MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM),
            kernel_values
                .mapv(|elem| elem as i64)
                .outer_iter()
                .flat_map(|row| {
                    row.iter()
                        .cloned()
                        .chain(std::iter::repeat(0i64).take(MATMULT_INTERNAL_DIM - num_kernel_values))
                        .collect::<Vec<_>>()
                })
                .collect()
            ).unwrap()
            .t()
            .to_owned();

        // Calculate the matrix product. Has dimensions (MATMULT_NUM_ROWS, MATMULT_NUM_COLS).
        let responses = rerouted_matrix.dot(&rh_multiplicand);

        // Calculate the thresholded responses, which are the responses minus the thresholds. We pad
        // the thresholded responses to the nearest power of two, since logup expects the number of
        // constrained values (which will be the digits of the decomps of the threshold responses)
        // to be a power of two.
        let thres_resp = pad_with(0, &(responses - &thresholds_matrix).into_iter().collect_vec());

        // Calculate the complementary digital decompositions of the thresholded responses.
        // Both vectors have the same length as thres_resp.
        let (digits, code): (Vec<_>, Vec<_>) = thres_resp
            .into_iter()
            .map(|value| complementary_decomposition::<BASE, NUM_DIGITS>(value).unwrap())
            .unzip();

        // Count the number of times each digit occurs.
        let mut digit_multiplicities: Vec<usize> = vec![0; BASE as usize];
        digits.iter().for_each(|decomp| decomp.iter().for_each(|&digit| {
            digit_multiplicities[digit as usize] += 1;
        }));

        // Derive the padded image MLE.
        // Note that this padding has nothing to do with the padding of the thresholded responses.
        let image_matrix_mle: Vec<F> = pad_with(0, &image.into_iter().collect_vec()).into_iter().map(|v| F::from(v as u64)).collect_vec();

        // Convert the iris code to field elements (this is already padded by construction).
        let code: Vec<F> = code
            .into_iter()
            .map(|elem| F::from(elem as u64))
            .collect_vec();

        // Flatten the kernel values, convert to field.  (Already padded)
        let kernel_values: Vec<F> = rh_multiplicand
            .into_iter()
            .map(i64_to_field)
            .collect_vec();

        // Flatten the thresholds matrix, convert to field and pad.
        let thresholds_matrix: Vec<F> = pad_with(F::ZERO,
            &thresholds_matrix
            .into_iter()
            .map(i64_to_field)
            .collect_vec()
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
        assert!(MATMULT_NUM_ROWS.is_power_of_two());
        assert!(MATMULT_NUM_COLS.is_power_of_two());
        assert!(MATMULT_INTERNAL_DIM.is_power_of_two());
        assert!(self.to_reroute.len().is_power_of_two());
        assert_eq!(self.rh_matmult_multiplicand.len(), MATMULT_INTERNAL_DIM * MATMULT_NUM_COLS);
        self.digits.get_mle_refs().iter().for_each(|mle| {
            assert_eq!(mle.original_num_vars(), log2(MATMULT_NUM_ROWS * MATMULT_NUM_COLS) as usize);
        });
        assert_eq!(self.sign_bits.len(), MATMULT_NUM_ROWS * MATMULT_NUM_COLS);
        assert_eq!(self.digit_multiplicities.len(), BASE as usize);
        assert_eq!(self.to_sub_from_matmult.len(), MATMULT_NUM_ROWS * MATMULT_NUM_COLS);


    }
}

/// Loads the witnesses for a run of the iris code circuit from disk for either the iris or mask case.
/// 
/// # Arguments:
///   `constant_data_path` is the path to the root folder (containing the wirings, and subfolders).
///   `image_path` is the path to an image file (could be the iris or the mask).
///   `is_mask` indicates whether to load the files for the mask or the iris.
pub fn load_worldcoin_data<F: FieldExt, const MATMULT_NUM_ROWS: usize, const MATMULT_NUM_COLS: usize, const MATMULT_INTERNAL_DIM: usize, const BASE: u64, const NUM_DIGITS: usize>(constant_data_folder: PathBuf, image_path: PathBuf, is_mask: bool) -> CircuitData<F, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS> {
    let wirings: Array2<u32> = read_npy(&constant_data_folder.join("wirings.npy")).unwrap();
    assert_eq!(wirings.dim().1, 4);

    let data_folder = constant_data_folder.join(if is_mask { "mask" } else { "iris" });
    let image: Array2<u8> = read_npy(image_path).unwrap();

    let kernel_values: Array3<i32> =
        read_npy(&data_folder.join("kernel_values.npy")).unwrap();

    let thresholds: Array2<i64> = read_npy(&data_folder.join("thresholds.npy")).unwrap();

    CircuitData::build_worldcoin_circuit_data(
        image,
        kernel_values,
        thresholds,
        wirings
    )
}

#[test]
/// Simply checks that the test files for v2 are available (in both the iris and mask case) and that
/// the CircuitData instance can be constructed.
fn test_load_worldcoin_data_v2() {
    use super::parameters_v2::{CONSTANT_DATA_FOLDER, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS};
    // iris
    let path = Path::new(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("iris/test_image.npy");
    let data = load_worldcoin_data::<Fr, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS>(path.clone(), image_path, false);
    data.ensure_guarantees();
    // mask
    let image_path = path.join("mask/test_image.npy");
    let data = load_worldcoin_data::<Fr, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS>(path.clone(), image_path, true);
    data.ensure_guarantees();
}

#[test]
/// Simply checks that the test files for v3 are available (in both the iris and mask case) and that
/// the CircuitData instance can be constructed.
fn test_load_worldcoin_data_v3() {
    use super::parameters_v3::{CONSTANT_DATA_FOLDER, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS};
    // iris
    let path = Path::new(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("iris/test_image.npy");
    let data = load_worldcoin_data::<Fr, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS>(path.clone(), image_path, false);
    data.ensure_guarantees();
    // mask
    let image_path = path.join("mask/test_image.npy");
    let data = load_worldcoin_data::<Fr, MATMULT_NUM_ROWS, MATMULT_NUM_COLS, MATMULT_INTERNAL_DIM, BASE, NUM_DIGITS>(path.clone(), image_path, true);
    data.ensure_guarantees();
}