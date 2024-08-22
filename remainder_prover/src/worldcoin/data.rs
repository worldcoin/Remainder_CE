use std::path::{Path, PathBuf};

use itertools::Itertools;
use ndarray::{Array, Array1, Array2, Array3};
use ndarray_npy::read_npy;
use remainder_shared_types::FieldExt;

use crate::layer::LayerId;
use crate::mle::circuit_mle::{to_flat_mles, FlatMles};
use crate::utils::digital_decomposition::complementary_decomposition;
use super::{BASE, NUM_DIGITS};

/// Generate toy data for the worldcoin circuit.
/// Image is 2x2, and there are two placements of two 2x1 kernels (1, 2).T and (3, 4).T
pub fn tiny_worldcoin_data<F: FieldExt>() -> WorldcoinCircuitData<F> {
    let image_shape = (2, 2);
    let kernel_shape = (2, 2, 1);
    let response_shape = (2, 2);
    WorldcoinCircuitData::new(
        Array2::from_shape_vec(image_shape, vec![3, 1, 4, 9]).unwrap(),
        Array3::from_shape_vec(kernel_shape, vec![1, 2, 3, 4]).unwrap(),
        vec![0],
        vec![0, 1],
        Array2::from_shape_vec(response_shape, vec![0, 0, 0, 0]).unwrap(),
        false,
    )
}

// FIXME what should the iris code be?
/// Generate toy data for the worldcoin circuit.
/// Image is 3x3:
///  3 1 4
///  1 5 9
///  2 6 5
/// There are four 2x2 kernels
///  1 2    2 7    2 3    3 -3
///  3 4    1 8    5 7    -2 0
/// There are four placements of the kernels.
/// Threshold values are non-trivial.
pub fn medium_worldcoin_data<F: FieldExt>() -> WorldcoinCircuitData<F> {
    let image_shape = (3, 3);
    let kernel_shape = (4, 2, 2);
    let response_shape = (4, 4);
    WorldcoinCircuitData::new(
        Array2::from_shape_vec(image_shape, vec![3, 1, 4, 1, 5, 9, 2, 6, 5]).unwrap(),
        Array3::from_shape_vec(
            kernel_shape,
            vec![1, 2, 3, 4, 2, 7, 1, 8, 2, 3, 5, 7, 3, -3, -2, 0],
        )
        .unwrap(),
        vec![0, 2],
        vec![0, 2],
        Array2::from_shape_vec(response_shape, vec![
            5, 5, 5, 5,
            5, 5, 5, 5,
            -5, -5, -5, -5,
            -5, -5, -5, -5,
            ]).unwrap(),
        false,
    )
}

#[derive(Debug, Clone)]
/// Used for instantiating the Worldcoin circuit.
/// Kernel placements are specified by the _product_ placements_row_idxs x placements_col_idxs.
pub struct WorldcoinCircuitData<F: FieldExt> {
    /// Row-major flattening of the input image (with no padding applied before or after).
    pub image_matrix_mle: Vec<F>,
    /// The reroutings from the input image to the matrix multiplicand "A", as pairs of gate labels.
    pub reroutings: Vec<(usize, usize)>,
    /// The number of kernel placements
    pub num_placements: usize,
    /// The row-major flattening of the reshaped kernel matrix, i.e. of the matrix multiplicand "B".
    pub kernel_matrix_mle: Vec<F>,
    /// (num_kernel_values, num_kernels)
    pub kernel_matrix_dims: (usize, usize),
    /// The digits of the complementary digital decompositions (base BASE) of the response - threshold + int(equality_allowed).
    pub digits: FlatMles<F, NUM_DIGITS>,
    /// The resulting binary code.  This is the iris code (if processing the iris image) or the mask
    /// code (if processing the mask).
    /// Equals the bits of the complementary digital decompositions of the values
    ///     response - threshold + int(equality_allowed)).
    /// Has dimensions num_placements x num_kernels (it is not padded).
    pub code: Vec<F>,
    /// The number of times each digit 0 .. BASE - 1 occurs in the complementary digital decompositions
    pub digit_multiplicities: Vec<F>,
    /// The matrix of thresholds of dimensions num_placements x num_kernels
    pub thresholds_matrix: Vec<F>,
    /// Whether equality of response and threshold results in a 1 (true) or 0 (false)
    pub equality_allowed: F,
}

/// Loads the witness for the v2 Worldcoin data from disk for either the iris or mask case.
/// Expects the following files:
/// + `image.npy` - (i64) the quantized input image (could be the iris or the mask)
/// + `padded_kernel_values.npy` - (i64) the padded kernel values (quantized)
/// + Placements, specified by the product of placements_row_idxs x placements_col_idxs:
///   - `placements_top_left_row_idxs.npy` - (i32) the row indices of the top-left corner of the placements of the padded kernels
///   - `placements_top_left_col_idxs.npy` - (i32) the column indices of the top-left corner of the placements of the padded kernels
/// + `thresholds.npy` - (i64) the thresholds for each placement and kernel combination (so has shape (num_placements, num_kernels)).
pub fn load_data<F: FieldExt>(data_directory: PathBuf) -> WorldcoinCircuitData<F> {
    let image: Array2<i64> =
        read_npy(Path::new(&data_directory.join("image.npy"))).unwrap();

    let kernel_values: Array3<i64> =
        read_npy(&data_directory.join("padded_kernel_values.npy")).unwrap();

    let placements_row_idxs: Vec<i32> =
        read_npy::<&PathBuf, Array1<i32>>(&data_directory.join("placements_top_left_row_idxs.npy"))
            .unwrap()
            .to_vec();

    let placements_col_idxs: Vec<i32> =
        read_npy::<&PathBuf, Array1<i32>>(&data_directory.join("placements_top_left_col_idxs.npy"))
            .unwrap()
            .to_vec();

    let thresholds: Array2<i64> = read_npy(&data_directory.join("thresholds.npy")).unwrap();

    WorldcoinCircuitData::new(
        image,
        kernel_values,
        placements_row_idxs,
        placements_col_idxs,
        thresholds,
        false // FIXME differentiate between iris and mask
    )
}

/// Helper function for conversion to field elements, handling negative values.
pub fn i64_to_field<F: FieldExt>(value: i64) -> F {
    if value >= 0 {
        F::from(value as u64)
    } else {
        F::from(value.unsigned_abs()).neg()
    }
}

impl<F: FieldExt> WorldcoinCircuitData<F> {
    /// Create a new instance. Note that every _combination_ from placements_row_idxs x
    /// placements_col_idxs specifies a kernel placement in the image (via the top-left coordinate).
    ///
    /// # Arguments:
    /// + `image` is the quantized input image (100x400 for v2).
    /// + `kernel_values` is the matrix of padded, quantized kernel values of shape (kernel_num_rows *
    ///   kernel_num_cols, num_kernels).
    /// + `placements_row_idxs` gives the row coordinate of the top-left corner of each placement of
    ///   the kernels (can be negative)
    /// + `placements_col_idxs` gives the column coordinate of the top-left corner of each placement
    ///   of the kernels (can be negative)
    /// + `thresholds_matrix` specifies the threshold to use for each kernel and kernel placement; these
    ///   will be all zeroes for the iris code, but non-zero values for the mask.
    ///
    /// # Requires:
    /// + `thresholds_matrix.dim() == (num_placements, num_kernels)`
    /// + `num_kernels.is_power_of_two()`
    /// + The dimensions of all kernels are the same, and are each a power of two.
    pub fn new(
        image: Array2<i64>,
        kernel_values: Array3<i64>,
        placements_row_idxs: Vec<i32>,
        placements_col_idxs: Vec<i32>,
        thresholds_matrix: Array2<i64>,
        equality_allowed: bool,
    ) -> Self {
        let (im_num_rows, im_num_cols) = image.dim();
        let (num_kernels, kernel_num_rows, kernel_num_cols) = kernel_values.dim();
        let num_placements = placements_row_idxs.len() * placements_col_idxs.len();
        let num_kernel_values = kernel_num_rows * kernel_num_cols;
        assert_eq!(thresholds_matrix.dim(), (num_placements, num_kernels));
        assert!(num_kernels.is_power_of_two());
        assert!(num_kernel_values.is_power_of_two());

        // convert the kernel placements to wirings
        let mut wirings: Vec<(usize, usize, usize, usize)> = Vec::new();
        for (i, placement_row_idx) in placements_row_idxs.iter().enumerate() {
            for (j, placement_col_idx) in placements_col_idxs.iter().enumerate() {
                let placement_idx = i * placements_col_idxs.len() + j;
                for row_idx in 0..kernel_num_rows {
                    let image_row_idx = placement_row_idx + (row_idx as i32);
                    if (image_row_idx < 0) || (image_row_idx as usize >= im_num_rows) {
                        continue; // zero padding vertically, so if row is out of bounds, then nothing to do
                    }
                    for col_idx in 0..kernel_num_cols {
                        // wrap around horizontally
                        let mut image_col_idx =
                            (placement_col_idx + (col_idx as i32)) % (im_num_cols as i32);
                        // adjust if the remainder is negative
                        if image_col_idx < 0 {
                            image_col_idx += im_num_cols as i32;
                        }
                        let flattened_kernel_idx = row_idx * kernel_num_cols + col_idx;
                        wirings.push((
                            placement_idx,
                            flattened_kernel_idx,
                            image_row_idx as usize,
                            image_col_idx as usize,
                        ));
                    }
                }
            }
        }
        // Derive the re-routings from the wirings (this is what is needed for identity gate)
        let mut reroutings = Vec::new();
        let matrix_a_num_cols = num_kernel_values;
        for (a_row, a_col, im_row, im_col) in &wirings {
            let a_gate_label = a_row * (matrix_a_num_cols) + a_col;
            let im_gate_label = im_row * im_num_cols + im_col;
            reroutings.push((a_gate_label, im_gate_label));
        }

        // Calculate the left-hand side of the matrix multiplication
        let mut rerouted_matrix: Array2<i64> =
            Array::zeros((num_placements, kernel_num_rows * kernel_num_cols));
        for (a_row, a_col, im_row, im_col) in &wirings {
            rerouted_matrix[[*a_row, *a_col]] = image[[*im_row, *im_col]];
        }

        // Reshape kernel values to have dimensions (num_kernel_values, num_kernels).
        // This is the RHS of the matrix multiplication.
        let kernel_matrix: Array2<i64> = kernel_values
            .into_shape((num_kernels, num_kernel_values))
            .unwrap()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap()
            .t()
            .to_owned();

        // Calculate the matrix product. Has dimensions (num_placements, num_kernels).
        let responses = rerouted_matrix.dot(&kernel_matrix);

        // Calculate the complementary digital decompositions of the thresholded responses
        // Both vectors have length num_placements * num_kernels.
        let thres_resp = responses - &thresholds_matrix + (if equality_allowed { 1 } else { 0 });
        let (digits, code): (Vec<_>, Vec<_>) = thres_resp
            .into_iter()
            .map(|value| complementary_decomposition::<BASE, NUM_DIGITS>(value).unwrap())
            .unzip();

        // Count the number of times each digit occurs
        let mut digit_multiplicities: Vec<usize> = vec![0; BASE as usize];
        digits.iter().for_each(|decomp| decomp.iter().for_each(|&digit| {
            digit_multiplicities[digit as usize] += 1;
        }));

        // Derive the padded image MLE.
        // Note that this padding has nothing to do with the padding of the thresholded responses.
        let mut flattened_input_image: Vec<F> = image
            .outer_iter()
            .map(|x| x.to_vec())
            .flat_map(|row| row.into_iter().map(i64_to_field).collect_vec())
            .collect_vec();
        let im_len_power_of_2 = flattened_input_image.len().next_power_of_two();
        let padding_amount = im_len_power_of_2 - flattened_input_image.len();
        let padding_values = vec![F::ZERO; padding_amount];
        flattened_input_image.extend(padding_values);
        let image_matrix_mle = flattened_input_image;

        // Convert the iris code to field elements
        let code: Vec<F> = code
            .into_iter()
            .map(|elem| F::from(elem as u64))
            .collect_vec();

        // Convert the kernel matrix to an MLE
        let kernel_matrix_mle: Vec<F> = kernel_matrix
            .outer_iter()
            .map(|row| row.to_vec())
            .flat_map(|row| row.into_iter().map(i64_to_field).collect_vec())
            .collect_vec();

        // Convert the thresholds matrix to an MLE
        let thresholds_matrix: Vec<F> = thresholds_matrix
            .outer_iter()
            .map(|row| row.to_vec().iter().map(|v| i64_to_field(*v)).collect_vec())
            .flatten()
            .collect_vec();

        // Create MLE for digit multiplicities
        let digit_multiplicities = digit_multiplicities
            .into_iter()
            .map(|x| F::from(x as u64))
            .collect_vec();

        // FlatMles for the digits
        let digits: FlatMles<F, NUM_DIGITS> = FlatMles::new_from_raw(
            to_flat_mles(
                digits
                    .into_iter()
                    .map(|vals| {
                        let vals: [F; NUM_DIGITS] = vals
                            .into_iter()
                            .map(|x| F::from(x as u64))
                            .collect_vec()
                            .try_into()
                            .unwrap();
                        vals
                    })
                    .collect_vec(),
            ),
            LayerId::Input(0),
        );

        let equality_allowed = if equality_allowed { F::ONE } else { F::ZERO };

        WorldcoinCircuitData {
            image_matrix_mle,
            reroutings,
            num_placements,
            kernel_matrix_mle,
            kernel_matrix_dims: (num_kernel_values, num_kernels),
            digits,
            code,
            digit_multiplicities,
            thresholds_matrix,
            equality_allowed,
        }
    }
}

/// Return the first power of two that is greater than or equal to the argument, or None if this
/// would exceed the range of a u32.
pub fn next_power_of_two(n: usize) -> Option<usize> {
    if n == 0 {
        return Some(1);
    }

    for pow in 1..32 {
        let value = 1_usize << (pow - 1);
        if value >= n {
            return Some(value);
        }
    }
    None
}

#[cfg(test)]
mod test {
    use ndarray::{Array2, Array3};
    use ndarray_npy::read_npy;
    use remainder_shared_types::Fr;
    use std::path::Path;

    use crate::{
        mle::{circuit_mle::CircuitMle, Mle},
        worldcoin::data::next_power_of_two,
    };

    use super::{load_data, medium_worldcoin_data, WorldcoinCircuitData};

    #[test]
    fn test_circuit_data_creation_v2_iris_and_mask() {
        let iris_path = Path::new("worldcoin_witness_data/iris").to_path_buf();
        let mask_path = Path::new("worldcoin_witness_data/mask").to_path_buf();
        for path in vec![iris_path, mask_path] {
            dbg!(&path);
            let data: WorldcoinCircuitData<Fr> = load_data(path.clone());
            // Check things that should be generically true
            assert_eq!(data.code.len(), data.thresholds_matrix.len());
            assert_eq!(
                data.kernel_matrix_mle.len(),
                data.kernel_matrix_dims.0 * data.kernel_matrix_dims.1
            );
            // there should be any many digits in the kth position as there are elements in the _padded_ iris code
            let expected_digits_length = next_power_of_two(data.code.len()).unwrap();
            for digits in data.digits.get_mle_refs().iter() {
                assert_eq!(
                    digits.get_padded_evaluations().len(),
                    expected_digits_length
                );
            }
            // Check things that should be true for this dataset
            assert_eq!(data.num_placements, 16 * 200);
            assert_eq!(data.code.len(), 16 * 200 * 4);
            assert_eq!(
                data.image_matrix_mle.len(),
                next_power_of_two(100 * 400).unwrap()
            );
            assert_eq!(data.kernel_matrix_mle.len(), 32 * 64 * 4);

            // Load the iris code as calculated in Python, check it's the same as we derive.
            let num_kernels = data.kernel_matrix_dims.1;
            let expected_iris_code3d: Array3<bool> =
                read_npy(&path.join("code.npy")).unwrap();
            let expected_iris_code: Array2<bool> = expected_iris_code3d
                .into_shape((num_kernels, data.num_placements))
                .unwrap()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap()
                .t()
                .to_owned();
            let expected_flattened = expected_iris_code
                .outer_iter()
                .flat_map(|row| row.to_vec())
                .collect::<Vec<bool>>();
            let expected_flattened: Vec<Fr> = expected_flattened
                .iter()
                .map(|&b| Fr::from(b as u64))
                .collect();
            assert_eq!(data.code, expected_flattened);
        }
    }

    #[test]
    fn test_circuit_data_creation_medium_dataset() {
        let data = medium_worldcoin_data::<Fr>();
        // Check things that should be generically true
        assert_eq!(data.code.len(), data.thresholds_matrix.len());
        assert_eq!(
            data.kernel_matrix_mle.len(),
            data.kernel_matrix_dims.0 * data.kernel_matrix_dims.1
        );
        for digits in data.digits.get_mle_refs().iter() {
            assert_eq!(digits.get_padded_evaluations().len(), data.code.len());
        }
        // Check things that should be true for this dataset
        assert_eq!(data.num_placements, 4);
        assert_eq!(data.code.len(), 16);
        assert_eq!(data.image_matrix_mle.len(), 16); // 16 is the nearest power of two to 9
        assert_eq!(data.kernel_matrix_mle.len(), 16);
    }
}
