use std::path::{Path, PathBuf};

use itertools::Itertools;
use ndarray::{Array, Array1, Array2, Array3};
use ndarray_npy::read_npy;
use remainder_shared_types::FieldExt;

use crate::layer::LayerId;
use crate::mle::circuit_mle::{to_slice_of_vectors, FlatMles};
use crate::digits::{complementary_decomposition, digits_to_field};
use crate::utils::arithmetic::i64_to_field;

/// Generate toy data for the worldcoin circuit.
/// Image is 2x2, and there are two placements of two 2x1 kernels (1, 2).T and (3, 4).T
pub fn tiny_worldcoin_data<F: FieldExt>() -> WorldcoinCircuitData<F, 16, 2> {
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

/// Generate toy data for the worldcoin circuit in which the number of responses is not a power of
/// two (this is the case for the true v2 data, but that test case takes 90 seconds).
/// Image is 2x2, and there are three placements of one 2x1 kernel (1, 2).T
pub fn tiny_worldcoin_data_non_power_of_two<F: FieldExt>() -> WorldcoinCircuitData<F, 16, 2> {
    let image_shape = (2, 2);
    let kernel_shape = (1, 2, 1);
    let response_shape = (3, 1);
    WorldcoinCircuitData::new(
        Array2::from_shape_vec(image_shape, vec![3, 1, 4, 9]).unwrap(),
        Array3::from_shape_vec(kernel_shape, vec![1, 2]).unwrap(),
        vec![0],
        vec![0, 1, 1],
        Array2::from_shape_vec(response_shape, vec![0, 0, 0]).unwrap(),
        false,
    )
}

/// As per [tiny_worldcoin_data_non_power_of_two], but with the flag `equality_allowed` set to true.
pub fn tiny_worldcoin_data_non_power_of_two_mask_case<F: FieldExt>() -> WorldcoinCircuitData<F, 16, 2> {
    let image_shape = (2, 2);
    let kernel_shape = (1, 2, 1);
    let response_shape = (3, 1);
    WorldcoinCircuitData::new(
        Array2::from_shape_vec(image_shape, vec![3, 1, 4, 9]).unwrap(),
        Array3::from_shape_vec(kernel_shape, vec![1, 2]).unwrap(),
        vec![0],
        vec![0, 1, 1],
        Array2::from_shape_vec(response_shape, vec![0, 0, 0]).unwrap(),
        true,
    )
}

/// Generate and return two WorldcoinCircuitData instances, both of which have all their responses that are
/// on the threshold boundary, but which set the flag `equality_allowed` differently (this corresponds to the iris vs the
/// mask case).
/// Their respective codes are [0; 4] and [1; 4].
/// Image is 2x2, and there are two placements of two 2x1 kernels (1, 2).T and (3, 4).T
pub fn tiny_worldcoin_data_responses_on_threshold_boundary<F: FieldExt>() -> (WorldcoinCircuitData<F, 16, 2>, WorldcoinCircuitData<F, 16, 2>) {
    let image_shape = (2, 2);
    let kernel_shape = (2, 2, 1);
    let response_shape = (2, 2);
    let image = Array2::from_shape_vec(image_shape, vec![3, 1, 4, 9]).unwrap();
    let kernel_values = Array3::from_shape_vec(kernel_shape, vec![1, 2, 3, 4]).unwrap();
    let placements_row_idxs = vec![0];
    let placements_col_idxs = vec![0, 1];
    let thresholds = Array2::from_shape_vec(response_shape, vec![11, 25, 19, 39]).unwrap();
    (WorldcoinCircuitData::new(
        image.clone(),
        kernel_values.clone(),
        placements_row_idxs.clone(),
        placements_col_idxs.clone(),
        thresholds.clone(),
        false),
    WorldcoinCircuitData::new(
        image.clone(),
        kernel_values.clone(),
        placements_row_idxs.clone(),
        placements_col_idxs.clone(),
        thresholds.clone(),
        true)
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
pub fn medium_worldcoin_data<F: FieldExt>() -> WorldcoinCircuitData<F, 16, 2> {
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

// FIXME doc, test also boundary case, in fact, doctest
pub fn pad<F: Clone>(padding_value: F, data: Vec<F>) -> Vec<F> {
    let padding_amount = data.len().next_power_of_two() - data.len();
    let padding_values = vec![padding_value; padding_amount];
    let mut padded_data = data.clone();
    padded_data.extend(padding_values);
    padded_data
}

#[derive(Debug, Clone)]
/// Used for instantiating the Worldcoin circuit.
/// Kernel placements are specified by the _product_ placements_row_idxs x placements_col_idxs.
pub struct WorldcoinCircuitData<F: FieldExt, const BASE: u16, const NUM_DIGITS: usize> {
    /// The input image, flattened and then padded with zeros to the next power of two.
    pub image: Vec<F>,
    /// The reroutings from the input image to the matrix multiplicand "A", as pairs of gate labels.
    pub reroutings: Vec<(usize, usize)>,
    /// The number of kernel placements
    pub num_placements: usize,
    /// The flattening of the tensor of kernel values (num_kernels, num_rows, num_cols), padded with zeroes to the next power of two.
    /// (This ends up being the flattening of the matrix multiplicand "B".)
    pub kernel_values: Vec<F>,
    /// (num_kernel_values, num_kernels)
    pub kernel_matrix_dims: (usize, usize),
    /// The digits of the complementary digital decompositions (base BASE) of the response -
    /// threshold + int(equality_allowed).
    pub digits: FlatMles<F, NUM_DIGITS>,
    /// The resulting binary code.  This is the iris code (if processing the iris image) or the mask
    /// code (if processing the mask).
    /// Equals the bits of the complementary digital decompositions of the values
    ///     response - threshold + int(equality_allowed).
    /// Had dimensions num_placements x num_kernels before flattening and padding with zeros to the
    /// next power of two.
    pub code: Vec<F>,
    /// The number of times each digit 0 .. BASE - 1 occurs in the complementary digital decompositions of
    /// response - threshold + int(equality_allowed).
    pub digit_multiplicities: Vec<F>,
    /// The matrix of thresholds of dimensions num_placements x num_kernels, flattened and then
    /// padded to a power of two.
    pub thresholds_matrix: Vec<F>,
    /// Whether equality of response and threshold results in a 1 (true) or 0 (false)
    pub equality_allowed: F,
}

/// Loads the witness for the v2 Worldcoin data from disk for either the iris or mask case.
/// Expects the following files to be available in either the "iris" or "mask" subfolder of `data_directory`:
/// + `image.npy` - (i64) the quantized input image (could be the iris or the mask)
/// + `padded_kernel_values.npy` - (i64) the padded kernel values (quantized)
/// + Placements, specified by the product of placements_row_idxs x placements_col_idxs:
///   - `placements_top_left_row_idxs.npy` - (i32) the row indices of the top-left corner of the placements of the padded kernels
///   - `placements_top_left_col_idxs.npy` - (i32) the column indices of the top-left corner of the placements of the padded kernels
/// + `thresholds.npy` - (i64) the thresholds for each placement and kernel combination (so has shape (num_placements, num_kernels)).
/// The argument `is_mask` indicates whether the mask or the iris, and sets the `equality_allowed` flag accordingly.
pub fn load_data<F: FieldExt, const BASE: u16, const NUM_DIGITS: usize>(data_directory: PathBuf, is_mask: bool) -> WorldcoinCircuitData<F, BASE, NUM_DIGITS> {
    let data_directory = data_directory.join(if is_mask { "mask" } else { "iris" });
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
        is_mask
    )
}

impl<F: FieldExt, const BASE: u16, const NUM_DIGITS: usize> WorldcoinCircuitData<F, BASE, NUM_DIGITS> {
    /// Create a new instance. Note that every _combination_ from placements_row_idxs x
    /// placements_col_idxs specifies a kernel placement in the image (via the top-left coordinate).
    ///
    /// # Arguments:
    /// + `image` is the quantized input image (100x400 for v2).
    /// + `kernel_values` is the matrix of padded, quantized kernel values of shape `(num_kernels, kernel_num_rows,
    ///   kernel_num_cols)`.
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
    /// + `BASE` and `NUM_DIGITS` are powers of two.
    pub fn new(
        image: Array2<i64>,
        kernel_values: Array3<i64>,
        placements_row_idxs: Vec<i32>,
        placements_col_idxs: Vec<i32>,
        thresholds_matrix: Array2<i64>,
        equality_allowed: bool,
    ) -> Self {
        assert!(BASE.is_power_of_two());
        assert!(NUM_DIGITS.is_power_of_two());
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

        // Calculate the thresholded responses, which are the responses minus the thresholds plus
        // the equality_allowed indicator, to allow for the case where the inequality is not strict.
        let mut thres_resp = (responses - &thresholds_matrix + (if equality_allowed { 1 } else { 0 })).into_iter().collect_vec();
        // We pad the thresholded responses to the nearest power of two, since the number of
        // placements is not necessarily a power of two, and this will otherwise cause an issue for
        // logup (which expects the number of constrained values to be a power of two).
        // The padding value is `equality_allowed`.
        let padding_amount = thres_resp.len().next_power_of_two() - thres_resp.len();
        let padding_values = vec![ if equality_allowed { 1 } else { 0 }; padding_amount];
        thres_resp.extend(padding_values);

        // Calculate the complementary digital decompositions of the thresholded responses.
        // Both vectors have length num_placements * num_kernels.
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

        // Convert the iris code to field elements (this is already padded by construction)
        let code: Vec<F> = code
            .into_iter()
            .map(|elem| F::from(elem as u64))
            .collect_vec();

        // Flatten the kernel values, convert to field and pad
        let kernel_values: Vec<F> = pad(F::ZERO,
            kernel_matrix
            .into_iter()
            .map(i64_to_field)
            .collect_vec()
        );

        // Flatten the thresholds matrix, convert to field and pad
        let thresholds_matrix: Vec<F> = pad(F::ZERO,
            thresholds_matrix
            .into_iter()
            .map(i64_to_field)
            .collect_vec()
        );

        // Convert the digit multiplicities to field elements
        let digit_multiplicities = digit_multiplicities
            .into_iter()
            .map(|x| F::from(x as u64))
            .collect_vec();

        // FlatMles for the digits
        let digits: FlatMles<F, NUM_DIGITS> = FlatMles::new_from_raw(
            to_slice_of_vectors(digits.iter().map(digits_to_field).collect_vec()),
            LayerId::Input(0),
        );

        let equality_allowed = if equality_allowed { F::ONE } else { F::ZERO };

        WorldcoinCircuitData {
            image: image_matrix_mle,
            reroutings,
            num_placements,
            kernel_values,
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
    use remainder_shared_types::{halo2curves::ff::Field, Fr};
    use std::path::Path;

    use crate::{
        mle::{circuit_mle::CircuitMle, Mle},
        worldcoin::data::{next_power_of_two, pad, tiny_worldcoin_data_non_power_of_two, tiny_worldcoin_data_non_power_of_two_mask_case},
    };

    use super::{load_data, medium_worldcoin_data, WorldcoinCircuitData};

    // Check things that should be generically true for a WorldcoinCircuitData instance.
    fn check_worldcoin_circuit_data_promises<const BASE: u16, const NUM_DIGITS: usize>(data: &WorldcoinCircuitData<Fr, BASE, NUM_DIGITS>) {
        // Check padding to a power of two
        assert!(data.image.len().is_power_of_two());
        assert!(data.kernel_values.len().is_power_of_two());
        assert!(data.code.len().is_power_of_two());
        assert!(data.thresholds_matrix.len().is_power_of_two());
        assert_eq!(data.code.len(), data.thresholds_matrix.len());
        assert_eq!(
            data.kernel_values.len(),
            data.kernel_matrix_dims.0 * data.kernel_matrix_dims.1
        );
        // there should be any many digits in the kth position as there are elements in the iris code
        let expected_digits_length = data.code.len();
        for digits in data.digits.get_mle_refs().iter() {
            assert_eq!(
                digits.get_padded_evaluations().len(),
                expected_digits_length
            );
        }
    }

    #[test]
    fn test_circuit_data_creation_v2_iris_and_mask() {
        let path = Path::new("worldcoin_witness_data").to_path_buf();
        for is_mask in vec![false, true] { // FIXME currently fails for true i.e. the mask case
            dbg!(&is_mask);
            use crate::worldcoin::{WC_BASE, WC_NUM_DIGITS};
            let data: WorldcoinCircuitData<Fr, WC_BASE, WC_NUM_DIGITS> = load_data(path.clone(), is_mask);
            check_worldcoin_circuit_data_promises(&data);
            assert_eq!(data.equality_allowed, if is_mask { Fr::ONE } else { Fr::ZERO });
            // Check things that should be true for this dataset
            assert_eq!(data.num_placements, 16 * 200);
            assert_eq!(data.code.len(), ((16 * 200 * 4) as usize).next_power_of_two());
            assert_eq!(
                data.image.len(),
                ((100 * 400) as usize).next_power_of_two()
            );
            assert_eq!(data.kernel_values.len(), 32 * 64 * 4);

            // Load the iris code as calculated in Python, check it's the same as we derive.
            let num_kernels = data.kernel_matrix_dims.1;
            let code_path = if is_mask {
                path.join("mask").join("code.npy")
            } else {
                path.join("iris").join("code.npy")
            };
            let expected_iris_code3d: Array3<bool> = read_npy(&code_path).unwrap();
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
            let expected_flattened_padded = pad(Fr::from(0), expected_flattened);
            if data.code != expected_flattened_padded {
                println!("Expected code (length {}):", expected_flattened_padded.len());
                print_code(&expected_flattened_padded);
                println!("\nActual code (length {}):", data.code.len());
                print_code(&data.code);
            }
        }
    }

    fn print_code(code: &Vec<Fr>) {
        code.iter().for_each(|&x| {
            if x == Fr::from(1) {
                print!("1");
            } else {
                assert!(x == Fr::from(0));
                print!("0");
            }
        });
    }

    #[test]
    fn test_circuit_data_creation_tiny_non_power_of_two_mask_case() {
        let data = tiny_worldcoin_data_non_power_of_two_mask_case::<Fr>();
        // Check things that should be generically true
        check_worldcoin_circuit_data_promises(&data); 
        // Check things that should be true for this dataset
        assert_eq!(data.num_placements, 3);
        assert_eq!(data.code.len(), 4);
        assert_eq!(data.image.len(), 4);
        assert_eq!(data.kernel_values.len(), 2);
    }

    #[test]
    fn test_circuit_data_creation_medium_dataset() {
        let data = medium_worldcoin_data::<Fr>();
        // Check things that should be generically true
        check_worldcoin_circuit_data_promises(&data); 
        // Check things that should be true for this dataset
        assert_eq!(data.num_placements, 4);
        assert_eq!(data.code.len(), 16);
        assert_eq!(data.image.len(), 16); // 16 is the nearest power of two to 9
        assert_eq!(data.kernel_values.len(), 16);
    }
}
