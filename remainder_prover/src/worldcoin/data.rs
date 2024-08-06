use std::{
    fs,
    io::BufWriter,
    marker::PhantomData,
    path::{Path, PathBuf},
};

use ark_serialize::Read;
use base64::{engine::general_purpose::STANDARD, Engine};
use itertools::Itertools;
use ndarray::{Array, Array1, Array2, Array3};
use ndarray_npy::read_npy;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use remainder_shared_types::FieldExt;
use remainder_shared_types::{curves::PrimeOrderCurve, layer::LayerId};
use serde_json::Value;

use crate::{
    mle::circuit_mle::{to_flat_mles, FlatMles},
    worldcoin::digit_decomposition::digital_decomposition,
};
use crate::worldcoin::digit_decomposition::{BASE, NUM_DIGITS};
use remainder_shared_types::HasByteRepresentation;

#[derive(Debug, Clone)]
/// Used for instantiating [super::circuits::WorldcoinCircuit].
/// Instantiated from WorldCoin data.
pub struct WorldcoinCircuitData<F: FieldExt> {
    /// Row-major flattening of the input image.
    pub image_matrix_mle: Vec<F>,
    /// The reroutings from the input image to the matrix multiplicand "A", as pairs of gate labels.
    pub reroutings: Vec<(usize, usize)>,
    /// The number of kernel placements
    pub num_placements: usize,
    /// The row-major flattening of [WorldcoinData::kernel_matrix]
    pub kernel_matrix_mle: Vec<F>,
    /// (kernel_num_rows * kernel_num_cols, num_kernels)
    pub kernel_matrix_dims: (usize, usize),
    /// The digital decompositions base BASE of the abs value of the responses
    pub digits: FlatMles<F, NUM_DIGITS>,
    /// The iris code = the sign bits of the responses
    pub iris_code: Vec<F>,
    /// The number of times each digit 0 .. BASE - 1 occurs in the digital decompositions
    pub digit_multiplicities: Vec<F>,
}

/// Witness data for the Worldcoin circuit, before conversion to MLEs.
#[derive(Debug)]
pub struct WorldcoinData<F: FieldExt> {
    /// Quantized input image (typically 100x400)
    pub image: Array2<i64>,
    /// Matrix of quantized kernel values of shape (kernel_num_rows * kernel_num_cols, num_kernels)
    pub kernel_matrix: Array2<i64>,
    /// Row coordinate of the top-left corner of each placement of the kernels (can be negative)
    pub placements_row_idxs: Vec<i32>,
    /// Column coordinate of the top-left corner of each placement of the kernels (can be negative)
    pub placements_col_idxs: Vec<i32>,
    /// Indices defining the wiring from input image to the matrix multiplicand "A".
    /// Each tuple is (A_row, A_column, image_row, image_column)
    pub wirings: Vec<(usize, usize, usize, usize)>,
    /// Result of applying the convolutions at the specified placements, or of performing
    /// the equivalent matrix multiplication. Has shape (number of placements, num_kernels)
    pub responses: Array2<i64>,
    /// Result of thresholding the responses.
    /// Has dimensions (number of placements * num_kernels)
    pub iris_code: Vec<bool>,
    pub _marker: PhantomData<F>,
    // FIXME PhantomData needed?
}

impl<F: FieldExt> WorldcoinData<F> {
    /// Create a new instance of WorldcoinData. kernel_values has dimensions (num_kernels,
    /// kernel_num_rows, kernel_num_cols). Every _combination_ from placements_row_idxs x
    /// placements_col_idxs specifies a kernel placement in the image (via the top-left coordinate).
    pub fn new(
        image: Array2<i64>,
        kernel_values: Array3<i64>,
        placements_row_idxs: Vec<i32>,
        placements_col_idxs: Vec<i32>,
    ) -> Self {
        let (im_num_rows, im_num_cols) = image.dim();
        let (num_kernels, kernel_num_rows, kernel_num_cols) = kernel_values.dim();

        // has dimensions (kernel_num_rows * kernel_num_cols, num_kernels)
        let kernel_matrix: Array2<i64> = kernel_values
            .into_shape((num_kernels, kernel_num_rows * kernel_num_cols))
            .unwrap()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap()
            .t()
            .to_owned();

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

        // build the rerouted matrix as an array
        let num_placements = placements_row_idxs.len() * placements_col_idxs.len();
        let mut rerouted_matrix: Array2<i64> =
            Array::zeros((num_placements, kernel_num_rows * kernel_num_cols));
        for (a_row, a_col, im_row, im_col) in &wirings {
            rerouted_matrix[[*a_row, *a_col]] = image[[*im_row, *im_col]];
        }

        // calculate the matrix product and the iris code
        // both have dimensions (kernel_placements.len(), num_kernels)
        let responses = rerouted_matrix.dot(&kernel_matrix);
        let iris_code = responses.mapv(|x| x > 0);
        let flattened_iris_code: Vec<bool> = iris_code
            .outer_iter()
            .flat_map(|row| row.to_vec())
            .collect();

        Self {
            image,
            kernel_matrix,
            placements_row_idxs,
            placements_col_idxs,
            wirings,
            responses,
            iris_code: flattened_iris_code,
            _marker: PhantomData,
        }
    }
}

/// Test that WorldcoinData::new() derives data as expected.
#[test]
fn test_worldcoin_data_creation() {
    use remainder_shared_types::Fr;
    let image_shape = (2, 2);
    let kernel_shape = (1, 2, 1);
    let response_shape = (image_shape.0, kernel_shape.2);
    let data = WorldcoinData::<Fr>::new(
        Array2::from_shape_vec(image_shape, vec![3, 1, 4, 9]).unwrap(),
        Array3::from_shape_vec(kernel_shape, vec![1, 2]).unwrap(),
        vec![0],
        vec![0, 1],
    );
    let expected_responses = Array2::from_shape_vec(response_shape, vec![1 * 3 + 2 * 4, 1 * 1 + 2 * 9]).unwrap(); // 11, 19
    assert_eq!(expected_responses, data.responses);
    // rewirings should transpose the matrix
    let expected_wirings = vec![(0, 0, 0, 0), (0, 1, 1, 0), (1, 0, 0, 1), (1, 1, 1, 1)];
    assert_eq!(expected_wirings, data.wirings);
    // Both response values are positive, so expected iris code is [true, true]
    let expected_iris_code = vec![true, true];
    assert_eq!(expected_iris_code, data.iris_code);
}

/// Loads the v2 Worldcoin data from disk, and checks our computation of the iris code against the
/// expected iris code.
pub fn load_data<F: FieldExt>(data_directory: PathBuf) -> WorldcoinData<F> {
    let image: Array2<i64> =
        read_npy(Path::new(&data_directory.join("quantized_image.npy"))).unwrap();

    let kernel_values: Array3<i64> =
        read_npy(&data_directory.join("quantized_kernels.npy")).unwrap();

    // read in the kernel placements
    let placements_row_idxs: Vec<i32> =
        read_npy::<&PathBuf, Array1<i32>>(&data_directory.join("placements_top_left_row_idxs.npy"))
            .unwrap()
            .to_vec();

    let placements_col_idxs: Vec<i32> =
        read_npy::<&PathBuf, Array1<i32>>(&data_directory.join("placements_top_left_col_idxs.npy"))
            .unwrap()
            .to_vec();

    let (num_kernels, _, _) = kernel_values.dim();
    let num_placements = placements_row_idxs.len() * placements_col_idxs.len();

    let result = WorldcoinData::new(
        image,
        kernel_values,
        placements_row_idxs,
        placements_col_idxs,
    );

    // sanity check: load the iris code as calculated in Python, check it's the same
    let expected_iris_code3d: Array3<bool> =
        read_npy(&data_directory.join("iris_code_for_sanity_check.npy")).unwrap();
    let expected_iris_code: Array2<bool> = expected_iris_code3d
        .into_shape((num_kernels, num_placements))
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()
        .t()
        .to_owned();
    let expected_flattened = expected_iris_code
        .outer_iter()
        .flat_map(|row| row.to_vec())
        .collect::<Vec<bool>>();
    assert_eq!(result.iris_code, expected_flattened);
    result
}

impl<F: FieldExt> From<&WorldcoinData<F>> for WorldcoinCircuitData<F> {
    fn from(data: &WorldcoinData<F>) -> Self {
        let mut flattened_input_image: Vec<F> = data
            .image
            .outer_iter()
            .map(|x| x.to_vec())
            .flat_map(|row| {
                row.into_iter()
                    .map(|elem| {
                        if elem < 0 {
                            F::from(elem.abs() as u64).neg()
                        } else {
                            F::from(elem.abs() as u64)
                        }
                    })
                    .collect_vec()
            })
            .collect_vec();
        let im_len_power_of_2 = flattened_input_image.len().next_power_of_two();
        let padding_amount = im_len_power_of_2 - flattened_input_image.len();
        let padding_values = vec![F::ZERO; padding_amount];
        flattened_input_image.extend(padding_values);
        let image_matrix_mle = flattened_input_image;

        let (_im_num_rows, im_num_cols) = data.image.dim();

        let num_placements = data.placements_row_idxs.len() * data.placements_col_idxs.len();

        let flattened_kernel_matrix: Vec<F> = data
            .kernel_matrix
            .outer_iter()
            .map(|row| row.to_vec())
            .flat_map(|row| {
                row.into_iter()
                    .map(|elem| {
                        if elem < 0 {
                            F::from(elem.abs() as u64).neg()
                        } else {
                            F::from(elem.abs() as u64)
                        }
                    })
                    .collect_vec()
            })
            .collect_vec();
        let kernel_matrix_mle = flattened_kernel_matrix;
        let (kernel_matrix_rows, _kernel_matrix_cols) = data.kernel_matrix.dim();
        let kernel_matrix_dims = data.kernel_matrix.dim();

        let mut reroutings = Vec::new();
        let matrix_a_num_cols = kernel_matrix_rows;
        for (a_row, a_col, im_row, im_col) in &data.wirings {
            let a_gate_label = a_row * (matrix_a_num_cols) + a_col;
            let im_gate_label = im_row * im_num_cols + im_col;
            reroutings.push((a_gate_label, im_gate_label));
        }

        // Calculate the signed digital decompositions of the data.responses
        let responses_vec_vec: Vec<Vec<i64>> = data
            .responses
            .outer_iter()
            .map(|row| row.to_vec())
            .collect_vec();

        // Manually inserting padding - this is fine since it is a public input.
        let pad_length =
            next_power_of_two(responses_vec_vec.len()).unwrap() - responses_vec_vec.len();
        let num_filters = data.kernel_matrix.dim().1;
        let responses_vec_vec: Vec<Vec<i64>> = responses_vec_vec
            .into_iter()
            .chain(vec![vec![0i64; num_filters]; pad_length].into_iter())
            .collect();

        let responses_digital_decomps: Vec<Vec<[u16; NUM_DIGITS]>> = responses_vec_vec
            .into_par_iter()
            .map(|row| {
                row.into_par_iter()
                    .map(|value| digital_decomposition(value.abs() as u64))
                    .collect()
            })
            .collect();
        let responses_digital_decomps: Vec<[u16; NUM_DIGITS]> = responses_digital_decomps
            .into_iter()
            .flat_map(|x| x)
            .collect();

        // count the number of times each digit appears in the digits
        let mut digit_multiplicities: Vec<usize> = vec![0; BASE as usize];
        for digit in responses_digital_decomps
            .clone()
            .into_iter()
            .flat_map(|x| x.into_iter())
        {
            digit_multiplicities[digit as usize] += 1;
        }
        // turn into an MLE
        let digit_multiplicities = digit_multiplicities
            .into_iter()
            .map(|x| F::from(x as u64))
            .collect_vec();

        let digits: FlatMles<F, NUM_DIGITS> = FlatMles::new_from_raw(
            to_flat_mles(
                responses_digital_decomps
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

        // Convert the iris code to a DenseMle
        let flattened_iris_code_matrix: Vec<F> = data
            .iris_code
            .iter()
            .map(|elem| F::from(*elem as u64))
            .collect_vec();
        let iris_code = flattened_iris_code_matrix;

        WorldcoinCircuitData {
            image_matrix_mle,
            reroutings,
            num_placements,
            kernel_matrix_mle,
            kernel_matrix_dims,
            digits,
            iris_code,
            digit_multiplicities,
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

/// Helper function for buffered writing to file.
pub fn write_bytes_to_file(filename: &str, bytes: &[u8]) {
    let file = fs::File::create(filename).unwrap();
    let bw = BufWriter::new(file);
    serde_json::to_writer(bw, &bytes).unwrap();
}

/// Helper function for buffered reading from file.
pub fn read_bytes_from_file(filename: &str) -> Vec<u8> {
    let mut file = std::fs::File::open(filename).unwrap();
    let initial_buffer_size = file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
    let mut bufreader = Vec::with_capacity(initial_buffer_size);
    file.read_to_end(&mut bufreader).unwrap();
    serde_json::de::from_slice(&bufreader[..]).unwrap()
}

/// Helper function to get iris code specifically by deserializing base64 and the speciifc key in json
pub fn read_iris_code_from_file_with_key(filename: &str, key: &str) -> Vec<bool> {
    let mut file = std::fs::File::open(filename).unwrap();
    let initial_buffer_size = file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
    let mut bufreader = Vec::with_capacity(initial_buffer_size);
    file.read_to_end(&mut bufreader).unwrap();
    let v: Value = serde_json::de::from_slice(&bufreader[..]).unwrap();
    let base64_string = v[key].as_str().unwrap();

    let bits: Vec<bool> = STANDARD
        .decode(base64_string)
        .unwrap()
        .iter()
        .flat_map(|byte| (0..8).rev().map(move |i| byte & (1 << i) != 0))
        .collect();
    bits
}

/// Helper function for buffered reading from file.
pub fn read_bytes_from_file_bin(filename: &str) -> Vec<u8> {
    let mut file = std::fs::File::open(filename).unwrap();
    let initial_buffer_size = file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
    let mut bufreader = Vec::with_capacity(initial_buffer_size);
    file.read_to_end(&mut bufreader).unwrap();
    bufreader
}

/// helper function to read a stream of bytes as a hyrax commitment
pub fn deserialize_commitment_from_bytes<C: PrimeOrderCurve>(bytes: &[u8]) -> Vec<C> {
    let commitment: Vec<C> = bytes
        .chunks(C::COMPRESSED_CURVE_POINT_BYTEWIDTH)
        .map(|chunk| C::from_bytes_compressed(chunk))
        .collect_vec();
    commitment
}

/// helper function to read a stream of bytes as blinding factors
pub fn deserialize_blinding_factors_from_bytes<C: PrimeOrderCurve>(bytes: &[u8]) -> Vec<C::Scalar> {
    let blinding_factors: Vec<C::Scalar> = bytes
        .chunks(C::SCALAR_ELEM_BYTEWIDTH)
        .map(|chunk| C::Scalar::from_bytes_le(chunk.to_vec()))
        .collect_vec();
    blinding_factors
}
#[cfg(test)]
mod test {
    use remainder_shared_types::Fr;
    use std::path::Path;

    use super::{load_data, WorldcoinCircuitData, WorldcoinData};

    #[test]
    fn test_load_data() {
        let data: WorldcoinData<Fr> = load_data(Path::new("worldcoin_witness_data").to_path_buf());
        let num_placements = data.placements_row_idxs.len() * data.placements_col_idxs.len();
        let num_kernels = data.kernel_matrix.dim().1;
        assert_eq!(data.responses.dim(), (num_placements, num_kernels));
        assert_eq!(data.iris_code.len(), (num_placements * num_kernels));
    }

    #[test]
    fn test_conversion_to_circuit_data() {
        let data: WorldcoinData<Fr> = load_data(Path::new("worldcoin_witness_data").to_path_buf());
        let _circuit_data: WorldcoinCircuitData<Fr> = (&data).into();
    }
}
