/// For V3 iris code stuff!
use std::{
    fs,
    io::BufWriter,
    marker::PhantomData,
    path::{Path, PathBuf},
    process::exit,
};

use ark_serialize::Read;
use ark_std::log2;
use base64::{engine::general_purpose::STANDARD, Engine};
use itertools::Itertools;
use ndarray::{Array, Array1, Array2, Array3, Axis};
use ndarray_npy::read_npy;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use remainder_shared_types::{curves::PrimeOrderCurve, layer::LayerId, FieldExt};
use serde::de;
use serde_json::Value;

use crate::{
    mle::circuit_mle::{to_flat_mles, FlatMles},
    worldcoin::digit_decomposition::digital_decomposition,
};
use crate::{
    mle::dense::DenseMle,
    worldcoin::digit_decomposition::{BASE, NUM_DIGITS},
};

use super::data::WorldcoinCircuitData;
use remainder_shared_types::HasByteRepresentation;

pub const IRIS_CODE_V3_FLATTENED_LENGTH: usize = 16 * 256;
pub const IRIS_CODE_V3_IMG_ROWS: usize = 128;
pub const IRIS_CODE_V3_IMG_COLS: usize = 1024;

/// Witness data for the Worldcoin circuit, before conversion to MLEs.
/// Note that a few of these data types are different for the V3 version!
pub struct WorldcoinDataV3<F: FieldExt> {
    /// Input image (128 x 1024)
    image: Array2<u8>,
    /// Result of applying the convolutions at the specified placements, or of performing
    /// the equivalent matrix multiplication. Has shape (number of placements, num_kernels)
    responses: Array2<i64>,
    /// Result of thresholding the responses.
    /// Has dimensions (number of placements * num_kernels)
    iris_code: Vec<bool>,
    _marker: PhantomData<F>,
}

/// Every IrisCode version has a fixed set of kernel data. This just represents
/// one of the kernels for the V3 IrisCode model.
pub struct IrisCodeV3KernelData {
    /// The kernel number within all the kernels
    kernel_idx: usize,
    /// Matrix of quantized kernel values of shape (kernel_num_rows * kernel_num_cols, num_kernels)
    kernel_matrix: Array2<i64>,
    /// Row coordinate of the top-left corner of each placement of the kernels (can be negative)
    placements_row_idxs: Vec<i32>,
    /// Column coordinate of the top-left corner of each placement of the kernels (can be negative)
    placements_col_idxs: Vec<i32>,
    /// Indices defining the wiring from input image to the matrix multiplicand "A".
    /// Each tuple is (A_row, A_column, image_row, image_column)
    wirings: Vec<(usize, usize, usize, usize)>,
}

/// TODO(ryancao): Replace the hard-coded value for multiple kernels!
///
/// Okay so the plan is that we are going to make this all load in a *single*
/// filter at a time, since the filters are different sizes anyway
pub fn load_kernel_data_iriscode_v3<F: FieldExt>(
    data_directory: PathBuf,
    kernel_idx: usize,
    kernel_type: &str,
) -> IrisCodeV3KernelData {
    // --- Kernel values depend on the kernel number and type ---
    let kernel_values: Array2<i64> = read_npy(&data_directory.join(format!(
        "v3_padded_kernel_{}_{}.npy",
        kernel_idx, kernel_type
    )))
    .unwrap();
    let (kernel_num_rows, kernel_num_cols) = kernel_values.dim();

    // --- Flatten the kernel so we get `B` ---
    let kernel_matrix: Array2<i64> = kernel_values
        .into_shape((1, kernel_num_rows * kernel_num_cols))
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()
        .t()
        .to_owned();

    // --- Read in wirings + probe schema (wirings are the same regardless of kernel type) ---
    let wirings_arr: Array2<i64> =
        read_npy(&data_directory.join(format!("actual_v3_wirings_kernel_{}.npy", kernel_idx)))
            .unwrap();

    // --- The actual wirings are permuted from the Python script stuff ---
    // TODO(ryancao): Get rid of the permutation lol
    let wirings = wirings_arr
        .outer_iter()
        .map(|x| (x[2] as usize, x[3] as usize, x[0] as usize, x[1] as usize))
        .collect_vec();

    // --- Hmmm okay we have an issue here with the whole multiple probe schema rows/cols thing ---
    let probe_schema_rows: Array1<i64> =
        read_npy(&data_directory.join(format!("v3_probe_schema_rows_{}.npy", kernel_idx))).unwrap();

    let probe_schema_cols: Array1<i64> =
        read_npy(&data_directory.join(format!("v3_probe_schema_cols_{}.npy", kernel_idx))).unwrap();

    let placements_row_idxs = probe_schema_rows.map(|x| *x as i32).into_raw_vec();
    let placements_col_idxs = probe_schema_cols.map(|x| *x as i32).into_raw_vec();

    IrisCodeV3KernelData {
        kernel_idx,
        kernel_matrix,
        placements_row_idxs,
        placements_col_idxs,
        wirings,
    }
}

/// We load:
/// - The image
/// - The blinding factors
/// - The commitment
///
/// We compute:
/// - The iris code response
///
/// We (optionally) check:
/// - The manually computed iris code response against the expected iris code response
pub fn load_data_from_serialized_inputs_iriscode_v3<P: PrimeOrderCurve>(
    image_file_path: &str,
    blinding_factors_file_path: &str,
    commitment_file_path: &str,
    maybe_iris_code_file_path: Option<&str>,
    maybe_iris_code_key_value: Option<&str>,
    im_num_rows: usize,
    im_num_cols: usize,
    kernel_data: &IrisCodeV3KernelData,
) -> (Vec<P>, Vec<P::Scalar>, WorldcoinDataV3<P::Scalar>) {
    // --- Load the commitment + blinding factors first ---
    let serialized_commitment_from_file = read_bytes_from_file_bin(commitment_file_path);
    let serialized_blinding_factors_from_file =
        read_bytes_from_file_bin(blinding_factors_file_path);

    let commitment_from_file: Vec<P> =
        deserialize_commitment_from_bytes(&serialized_commitment_from_file);
    let blinding_factors_from_file: Vec<P::Scalar> =
        deserialize_blinding_factors_from_bytes::<P>(&serialized_blinding_factors_from_file);

    // --- Get the image and iris codes from the provided binaries ---
    let image_data = read_bytes_from_file_bin(image_file_path);
    assert_eq!(image_data.len(), im_num_rows * im_num_cols);
    let reshaped_img_from_bytes: Array2<u8> =
        Array2::from_shape_vec((im_num_rows, im_num_cols), image_data).unwrap();

    // dbg!(&reshaped_img_from_bytes);

    // build the rerouted matrix as an array
    let iris_response_num_rows = kernel_data.placements_row_idxs.len();
    let iris_response_num_cols = kernel_data.placements_col_idxs.len();
    let num_placements = iris_response_num_rows * iris_response_num_cols;

    let kernel_num_rows = kernel_data.kernel_matrix.shape()[0];
    let kernel_num_cols = kernel_data.kernel_matrix.shape()[1];
    let mut rerouted_matrix: Array2<i64> =
        Array::zeros((num_placements, kernel_num_rows * kernel_num_cols));

    for wiring_row in kernel_data.wirings.iter() {
        // img_src_row_idx, img_src_col_idx, flattened_result_idx, flattened_dst_idx is the original
        // TODO(ryancao): maybe we should just replace the type of `rerouted_matrix` as well, since `image` is now u8
        rerouted_matrix[[wiring_row.0, wiring_row.1]] =
            reshaped_img_from_bytes[[wiring_row.2, wiring_row.3]] as i64;
    }

    // calculate the matrix product and the iris code
    // both have dimensions (kernel_placements.len(), num_kernels)
    let computed_responses = rerouted_matrix.dot(&kernel_data.kernel_matrix);
    let computed_iris_code = computed_responses.mapv(|x| x > 0);
    let flattened_iris_code: Vec<bool> = computed_iris_code
        .outer_iter()
        .flat_map(|row| row.to_vec())
        .collect();

    // sanity check: load the iris code as calculated in Python, check it's the same
    if let (Some(iris_code_file_path), Some(iris_code_key_value)) =
        (maybe_iris_code_file_path, maybe_iris_code_key_value)
    {
        let iris_code_vec_from_file =
            read_iris_code_from_file_with_key(iris_code_file_path, iris_code_key_value)
                .into_iter()
                .skip(IRIS_CODE_V3_FLATTENED_LENGTH * kernel_data.kernel_idx)
                .take(IRIS_CODE_V3_FLATTENED_LENGTH)
                .collect_vec();
        assert_eq!(computed_iris_code.into_raw_vec(), iris_code_vec_from_file);
    }

    let worldcoin_data_v3: WorldcoinDataV3<P::Scalar> = WorldcoinDataV3 {
        image: reshaped_img_from_bytes,
        responses: computed_responses,
        iris_code: flattened_iris_code,
        _marker: PhantomData,
    };

    (
        commitment_from_file,
        blinding_factors_from_file,
        worldcoin_data_v3,
    )
}

/// TODO(ryancao): This code is SUPER redundant with the `from` function for
/// `WorldcoinData`... Any chance we can just do a simple conversion to the other?
pub fn convert_to_circuit_data<F: FieldExt>(
    image_responses_data: &WorldcoinDataV3<F>,
    kernel_data: &IrisCodeV3KernelData,
) -> WorldcoinCircuitData<F> {
    let mut flattened_input_image: Vec<F> = image_responses_data
        .image
        .outer_iter()
        .map(|x| x.to_vec())
        .flat_map(|row| {
            row.into_iter()
                .map(|elem| F::from(elem as u64))
                .collect_vec()
        })
        .collect_vec();
    let im_len_power_of_2 = flattened_input_image.len().next_power_of_two();
    let padding_amount = im_len_power_of_2 - flattened_input_image.len();
    let padding_values = vec![F::ZERO; padding_amount];
    flattened_input_image.extend(padding_values);
    let image_matrix_mle = flattened_input_image;

    let (_im_num_rows, im_num_cols) = image_responses_data.image.dim();

    let num_placements =
        kernel_data.placements_row_idxs.len() * kernel_data.placements_col_idxs.len();

    let flattened_kernel_matrix: Vec<F> = kernel_data
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
    let (kernel_matrix_rows, _kernel_matrix_cols) = kernel_data.kernel_matrix.dim();
    let kernel_matrix_dims = kernel_data.kernel_matrix.dim();

    let mut reroutings = Vec::new();
    let matrix_a_num_cols = kernel_matrix_rows;
    for (a_row, a_col, im_row, im_col) in &kernel_data.wirings {
        let a_gate_label = a_row * (matrix_a_num_cols) + a_col;
        let im_gate_label = im_row * im_num_cols + im_col;
        reroutings.push((a_gate_label, im_gate_label));
    }

    // Calculate the signed digital decompositions of the data.responses
    let responses_vec_vec: Vec<Vec<i64>> = image_responses_data
        .responses
        .outer_iter()
        .map(|row| row.to_vec())
        .collect_vec();

    // FIXME manually inserting padding - must this be removed for soundness?
    // (Not worried about this for now, since according to Yi Chen, in the next iteration the number of responses will be a power of two anyway.)
    // (The logUp implementation currently requires witness length to be a power of two)
    let pad_length = next_power_of_two(responses_vec_vec.len()).unwrap() - responses_vec_vec.len();
    let num_filters = kernel_data.kernel_matrix.dim().1;
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
    let flattened_iris_code_matrix: Vec<F> = image_responses_data
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
    use std::path::Path;

    use itertools::Itertools;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use remainder_shared_types::halo2curves::bn256::G1 as Bn256Point;
    use remainder_shared_types::Fr;

    use crate::worldcoin::{
        data::WorldcoinCircuitData,
        data_v3::{
            convert_to_circuit_data, load_data_from_serialized_inputs_iriscode_v3,
            load_kernel_data_iriscode_v3, IrisCodeV3KernelData, WorldcoinDataV3,
            IRIS_CODE_V3_IMG_COLS, IRIS_CODE_V3_IMG_ROWS,
        },
    };

    #[test]
    fn test_load_data_iriscode_v3() {
        const NUM_KERNELS: usize = 2;
        for kernel_idx in 0..NUM_KERNELS {
            // --- First load the kernel data ---
            let kernel_data: IrisCodeV3KernelData = load_kernel_data_iriscode_v3::<Fr>(
                Path::new("worldcoin_backfill_test")
                    .join(Path::new("v3_stuff"))
                    .to_path_buf(),
                kernel_idx,
                "imag",
            );
            // --- Next, load the image + responses data ---
            let (_commitment, _blinding_factors, image_responses_data) =
                load_data_from_serialized_inputs_iriscode_v3::<Bn256Point>(
                    "worldcoin_backfill_test/mehmet_pcp_example/left_normalized_image.bin",
                    "worldcoin_backfill_test/mehmet_pcp_example/left_normalized_image_blinding_factors.bin",
                    "worldcoin_backfill_test/mehmet_pcp_example/left_normalized_image_commitment.bin",
                    None,
                    None,
                    IRIS_CODE_V3_IMG_ROWS,
                    IRIS_CODE_V3_IMG_COLS,
                    &kernel_data,
                );
            let _circuit_data = convert_to_circuit_data(&image_responses_data, &kernel_data);

            // --- Sanitychecks ---
            let num_placements =
                kernel_data.placements_row_idxs.len() * kernel_data.placements_col_idxs.len();
            let num_kernels = kernel_data.kernel_matrix.dim().1;
            assert_eq!(
                image_responses_data.responses.dim(),
                (num_placements, num_kernels)
            );
            assert_eq!(
                image_responses_data.iris_code.len(),
                (num_placements * num_kernels)
            );
        }
    }
}
