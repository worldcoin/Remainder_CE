/// The number of rows of the result of the matrix multiplication
pub const MATMULT_NUM_ROWS: usize = 1 << 12;
/// The number of columns of the result of the matrix multiplication
pub const MATMULT_NUM_COLS: usize = 4;
/// The internal dimension of the matrix multiplication
pub const MATMULT_INTERNAL_DIM: usize = 1 << 11;

// Constants defining the digit decomposition of the WC circuit.
const LOG_NUM_DIGITS: usize = 2;
/// The number of digits in the complementary decomposition of the thresholded responses.
pub const NUM_DIGITS: usize = (1 << LOG_NUM_DIGITS) as usize;
/// The base of the complementary decomposition of the thresholded responses.
pub const BASE: u64 = 256;

/// Where to look for the thresholds and the kernel values, i.e. the data that is the same for every
/// run of the circuit.
/// Contains a file `wirings.npy`, a 2d array of type u32 encoding the input `wirings` of
/// [CircuitData::build_worldcoin_circuit_data].
/// Contains two subfolders "iris" and "mask" each containing:
/// + `thresholds.npy` - (i64) the thresholds for each placement and kernel combination (so has shape (num_placements, num_kernels)).
/// + `kernel_values.npy` - (i64) the padded kernel values (quantized)
pub const CONSTANT_DATA_FOLDER: &str = "../worldcoin/v2/";