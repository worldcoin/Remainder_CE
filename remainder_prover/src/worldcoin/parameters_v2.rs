/// The number of variables for the rows of the result of the matrix multiplication
pub const MATMULT_ROWS_NUM_VARS: usize = 12;
/// The number of variables for the columns of the result of the matrix multiplication
pub const MATMULT_COLS_NUM_VARS: usize = 2;
/// The number of internal dimension variables of the matrix multiplication
pub const MATMULT_INTERNAL_DIM_NUM_VARS: usize = 11;

// Constants defining the digit decomposition of the WC circuit.
const LOG_NUM_DIGITS: usize = 2;
/// The number of digits in the complementary decomposition of the thresholded responses.
pub const NUM_DIGITS: usize = (1 << LOG_NUM_DIGITS) as usize;
/// The base of the complementary decomposition of the thresholded responses.
pub const BASE: u64 = 256;

/// Where to look for the thresholds and the kernel values, i.e. the data that is the same for every
/// run of the circuit.
/// Contains a file `wirings.npy`, a 2d array of type u16 encoding the input `wirings` of
/// [remainder::worldcoin::data::CircuitData::build_worldcoin_circuit_data].
/// Contains two subfolders "iris" and "mask" each containing:
/// + `thresholds.npy` - (i64) the thresholds for each placement and kernel combination (so has
///   shape (num_placements, num_kernels)).
/// + `kernel_values.npy` - (i64) the 3d array of kernel values with dimensions (num_kernels,
///   num_kernel_rows, num_kernel_cols).
pub const CONSTANT_DATA_FOLDER: &str = "worldcoin/v2/";

pub static WIRINGS_BYTES: &'static [u8] = include_bytes!("constants/v2/wirings.bin");