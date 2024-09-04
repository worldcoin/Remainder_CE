// Constants defining the digit decomposition of the WC circuit.
const WC_LOG_NUM_DIGITS: usize = 2;
/// The number of digits in the complementary decomposition of the thresholded responses.
pub const WC_NUM_DIGITS: usize = (1 << WC_LOG_NUM_DIGITS) as usize;
/// The base of the complementary decomposition of the thresholded responses.
pub const WC_BASE: u64 = 256;

/// The number of kernels
pub const NUM_KERNELS: usize = 4;

/// The number of thresholds (= length of the result of matmult, flattened)
pub const NUM_THRESHOLDS: usize = 1 << 14;

/// Where to look for the thresholds and the kernel values, i.e. the data that is the same for every
/// run of the v3 circuit.
/// Contains two subfolders "iris" and "mask" each containing:
/// + `thresholds.npy` - (i64) the thresholds for each placement and kernel combination (so has shape (num_placements, num_kernels)).
/// + `kernel_values.npy` - (i64) the padded kernel values (quantized)
pub const CONSTANT_DATA_FOLDER: &str = "../worldcoin/v2/";