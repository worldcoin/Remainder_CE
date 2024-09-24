// Constants defining the digit decomposition of the WC circuit.
const WC_LOG_NUM_DIGITS: usize = 2;
/// The number of digits in the complementary decomposition of the thresholded responses.
pub const WC_NUM_DIGITS: usize = (1 << WC_LOG_NUM_DIGITS) as usize;
/// The base of the complementary decomposition of the thresholded responses.
pub const WC_BASE: u64 = 256;

/// The row coordinate of the top-left corner of each placement of the kernels (can be negative)
pub const PLACEMENTS_ROW_IDXS: [i32; 16] = [
    -13, -7, 0, 6, 12, 18, 25, 31, 37, 43, 50, 56, 62, 68, 75, 81,
];

/// The column coordinate of the top-left corner of each placement of the kernels (can be negative)
pub const PLACEMENTS_COL_IDXS: [i32; 200] = [
    -32, -30, -28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10,
    12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58,
    60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104,
    106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142,
    144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180,
    182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218,
    220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256,
    258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294,
    296, 298, 300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332,
    334, 336, 338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366,
];

/// The number of kernels
pub const NUM_KERNELS: usize = 4;

/// The number of rows in the kernels (must be a power of two)
pub const NUM_KERNEL_ROWS: usize = 32;

/// The number of columns in the kernels (must be a power of two)
pub const NUM_KERNEL_COLS: usize = 64;

/// The number of thresholds (= length of the result of matmult, flattened)
pub const NUM_THRESHOLDS: usize = 1 << 14;

/// Where to look for the thresholds and the kernel values, i.e. the data that is the same for every
/// run of the v3 circuit.
/// Contains two subfolders "iris" and "mask" each containing:
/// + `thresholds.npy` - (i64) the thresholds for each placement and kernel combination (so has shape (num_placements, num_kernels)).
/// + `padded_kernel_values.npy` - (i64) the padded kernel values (quantized)
pub const CONSTANT_DATA_FOLDER: &str = "worldcoin/v2/";
