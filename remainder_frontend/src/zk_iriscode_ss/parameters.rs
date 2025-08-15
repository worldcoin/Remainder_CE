/// The number of variables of rows of the result of the matrix multiplication
pub const MATMULT_ROWS_NUM_VARS: usize = 8;
/// The number of variables of columns of the result of the matrix multiplication
pub const MATMULT_COLS_NUM_VARS: usize = 2;
/// The number of internal dimension variables of the matrix multiplication
pub const MATMULT_INTERNAL_DIM_NUM_VARS: usize = 10;

// Constants defining the digit decomposition of the WC circuit.
const LOG_NUM_DIGITS: usize = 2;
/// The number of digits in the complementary decomposition of the thresholded responses.
pub const NUM_DIGITS: usize = (1 << LOG_NUM_DIGITS) as usize;
/// The base of the complementary decomposition of the thresholded responses.
pub const BASE: u64 = 256;

/// The number of rows in the image
pub const IM_NUM_ROWS: usize = 128;

/// The number of columns in the image
pub const IM_NUM_COLS: usize = 1024;

/// The log number of columns to use for the Hyrax commitment to the iris/mask code
pub const IRISCODE_COMMIT_LOG_NUM_COLS: usize = 7;

/// The log number of columns to use for the Hyrax commitment to the Shamir secret share
/// polynomial slopes
pub const SHAMIR_SECRET_SHARE_SLOPE_LOG_NUM_COLS: usize = 7;

/// The length of the unpadded iris code
pub const IRISCODE_LEN: usize = 4 * 16 * 256;

/// The number of variables in the MLE getting rerouted (typically the image input)
pub const IM_NUM_VARS: usize =
    (IM_NUM_ROWS.next_power_of_two().ilog2() + IM_NUM_COLS.next_power_of_two().ilog2()) as usize;

/// The wirings from the RLC'd image (2d) to the LH matrix multiplicand (2d), first a flattened u16 array,
/// then serialized as bytes
pub static LH_MATRIX_WIRINGS: &[u8] =
    include_bytes!("constants/v3-split-images/image_strip_to_lh_matrix_wirings.bin");

/// IMAGE STRIPS: the image is decomposed into strips which are RLC'd together.
/// The number of rows in a strip of the image
pub const IM_STRIP_NUM_ROWS: usize = 32;
/// The number of columns in a strip of the image
pub const IM_STRIP_NUM_COLS: usize = 1024;
/// The number of image strips = the number of MLEs being RLC'd
pub const LOG_NUM_STRIPS: usize = 4;

/// The wirings for building the 16 image strips, first a flattened u16 array, then serialized as bytes
pub static IMAGE_STRIP_WIRINGS: [&[u8]; 1 << LOG_NUM_STRIPS] = [
    include_bytes!("constants/v3-split-images/image_strip_wirings_0.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_1.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_2.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_3.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_4.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_5.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_6.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_7.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_8.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_9.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_10.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_11.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_12.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_13.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_14.bin"),
    include_bytes!("constants/v3-split-images/image_strip_wirings_15.bin"),
];

/// The thresholds for the iris circuit, first flattened as a 1d i64 array, then serialized as bytes.
pub static IRIS_THRESHOLDS: &[u8] = include_bytes!("constants/v3-split-images/iris/thresholds.bin");

/// The thresholds for the mask circuit, first flattened as a 1d i64 array, then serialized as bytes.
pub static MASK_THRESHOLDS: &[u8] = include_bytes!("constants/v3-split-images/mask/thresholds.bin");

/// The RH multiplicand for the iris circuit, first flattened as a 1d i32 array, then serialized as bytes.
pub static IRIS_RH_MULTIPLICAND: &[u8] =
    include_bytes!("constants/v3-split-images/iris/rh_multiplicand.bin");

/// The RH multiplicand for the mask circuit, first flattened as a 1d i32 array, then serialized as bytes.
pub static MASK_RH_MULTIPLICAND: &[u8] =
    include_bytes!("constants/v3-split-images/mask/rh_multiplicand.bin");
