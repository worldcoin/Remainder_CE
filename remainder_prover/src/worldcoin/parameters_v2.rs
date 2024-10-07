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

/// The number of rows in the image
pub const IM_NUM_ROWS: usize = 100;

/// The number of columns in the image
pub const IM_NUM_COLS: usize = 400;

pub const TO_REROUTE_NUM_VARS: usize = 7 + 9;

/// The wirings from the image (2d) to the LH matrix multiplicand (2d), first a flattened u16 array,
/// then serialized as bytes
pub static WIRINGS: &[u8] = include_bytes!("constants/v2/wirings.bin");

/// The thresholds for the iris circuit, first flattened as a 1d i64 array, then serialized as bytes.
pub static IRIS_THRESHOLDS: &[u8] = include_bytes!("constants/v2/iris/thresholds.bin");

/// The thresholds for the mask circuit, first flattened as a 1d i64 array, then serialized as bytes.
pub static MASK_THRESHOLDS: &[u8] = include_bytes!("constants/v2/mask/thresholds.bin");

/// The RH multiplicand for the iris circuit, first flattened as a 1d i32 array, then serialized as bytes.
pub static IRIS_RH_MULTIPLICAND: &[u8] =
    include_bytes!("constants/v2/iris/rh_multiplicand.bin");

/// The RH multiplicand for the mask circuit, first flattened as a 1d i32 array, then serialized as bytes.
pub static MASK_RH_MULTIPLICAND: &[u8] =
    include_bytes!("constants/v2/mask/rh_multiplicand.bin");
