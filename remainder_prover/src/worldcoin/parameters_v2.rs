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

pub const IM_NUM_ROWS: usize = 100;
pub const IM_NUM_COLS: usize = 400;
pub static WIRINGS: &'static [u8] = include_bytes!("constants/v2/wirings.bin"); 
pub static IRIS_THRESHOLDS: &'static [u8] = include_bytes!("constants/v2/iris/thresholds.bin");
pub static MASK_THRESHOLDS: &'static [u8] = include_bytes!("constants/v2/mask/thresholds.bin");
pub static IRIS_RH_MULTIPLICAND: &'static [u8] = include_bytes!("constants/v2/iris/rh_multiplicand.bin");
pub static MASK_RH_MULTIPLICAND: &'static [u8] = include_bytes!("constants/v2/mask/rh_multiplicand.bin");