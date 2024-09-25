use ndarray::Array2;

pub struct Parameters {
    /// The number of variables for the rows of the result of the matrix multiplication
    /// In iris code: rows index kernel placements.
    pub matmult_rows_num_vars: usize,
    /// The number of variables for the columns of the result of the matrix multiplication
    /// In iris code: columns index kernels.
    pub matmult_cols_num_vars: usize,
    /// The number of internal dimension variables of the matrix multiplication
    /// In iris code: the internal dimension indexes the values of the kernel.
    pub matmult_internal_dim_num_vars: usize,
    /// The number of digits in the complementary decomposition of the thresholded responses.
    pub num_digits: usize,
    /// The base of the complementary decomposition of the thresholded responses.
    pub base: u64,
    /// A flattened 2d array of u16s encoding the input `wirings` of [remainder::worldcoin::data::CircuitData::build_worldcoin_circuit_data].
    pub wirings_bytes: &'static [u8],
    /// A flattened 2d array of i64s encoding the thresholds for each placement and kernel combination (so has shape `(num_placements, num_kernels)`).
    pub thresholds_bytes: &'static [u8],
    /// A flattened 2d array of i64s encoding the right-hand multiplicand for the matrix multiplication.
    pub rh_multiplicand_bytes: &'static [u8],
    // FIXME(Ben)
    /// The number of rows in the input image.
    pub im_num_rows: usize,
    // FIXME(Ben)
    /// The number of columns in the input image.
    pub im_num_cols: usize,
}

pub const V2_IRIS: Parameters = Parameters {
    matmult_rows_num_vars: 12,
    matmult_cols_num_vars: 2,
    matmult_internal_dim_num_vars: 11,
    num_digits: 4,
    base: 256,
    wirings_bytes: include_bytes!("constants/v2/wirings.bin"),
    thresholds_bytes: include_bytes!("constants/v2/iris/thresholds.bin"),
    rh_multiplicand_bytes: include_bytes!("constants/v2/iris/rh_multiplicand.bin"),
    im_num_rows: 100,
    im_num_cols: 400,
};

pub const V2_MASK: Parameters = Parameters {
    matmult_rows_num_vars: 12,
    matmult_cols_num_vars: 2,
    matmult_internal_dim_num_vars: 11,
    num_digits: 4,
    base: 256,
    wirings_bytes: include_bytes!("constants/v2/wirings.bin"),
    thresholds_bytes: include_bytes!("constants/v2/mask/thresholds.bin"),
    rh_multiplicand_bytes: include_bytes!("constants/v2/mask/rh_multiplicand.bin"),
    im_num_rows: 100,
    im_num_cols: 400,
};

pub const V3_IRIS: Parameters = Parameters {
    matmult_rows_num_vars: 12,
    matmult_cols_num_vars: 2,
    matmult_internal_dim_num_vars: 11,
    num_digits: 4,
    base: 256,
    wirings_bytes: include_bytes!("constants/v3/wirings.bin"),
    thresholds_bytes: include_bytes!("constants/v3/iris/thresholds.bin"),
    rh_multiplicand_bytes: include_bytes!("constants/v3/iris/rh_multiplicand.bin"),
    im_num_rows: 128,
    im_num_cols: 1024,
};

pub const V3_MASK: Parameters = Parameters {
    matmult_rows_num_vars: 12,
    matmult_cols_num_vars: 2,
    matmult_internal_dim_num_vars: 11,
    num_digits: 4,
    base: 256,
    wirings_bytes: include_bytes!("constants/v3/wirings.bin"),
    thresholds_bytes: include_bytes!("constants/v3/mask/thresholds.bin"),
    rh_multiplicand_bytes: include_bytes!("constants/v3/mask/rh_multiplicand.bin"),
    im_num_rows: 128,
    im_num_cols: 1024,
};

// FIXME(Ben) document
pub fn decode_wirings(wirings_bytes: &[u8]) -> Vec<(u16, u16, u16, u16)> {
    assert!(wirings_bytes.len() % 8 == 0); // 4 u16s
    // Process the data in chunks of 8 bytes (one row)
    wirings_bytes
        .chunks_exact(8)
        .map(|row_bytes| {
            // Convert each pair of bytes to an u16 value
            let a = u16::from_le_bytes(row_bytes[0..2].try_into().unwrap());
            let b = u16::from_le_bytes(row_bytes[2..4].try_into().unwrap());
            let c = u16::from_le_bytes(row_bytes[4..6].try_into().unwrap());
            let d = u16::from_le_bytes(row_bytes[6..8].try_into().unwrap());
            (a, b, c, d)
        })
        .collect()
}

// FIXME(Ben)
pub fn decode_thresholds(parameters: Parameters) -> Array2<i64> {
    let vec_i64 = decode_i64(parameters.thresholds_bytes);
    let num_rows = 1 << parameters.matmult_rows_num_vars;
    let num_cols = 1 << parameters.matmult_cols_num_vars;
    Array2::from_shape_vec((num_rows, num_cols), vec_i64).unwrap()
}

// FIXME(Ben)
pub fn decode_rh_multiplicand(parameters: Parameters) -> Array2<i64> {
    let vec_i64 = decode_i64(parameters.rh_multiplicand_bytes);
    let num_rows = 1 << parameters.matmult_internal_dim_num_vars;
    let num_cols = 1 << parameters.matmult_cols_num_vars;
    Array2::from_shape_vec((num_rows, num_cols), vec_i64).unwrap()
}

// FIXME(Ben)
fn decode_i64(bytes: &[u8]) -> Vec<i64> {
    assert!(bytes.len() % 8 == 0); // 8 bytes per i64
    bytes
        .chunks_exact(8)
        .map(|row_bytes| {
            i64::from_le_bytes(row_bytes.try_into().unwrap())
        })
        .collect()
}