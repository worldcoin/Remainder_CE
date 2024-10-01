// FIXME(Ben) should this file go somewhere else?  are the doc comments below better than those in v2, v3?

    /// The number of variables for the rows of the result of the matrix multiplication
    /// In iris code: rows index kernel placements.
    
    /// The number of variables for the columns of the result of the matrix multiplication
    /// In iris code: columns index kernels.
    
    /// The number of internal dimension variables of the matrix multiplication
    /// In iris code: the internal dimension indexes the values of the kernel.
    
    /// The number of digits in the complementary decomposition of the thresholded responses.
    
    /// The base of the complementary decomposition of the thresholded responses.
    
    /// A flattened 2d array of u16s encoding the input `wirings` of [remainder::worldcoin::data::CircuitData::build_worldcoin_circuit_data].

// FIXME(Ben)
#[allow(long_running_const_eval)]
pub const fn static_decode_wirings<const N: usize>(bytes: &[u8]) -> [(u16, u16, u16, u16); N] {
    let mut result = [(0, 0, 0, 0); N];
    let mut row_idx = 0;
    while row_idx < N {
        // index into the bytes array
        let idx = 2 * (row_idx * 4);
        result[row_idx].0 = (bytes[idx + 0] as u16) | ((bytes[idx + 1] as u16) << 8);
        result[row_idx].1 = (bytes[idx + 2] as u16) | ((bytes[idx + 3] as u16) << 8);
        result[row_idx].2 = (bytes[idx + 4] as u16) | ((bytes[idx + 5] as u16) << 8);
        result[row_idx].3 = (bytes[idx + 6] as u16) | ((bytes[idx + 7] as u16) << 8);
        row_idx += 1;
    }
    result
}

// FIXME(Ben)
pub const fn static_decode_i64_slice<const N: usize>(bytes: &[u8]) -> [i64; N] {
    let mut result = [0; N];
    let mut out_idx = 0;
    while out_idx < N {
        // index into the bytes array
        let idx = 8 * out_idx;
        result[out_idx] = (bytes[idx + 0] as i64) | ((bytes[idx + 1] as i64) << 8) | ((bytes[idx + 2] as i64) << 16) | ((bytes[idx + 3] as i64) << 24) | ((bytes[idx + 4] as i64) << 32) | ((bytes[idx + 5] as i64) << 40) | ((bytes[idx + 6] as i64) << 48) | ((bytes[idx + 7] as i64) << 56);
        out_idx += 1;
    }
    result
}

// FIXME(Ben) document
pub fn decode_wirings(parameters: &Parameters) -> Vec<(u16, u16, u16, u16)> {
    assert!(parameters.wirings_bytes.len() % 8 == 0); // 4 u16s
    // Process the data in chunks of 8 bytes (one row)
    parameters.wirings_bytes
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
pub fn decode_thresholds(parameters: &Parameters) -> Array2<i64> {
    let vec_i64 = decode_i64(parameters.thresholds_bytes);
    let num_cols = 1 << parameters.matmult_cols_num_vars;
    let num_rows = vec_i64.len() / num_cols;
    Array2::from_shape_vec((num_rows, num_cols), vec_i64).unwrap()
}

// FIXME(Ben)
pub fn decode_rh_multiplicand(parameters: &Parameters) -> Array2<i64> {
    let vec_i64 = decode_i64(parameters.rh_multiplicand_bytes);
    let num_cols = 1 << parameters.matmult_cols_num_vars;
    let num_rows = vec_i64.len() / num_cols;
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