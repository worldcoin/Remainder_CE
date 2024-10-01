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
pub fn decode_i32_array(bytes: &[u8]) -> Vec<i32> {
    assert!(bytes.len() % 4 == 0); // 4 bytes per i32
    bytes
        .chunks_exact(4)
        .map(|row_bytes| {
            i32::from_le_bytes(row_bytes.try_into().unwrap())
        })
        .collect()
}

// FIXME(Ben)
pub fn decode_i64_array(bytes: &[u8]) -> Vec<i64> {
    assert!(bytes.len() % 8 == 0); // 8 bytes per i64
    bytes
        .chunks_exact(8)
        .map(|row_bytes| {
            i64::from_le_bytes(row_bytes.try_into().unwrap())
        })
        .collect()
}