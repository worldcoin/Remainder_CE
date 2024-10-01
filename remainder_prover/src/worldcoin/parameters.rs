// FIXME(Ben) should this file go somewhere else?

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