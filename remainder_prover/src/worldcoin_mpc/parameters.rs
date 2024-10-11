use remainder_shared_types::Fr;

/// The dimension of the encoding matrix
pub const ENCODING_MATRIX_NUM_VARS: usize = 2;

/// The number of length-4 chunks of the iris code
pub const IRIS_DATAPARALLEL_NUM_VARS: usize = 3200;

/// The modulo of the ring Z/2^16Z (used for our galois ring GR4)
pub const GR4_MODULUS: u64 = 65536;

/// The number of bits of the element in the ring Z/2^16Z
pub const GR4_ELEM_BIT_LENGTH: u64 = 16;

/// The actual encoding matrix
/// It's a square matrix, meaning the size of it is 2^(ENCODING_MATRIX_NUM_VARS)^2
/// a.k.a. 2^(ENCODING_MATRIX_NUM_VARS*2)
/// a little hack to use from_raw to make sure this encoding matrix can be declared
/// as a const
pub const ENCODING_MATRIX: [Fr; 1 << (ENCODING_MATRIX_NUM_VARS * 2)] = [
    Fr::one(),
    Fr::zero(),
    Fr::zero(),
    Fr::zero(),
    Fr::from_raw([58082, 0, 0, 0]),
    Fr::one(),
    Fr::zero(),
    Fr::zero(),
    Fr::from_raw([60579, 0, 0, 0]),
    Fr::from_raw([25194, 0, 0, 0]),
    Fr::one(),
    Fr::zero(),
    Fr::from_raw([17325, 0, 0, 0]),
    Fr::from_raw([51956, 0, 0, 0]),
    Fr::from_raw([57011, 0, 0, 0]),
    Fr::one(),
];

/// The number of wirings for the galois ring GR4 multiplication
/// GR4: GR(2^16, 4) is a Galois extension of Z/2^16Z over the monic
/// polynimial x^4 - x - 1
/// The formula for multiplying two GR4 ring elements is:
/// say a = a0, a1, a2, a3
/// and b = b0, b1, b2, b3
/// then a * b = [ a3*b1 + a2*b2 + a1*b3 + a0*b0,
///                a3*b2 + a2*b3 + a3*b1 + a2*b2 + a1*b3 + a1*b0 + a0*b1,
///                a3*b3 + a3*b2 + a2*b3 + a2*b0 + a1*b1 + a0*b2,
///                a3*b3 + a3*b0 + a2*b1 + a1*b2 + a0*b3]
/// Thus, the number of wirings for each coefficient is: sum(4, 7, 6, 5) = 22
pub const GR4_NUM_WIRINGS: usize = 22;

/// The actual wirings for the galois ring GR4 multiplication
pub const GR4_MULTIPLICATION_WIRINGS: [(usize, usize, usize); GR4_NUM_WIRINGS] = [
    // a*b[0]
    (0, 0, 0),
    (0, 1, 3),
    (0, 2, 2),
    (0, 3, 1),
    // a*b[1]
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 3),
    (1, 2, 2),
    (1, 3, 1),
    (1, 2, 3),
    (1, 3, 2),
    // a*b[2]
    (2, 0, 2),
    (2, 1, 1),
    (2, 2, 0),
    (2, 2, 3),
    (2, 3, 2),
    (2, 3, 3),
    // a*b[4]
    (3, 0, 3),
    (3, 1, 2),
    (3, 2, 1),
    (3, 3, 0),
    (3, 3, 3),
];
