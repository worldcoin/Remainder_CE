use remainder_shared_types::Fr;

/// The dimension of the encoding matrix (row)
pub const ENCODING_MATRIX_NUM_VARS_ROWS: usize = 2;
/// The dimension of the encoding matrix (column)
pub const ENCODING_MATRIX_NUM_VARS_COLS: usize = 2;

/// The modulo of the ring Z/2^16Z (used for our galois ring GR4)
pub const GR4_MODULUS: u64 = 65536;

/// The number of parties that we are secret sharing over
pub const NUM_PARTIES: usize = 3;

/// The number of bits of the element in the ring Z/2^16Z
pub const GR4_ELEM_BIT_LENGTH: u64 = 16;

/// The number of 4 chunks in the iris/mask code for one eye
/// the number of 16384 comes from: 3200 -> padding -> 4096 * 4 = 16384
pub const MPC_NUM_IRIS_4_CHUNKS: usize = 16384 / 4;

/// The actual encoding matrix
/// It's a square matrix, meaning the size of it is 2^(ENCODING_MATRIX_NUM_VARS)^2
/// a.k.a. 2^(ENCODING_MATRIX_NUM_VARS*2)
/// a little hack to use from_raw to make sure this encoding matrix can be declared
/// as a const
pub const ENCODING_MATRIX: [Fr; 1
    << (ENCODING_MATRIX_NUM_VARS_ROWS + ENCODING_MATRIX_NUM_VARS_COLS)] = [
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

/// The actual encoding matrix, but in u64
pub const ENCODING_MATRIX_U64: [u64; 1
    << (ENCODING_MATRIX_NUM_VARS_ROWS + ENCODING_MATRIX_NUM_VARS_COLS)] = [
    1, 0, 0, 0, 58082, 1, 0, 0, 60579, 25194, 1, 0, 17325, 51956, 57011, 1,
];

/// The actual encoding matrix transposed, also in u64
pub const ENCODING_MATRIX_U64_TRANSPOSE: [u64; 1
    << (ENCODING_MATRIX_NUM_VARS_ROWS + ENCODING_MATRIX_NUM_VARS_COLS)] = [
    1, 58082, 60579, 17325, 0, 1, 25194, 51956, 0, 0, 1, 57011, 0, 0, 0, 1,
];

/// The number of wirings for the galois ring GR4 multiplication
/// GR4: GR(2^16, 4) is a Galois extension of Z/2^16Z over the monic
/// polynomial x^4 - x - 1
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
pub const GR4_MULTIPLICATION_WIRINGS: [(u32, u32, u32); GR4_NUM_WIRINGS] = [
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

/// The actual evaluation points used by the three parties
/// They are 1, x, 1 + x
pub const EVALUATION_POINTS_U64: [[u64; 4]; 3] = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]];

/// Test data given by Inversed.
/// Because they give only the masked iris codes, we need to generate random iris codes, and
/// calculate the corresponding mask codes to produce their masked iris codes.
pub const TEST_MASKED_IRIS_CODES: [[u64; 4]; 20] = [
    [0, 0, 1, 1],
    [0, 1, 65535, 0],
    [65535, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [0, 65535, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 1, 1],
    [0, 1, 1, 65535],
    [0, 0, 0, 1],
    [65535, 65535, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [65535, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 65535, 0, 0],
    [0, 1, 0, 0],
    [0, 65535, 0, 0],
];

/// Test data given by Inversed.
/// These are the result of encoding the masked iris code quadruplets into a GR4 element
pub const TEST_GR4_ELEMENTS: [[u64; 4]; 20] = [
    [0, 0, 1, 57012],
    [0, 1, 25193, 60481],
    [65535, 7455, 30151, 34631],
    [0, 0, 1, 57012],
    [1, 58083, 20237, 3745],
    [0, 65535, 40343, 5056],
    [1, 58082, 60579, 17325],
    [1, 58082, 60580, 8801],
    [0, 1, 25195, 43430],
    [0, 0, 0, 1],
    [65535, 7453, 45299, 61791],
    [1, 58083, 20237, 3746],
    [0, 0, 1, 57011],
    [1, 58082, 60579, 17325],
    [1, 58082, 60579, 17325],
    [65535, 7455, 30152, 26107],
    [1, 58082, 60579, 17326],
    [1, 58081, 35385, 30905],
    [0, 1, 25194, 51956],
    [0, 65535, 40342, 13580],
];

/// Test data given by Inversed.
/// These represents the slopes of the one degree polynomial that encodes
/// the secret shares.
pub const TEST_RANDOMNESSES: [[u64; 4]; 20] = [
    [62791, 4515, 39759, 11512],
    [36426, 12484, 23448, 55897],
    [8790, 45739, 57847, 1581],
    [28451, 64021, 7347, 60655],
    [60790, 59010, 16989, 23542],
    [19979, 8251, 55079, 46499],
    [61291, 64096, 12598, 40943],
    [62320, 28246, 40494, 33118],
    [61171, 64822, 29618, 2235],
    [46726, 22873, 29045, 1091],
    [10943, 58396, 27495, 9620],
    [13950, 36926, 41062, 21391],
    [22845, 50864, 2278, 14600],
    [40492, 7720, 50764, 49223],
    [64506, 20573, 50634, 7478],
    [46864, 56026, 58160, 58665],
    [13565, 42432, 52136, 11986],
    [21898, 26161, 4811, 24926],
    [465, 13962, 3685, 38102],
    [61856, 254, 16676, 4017],
];

/// Test data given by Inversed.
/// These represents the expected shares for each of the three parties
pub const TEST_SHARES: [[[u64; 4]; 20]; 3] = [
    [
        [62791, 4515, 39760, 2988],
        [36426, 12485, 48641, 50842],
        [8789, 53194, 22462, 36212],
        [28451, 64021, 7348, 52131],
        [60791, 51557, 37226, 27287],
        [19979, 8250, 29886, 51555],
        [61292, 56642, 7641, 58268],
        [62321, 20792, 35538, 41919],
        [61171, 64823, 54813, 45665],
        [46726, 22873, 29045, 1092],
        [10942, 313, 7258, 5875],
        [13951, 29473, 61299, 25137],
        [22845, 50864, 2279, 6075],
        [40493, 266, 45807, 1012],
        [64507, 13119, 45677, 24803],
        [46863, 63481, 22776, 19236],
        [13566, 34978, 47179, 29312],
        [21899, 18706, 40196, 55831],
        [465, 13963, 28879, 24522],
        [61856, 253, 57018, 17597],
    ],
    [
        [11512, 8767, 4516, 31235],
        [55897, 26788, 37677, 18393],
        [1580, 17826, 10354, 26942],
        [60655, 23570, 64022, 64359],
        [23543, 11343, 13711, 20734],
        [46499, 941, 48594, 60135],
        [40944, 29244, 59139, 29923],
        [33119, 22448, 23290, 49295],
        [2235, 63407, 24481, 7512],
        [1091, 47817, 22873, 29046],
        [9619, 28016, 38159, 23750],
        [21392, 27888, 57163, 44808],
        [14600, 37445, 50865, 59289],
        [49224, 16725, 2763, 2553],
        [7479, 64530, 15616, 2423],
        [58664, 47448, 20642, 18731],
        [11987, 18097, 37475, 3926],
        [24927, 39369, 61546, 35716],
        [38102, 38568, 39156, 55641],
        [4017, 336, 40596, 30256],
    ],
    [
        [8767, 13282, 44275, 42747],
        [26787, 39272, 61125, 8754],
        [10370, 63565, 2665, 28523],
        [23570, 22055, 5833, 59478],
        [18797, 4817, 30700, 44276],
        [942, 9192, 38137, 41098],
        [36699, 27804, 6201, 5330],
        [29903, 50694, 63784, 16877],
        [63406, 62693, 54099, 9747],
        [47817, 5154, 51918, 30137],
        [20562, 20876, 118, 33370],
        [35342, 64814, 32689, 663],
        [37445, 22773, 53143, 8353],
        [24180, 24445, 53527, 51776],
        [6449, 19567, 714, 9901],
        [39992, 37938, 13266, 11860],
        [25552, 60529, 24075, 15912],
        [46825, 65530, 821, 60642],
        [38567, 52530, 42841, 28207],
        [337, 590, 57272, 34273],
    ],
];
