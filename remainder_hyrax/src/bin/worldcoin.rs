// use std::path::Path;

// use remainder::worldcoin::{data::{load_worldcoin_data_v2, wirings_to_reroutings, IriscodeCircuitData}, parameters::decode_wirings};
// use remainder_shared_types::{
//     halo2curves::{bn256::G1 as Bn256Point, group::Group, CurveExt},
//     transcript::{
//         ec_transcript::{ECTranscriptReader, ECTranscriptWriter},
//         poseidon_transcript::PoseidonSponge,
//     },
// };

// use remainder_hyrax::{
//     hyrax_gkr::HyraxProver, hyrax_worldcoin::build_hyrax_circuit_hyrax_input_layer,
//     pedersen::PedersenCommitter, utils::vandermonde::VandermondeInverse,
// };
// type Scalar = <Bn256Point as Group>::Scalar;
// type Base = <Bn256Point as CurveExt>::Base;

// use remainder_hyrax::hyrax_worldcoin::test_worldcoin::{test_hyrax_worldcoin_public_input_layer, test_hyrax_worldcoin_hyrax_input_layer};

fn main() {
    //FIXME(Ben)
    // use remainder::worldcoin::parameters_v2::{
    //     IM_NUM_COLS,
    //     BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
    //     MATMULT_ROWS_NUM_VARS, NUM_DIGITS, WIRINGS
    // };
    // let image_path = Path::new("../remainder_prover/src/worldcoin/constants/v2/iris/test_image.npy").to_path_buf();
    // let data = load_worldcoin_data_v2::<
    //     Scalar,
    //     MATMULT_ROWS_NUM_VARS,
    //     MATMULT_COLS_NUM_VARS,
    //     MATMULT_INTERNAL_DIM_NUM_VARS,
    //     BASE,
    //     NUM_DIGITS,
    // >(image_path, false);
    // let reroutings = wirings_to_reroutings(&decode_wirings(WIRINGS), IM_NUM_COLS, 1 << MATMULT_INTERNAL_DIM_NUM_VARS);
    // test_hyrax_worldcoin_hyrax_input_layer(data, reroutings, 512);
}
