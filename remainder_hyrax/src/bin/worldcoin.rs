use std::path::Path;

use remainder::worldcoin::data::{load_worldcoin_data_v2, CircuitData};
use remainder_shared_types::{
    halo2curves::{bn256::G1 as Bn256Point, group::Group, CurveExt},
    transcript::{
        ec_transcript::{ECTranscriptReader, ECTranscriptWriter},
        poseidon_transcript::PoseidonSponge,
    },
};

use remainder_hyrax::{
    hyrax_gkr::HyraxProver, hyrax_worldcoin::build_hyrax_circuit_hyrax_input_layer,
    pedersen::PedersenCommitter, utils::vandermonde::VandermondeInverse,
};
type Scalar = <Bn256Point as Group>::Scalar;
type Base = <Bn256Point as CurveExt>::Base;

/// Helper function that runs the Hyrax Worldcoin test against a given data set
/// with hyrax input layers when data needs to be blinded.
fn test_hyrax_worldcoin_hyrax_input_layer<
    const MATMULT_ROWS_NUM_VARS: usize,
    const MATMULT_COLS_NUM_VARS: usize,
    const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
    const BASE: u64,
    const NUM_DIGITS: usize,
>(
    data: CircuitData<
        Scalar,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >,
    num_generators: usize,
) {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("");
    let blinding_rng = &mut rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    let committer = PedersenCommitter::<Bn256Point>::new(
        num_generators + 1,
        "hi why is this not working, please help me",
        None,
    );
    let mut hyrax_prover = HyraxProver::new(&committer, blinding_rng, converter);

    let witness_function = build_hyrax_circuit_hyrax_input_layer(data, None);

    let (mut circuit_description, hyrax_proof) =
        hyrax_prover.prove_gkr_circuit(witness_function, &mut prover_transcript);

    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());

    hyrax_prover.verify_gkr_circuit(
        &hyrax_proof,
        &mut circuit_description,
        &mut verifier_transcript,
    );
}

fn main() {
    use remainder::worldcoin::parameters_v2::{
        BASE, MATMULT_COLS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    };
    let image_path = Path::new("../worldcoin/v2/iris/test_image.npy").to_path_buf();
    let data = load_worldcoin_data_v2::<
        Scalar,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(image_path, false);
    test_hyrax_worldcoin_hyrax_input_layer(data, 512);
}
