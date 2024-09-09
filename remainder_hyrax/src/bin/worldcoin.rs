use std::path::Path;

use remainder::worldcoin::{
    circuits::build_circuit,
    data::{load_worldcoin_data, CircuitData},
};

use remainder_shared_types::{
    halo2curves::{bn256::G1 as Bn256Point, group::Group, CurveExt},
    transcript::{
        ec_transcript::{ECTranscriptReader, ECTranscriptWriter},
        poseidon_transcript::PoseidonSponge,
    },
};

use remainder_hyrax::{
    hyrax_gkr::HyraxCircuit, pedersen::PedersenCommitter, utils::vandermonde::VandermondeInverse,
};
type Scalar = <Bn256Point as Group>::Scalar;
type Base = <Bn256Point as CurveExt>::Base;

/// Helper function that runs the Hyrax Worldcoin test against a given data set.
fn test_hyrax_worldcoin<
    const MATMULT_NUM_ROWS: usize,
    const MATMULT_NUM_COLS: usize,
    const MATMULT_INTERNAL_DIM: usize,
    const BASE: u64,
    const NUM_DIGITS: usize,
>(
    data: CircuitData<
        Scalar,
        MATMULT_NUM_ROWS,
        MATMULT_NUM_COLS,
        MATMULT_INTERNAL_DIM,
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

    let mut circuit = build_circuit(data);

    let (hyrax_proof, input_commits, circuit_description) = HyraxCircuit::prove_gkr_circuit(
        &mut circuit,
        &committer,
        None,
        None,
        None,
        blinding_rng,
        converter,
        &mut prover_transcript,
    );

    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());

    HyraxCircuit::verify_gkr_circuit(
        hyrax_proof,
        input_commits,
        &circuit_description,
        &committer,
        &mut verifier_transcript,
    );
}

fn main() {
    use remainder::worldcoin::parameters_v2::{
        BASE, CONSTANT_DATA_FOLDER, MATMULT_INTERNAL_DIM, MATMULT_NUM_COLS, MATMULT_NUM_ROWS,
        NUM_DIGITS,
    };
    let path = Path::new("./").join(CONSTANT_DATA_FOLDER).to_path_buf();
    let image_path = path.join("iris/test_image.npy");
    let data = load_worldcoin_data::<
        Scalar,
        MATMULT_NUM_ROWS,
        MATMULT_NUM_COLS,
        MATMULT_INTERNAL_DIM,
        BASE,
        NUM_DIGITS,
    >(path.clone(), image_path, false);
    test_hyrax_worldcoin(data, 100);
}
