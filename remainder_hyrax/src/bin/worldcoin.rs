use std::path::Path;

use ark_std::{end_timer, start_timer};
use remainder::worldcoin::{
    circuits::build_circuit,
    data::{load_data, medium_worldcoin_data, tiny_worldcoin_data, WorldcoinCircuitData},
};
use remainder::worldcoin::{WC_BASE, WC_NUM_DIGITS};
use remainder_shared_types::{
    halo2curves::{bn256::G1 as Bn256Point, group::Group, CurveExt},
    transcript::{
        ec_transcript::{ECTranscriptReader, ECTranscriptWriter},
        poseidon_transcript::PoseidonSponge,
        test_transcript::TestSponge,
    },
    FieldExt, Poseidon,
};

use remainder_hyrax::{
    hyrax_gkr::HyraxCircuit, pedersen::PedersenCommitter, utils::vandermonde::VandermondeInverse,
};
type Scalar = <Bn256Point as Group>::Scalar;
type Base = <Bn256Point as CurveExt>::Base;

/// Helper function that runs the Hyrax Worldcoin test against a given data set.
fn test_hyrax_worldcoin<const BASE: u16, const NUM_DIGITS: usize>(
    data: WorldcoinCircuitData<Scalar, BASE, NUM_DIGITS>,
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
    let data: WorldcoinCircuitData<Scalar, WC_BASE, WC_NUM_DIGITS> =
        load_data(Path::new("worldcoin_witness_data").to_path_buf(), false);
    test_hyrax_worldcoin(data, 100);
}
