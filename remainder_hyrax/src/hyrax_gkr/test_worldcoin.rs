use std::path::Path;

use remainder::worldcoin::{
    circuits::build_circuit,
    data::{load_data, WorldcoinCircuitData},
};
use remainder_shared_types::{
    halo2curves::{bn256::G1 as Bn256Point, group::Group, CurveExt},
    transcript::{ec_transcript::ECTranscriptWriter, poseidon_transcript::PoseidonSponge},
    FieldExt, Poseidon,
};

use crate::{pedersen::PedersenCommitter, utils::vandermonde::VandermondeInverse};
type Scalar = <Bn256Point as Group>::Scalar;
type Base = <Bn256Point as CurveExt>::Base;

#[test]
fn test_hyrax_worldcoin_tiny() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test small regular identity matmult circuit");
    let blinding_rng = &mut rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );
    let data: WorldcoinCircuitData<Scalar> =
        load_data(Path::new("worldcoin_witness_data").to_path_buf());
    let circuit = build_circuit(data);
}
