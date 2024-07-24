#[cfg(test)]
mod tests {
    use crate::prover::helpers::test_circuit;
    // use crate::worldcoin::circuits::{WorldcoinCircuit, WorldcoinCircuitPrecommit};
    use crate::worldcoin::data::{
        load_data, load_data_from_serialized_inputs, WorldcoinCircuitData, WorldcoinData,
    };
    use crate::worldcoin::data_v3::{
        convert_to_circuit_data, load_data_from_serialized_inputs_iriscode_v3,
        load_kernel_data_iriscode_v3, IrisCodeV3KernelData, WorldcoinDataV3, IRIS_CODE_V3_IMG_COLS,
        IRIS_CODE_V3_IMG_ROWS,
    };
    use remainder_shared_types::halo2curves::bn256::G1 as Bn256Point;
    use remainder_shared_types::Fr;
    use std::marker::PhantomData;
    use std::path::Path;

    use chrono;
    use log::LevelFilter;
    use std::io::Write;

    #[test]
    fn test_worldcoin_circuit() {
        env_logger::Builder::new()
            .format(|buf, record| {
                writeln!(
                    buf,
                    "----> {}:{} {} [{}]:\n{}",
                    record.file().unwrap_or("unknown"),
                    record.line().unwrap_or(0),
                    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
                    record.level(),
                    record.args()
                )
            })
            .filter(None, LevelFilter::Error)
            .init();

        // --- This is for V2 stuff ---
        let data: WorldcoinData<Fr> = load_data(Path::new("worldcoin_witness_data").to_path_buf());
        let worldcoin_circuit_data: WorldcoinCircuitData<Fr> = (&data).into();
        // let worldcoin_circuit: WorldcoinCircuit<Bn256Point> = WorldcoinCircuit {
        //     worldcoin_circuit_data,
        // };

        let proof_filepath = format!(
            "worldcoin_backfill_test/v3_stuff/live_commit_proof_v2_kernel_{}.json",
            0
        );
        // test_circuit::<Bn256Point, Fr, Fr, WorldcoinCircuit<Bn256Point>>(
        //     worldcoin_circuit,
        //     Some(Path::new(&proof_filepath)),
        // );
    }

    #[test]
    fn test_worldcoin_circuit_precommit() {
        env_logger::Builder::new()
            .format(|buf, record| {
                writeln!(
                    buf,
                    "----> {}:{} {} [{}]:\n{}",
                    record.file().unwrap_or("unknown"),
                    record.line().unwrap_or(0),
                    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
                    record.level(),
                    record.args()
                )
            })
            .filter(None, LevelFilter::Error)
            .init();

        // --- V2 stuff ---
        let (commitment, blinding_factors, data): (Vec<Bn256Point>, Vec<Fr>, WorldcoinData<Fr>) =
            load_data_from_serialized_inputs::<Bn256Point>(
                "worldcoin_witness_data/right_normalized_image_resized.bin",
                "worldcoin_witness_data/right_normalized_image_blinding_factors_resized.bin",
                "worldcoin_witness_data/right_normalized_image_commitment_resized.bin",
                "worldcoin_witness_data/iris_codes.json",
                "right_iris_code",
                Path::new("worldcoin_witness_data").to_path_buf(),
                (100) as usize,
                (400) as usize,
            );
        let worldcoin_circuit_data: WorldcoinCircuitData<Fr> = (&data).into();

        // let worldcoin_circuit_with_precommit: WorldcoinCircuitPrecommit<Bn256Point> =
        //     WorldcoinCircuitPrecommit {
        //         worldcoin_circuit_data,
        //         hyrax_precommit: commitment,
        //         blinding_factors_matrix: blinding_factors,
        //         log_num_cols: 9,
        //     };

        // test_circuit::<Bn256Point, Fr, Fr, WorldcoinCircuitPrecommit<Bn256Point>>(
        //     worldcoin_circuit_with_precommit,
        //     Some(Path::new("worldcoin_witness_data/proof_precommit_2.json")),
        // );
    }
}
