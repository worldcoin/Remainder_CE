#[cfg(test)]
mod tests {
    use crate::layouter::compiling::LayouterCircuit;
    use crate::layouter::component::Component;
    use crate::layouter::component::ComponentSet;
    use crate::layouter::nodes::circuit_inputs::InputShred;
    use crate::layouter::nodes::circuit_inputs::{InputLayerNode, InputLayerType};
    use crate::layouter::nodes::circuit_outputs::OutputNode;
    use crate::layouter::nodes::identity_gate::IdentityGateNode;
    use crate::layouter::nodes::matmult::MatMultNode;
    use crate::layouter::nodes::node_enum::NodeEnum;
    use crate::layouter::nodes::sector::Sector;
    use crate::layouter::nodes::ClaimableNode;
    use crate::mle::circuit_mle::CircuitMle;
    use crate::mle::evals::MultilinearExtension;
    use crate::prover::helpers::test_circuit;
    use crate::utils::get_input_shred_from_vec;
    use crate::utils::pad_to_nearest_power_of_two;
    use crate::worldcoin::components::DigitRecompComponent;
    use crate::worldcoin::components::IdentityGateComponent;
    use crate::worldcoin::components::SignedRecompComponent;
    // use crate::worldcoin::circuits::{WorldcoinCircuit, WorldcoinCircuitPrecommit};
    use crate::worldcoin::data::{
        load_data, load_data_from_serialized_inputs, WorldcoinCircuitData, WorldcoinData,
    };
    use crate::worldcoin::data_v3::{
        convert_to_circuit_data, load_data_from_serialized_inputs_iriscode_v3,
        load_kernel_data_iriscode_v3, IrisCodeV3KernelData, WorldcoinDataV3, IRIS_CODE_V3_IMG_COLS,
        IRIS_CODE_V3_IMG_ROWS,
    };
    use crate::worldcoin::digit_decomposition::BASE;
    use itertools::Itertools;
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
        let WorldcoinCircuitData {
            mut image_matrix_mle,
            reroutings: wirings,
            num_placements,
            mut kernel_matrix_mle,
            kernel_matrix_dims,
            mut digits,
            mut iris_code,
            mut digit_multiplicities,
        } = (&data).into();
        // let worldcoin_circuit: WorldcoinCircuit<Bn256Point> = WorldcoinCircuit {
        //     worldcoin_circuit_data,
        // };

        let proof_filepath = format!(
            "worldcoin_backfill_test/v3_stuff/live_commit_proof_v2_kernel_{}.json",
            0
        );

        let circuit = LayouterCircuit::new(|ctx| {
            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let input_shred_matrix_a =
                get_input_shred_from_vec(image_matrix_mle.clone(), ctx, &input_layer);

            let matrix_a = IdentityGateNode::new(ctx, &input_shred_matrix_a, wirings.clone());
            let (filter_num_rows, filter_num_cols) = kernel_matrix_dims;
            let matrix_a_num_rows_cols = (num_placements, filter_num_rows);
            let matrix_b = get_input_shred_from_vec(kernel_matrix_mle.clone(), ctx, &input_layer);
            let matrix_b_num_rows_cols = (filter_num_rows, filter_num_cols);

            let result_of_matmult = MatMultNode::new(
                ctx,
                &matrix_a,
                matrix_a_num_rows_cols,
                &matrix_b,
                matrix_b_num_rows_cols,
            );

            let digits_input_shreds = digits.make_input_shreds(ctx, &input_layer);
            let digits_input_refs = digits_input_shreds
                .iter()
                .map(|shred| shred as &dyn ClaimableNode<F = Fr>)
                .collect_vec();
            let recomp_of_abs_value =
                DigitRecompComponent::new(ctx, &digits_input_refs, BASE as u64);

            let iris_code_input_shred = get_input_shred_from_vec(
                pad_to_nearest_power_of_two(iris_code.clone()),
                ctx,
                &input_layer,
            );
            let recomp_check_builder = SignedRecompComponent::new(
                ctx,
                &result_of_matmult,
                &iris_code_input_shred,
                &recomp_of_abs_value.recomp_sector,
            );

            let output = OutputNode::new_zero(ctx, &recomp_check_builder.signed_recomp_sector);

            let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
                input_layer.into(),
                input_shred_matrix_a.into(),
                matrix_a.into(),
                matrix_b.into(),
                result_of_matmult.into(),
                recomp_of_abs_value.recomp_sector.into(),
                iris_code_input_shred.into(),
                recomp_check_builder.signed_recomp_sector.into(),
                output.into(),
            ];

            all_nodes.extend(digits_input_shreds.into_iter().map(|shred| shred.into()));

            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes)
        });

        test_circuit(circuit, None);

        // test_circuit::<Bn256Point, Fr, Fr, WorldcoinCircuit<Bn256Point>>(
        //     worldcoin_circuit,
        //     Some(Path::new(&proof_filepath)),
        // );
    }

    #[test]
    #[ignore]
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
