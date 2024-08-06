#[cfg(test)]
mod tests {
    use crate::layouter::compiling::LayouterCircuit;
    use crate::layouter::component::ComponentSet;
    use crate::layouter::nodes::circuit_inputs::{InputLayerNode, InputLayerType};
    use crate::layouter::nodes::circuit_outputs::OutputNode;
    use crate::layouter::nodes::node_enum::NodeEnum;
    use crate::layouter::nodes::ClaimableNode;
    use crate::prover::helpers::test_circuit;
    use crate::utils::get_input_shred_from_vec;
    use crate::worldcoin::circuits::build_circuit;
    use crate::worldcoin::components::{BitsAreBinary, DigitalRecompositionComponent};
    use crate::worldcoin::components::SignCheckerComponent;
    use crate::worldcoin::data::{
        load_data, WorldcoinCircuitData, WorldcoinData,
    };
    use itertools::Itertools;
    use ndarray::{Array2, Array3};
    use remainder_shared_types::Fr;
    use std::path::Path;

    #[test]
    fn test_sign_checker() {
        let values = vec![ Fr::from(3u64), Fr::from(2u64).neg() ];
        let abs_values = vec![ Fr::from(3u64), Fr::from(2u64)];
        let sign_bits = vec![
            Fr::from(1u64), // positive
            Fr::from(0u64), // negative
        ];

        let circuit = LayouterCircuit::new(|ctx| {
            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let abs_values_shred = get_input_shred_from_vec(abs_values.clone(), ctx, &input_layer);
            let sign_bits_input_shred = get_input_shred_from_vec(sign_bits.clone(), ctx, &input_layer);
            let values_input_shred = get_input_shred_from_vec(values.clone(), ctx, &input_layer);

            let sign_checker = SignCheckerComponent::new(
                ctx,
                &values_input_shred,
                &sign_bits_input_shred,
                &abs_values_shred,
            );

            let output = OutputNode::new_zero(ctx, &sign_checker.sector);

            let all_nodes: Vec<NodeEnum<Fr>> = vec![
                input_layer.into(),
                abs_values_shred.into(),
                sign_bits_input_shred.into(),
                values_input_shred.into(),
                sign_checker.sector.into(),
                output.into(),
            ];

            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes)
        });

        test_circuit(circuit, None);
    }

    #[test]
    fn test_recomposition() {
        let base = 16;
        // a length 2 decomposition of four values
        let digits = vec![
            vec![ // MSBs
                Fr::from(1u64),
                Fr::from(0u64),
                Fr::from(2u64),
                Fr::from(3u64),
            ],
            vec![ // LSBs
                Fr::from(3u64),
                Fr::from(2u64),
                Fr::from(1u64),
                Fr::from(0u64),
            ],
        ];
        let sign_bits = vec![ // 1 means positive, 0 means negative
            Fr::from(1u64),
            Fr::from(0u64),
            Fr::from(1u64),
            Fr::from(0u64),
        ];
        let expected = vec![
            Fr::from(19u64),
            Fr::from(2u64).neg(),
            Fr::from(33u64),
            Fr::from(48u64).neg(),
        ];

        let circuit = LayouterCircuit::new(|ctx| {
            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let digits_input_shreds = digits.iter().map(|digits_at_place| get_input_shred_from_vec(digits_at_place.clone(), ctx, &input_layer)).collect_vec();
            let sign_bits_input_shred = get_input_shred_from_vec(sign_bits.clone(), ctx, &input_layer);
            let expected_input_shred = get_input_shred_from_vec(expected.clone(), ctx, &input_layer);

            let digits_input_refs = digits_input_shreds
                .iter()
                .map(|shred| shred as &dyn ClaimableNode<F = Fr>)
                .collect_vec();
            let recomp_of_abs_value = DigitalRecompositionComponent::new(ctx, &digits_input_refs, base);

            let signed_recomp_checker = SignCheckerComponent::new(
                ctx,
                &expected_input_shred,
                &sign_bits_input_shred,
                &recomp_of_abs_value.sector,
            );

            let output = OutputNode::new_zero(ctx, &signed_recomp_checker.sector);

            let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
                input_layer.into(),
                sign_bits_input_shred.into(),
                expected_input_shred.into(),
                recomp_of_abs_value.sector.into(),
                signed_recomp_checker.sector.into(),
                output.into(),
            ];

            all_nodes.extend(digits_input_shreds.into_iter().map(|shred| shred.into()));

            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes)
        });

        test_circuit(circuit, None);
    }

    #[test]
    #[should_panic]
    fn test_bits_are_binary_soundness() {
        let bits = vec![ Fr::from(3u64)];
        let circuit = LayouterCircuit::new(|ctx| {
            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let shred = get_input_shred_from_vec(bits.clone(), ctx, &input_layer);
            let component = BitsAreBinary::new(ctx, &shred);
            let output = OutputNode::new_zero(ctx, &component.sector);
            let all_nodes: Vec<NodeEnum<Fr>> = vec![
                input_layer.into(),
                shred.into(),
                component.sector.into(),
                output.into(),
            ];
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes)
        });
        test_circuit(circuit, None);
    }

    #[test]
    fn test_bits_are_binary() {
        let bits = vec![ Fr::from(1u64), Fr::from(1u64), Fr::from(1u64), Fr::from(0u64), ];
        let circuit = LayouterCircuit::new(|ctx| {
            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let shred = get_input_shred_from_vec(bits.clone(), ctx, &input_layer);
            let component = BitsAreBinary::new(ctx, &shred);
            let output = OutputNode::new_zero(ctx, &component.sector);
            let all_nodes: Vec<NodeEnum<Fr>> = vec![
                input_layer.into(),
                shred.into(),
                component.sector.into(),
                output.into(),
            ];
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes)
        });
        test_circuit(circuit, None);
    }

    /// Generate toy data for the worldcoin circuit.
    /// Image is 2x2, and there are two placements of two 2x1 kernels.
    pub fn toy_worldcoin_circuit_data() -> WorldcoinCircuitData<Fr> {
        let image_shape = (2, 2);
        let kernel_shape = (2, 2, 1);
        let data = WorldcoinData::new(
            Array2::from_shape_vec(image_shape, vec![3, 1, 4, 9]).unwrap(),
            Array3::from_shape_vec(kernel_shape, vec![1, 2, 3, 4]).unwrap(),
            vec![0],
            vec![0, 1],
        );
        dbg!(&data);
        (&data).into()
    }

    #[test]
    fn test_worldcoin_circuit_toy_data() {
        let circuit = build_circuit(toy_worldcoin_circuit_data());
        test_circuit(circuit, Some(Path::new("worldcoin_witness_data/blah.json")));
    }
    
    #[test]
    fn test_worldcoin_circuit() {
        let data: WorldcoinData = load_data(Path::new("worldcoin_witness_data").to_path_buf());
        let circuit = build_circuit::<Fr>((&data).into());
        test_circuit(circuit, None);
    }

}
