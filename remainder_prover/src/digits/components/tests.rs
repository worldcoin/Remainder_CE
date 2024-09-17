use super::{BitsAreBinary, ComplementaryRecompChecker, UnsignedRecomposition};
use crate::components::EqualityChecker;
use crate::digits::{complementary_decomposition, digits_to_field};
use crate::layer::LayerId;
use crate::layouter::compiling::LayouterCircuit;
use crate::layouter::component::ComponentSet;
use crate::layouter::nodes::circuit_inputs::{
    InputLayerData, InputLayerNode, InputLayerType, InputShred, InputShredData,
};
use crate::layouter::nodes::circuit_outputs::OutputNode;
use crate::layouter::nodes::node_enum::NodeEnum;
use crate::layouter::nodes::CircuitNode;
use crate::mle::bundled_input_mle::{to_slice_of_vectors, BundledInputMle, FlatMles};
use crate::prover::helpers::test_circuit;
use crate::utils::arithmetic::i64_to_field;
use crate::utils::get_input_shred_and_data_from_vec;
use ark_std::iterable::Iterable;
use itertools::Itertools;
use remainder_shared_types::ff_field;
use remainder_shared_types::Fr;

#[test]
fn test_complementary_recomposition_vertical() {
    let values = [-3, -2, -1, 0, 1, 2, 3, 4];
    let (digits_raw, bits): (Vec<_>, Vec<_>) = values
        .clone()
        .into_iter()
        .map(|value| complementary_decomposition::<2, 2>(value).unwrap())
        .unzip();

    // FlatMles for the digits
    let digits: FlatMles<Fr, 2> = FlatMles::new_from_raw(
        to_slice_of_vectors(digits_raw.iter().map(digits_to_field).collect_vec()),
        LayerId::Input(0),
    );

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (digits_input_shreds, digits_input_shreds_data) =
            digits.make_input_shred_and_data(ctx, &input_layer);
        let digits_refs = digits_input_shreds
            .iter()
            .map(|shred| shred as &dyn CircuitNode)
            .collect_vec();
        let (bits_input_shred, bits_input_shred_data) = get_input_shred_and_data_from_vec(
            bits.iter()
                .map(|b| if *b { Fr::ONE } else { Fr::ZERO })
                .collect(),
            ctx,
            &input_layer,
        );
        let (values_input_shred, values_input_shred_data) = get_input_shred_and_data_from_vec(
            values.iter().map(|value| i64_to_field(value)).collect(),
            ctx,
            &input_layer,
        );

        let recomp = UnsignedRecomposition::new(ctx, &digits_refs, 2);
        let comp_checker = ComplementaryRecompChecker::new(
            ctx,
            &values_input_shred,
            &bits_input_shred,
            &&recomp.sector,
            2,
            2,
        );

        let output = OutputNode::new_zero(ctx, &comp_checker.sector);

        let mut input_shred_data = vec![values_input_shred_data, bits_input_shred_data];
        input_shred_data.extend(digits_input_shreds_data);
        let input_layer_data = InputLayerData::new(input_layer.id(), input_shred_data, None);
        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            values_input_shred.into(),
            bits_input_shred.into(),
            recomp.sector.into(),
            comp_checker.sector.into(),
            output.into(),
        ];
        all_nodes.extend(digits_input_shreds.into_iter().map(|shred| shred.into()));

        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
            vec![input_layer_data],
        )
    });

    test_circuit(circuit, None);
}

#[test]
fn test_unsigned_recomposition() {
    let base: u64 = 16;
    let num_digits = 2;
    let digits = vec![
        vec![
            // MSBs
            Fr::from(1u64),
            Fr::from(0u64),
            Fr::from(2u64),
            Fr::from(3u64),
        ],
        vec![
            // LSBs
            Fr::from(3u64),
            Fr::from(2u64),
            Fr::from(1u64),
            Fr::from(0u64),
        ],
    ];
    assert_eq!(digits.len(), num_digits);
    let expected = vec![Fr::from(19), Fr::from(2), Fr::from(33), Fr::from(48)];

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (digits_input_shreds, digits_input_shreds_data): (
            Vec<InputShred>,
            Vec<InputShredData<Fr>>,
        ) = digits
            .iter()
            .map(|digits_at_place| {
                get_input_shred_and_data_from_vec(digits_at_place.clone(), ctx, &input_layer)
            })
            .unzip();
        let (expected_input_shred, expected_input_shred_data) =
            get_input_shred_and_data_from_vec(expected.clone(), ctx, &input_layer);

        let digits_input_refs = digits_input_shreds
            .iter()
            .map(|shred| shred as &dyn CircuitNode)
            .collect_vec();
        let recomp = UnsignedRecomposition::new(ctx, &digits_input_refs, base);

        let equality_checker = EqualityChecker::new(ctx, &expected_input_shred, &&recomp.sector);
        let output = OutputNode::new_zero(ctx, &equality_checker.sector);

        let mut input_shred_data = vec![expected_input_shred_data];
        input_shred_data.extend(digits_input_shreds_data);
        let input_layer_data = InputLayerData::new(input_layer.id(), input_shred_data, None);

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            expected_input_shred.into(),
            recomp.sector.into(),
            equality_checker.sector.into(),
            output.into(),
        ];

        all_nodes.extend(digits_input_shreds.into_iter().map(|shred| shred.into()));

        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
            vec![input_layer_data],
        )
    });

    test_circuit(circuit, None);
}

#[test]
fn test_complementary_recomposition() {
    let base: u64 = 16;
    let num_digits = 2;
    let base_pow = base.pow(num_digits as u32);
    let digits = vec![
        vec![
            // MSBs
            Fr::from(1u64),
            Fr::from(0u64),
            Fr::from(2u64),
            Fr::from(3u64),
        ],
        vec![
            // LSBs
            Fr::from(3u64),
            Fr::from(2u64),
            Fr::from(1u64),
            Fr::from(0u64),
        ],
    ];
    assert_eq!(digits.len(), num_digits);
    let bits = vec![
        // 1 iff strictly positive
        Fr::from(1u64),
        Fr::from(0u64),
        Fr::from(1u64),
        Fr::from(0u64),
    ];
    let expected = vec![
        Fr::from((base_pow - 19) as u64),
        Fr::from(2u64).neg(),
        Fr::from((base_pow - 33) as u64),
        Fr::from(48u64).neg(),
    ];

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (digits_input_shreds, digits_input_shreds_data): (
            Vec<InputShred>,
            Vec<InputShredData<Fr>>,
        ) = digits
            .iter()
            .map(|digits_at_place| {
                get_input_shred_and_data_from_vec(digits_at_place.clone(), ctx, &input_layer)
            })
            .unzip();
        let (bits_input_shred, bits_input_shred_data) =
            get_input_shred_and_data_from_vec(bits.clone(), ctx, &input_layer);
        let (expected_input_shred, expected_input_shred_data) =
            get_input_shred_and_data_from_vec(expected.clone(), ctx, &input_layer);

        let digits_input_refs = digits_input_shreds
            .iter()
            .map(|shred| shred as &dyn CircuitNode)
            .collect_vec();
        let unsigned_recomp = UnsignedRecomposition::new(ctx, &digits_input_refs, base);

        let signed_recomp_checker = ComplementaryRecompChecker::new(
            ctx,
            &expected_input_shred,
            &bits_input_shred,
            &&unsigned_recomp.sector,
            base,
            num_digits,
        );

        let output = OutputNode::new_zero(ctx, &signed_recomp_checker.sector);

        let mut input_shred_data = vec![bits_input_shred_data, expected_input_shred_data];
        input_shred_data.extend(digits_input_shreds_data);
        let input_layer_data = InputLayerData::new(input_layer.id(), input_shred_data, None);

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            bits_input_shred.into(),
            expected_input_shred.into(),
            unsigned_recomp.sector.into(),
            signed_recomp_checker.sector.into(),
            output.into(),
        ];

        all_nodes.extend(digits_input_shreds.into_iter().map(|shred| shred.into()));

        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
            vec![input_layer_data],
        )
    });

    test_circuit(circuit, None);
}

#[test]
#[should_panic]
fn test_bits_are_binary_soundness() {
    let bits = vec![Fr::from(3u64)];
    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (shred, shred_data) =
            get_input_shred_and_data_from_vec(bits.clone(), ctx, &input_layer);
        let component = BitsAreBinary::new(ctx, &shred);
        let output = OutputNode::new_zero(ctx, &component.sector);
        let input_shred_data = vec![shred_data];
        let input_layer_data = InputLayerData::new(input_layer.id(), input_shred_data, None);
        let all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            shred.into(),
            component.sector.into(),
            output.into(),
        ];
        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
            vec![input_layer_data],
        )
    });
    test_circuit(circuit, None);
}

#[test]
fn test_bits_are_binary() {
    let bits = vec![
        Fr::from(1u64),
        Fr::from(1u64),
        Fr::from(1u64),
        Fr::from(0u64),
    ];
    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (shred, shred_data) =
            get_input_shred_and_data_from_vec(bits.clone(), ctx, &input_layer);
        let component = BitsAreBinary::new(ctx, &shred);
        let output = OutputNode::new_zero(ctx, &component.sector);
        let input_shred_data = vec![shred_data];
        let input_layer_data = InputLayerData::new(input_layer.id(), input_shred_data, None);
        let all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            shred.into(),
            component.sector.into(),
            output.into(),
        ];
        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
            vec![input_layer_data],
        )
    });
    test_circuit(circuit, None);
}
