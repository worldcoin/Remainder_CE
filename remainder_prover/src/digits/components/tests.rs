use std::collections::HashMap;

use super::{BitsAreBinary, ComplementaryRecompChecker, UnsignedRecomposition};
use crate::components::EqualityChecker;
use crate::digits::{complementary_decomposition, digits_to_field};
use crate::layer::LayerId;
use crate::layouter::nodes::circuit_inputs::{InputLayerNode, InputShred};
use crate::layouter::nodes::circuit_outputs::OutputNode;
use crate::layouter::nodes::node_enum::NodeEnum;
use crate::layouter::nodes::{CircuitNode, Context, NodeId};
use crate::mle::bundled_input_mle::{to_slice_of_vectors, BundledInputMle};
use crate::mle::evals::MultilinearExtension;
use crate::prover::generate_circuit_description;
use crate::prover::helpers::test_circuit_new;
use crate::utils::arithmetic::i64_to_field;
use ark_std::iterable::Iterable;
use itertools::Itertools;
use remainder_shared_types::ff_field;
use remainder_shared_types::Fr;

#[test]
fn test_complementary_recomposition_vertical() {
    let values = [-3, -2, -1, 0, 1, 2, 3, 4];
    let (digits_raw, bits): (Vec<_>, Vec<_>) = values
        .into_iter()
        .map(|value| complementary_decomposition::<2, 2>(value).unwrap())
        .unzip();

    // FlatMles for the digits
    let digits: BundledInputMle<Fr, 2> = BundledInputMle::new_from_raw(
        to_slice_of_vectors(digits_raw.iter().map(digits_to_field).collect_vec()),
        LayerId::Input(0),
    );

    let ctx = &Context::new();
    let input_layer = InputLayerNode::new(ctx, None);
    let (digits_input_shreds, digits_input_shreds_data) =
        digits.make_input_shred_and_data(ctx, &input_layer);

    let digits_ids = digits_input_shreds
        .iter()
        .map(|digit_input_shred| digit_input_shred.id())
        .collect_vec();
    let digits_refs = digits_input_shreds
        .iter()
        .map(|shred| shred as &dyn CircuitNode)
        .collect_vec();

    let bits_data = MultilinearExtension::new(
        bits.iter()
            .map(|b| if *b { Fr::ONE } else { Fr::ZERO })
            .collect(),
    );
    let bits_input_shred = InputShred::new(ctx, 3, &input_layer);
    let bits_input_shred_id = bits_input_shred.id();

    let values_data = MultilinearExtension::new(values.iter().map(i64_to_field).collect());
    let values_input_shred = InputShred::new(ctx, 3, &input_layer);
    let values_input_shred_id = values_input_shred.id();

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

    let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
        input_layer.into(),
        values_input_shred.into(),
        bits_input_shred.into(),
        recomp.sector.into(),
        comp_checker.sector.into(),
        output.into(),
    ];
    all_nodes.extend(digits_input_shreds.into_iter().map(|shred| shred.into()));

    let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
        generate_circuit_description(all_nodes).unwrap();

    let input_builder = move |(digits_data, bits_data, values_data): (
        Vec<MultilinearExtension<Fr>>,
        MultilinearExtension<Fr>,
        MultilinearExtension<Fr>,
    )| {
        let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> = HashMap::new();
        assert_eq!(digits_ids.len(), digits_data.len());
        digits_ids
            .iter()
            .zip(digits_data)
            .for_each(|(digit_input_shred_id, digit_data)| {
                input_shred_id_to_data.insert(*digit_input_shred_id, digit_data);
            });

        input_shred_id_to_data.insert(bits_input_shred_id, bits_data);
        input_shred_id_to_data.insert(values_input_shred_id, values_data);
        input_builder_from_shred_map(input_shred_id_to_data).unwrap()
    };

    let inputs = input_builder((digits_input_shreds_data, bits_data, values_data));

    test_circuit_new(&circ_desc, HashMap::new(), &inputs);
}

#[test]
fn test_unsigned_recomposition() {
    let base: u64 = 16;
    let num_digits = 2;
    let digits = [
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

    let ctx = &Context::new();
    let input_layer = InputLayerNode::new(ctx, None);
    let digits_input_shreds: Vec<InputShred> = digits
        .iter()
        .map(|_| InputShred::new(ctx, 2, &input_layer))
        .collect();
    let digits_input_shred_ids = digits_input_shreds
        .iter()
        .map(|input_shred| input_shred.id())
        .collect_vec();
    let expected_input_shred = InputShred::new(ctx, 2, &input_layer);
    let expected_input_shred_id = expected_input_shred.id();

    let digits_input_refs = digits_input_shreds
        .iter()
        .map(|shred| shred as &dyn CircuitNode)
        .collect_vec();
    let recomp = UnsignedRecomposition::new(ctx, &digits_input_refs, base);

    let equality_checker = EqualityChecker::new(ctx, &expected_input_shred, &&recomp.sector);
    let output = OutputNode::new_zero(ctx, &equality_checker.sector);

    let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
        input_layer.into(),
        expected_input_shred.into(),
        recomp.sector.into(),
        equality_checker.sector.into(),
        output.into(),
    ];

    all_nodes.extend(digits_input_shreds.into_iter().map(|shred| shred.into()));
    let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
        generate_circuit_description(all_nodes).unwrap();

    let input_builder = move |(digits_data, expected_data): (
        Vec<MultilinearExtension<Fr>>,
        MultilinearExtension<Fr>,
    )| {
        let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> = HashMap::new();
        assert_eq!(digits_input_shred_ids.len(), digits_data.len());
        digits_input_shred_ids.iter().zip(digits_data).for_each(
            |(digit_input_shred_id, digit_data)| {
                input_shred_id_to_data.insert(*digit_input_shred_id, digit_data);
            },
        );

        input_shred_id_to_data.insert(expected_input_shred_id, expected_data);
        input_builder_from_shred_map(input_shred_id_to_data).unwrap()
    };

    let inputs = input_builder((
        digits
            .into_iter()
            .map(MultilinearExtension::new)
            .collect_vec(),
        MultilinearExtension::new(expected),
    ));

    test_circuit_new(&circ_desc, HashMap::new(), &inputs);
}

#[test]
fn test_complementary_recomposition() {
    let base: u64 = 16;
    let num_digits = 2;
    let base_pow = base.pow(num_digits as u32);
    let digits = [
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
        Fr::from(base_pow - 19),
        Fr::from(2u64).neg(),
        Fr::from(base_pow - 33),
        Fr::from(48u64).neg(),
    ];

    let ctx = &Context::new();
    let input_layer = InputLayerNode::new(ctx, None);
    let digits_input_shreds: Vec<InputShred> = digits
        .iter()
        .map(|_| InputShred::new(ctx, 2, &input_layer))
        .collect();
    let digits_input_shred_ids = digits_input_shreds
        .iter()
        .map(|input_shred| input_shred.id())
        .collect_vec();
    let expected_input_shred = InputShred::new(ctx, 2, &input_layer);
    let expected_input_shred_id = expected_input_shred.id();
    let bits_input_shred = InputShred::new(ctx, 2, &input_layer);
    let bits_input_shred_id = bits_input_shred.id();

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

    let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
        input_layer.into(),
        bits_input_shred.into(),
        expected_input_shred.into(),
        unsigned_recomp.sector.into(),
        signed_recomp_checker.sector.into(),
        output.into(),
    ];

    all_nodes.extend(digits_input_shreds.into_iter().map(|shred| shred.into()));

    let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
        generate_circuit_description(all_nodes).unwrap();

    let input_builder = move |(digits_data, expected_data, bits_data): (
        Vec<MultilinearExtension<Fr>>,
        MultilinearExtension<Fr>,
        MultilinearExtension<Fr>,
    )| {
        let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> = HashMap::new();
        assert_eq!(digits_input_shred_ids.len(), digits_data.len());
        digits_input_shred_ids.iter().zip(digits_data).for_each(
            |(digit_input_shred_id, digit_data)| {
                input_shred_id_to_data.insert(*digit_input_shred_id, digit_data);
            },
        );

        input_shred_id_to_data.insert(expected_input_shred_id, expected_data);
        input_shred_id_to_data.insert(bits_input_shred_id, bits_data);
        input_builder_from_shred_map(input_shred_id_to_data).unwrap()
    };

    let inputs = input_builder((
        digits
            .into_iter()
            .map(MultilinearExtension::new)
            .collect_vec(),
        MultilinearExtension::new(expected),
        MultilinearExtension::new(bits),
    ));

    test_circuit_new(&circ_desc, HashMap::new(), &inputs);
}

#[test]
#[should_panic]
fn test_bits_are_binary_soundness() {
    let bits = vec![Fr::from(3u64)];
    let ctx = &Context::new();
    let input_layer = InputLayerNode::new(ctx, None);
    let bits_input_shred = InputShred::new(ctx, 0, &input_layer);
    let bits_input_shred_id = bits_input_shred.id();
    let component = BitsAreBinary::new(ctx, &bits_input_shred);
    let output = OutputNode::new_zero(ctx, &component.sector);

    let all_nodes: Vec<NodeEnum<Fr>> = vec![
        input_layer.into(),
        bits_input_shred.into(),
        component.sector.into(),
        output.into(),
    ];

    let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
        generate_circuit_description(all_nodes).unwrap();

    let input_builder = move |bits_data: MultilinearExtension<Fr>| {
        let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> = HashMap::new();
        input_shred_id_to_data.insert(bits_input_shred_id, bits_data);
        input_builder_from_shred_map(input_shred_id_to_data).unwrap()
    };

    let inputs = input_builder(MultilinearExtension::new(bits));

    test_circuit_new(&circ_desc, HashMap::new(), &inputs);
}

#[test]
fn test_bits_are_binary() {
    let bits = vec![
        Fr::from(1u64),
        Fr::from(1u64),
        Fr::from(1u64),
        Fr::from(0u64),
    ];
    let ctx = &Context::new();
    let input_layer = InputLayerNode::new(ctx, None);
    let bits_input_shred = InputShred::new(ctx, 2, &input_layer);
    let bits_input_shred_id = bits_input_shred.id();
    let component = BitsAreBinary::new(ctx, &bits_input_shred);
    let output = OutputNode::new_zero(ctx, &component.sector);
    let all_nodes: Vec<NodeEnum<Fr>> = vec![
        input_layer.into(),
        bits_input_shred.into(),
        component.sector.into(),
        output.into(),
    ];
    let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
        generate_circuit_description(all_nodes).unwrap();

    let input_builder = move |bits_data: MultilinearExtension<Fr>| {
        let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> = HashMap::new();
        input_shred_id_to_data.insert(bits_input_shred_id, bits_data);
        input_builder_from_shred_map(input_shred_id_to_data).unwrap()
    };

    let inputs = input_builder(MultilinearExtension::new(bits));

    test_circuit_new(&circ_desc, HashMap::new(), &inputs);
}
