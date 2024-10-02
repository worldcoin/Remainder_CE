use remainder::{
    layouter::{
        compiling::LayouterCircuit,
        component::ComponentSet,
        nodes::{
            circuit_inputs::{InputLayerNodeData, InputLayerNode, InputLayerType},
            fiat_shamir_challenge::FiatShamirChallengeNode,
            lookup::{LookupConstraint, LookupTable},
            node_enum::NodeEnum,
            CircuitNode,
        },
    },
    prover::helpers::test_circuit,
};
use remainder_shared_types::Fr;

pub mod utils;
use utils::get_input_shred_and_data_from_vec;

/// Test the case where there is only one LookupConstraint for the LookupTable i.e. just one constrained
/// MLE.
#[test]
pub fn single_shred_test() {
    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (table, table_data) = get_input_shred_and_data_from_vec(
            vec![Fr::from(0u64), Fr::from(1u64)],
            ctx,
            &input_layer,
        );
        let fiat_shamir_challenge_node = FiatShamirChallengeNode::new(ctx, 1);
        let lookup_table = LookupTable::new::<Fr>(ctx, &table, &fiat_shamir_challenge_node);
        let (constrained, constrained_data) = get_input_shred_and_data_from_vec(
            vec![
                Fr::from(0u64),
                Fr::from(1u64),
                Fr::from(1u64),
                Fr::from(1u64),
            ],
            ctx,
            &input_layer,
        );
        let (multiplicities, multiplicities_data) = get_input_shred_and_data_from_vec(
            vec![Fr::from(1u64), Fr::from(3u64)],
            ctx,
            &input_layer,
        );
        let input_data = InputLayerNodeData::new(
            input_layer.id(),
            vec![table_data, constrained_data, multiplicities_data],
            None,
        );
        let lookup_constraint =
            LookupConstraint::new::<Fr>(ctx, &lookup_table, &constrained, &multiplicities);

        let nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            fiat_shamir_challenge_node.into(),
            table.into(),
            lookup_table.into(),
            constrained.into(),
            multiplicities.into(),
            lookup_constraint.into(),
        ];
        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(nodes),
            vec![input_data],
        )
    });
    test_circuit(circuit, None)
}

/// Test the lookup functionality when there are multiple LookupConstraints for the same LookupTable.
#[test]
pub fn multi_shred_test() {
    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (table, table_data) = get_input_shred_and_data_from_vec(
            vec![Fr::from(3u64), Fr::from(4u64)],
            ctx,
            &input_layer,
        );
        let fiat_shamir_challenge_node = FiatShamirChallengeNode::new(ctx, 1);
        let lookup_table = LookupTable::new::<Fr>(ctx, &table, &fiat_shamir_challenge_node);

        let (constrained_0, constrained_0_data) = get_input_shred_and_data_from_vec(
            vec![
                Fr::from(3u64),
                Fr::from(3u64),
                Fr::from(3u64),
                Fr::from(4u64),
            ],
            ctx,
            &input_layer,
        );
        let (multiplicities_0, multiplicities_0_data) = get_input_shred_and_data_from_vec(
            vec![Fr::from(3u64), Fr::from(1u64)],
            ctx,
            &input_layer,
        );
        let lookup_constraint_0 =
            LookupConstraint::new::<Fr>(ctx, &lookup_table, &constrained_0, &multiplicities_0);

        let (constrained_1, constrained_1_data) = get_input_shred_and_data_from_vec(
            vec![
                Fr::from(4u64),
                Fr::from(4u64),
                Fr::from(4u64),
                Fr::from(4u64),
            ],
            ctx,
            &input_layer,
        );
        let (multiplicities_1, multiplicities_1_data) = get_input_shred_and_data_from_vec(
            vec![Fr::from(0u64), Fr::from(4u64)],
            ctx,
            &input_layer,
        );
        let lookup_constraint_1 =
            LookupConstraint::new::<Fr>(ctx, &lookup_table, &constrained_1, &multiplicities_1);

        let (constrained_2, constrained_2_data) = get_input_shred_and_data_from_vec(
            vec![
                Fr::from(4u64),
                Fr::from(3u64),
                Fr::from(3u64),
                Fr::from(4u64),
            ],
            ctx,
            &input_layer,
        );
        let (multiplicities_2, multiplicities_2_data) = get_input_shred_and_data_from_vec(
            vec![Fr::from(2u64), Fr::from(2u64)],
            ctx,
            &input_layer,
        );
        let lookup_constraint_2 =
            LookupConstraint::new::<Fr>(ctx, &lookup_table, &constrained_2, &multiplicities_2);

        let (constrained_3, constrained_3_data) = get_input_shred_and_data_from_vec(
            vec![
                Fr::from(4u64),
                Fr::from(4u64),
                Fr::from(4u64),
                Fr::from(3u64),
            ],
            ctx,
            &input_layer,
        );
        let (multiplicities_3, multiplicities_3_data) = get_input_shred_and_data_from_vec(
            vec![Fr::from(1u64), Fr::from(3u64)],
            ctx,
            &input_layer,
        );
        let lookup_constraint_3 =
            LookupConstraint::new::<Fr>(ctx, &lookup_table, &constrained_3, &multiplicities_3);

        let input_data = InputLayerNodeData::new(
            input_layer.id(),
            vec![
                table_data,
                constrained_0_data,
                multiplicities_0_data,
                constrained_1_data,
                multiplicities_1_data,
                constrained_2_data,
                multiplicities_2_data,
                constrained_3_data,
                multiplicities_3_data,
            ],
            None,
        );

        let nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            fiat_shamir_challenge_node.into(),
            table.into(),
            lookup_table.into(),
            constrained_0.into(),
            multiplicities_0.into(),
            lookup_constraint_0.into(),
            constrained_1.into(),
            multiplicities_1.into(),
            lookup_constraint_1.into(),
            constrained_2.into(),
            multiplicities_2.into(),
            lookup_constraint_2.into(),
            constrained_3.into(),
            multiplicities_3.into(),
            lookup_constraint_3.into(),
        ];
        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(nodes),
            vec![input_data],
        )
    });
    test_circuit(circuit, None)
}

/// Test that a panic occurs when the constrained MLE contains values not in the lookup table.
#[test]
#[should_panic]
pub fn test_not_satisfied() {
    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (table, table_data) = get_input_shred_and_data_from_vec(
            vec![Fr::from(0u64), Fr::from(1u64)],
            ctx,
            &input_layer,
        );
        let fiat_shamir_challenge_node = FiatShamirChallengeNode::new(ctx, 1);
        let lookup_table = LookupTable::new::<Fr>(ctx, &table, &fiat_shamir_challenge_node);
        let (constrained, constrained_data) = get_input_shred_and_data_from_vec(
            vec![
                Fr::from(3u64),
                Fr::from(1u64),
                Fr::from(1u64),
                Fr::from(1u64),
            ],
            ctx,
            &input_layer,
        );
        let (multiplicities, multiplicities_data) = get_input_shred_and_data_from_vec(
            vec![Fr::from(1u64), Fr::from(3u64)],
            ctx,
            &input_layer,
        );
        let lookup_constraint =
            LookupConstraint::new::<Fr>(ctx, &lookup_table, &constrained, &multiplicities);
        let input_layer_data = InputLayerNodeData::new(
            input_layer.id(),
            vec![table_data, constrained_data, multiplicities_data],
            None,
        );

        let nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            fiat_shamir_challenge_node.into(),
            table.into(),
            lookup_table.into(),
            constrained.into(),
            multiplicities.into(),
            lookup_constraint.into(),
        ];
        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(nodes),
            vec![input_layer_data],
        )
    });
    test_circuit(circuit, None)
}
