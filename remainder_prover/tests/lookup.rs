use remainder::{
    layouter::{
        compiling::LayouterCircuit,
        component::ComponentSet,
        nodes::{
            circuit_inputs::{InputLayerNode, InputLayerType},
            lookup::{LookupNode, LookupShred},
            node_enum::NodeEnum,
            CircuitNode,
        },
    },
    prover::helpers::test_circuit,
};
use remainder_shared_types::Fr;

pub mod utils;
use utils::get_input_shred_from_vec;

/// Test the case where there is only one LookupShred for the LookupNode i.e. just one constrained
/// MLE.
#[test]
pub fn single_shred_test() {
    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let table = get_input_shred_from_vec(vec![Fr::from(0u64), Fr::from(1u64)], ctx);
        let lookup_node = LookupNode::new(ctx, table.id());
        let constrained = get_input_shred_from_vec(
            vec![
                Fr::from(0u64),
                Fr::from(1u64),
                Fr::from(1u64),
                Fr::from(1u64),
            ],
            ctx,
        );
        let multiplicities = get_input_shred_from_vec(vec![Fr::from(1u64), Fr::from(3u64)], ctx);
        let lookup_shred = LookupShred::new(ctx, &lookup_node, &constrained, &multiplicities, false);

        let nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            table.into(),
            lookup_node.into(),
            constrained.into(),
            multiplicities.into(),
            lookup_shred.into(),
        ];
        ComponentSet::<NodeEnum<Fr>>::new_raw(nodes)
    });
    test_circuit(circuit, None)
}

/// Test the lookup functionality when there are multiple LookupShreds for the same LookupNode.
#[test]
pub fn multi_shred_test() {
    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let table = get_input_shred_from_vec(vec![Fr::from(3u64), Fr::from(4u64)], ctx);
        let lookup_node = LookupNode::new(ctx, table.id());

        let constrained_0 = get_input_shred_from_vec(
            vec![
                Fr::from(3u64),
                Fr::from(3u64),
                Fr::from(3u64),
                Fr::from(4u64),
            ],
            ctx,
        );
        let multiplicities_0 = get_input_shred_from_vec(vec![Fr::from(3u64), Fr::from(1u64)], ctx);
        let lookup_shred_0 = LookupShred::new(ctx, &lookup_node, &constrained_0, &multiplicities_0, false);

        let constrained_1 = get_input_shred_from_vec(
            vec![
                Fr::from(4u64),
                Fr::from(4u64),
                Fr::from(4u64),
                Fr::from(4u64),
            ],
            ctx,
        );
        let multiplicities_1 = get_input_shred_from_vec(vec![Fr::from(0u64), Fr::from(4u64)], ctx);
        let lookup_shred_1 = LookupShred::new(ctx, &lookup_node, &constrained_1, &multiplicities_1, false);

        let constrained_2 = get_input_shred_from_vec(
            vec![
                Fr::from(4u64),
                Fr::from(3u64),
                Fr::from(3u64),
                Fr::from(4u64),
            ],
            ctx,
        );
        let multiplicities_2 = get_input_shred_from_vec(vec![Fr::from(2u64), Fr::from(2u64)], ctx);
        let lookup_shred_2 = LookupShred::new(ctx, &lookup_node, &constrained_2, &multiplicities_2, false);

        let constrained_3 = get_input_shred_from_vec(
            vec![
                Fr::from(4u64),
                Fr::from(4u64),
                Fr::from(4u64),
                Fr::from(3u64),
            ],
            ctx,
        );
        let multiplicities_3 = get_input_shred_from_vec(vec![Fr::from(1u64), Fr::from(3u64)], ctx);
        let lookup_shred_3 = LookupShred::new(ctx, &lookup_node, &constrained_3, &multiplicities_3, false);

        let nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            table.into(),
            lookup_node.into(),
            constrained_0.into(),
            multiplicities_0.into(),
            lookup_shred_0.into(),
            constrained_1.into(),
            multiplicities_1.into(),
            lookup_shred_1.into(),
            constrained_2.into(),
            multiplicities_2.into(),
            lookup_shred_2.into(),
            constrained_3.into(),
            multiplicities_3.into(),
            lookup_shred_3.into(),
        ];
        ComponentSet::<NodeEnum<Fr>>::new_raw(nodes)
    });
    test_circuit(circuit, None)
}

/// Test that a panic occurs when the constrained MLE contains values not in the lookup table.
#[test]
#[should_panic]
pub fn test_not_satisfied() {
    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let table = get_input_shred_from_vec(vec![Fr::from(0u64), Fr::from(1u64)], ctx);
        let lookup_node = LookupNode::new(ctx, table.id());
        let constrained = get_input_shred_from_vec(
            vec![
                Fr::from(3u64),
                Fr::from(1u64),
                Fr::from(1u64),
                Fr::from(1u64),
            ],
            ctx,
        );
        let multiplicities = get_input_shred_from_vec(vec![Fr::from(1u64), Fr::from(3u64)], ctx);
        let lookup_shred = LookupShred::new(ctx, &lookup_node, &constrained, &multiplicities, false);

        let nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            table.into(),
            lookup_node.into(),
            constrained.into(),
            multiplicities.into(),
            lookup_shred.into(),
        ];
        ComponentSet::<NodeEnum<Fr>>::new_raw(nodes)
    });
    test_circuit(circuit, None)
}
