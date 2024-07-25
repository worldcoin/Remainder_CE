use std::collections::HashMap;

use remainder_shared_types::Fr;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layouter::{layouting::DAGError, nodes::NodeId},
    mle::evals::MultilinearExtension,
};

use crate::layouter::nodes::CircuitNode;

use super::{
    super::nodes::{
        circuit_inputs::{InputLayerNode, InputLayerType, InputShred},
        circuit_outputs::OutputNode,
        debug::DebugNode,
        node_enum::NodeEnum,
        sector::Sector,
        Context,
    },
    topo_sort,
};

#[test]
fn test_topo_sort_with_cycle_include_children() {
    //    input_layer
    //         /\
    //   input_shred_0        input_shred_3    ->    debug_node_0 -> debug_node_1 -> debug_node_0
    //   input_shred_1        input_shred_4
    //   input_shred_2
    //
    //
    //                                         ->   sector_0 ->   sector_1  ->output_0
    //                                         ->   sector_2->  output_1
    //
    //
    //
    //
    let ctx = Context::new();
    let dummy_data = MultilinearExtension::new_zero();

    let input_node = InputLayerNode::new(&ctx, None, InputLayerType::PublicInputLayer);

    // node ids: [1, 2, 3]
    let input_shred_0 = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_1 = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_2 = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_vec = vec![
        input_shred_0.clone(),
        input_shred_1.clone(),
        input_shred_2.clone(),
    ];
    // node id: [4]
    let input_layer_node = InputLayerNode::new(
        &ctx,
        Some(input_shred_vec),
        InputLayerType::PublicInputLayer,
    );

    // node ids: [5, 6]
    let input_shred_3_node = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_4_node = InputShred::new(&ctx, dummy_data.clone(), &input_node);

    // for now, use debug node as the fake gate layer
    // node id: [7]
    let mut debug_node_0 = DebugNode::new(
        &ctx,
        "debug".to_string(),
        &[&input_layer_node, &input_shred_3_node],
    );

    // node id: [8]
    let sector_0_node = Sector::new(
        &ctx,
        &[&input_shred_0.clone(), &input_shred_1.clone()],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
        |_data| MultilinearExtension::new_zero(),
    );

    // node id: [9]
    let sector_1_node = Sector::new(
        &ctx,
        &[&input_shred_2, &input_shred_4_node, &sector_0_node],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
        |_data| MultilinearExtension::new_zero(),
    );

    // node id: [10]
    let output_0_node = OutputNode::new(&ctx, &sector_1_node);

    // node id: [11]
    let sector_2_node = Sector::new(
        &ctx,
        &[&input_shred_0.clone(), &input_shred_1.clone()],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
        |_data| MultilinearExtension::new_zero(),
    );

    // node id: [12]
    let output_1_node = OutputNode::new(&ctx, &sector_2_node);

    // adding a cycle here
    // node id: [13]
    let debug_node_1 = DebugNode::new(
        &ctx,
        "debug".to_string(),
        &[&input_layer_node, &debug_node_0],
    );

    debug_node_0.add_node(&debug_node_1);

    let mut nodes = vec![
        NodeEnum::InputLayer(input_layer_node),
        NodeEnum::InputShred(input_shred_3_node),
        NodeEnum::InputShred(input_shred_4_node),
        NodeEnum::Debug(debug_node_0),
        NodeEnum::Debug(debug_node_1),
        NodeEnum::Sector(sector_0_node),
        NodeEnum::Sector(sector_1_node),
        NodeEnum::Output(output_0_node),
        NodeEnum::Sector(sector_2_node),
        NodeEnum::Output(output_1_node),
    ];

    nodes.reverse();

    let out = topo_sort(nodes);

    assert!(matches!(out, Err(DAGError::DAGCycle)))
}

#[test]
fn test_topo_sort_with_cycle_no_children() {
    //    input_layer
    //         /\
    //   input_shred_0        input_shred_3    ->    debug_node_0 -> debug_node_1 -> debug_node_0
    //   input_shred_1        input_shred_4
    //   input_shred_2
    //
    //
    //                                         ->   sector_0 ->   sector_1  ->output_0
    //                                         ->   sector_2->  output_1
    //
    //
    //
    //
    let ctx = Context::new();
    let dummy_data = MultilinearExtension::new_zero();

    let input_node = InputLayerNode::new(&ctx, None, InputLayerType::PublicInputLayer);

    // node ids: [1, 2, 3]
    let input_shred_0 = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_1 = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_2 = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_vec = vec![
        input_shred_0.clone(),
        input_shred_1.clone(),
        input_shred_2.clone(),
    ];
    // node id: [4]
    let input_layer_node = InputLayerNode::new(
        &ctx,
        Some(input_shred_vec),
        InputLayerType::PublicInputLayer,
    );

    // node ids: [5, 6]
    let input_shred_3_node = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_4_node = InputShred::new(&ctx, dummy_data.clone(), &input_node);

    // for now, use debug node as the fake gate layer
    // node id: [7]
    let mut debug_node_0 = DebugNode::new(
        &ctx,
        "debug".to_string(),
        &[&input_layer_node, &input_shred_3_node],
    );

    // node id: [8]
    let sector_0_node = Sector::new(
        &ctx,
        &[&input_shred_0.clone(), &input_shred_1.clone()],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
        |_data| MultilinearExtension::new_zero(),
    );

    // node id: [9]
    let sector_1_node = Sector::new(
        &ctx,
        &[&input_shred_2, &input_shred_4_node, &sector_0_node],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
        |_data| MultilinearExtension::new_zero(),
    );

    // node id: [10]
    let output_0_node = OutputNode::new(&ctx, &sector_1_node);

    // node id: [11]
    let sector_2_node = Sector::new(
        &ctx,
        &[&input_shred_0.clone(), &input_shred_1.clone()],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
        |_data| MultilinearExtension::new_zero(),
    );

    // node id: [12]
    let output_1_node = OutputNode::new(&ctx, &sector_2_node);

    // adding a cycle here
    // node id: [13]
    let debug_node_1 = DebugNode::new(
        &ctx,
        "debug".to_string(),
        &[&input_layer_node, &debug_node_0],
    );

    debug_node_0.add_node(&debug_node_1);

    let mut nodes = vec![
        NodeEnum::InputLayer(input_layer_node),
        NodeEnum::InputShred(input_shred_3_node),
        NodeEnum::InputShred(input_shred_4_node),
        NodeEnum::Debug(debug_node_0),
        NodeEnum::Debug(debug_node_1),
        NodeEnum::Sector(sector_0_node),
        NodeEnum::Sector(sector_1_node),
        NodeEnum::Output(output_0_node),
        NodeEnum::Sector(sector_2_node),
        NodeEnum::Output(output_1_node),
        NodeEnum::InputShred(input_shred_0),
        NodeEnum::InputShred(input_shred_1),
        NodeEnum::InputShred(input_shred_2),
    ];

    nodes.reverse();

    let out = topo_sort(nodes);

    assert!(matches!(out, Err(DAGError::DAGCycle)))
}

#[test]
fn test_topo_sort_without_cycle_no_children() {
    //    input_layer
    //         /\
    //   input_shred_0        input_shred_3        debug_node     sector_0    sector_1->output_0
    //   input_shred_1        input_shred_4                       sector_2->output_1
    //   input_shred_2
    //
    //
    //
    //
    //
    //
    //
    //
    let ctx = Context::new();
    let dummy_data = MultilinearExtension::new_zero();

    let input_node = InputLayerNode::new(&ctx, None, InputLayerType::PublicInputLayer);

    // node ids: [1, 2, 3]
    let input_shred_0 = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_1 = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_2 = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_vec = vec![
        input_shred_0.clone(),
        input_shred_1.clone(),
        input_shred_2.clone(),
    ];
    // node id: [4]
    let input_layer_node = InputLayerNode::new(
        &ctx,
        Some(input_shred_vec),
        InputLayerType::PublicInputLayer,
    );

    // node ids: [5, 6]
    let input_shred_3_node = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_4_node = InputShred::new(&ctx, dummy_data.clone(), &input_node);

    // for now, use debug node as the fake gate layer
    // node id: [7]
    let debug_node = DebugNode::new(
        &ctx,
        "debug".to_string(),
        &[&input_layer_node, &input_shred_3_node],
    );

    // node id: [8]
    let sector_0_node = Sector::new(
        &ctx,
        &[&input_shred_0.clone(), &input_shred_1.clone()],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
        |_data| MultilinearExtension::new_zero(),
    );

    // node id: [9]
    let sector_1_node = Sector::new(
        &ctx,
        &[&input_shred_2, &input_shred_4_node, &sector_0_node],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
        |_data| MultilinearExtension::new_zero(),
    );

    // node id: [10]
    let output_0_node = OutputNode::new(&ctx, &sector_1_node);

    // node id: [11]
    let sector_2_node = Sector::new(
        &ctx,
        &[&input_shred_0.clone(), &input_shred_1.clone()],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
        |_data| MultilinearExtension::new_zero(),
    );

    // node id: [12]
    let output_1_node = OutputNode::new(&ctx, &sector_2_node);

    let mut nodes = vec![
        NodeEnum::InputLayer(input_layer_node),
        NodeEnum::InputShred(input_shred_3_node),
        NodeEnum::InputShred(input_shred_4_node),
        NodeEnum::Debug(debug_node),
        NodeEnum::Sector(sector_0_node),
        NodeEnum::Sector(sector_1_node),
        NodeEnum::Output(output_0_node),
        NodeEnum::Sector(sector_2_node),
        NodeEnum::Output(output_1_node),
    ];

    nodes.reverse();

    let out = topo_sort(nodes).unwrap();

    let mut children_to_parent_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut id_to_index_map: HashMap<NodeId, usize> = HashMap::new();
    for (idx, node) in out.iter().enumerate() {
        id_to_index_map.insert(node.id(), idx);
        if let Some(children) = node.children() {
            for child in children.into_iter() {
                children_to_parent_map.insert(child, node.id());
            }
        }
    }

    for node in out.iter() {
        for node_source in node.sources().iter() {
            let node_source_parent = children_to_parent_map
                .get(node_source)
                .unwrap_or(node_source);

            assert!(id_to_index_map[node_source_parent] < id_to_index_map[&node.id()]);
        }
    }
}

#[test]
fn test_topo_sort_without_cycle_include_children() {
    //    input_layer
    //         /\
    //   input_shred_0        input_shred_3        debug_node     sector_0    sector_1->output_0
    //   input_shred_1        input_shred_4                       sector_2->output_1
    //   input_shred_2
    //
    //
    //
    //
    //
    //
    //
    //
    let ctx = Context::new();
    let dummy_data = MultilinearExtension::new_zero();

    let input_node = InputLayerNode::new(&ctx, None, InputLayerType::PublicInputLayer);

    // node ids: [1, 2, 3]
    let input_shred_0 = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_1 = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_2 = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_vec = vec![
        input_shred_0.clone(),
        input_shred_1.clone(),
        input_shred_2.clone(),
    ];
    // node id: [4]
    let input_layer_node = InputLayerNode::new(
        &ctx,
        Some(input_shred_vec),
        InputLayerType::PublicInputLayer,
    );

    // node ids: [5, 6]
    let input_shred_3_node = InputShred::new(&ctx, dummy_data.clone(), &input_node);
    let input_shred_4_node = InputShred::new(&ctx, dummy_data.clone(), &input_node);

    // for now, use debug node as the fake gate layer
    // node id: [7]
    let debug_node = DebugNode::new(
        &ctx,
        "debug".to_string(),
        &[&input_layer_node, &input_shred_3_node],
    );

    // node id: [8]
    let sector_0_node = Sector::new(
        &ctx,
        &[&input_shred_0.clone(), &input_shred_1.clone()],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
        |_data| MultilinearExtension::new_zero(),
    );

    // node id: [9]
    let sector_1_node = Sector::new(
        &ctx,
        &[&input_shred_2, &input_shred_4_node, &sector_0_node],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
        |_data| MultilinearExtension::new_zero(),
    );

    // node id: [10]
    let output_0_node = OutputNode::new(&ctx, &sector_1_node);

    // node id: [11]
    let sector_2_node = Sector::new(
        &ctx,
        &[&input_shred_0.clone(), &input_shred_1.clone()],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
        |_data| MultilinearExtension::new_zero(),
    );

    // node id: [12]
    let output_1_node = OutputNode::new(&ctx, &sector_2_node);

    let mut nodes = vec![
        NodeEnum::InputLayer(input_layer_node),
        NodeEnum::InputShred(input_shred_3_node),
        NodeEnum::InputShred(input_shred_4_node),
        NodeEnum::Debug(debug_node),
        NodeEnum::Sector(sector_0_node),
        NodeEnum::Sector(sector_1_node),
        NodeEnum::Output(output_0_node),
        NodeEnum::Sector(sector_2_node),
        NodeEnum::Output(output_1_node),
        NodeEnum::InputShred(input_shred_0),
        NodeEnum::InputShred(input_shred_1),
        NodeEnum::InputShred(input_shred_2),
    ];

    nodes.reverse();

    let out = topo_sort(nodes).unwrap();

    let mut children_to_parent_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut id_to_index_map: HashMap<NodeId, usize> = HashMap::new();
    for (idx, node) in out.iter().enumerate() {
        id_to_index_map.insert(node.id(), idx);
        if let Some(children) = node.children() {
            for child in children.into_iter() {
                children_to_parent_map.insert(child, node.id());
            }
        }
    }

    for node in out.iter() {
        for node_source in node.sources().iter() {
            let node_source_parent = children_to_parent_map
                .get(node_source)
                .unwrap_or(node_source);

            assert!(id_to_index_map[node_source_parent] < id_to_index_map[&node.id()]);
        }
    }
}
