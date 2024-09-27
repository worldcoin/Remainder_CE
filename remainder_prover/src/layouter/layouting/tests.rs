use std::collections::{HashMap, HashSet};

use quickcheck::{Arbitrary, TestResult};
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
    let dummy_data: MultilinearExtension<Fr> = MultilinearExtension::new_zero();

    let input_node = InputLayerNode::new(&ctx, None, InputLayerType::PublicInputLayer);

    // node ids: [1, 2, 3]
    let input_shred_0 = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
    let input_shred_1 = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
    let input_shred_2 = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
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
    let input_shred_3_node = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
    let input_shred_4_node = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);

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
    );

    // node id: [9]
    let sector_1_node = Sector::new(
        &ctx,
        &[&input_shred_2, &input_shred_4_node, &sector_0_node],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
    );

    // node id: [10]
    let output_0_node = OutputNode::new(&ctx, &sector_1_node);

    // node id: [11]
    let sector_2_node = Sector::new(
        &ctx,
        &[&input_shred_0.clone(), &input_shred_1.clone()],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
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
    let dummy_data: MultilinearExtension<Fr> = MultilinearExtension::new_zero();

    let input_node = InputLayerNode::new(&ctx, None, InputLayerType::PublicInputLayer);

    // node ids: [1, 2, 3]
    let input_shred_0 = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
    let input_shred_1 = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
    let input_shred_2 = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
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
    let input_shred_3_node = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
    let input_shred_4_node = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);

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
    );

    // node id: [9]
    let sector_1_node = Sector::new(
        &ctx,
        &[&input_shred_2, &input_shred_4_node, &sector_0_node],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
    );

    // node id: [10]
    let output_0_node = OutputNode::new(&ctx, &sector_1_node);

    // node id: [11]
    let sector_2_node = Sector::new(
        &ctx,
        &[&input_shred_0.clone(), &input_shred_1.clone()],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
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
    let dummy_data: MultilinearExtension<Fr> = MultilinearExtension::new_zero();

    let input_node = InputLayerNode::new(&ctx, None, InputLayerType::PublicInputLayer);

    // node ids: [1, 2, 3]
    let input_shred_0 = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
    let input_shred_1 = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
    let input_shred_2 = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
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
    let input_shred_3_node = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
    let input_shred_4_node = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);

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
    );

    // node id: [9]
    let sector_1_node = Sector::new(
        &ctx,
        &[&input_shred_2, &input_shred_4_node, &sector_0_node],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
    );

    // node id: [10]
    let output_0_node = OutputNode::new(&ctx, &sector_1_node);

    // node id: [11]
    let sector_2_node = Sector::new(
        &ctx,
        &[&input_shred_0.clone(), &input_shred_1.clone()],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
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
        if let Some(children) = node.subnodes() {
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
    let dummy_data: MultilinearExtension<Fr> = MultilinearExtension::new_zero();

    let input_node = InputLayerNode::new(&ctx, None, InputLayerType::PublicInputLayer);

    // node ids: [1, 2, 3]
    let input_shred_0 = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
    let input_shred_1 = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
    let input_shred_2 = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
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
    let input_shred_3_node = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);
    let input_shred_4_node = InputShred::new(&ctx, dummy_data.clone().num_vars(), &input_node);

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
    );

    // node id: [9]
    let sector_1_node = Sector::new(
        &ctx,
        &[&input_shred_2, &input_shred_4_node, &sector_0_node],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
    );

    // node id: [10]
    let output_0_node = OutputNode::new(&ctx, &sector_1_node);

    // node id: [11]
    let sector_2_node = Sector::new(
        &ctx,
        &[&input_shred_0.clone(), &input_shred_1.clone()],
        |ids| Expression::<Fr, AbstractExpr>::products(ids),
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
        if let Some(children) = node.subnodes() {
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

/// Dependency graph wrapper for Quickcheck.
#[derive(Clone, Debug)]
struct QDepGraph {
    pub repr: Vec<NodeEnum<Fr>>,
}

impl Arbitrary for QDepGraph {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        // Instance size.
        let n = 3; // g.size();

        // Number of input nodes (all are sources).
        let num_input_shreds = n;
        // Number of intermediate nodes.
        let num_sectors = 3 * n;
        // Number of output nodes (all are sinks).
        let num_outputs = n;

        let ctx = Context::new();
        let dummy_data = MultilinearExtension::<Fr>::new_zero();

        // Start with the input nodes.
        let mut graph: Vec<NodeEnum<Fr>> = (0..num_input_shreds)
            .map(|_| {
                let input_node = InputLayerNode::new(&ctx, None, InputLayerType::PublicInputLayer);
                let input_shred = InputShred::new(&ctx, dummy_data.num_vars(), &input_node);
                NodeEnum::InputShred(input_shred)
            })
            .collect();

        for _ in 0..num_sectors {
            let available_edges = graph.iter().map(|node| match node {
                NodeEnum::Sector(sector) => sector as &dyn CircuitNode,
                NodeEnum::InputShred(input_shred) => input_shred as &dyn CircuitNode,
                _ => panic!("Unexpected node type"),
            });

            let inputs: Vec<&dyn CircuitNode> = available_edges
                .filter(|_| *g.choose(&[true, false]).unwrap())
                .collect();

            let sector = Sector::new(&ctx, &inputs, |ids| {
                Expression::<Fr, AbstractExpr>::products(ids)
            });

            graph.extend([NodeEnum::Sector(sector)]);
        }

        let available_edges: Vec<&Sector<Fr>> = graph
            .iter()
            .skip(num_input_shreds)
            .map(|node| match node {
                NodeEnum::Sector(sector) => sector,
                _ => panic!("Unexpected node type"),
            })
            .collect();

        let mut output_nodes = vec![];

        for _ in 0..num_outputs {
            let input = *g.choose(&available_edges).unwrap();
            let output = OutputNode::new(&ctx, input);
            output_nodes.extend([NodeEnum::Output(output)]);
        }

        graph.extend(output_nodes);

        for i in 0..graph.len() {
            let choices: Vec<usize> = (0..=i).collect();
            let j = *g.choose(&choices).unwrap();

            graph.swap(i, j);
        }

        QDepGraph { repr: graph }
    }
}

#[quickcheck]
fn toposorted_property(graph: QDepGraph) -> TestResult {
    let sorted_graph = topo_sort(graph.repr).unwrap();

    let mut ids_seen = HashSet::new();

    TestResult::from_bool(sorted_graph.into_iter().all(|u_enum| {
        let (vs, u_id) = match u_enum {
            NodeEnum::Sector(u) => (u.sources(), u.id()),
            NodeEnum::InputShred(u) => (u.sources(), u.id()),
            NodeEnum::Output(u) => (u.sources(), u.id()),
            _ => panic!("Unexpected node type"),
        };
        ids_seen.insert(u_id);

        vs.into_iter().all(|v_id| !ids_seen.get(&v_id).is_none())
    }))
}
