//! Module for easily creating Circuits from re-usable components

use std::collections::{HashMap, HashSet};

use remainder_shared_types::FieldExt;
use tracing_subscriber::layer;

use self::nodes::{
    circuit_inputs::{InputLayerNode, InputShred},
    circuit_outputs::OutputNode,
    node_enum::NodeEnum,
    CircuitNode, NodeId,
};

pub mod component;
pub mod nodes;

pub fn topo_sort<F: FieldExt>(nodes: &Vec<NodeEnum<F>>) -> Vec<NodeEnum<F>> {
    let mut children_to_parent_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut id_to_index_map: HashMap<NodeId, usize> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        id_to_index_map.insert(node.id(), idx);
        if let Some(children) = node.children() {
            for child in children.into_iter() {
                children_to_parent_map.insert(child, node.id());
            }
        }
    }
    let mut edges_out: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
    let mut edges_in: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
    let mut starting_nodes = vec![];

    for node in nodes.iter() {
        let node_id = node.id();
        if node.sources().len() == 0 {
            let insert_node = children_to_parent_map.get(&node_id).unwrap_or(&node_id);
            starting_nodes.push(*insert_node);
        } else {
            for source in node.sources().iter() {
                let insert_source = children_to_parent_map.get(source).unwrap_or(source);
                let insert_dest = children_to_parent_map.get(&node_id).unwrap_or(&node_id);
                edges_out
                    .entry(*insert_source)
                    .or_insert_with(|| HashSet::new())
                    .insert(*insert_dest);
                edges_in
                    .entry(*insert_dest)
                    .or_insert_with(|| HashSet::new())
                    .insert(*insert_source);
            }
        }
    }

    let mut out = vec![];

    while starting_nodes.len() > 0 {
        let node = starting_nodes.pop().unwrap();
        out.push(nodes[id_to_index_map[&node]].clone());
        if let Some(dest_nodes) = edges_out.get(&node) {
            for dest_node in dest_nodes.iter() {
                // remove the edge between node and dest_node
                edges_in.get_mut(dest_node).unwrap().remove(&node);

                // if dest_node has no incoming edges, add it to starting_nodes
                if edges_in.get(dest_node).unwrap().len() == 0 {
                    starting_nodes.push(*dest_node);
                    edges_in.remove(dest_node);
                }
            }
            // remove all outgoing edges from node
            edges_out.remove(&node);
        }
    }

    assert!(edges_in.len() == 0, "Graph has a cycle");
    assert!(edges_out.len() == 0, "Graph has a cycle");

    out
}

/// Assigns circuit nodes in the circuit to different layers based on their dependencies
pub fn assign_layers<'a, F: FieldExt>(
    nodes: &'a Vec<NodeEnum<F>>,
) -> (
    Vec<&'a InputShred<F>>,
    Vec<&'a InputLayerNode<F>>,
    Vec<Vec<&'a NodeEnum<F>>>,
    Vec<&'a OutputNode<F>>,
    HashMap<NodeId, usize>,
) {
    // the input can exist either as a shred or as a layer
    let mut input_shreds = vec![];
    let mut input_layers = vec![];
    // stores all the output layers, there will be no dependencies within themselves
    let mut output_layers = vec![];
    // stores all intermediate layers, and there can be multiple nodes within a single layer
    let mut intermediate_layers: Vec<Vec<&NodeEnum<F>>> = vec![];
    // stores which layer each node belongs to
    let mut node_to_layer_id_map: HashMap<NodeId, usize> = HashMap::new();

    // find all input_shreds & input_layer_nodes
    // there will be #input_layers + 1 input layers
    for node in nodes.iter() {
        if let NodeEnum::InputShred(input_shred) = node {
            input_shreds.push(input_shred);
            // all input shreds are in the first layer
            node_to_layer_id_map.insert(input_shred.id(), 0);
        } else if let NodeEnum::InputLayer(input_layer) = node {
            input_layers.push(input_layer);
            for input_layer_shreds_id in input_layer.children().unwrap() {
                // all input shreds are in the first layer
                node_to_layer_id_map.insert(input_layer_shreds_id, 0);
            }
        } else if let NodeEnum::Output(output_node) = node {
            output_layers.push(output_node);
        };
    }
    intermediate_layers.push(vec![]); // the first layer is the input layer / is empty

    for node in nodes.iter() {
        // skip if it's non-intermediate node
        if matches!(
            node,
            NodeEnum::InputShred(_) | NodeEnum::InputLayer(_) | NodeEnum::Output(_)
        ) {
            continue;
        } else if matches!(node, NodeEnum::Debug(_)) {
            // if it's a standalone node, i.e. cannot be concat-ed with any other node into the same layer,
            // then just put it at the last layer
            let curr_layer_id = node_to_layer_id_map.values().max().unwrap() + 1;

            node_to_layer_id_map.insert(node.id(), curr_layer_id);
            intermediate_layers.push(vec![node]);
        } else if matches!(node, NodeEnum::Sector(_)) {
            // if it's a sector node, we need to find its parents
            let max_parent_layer_id = node
                .sources()
                .iter()
                .map(|node_id| node_to_layer_id_map.get(node_id).unwrap())
                .max();
            if let Some(layer_id) = max_parent_layer_id {
                let mut curr_layer_id = *layer_id + 1;

                while curr_layer_id < intermediate_layers.len() {
                    // checks if the current layer is a debug node / standalone layer
                    if !(intermediate_layers[curr_layer_id].len() == 1
                        && matches!(intermediate_layers[curr_layer_id][0], NodeEnum::Debug(_)))
                    {
                        // if not, then we can insert the sector node into this layer
                        intermediate_layers[curr_layer_id].push(&node);
                        break;
                    }
                    curr_layer_id += 1;
                }

                if curr_layer_id == intermediate_layers.len() {
                    intermediate_layers.push(vec![&node]);
                }
                node_to_layer_id_map.insert(node.id(), curr_layer_id);
            } else {
                panic!("Sector node has no parents");
            }
        };
    }

    // println!("input shreds {:?}", input_shreds);
    // println!("input layeres {:?}", input_layers);
    // println!("output layers {:?}", output_layers);
    // for (layer_num, layer) in intermediate_layers.iter().enumerate() {
    //     println!("layer #{:?}", layer_num);
    //     for (part_num, part) in layer.iter().enumerate() {
    //         println!("part #{:?}: {:?}", part_num, part);
    //     }
    // }
    // println!("node->id map {:?}", node_to_layer_id_map);

    (
        input_shreds,
        input_layers,
        intermediate_layers,
        output_layers,
        node_to_layer_id_map,
    )
}

#[cfg(test)]
pub mod tests {

    use std::collections::{hash_map::RandomState, HashMap, HashSet};

    use remainder_shared_types::Fr;

    use crate::{
        expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
        layouter::nodes::NodeId,
        mle::evals::MultilinearExtension,
    };

    use crate::layouter::nodes::CircuitNode;

    use super::{
        assign_layers,
        nodes::{
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
    fn test_topo_sort() {
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

        // node ids: [1, 2, 3]
        let input_shred_0 = InputShred::new(&ctx, dummy_data.clone(), None);
        let input_shred_1 = InputShred::new(&ctx, dummy_data.clone(), None);
        let input_shred_2 = InputShred::new(&ctx, dummy_data.clone(), None);
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
        let input_shred_3_node = InputShred::new(&ctx, dummy_data.clone(), None);
        let input_shred_4_node = InputShred::new(&ctx, dummy_data.clone(), None);

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

        let out = topo_sort(&nodes);

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
    fn test_assign_layers() {
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

        // node ids: [1, 2, 3]
        let input_shred_0 = InputShred::new(&ctx, dummy_data.clone(), None);
        let input_shred_1 = InputShred::new(&ctx, dummy_data.clone(), None);
        let input_shred_2 = InputShred::new(&ctx, dummy_data.clone(), None);
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
        let input_shred_3_node = InputShred::new(&ctx, dummy_data.clone(), None);
        let input_shred_4_node = InputShred::new(&ctx, dummy_data.clone(), None);

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

        let nodes = vec![
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
        let (input_shreds, input_layers, intermediate_layers, output_layers, _node_to_layer_id_map) =
            assign_layers(&nodes);

        // check input shreds
        assert_eq!(
            HashSet::<NodeId, RandomState>::from_iter(vec![
                NodeId::new_unsafe(5),
                NodeId::new_unsafe(6)
            ]),
            HashSet::from_iter(input_shreds.iter().map(|node| node.id()))
        );

        // check input layers
        assert_eq!(
            HashSet::<NodeId, RandomState>::from_iter(vec![NodeId::new_unsafe(4)]),
            HashSet::from_iter(input_layers.iter().map(|layer| layer.id()))
        );

        // check children of input layers
        assert_eq!(
            HashSet::<NodeId, RandomState>::from_iter(vec![
                NodeId::new_unsafe(1),
                NodeId::new_unsafe(2),
                NodeId::new_unsafe(3)
            ]),
            HashSet::from_iter(
                input_layers
                    .iter()
                    .flat_map(|layer| layer.children().unwrap())
            )
        );

        // check layer #0 of intermediate_layers
        assert_eq!(
            HashSet::<NodeId, RandomState>::from_iter(vec![]),
            HashSet::from_iter(intermediate_layers[0].iter().map(|node| node.id()))
        );

        // check layer #1 of intermediate_layers
        assert_eq!(
            HashSet::<NodeId, RandomState>::from_iter(vec![NodeId::new_unsafe(7)]),
            HashSet::from_iter(intermediate_layers[1].iter().map(|node| node.id()))
        );

        // check layer #2 of intermediate_layers
        assert_eq!(
            HashSet::<NodeId, RandomState>::from_iter(vec![
                NodeId::new_unsafe(8),
                NodeId::new_unsafe(11)
            ]),
            HashSet::from_iter(intermediate_layers[2].iter().map(|node| node.id()))
        );

        // check layer #3 of intermediate_layers
        assert_eq!(
            HashSet::<NodeId, RandomState>::from_iter(vec![NodeId::new_unsafe(9)]),
            HashSet::from_iter(intermediate_layers[3].iter().map(|node| node.id()))
        );

        // check output layers
        assert_eq!(
            HashSet::<NodeId, RandomState>::from_iter(vec![
                NodeId::new_unsafe(10),
                NodeId::new_unsafe(12)
            ]),
            HashSet::from_iter(output_layers.iter().map(|layer| layer.id()))
        );
    }
}
