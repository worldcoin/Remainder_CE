//! Defines the utilities for taking a list of nodes and turning it into a layedout circuit

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use remainder_shared_types::{layer::LayerId, FieldExt};
use thiserror::Error;

use crate::{
    input_layer::enum_input_layer::InputLayerEnum,
    layer::layer_enum::LayerEnum,
    layouter::nodes::sector::{Sector, SectorGroup},
    mle::{evals::MultilinearExtension, mle_enum::MleEnum},
    prover::proof_system::ProofSystem,
};

use super::nodes::{
    circuit_inputs::{InputLayerNode, InputShred, SealedInputNode},
    circuit_outputs::OutputNode,
    gate::GateNode,
    node_enum::{NodeEnum, NodeEnumGroup},
    split_node::SplitNode,
    CircuitNode, CompilableNode, Context, NodeGroup, NodeId, YieldNode,
};

/// A HashMap that records during circuit compilation where nodes live in the circuit and what data they yield
pub struct CircuitMap<'a, F>(
    pub(crate) HashMap<NodeId, (CircuitLocation, &'a MultilinearExtension<F>)>,
);

impl<'a, F> CircuitMap<'a, F> {
    pub(crate) fn new() -> Self {
        Self(HashMap::new())
    }
}

/// The location of a Node in the circuit
#[derive(Debug, Clone)]
pub struct CircuitLocation {
    /// The LayerId this node has been placed into
    pub layer_id: LayerId,
    /// Any prefix_bits neccessary to differenciate claims made onto
    /// this node from other nodes in the same layer
    pub prefix_bits: Vec<bool>,
}

impl CircuitLocation {
    /// Creates a new CircuitLocation
    pub fn new(layer_id: LayerId, prefix_bits: Vec<bool>) -> Self {
        Self {
            layer_id,
            prefix_bits,
        }
    }
}

///Errors to do with the DAG, sorting, assigning layers, etc.
#[derive(Error, Debug, Clone)]
pub enum DAGError {
    ///The DAG has a cycle
    #[error("The DAG has a cycle")]
    DAGCycle,
    ///node has no parent
    #[error("Node has no parent")]
    NodeHasNoParent,
    ///A NodeId exists that references a node that is not present in the DAG
    #[error("A NodeId exists that references a node that is not present in the DAG: Id = {0:?}")]
    DanglingNodeId(NodeId),
}

/// given a unsorted vector of NodeEnum, returns a topologically sorted vector of NodeEnum
/// uses the edge deletion algorithm to find the topological sort
/// possible future improvement: use the DFS algorithm (w/ timestamping) together with the
/// adjacency list representation of the graph, to avoid hashmaps.
/// w.r.t the parent/child nodes, we need to decide if children nodes are already exclueded,
/// or if they are included in the vec of [NodeEnum]
///
/// Does allow subgraphs
pub fn topo_sort<N: CircuitNode>(nodes: Vec<N>) -> Result<Vec<N>, DAGError> {
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

    let mut subgraph_nodes: HashSet<NodeId> = HashSet::new();

    for node in nodes.iter() {
        subgraph_nodes.insert(node.id());
        for node in node.children().iter().flatten() {
            subgraph_nodes.insert(*node);
        }
    }

    let mut edges_out: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
    let mut edges_in: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
    let mut starting_nodes = HashSet::new();

    for node in nodes.iter() {
        let node_id = node.id();
        let is_source = node
            .sources()
            .iter()
            .all(|node_id| !subgraph_nodes.contains(node_id));
        if is_source {
            let insert_node = children_to_parent_map.get(&node_id).unwrap_or(&node_id);
            starting_nodes.insert(*insert_node);
        } else {
            for source in node
                .sources()
                .iter()
                .filter(|node_id| subgraph_nodes.contains(node_id))
            {
                let insert_source = children_to_parent_map.get(source).unwrap_or(source);
                let insert_dest = children_to_parent_map.get(&node_id).unwrap_or(&node_id);
                edges_out
                    .entry(*insert_source)
                    .or_default()
                    .insert(*insert_dest);
                edges_in
                    .entry(*insert_dest)
                    .or_default()
                    .insert(*insert_source);
            }
        }
    }
    let mut nodes = nodes.into_iter().map(|node| Some(node)).collect_vec();
    let mut out: Vec<N> = vec![];

    let mut starting_nodes = starting_nodes.into_iter().collect::<Vec<NodeId>>();
    while let Some(node) = starting_nodes.pop() {
        out.push(nodes[id_to_index_map[&node]].take().unwrap());
        if let Some(dest_nodes) = edges_out.get(&node) {
            for dest_node in dest_nodes.iter() {
                // remove the edge between node and dest_node
                edges_in.get_mut(dest_node).unwrap().remove(&node);

                // if dest_node has no incoming edges, add it to starting_nodes
                if edges_in.get(dest_node).unwrap().is_empty() {
                    starting_nodes.push(*dest_node);
                    edges_in.remove(dest_node);
                }
            }
            // remove all outgoing edges from node
            edges_out.remove(&node);
        }
    }

    if !edges_in.is_empty() || !edges_out.is_empty() {
        return Err(DAGError::DAGCycle);
    }

    Ok(out)
}

enum IntermediateNode<F: FieldExt, Pf> {
    CompilableNode(Box<dyn CompilableNode<F, Pf>>),
    Sector(Sector<F>),
}

impl<F: FieldExt, Pf: ProofSystem<F>> IntermediateNode<F, Pf> {
    fn new<N: CompilableNode<F, Pf> + 'static>(node: N) -> Self {
        Self::CompilableNode(Box::new(node) as Box<dyn CompilableNode<F, Pf>>)
    }
}

impl<F: FieldExt, Pf: ProofSystem<F>> CircuitNode for IntermediateNode<F, Pf> {
    fn id(&self) -> NodeId {
        match self {
            IntermediateNode::CompilableNode(node) => node.id(),
            IntermediateNode::Sector(node) => node.id(),
        }
    }

    fn sources(&self) -> Vec<NodeId> {
        match self {
            IntermediateNode::CompilableNode(node) => node.sources(),
            IntermediateNode::Sector(node) => node.sources(),
        }
    }

    fn children(&self) -> Option<Vec<NodeId>> {
        match self {
            IntermediateNode::CompilableNode(node) => node.children(),
            IntermediateNode::Sector(node) => node.children(),
        }
    }
}

/// Current Core Layouter
/// Assigns circuit nodes in the circuit to different layers based on their dependencies
/// assumes that the nodes are topologically sorted
/// this algorithm is greedy and assigns nodes to the first available layer that it can,
/// without breaking any dependencies, and any restrictions that are imposed by the node type
/// (such as matmal / gate nodes cannot be combined with other nodes in the same layer)
/// This algorithm currently minimizes the number of layers, and does not account for
/// specific layer size / its constituent nodes's numvars, etc.

pub fn layout<
    F: FieldExt,
    Pf: ProofSystem<F, Layer = LayerEnum<F>, InputLayer = InputLayerEnum<F>, OutputLayer = MleEnum<F>>,
>(
    ctx: Context,
    nodes: Vec<NodeEnum<F>>,
) -> Result<Vec<Box<dyn CompilableNode<F, Pf>>>, DAGError> {
    let mut dag = NodeEnumGroup::new(nodes);

    //handle input_layers
    let out = {
        let input_shreds: Vec<InputShred<F>> = dag.get_nodes();
        let mut input_layers: Vec<InputLayerNode<F>> = dag.get_nodes();
        let sealed_inputs: Vec<SealedInputNode<F>> = dag.get_nodes();

        let mut input_layer_map: HashMap<NodeId, &mut InputLayerNode<F>> = HashMap::new();
        let default_input_layer = input_layers[0].id();

        for layer in input_layers.iter_mut() {
            input_layer_map.insert(layer.id(), layer);
        }

        // Add InputShreds to specified parents
        for input_shred in input_shreds {
            if let Some(input_layer_id) = input_shred.parent {
                let input_layer = input_layer_map
                    .get_mut(&input_layer_id)
                    .ok_or(DAGError::DanglingNodeId(input_layer_id))?;
                input_layer.add_shred(input_shred);
            } else {
                // This probably sucks (rethink default inputlayers)
                let input_layer = input_layer_map.get_mut(&default_input_layer).unwrap();
                input_layer.add_shred(input_shred);
            }
        }

        sealed_inputs
            .into_iter()
            .map(|input| Box::new(input) as Box<dyn CompilableNode<F, Pf>>)
            .chain(
                input_layers
                    .into_iter()
                    .map(|input| Box::new(input) as Box<dyn CompilableNode<F, Pf>>),
            )
    };

    //handle intermediate layers
    let out = {
        let sector_groups: Vec<SectorGroup<F>> = dag.get_nodes();
        let sectors: Vec<Sector<F>> = dag.get_nodes();
        let gates: Vec<GateNode<F>> = dag.get_nodes();
        let splits: Vec<SplitNode<F>> = dag.get_nodes();
        let other_layers = sector_groups
            .into_iter()
            .map(|node| IntermediateNode::new(node))
            .chain(gates.into_iter().map(|node| IntermediateNode::new(node)))
            .chain(splits.into_iter().map(|node| IntermediateNode::new(node)));

        let intermediate_nodes = other_layers
            .chain(
                sectors
                    .into_iter()
                    .map(|sector| IntermediateNode::Sector(sector)),
            )
            .collect_vec();

        //topo_sort all the nodes which can be immediently compiled and the sectors that need to be layedout before compilation
        let intermediate_nodes = topo_sort(intermediate_nodes)?;

        //collapse the sectors into sector_groups
        let (mut intermediate_nodes, final_sector_group) = intermediate_nodes.into_iter().fold(
            (vec![], None::<SectorGroup<F>>),
            |(mut layedout_nodes, curr_sector_group), node| {
                let curr_sector_group = match node {
                    IntermediateNode::CompilableNode(node) => {
                        if let Some(curr_sector_group) = curr_sector_group {
                            layedout_nodes
                                    .push(Box::new(curr_sector_group)
                                        as Box<dyn CompilableNode<F, Pf>>);
                        }
                        layedout_nodes.push(node);
                        None
                    }
                    IntermediateNode::Sector(node) => {
                        if let Some(mut sector_group) = curr_sector_group {
                            sector_group.add_sector(node);
                            Some(sector_group)
                        } else {
                            Some(SectorGroup::new(&ctx, vec![node]))
                        }
                    }
                };
                (layedout_nodes, curr_sector_group)
            },
        );

        if let Some(final_sector_group) = final_sector_group {
            intermediate_nodes.push(Box::new(final_sector_group));
        }
        out.chain(intermediate_nodes)
    };

    //handle output layers
    let out = {
        let output_layers: Vec<OutputNode<F>> = dag.get_nodes();

        out.chain(
            output_layers
                .into_iter()
                .map(|node| Box::new(node) as Box<dyn CompilableNode<F, Pf>>),
        )
    };

    Ok(out.collect())
}
