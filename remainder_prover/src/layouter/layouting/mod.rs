//! Defines the utilities for taking a list of nodes and turning it into a layedout circuit

#[cfg(test)]
mod tests;

use std::collections::{hash_map::Entry, BTreeSet, HashMap, HashSet};

use itertools::Itertools;
use remainder_shared_types::Field;
use thiserror::Error;

use crate::{
    expression::circuit_expr::MleDescription,
    layer::LayerId,
    layouter::nodes::sector::{Sector, SectorGroup},
    mle::{dense::DenseMle, evals::MultilinearExtension},
};

use super::nodes::{
    circuit_inputs::{InputLayerNode, InputShred},
    circuit_outputs::OutputNode,
    fiat_shamir_challenge::FiatShamirChallengeNode,
    gate::GateNode,
    identity_gate::IdentityGateNode,
    lookup::{LookupConstraint, LookupTable},
    matmult::MatMultNode,
    node_enum::{NodeEnum, NodeEnumGroup},
    split_node::SplitNode,
    CircuitNode, CompilableNode, Context, NodeGroup, NodeId, YieldNode,
};

/// A HashMap that records during circuit compilation where nodes live in the circuit and what data they yield.
#[derive(Debug)]
pub struct CircuitMap<F: Field>(pub(crate) HashMap<CircuitLocation, MultilinearExtension<F>>);
/// A map that maps layer ID to all the MLEs that are output from that layer. Together these MLEs are combined
/// along with the information from their prefix bits to form the layerwise bookkeeping table.
pub type LayerMap<F> = HashMap<LayerId, Vec<DenseMle<F>>>;

impl<F: Field> CircuitMap<F> {
    /// Create a new circuit map, which maps circuit location to the data stored at that location.removing
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Using the circuit location, which is a layer_id and prefix_bits tuple,
    /// get the data that exists here.
    pub fn get_data_from_location(
        &self,
        circuit_location: &CircuitLocation,
    ) -> Option<&MultilinearExtension<F>> {
        self.0.get(circuit_location)
    }

    /// An alias to [get_data_from_location] above,
    pub fn get_data_from_circuit_mle(
        &self,
        circuit_mle: &MleDescription<F>,
    ) -> Result<&MultilinearExtension<F>, DAGError> {
        let circuit_location =
            CircuitLocation::new(circuit_mle.layer_id(), circuit_mle.prefix_bits());
        let result = self
            .0
            .get(&circuit_location)
            .ok_or(DAGError::NoCircuitLocation);
        if let Ok(actual_result) = result {
            assert_eq!(actual_result.num_vars(), circuit_mle.num_free_vars());
        }
        result
    }

    /// Adds a new node to the CircuitMap
    pub fn add_node(&mut self, circuit_location: CircuitLocation, value: MultilinearExtension<F>) {
        self.0.insert(circuit_location, value);
    }

    /// Destructively convert this into a map that maps LayerId to the [MultilinearExtension]s
    /// that generate claims on this area. This is to aid in claim aggregation,
    /// so we know the parts of the layerwise bookkeeping table in order to aggregate claims
    /// on this layer.
    pub fn convert_to_layer_map(mut self) -> LayerMap<F> {
        let mut layer_map = HashMap::<LayerId, Vec<DenseMle<F>>>::new();
        self.0.drain().for_each(|(circuit_location, data)| {
            let corresponding_mle = DenseMle::new_with_prefix_bits(
                data,
                circuit_location.layer_id,
                circuit_location.prefix_bits,
            );
            if let Entry::Vacant(e) = layer_map.entry(circuit_location.layer_id) {
                e.insert(vec![corresponding_mle]);
            } else {
                layer_map
                    .get_mut(&circuit_location.layer_id)
                    .unwrap()
                    .push(corresponding_mle);
            }
        });
        layer_map
    }
}

impl<F: Field> Default for CircuitMap<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
/// A HashMap that associates the node ID to a tuple which contains the
/// circuit location of that node, and the "shape" of that node, which
/// is basically the number of variables of that MLE.
pub struct CircuitDescriptionMap(pub(crate) HashMap<NodeId, (CircuitLocation, usize)>);

impl CircuitDescriptionMap {
    pub(crate) fn new() -> Self {
        Self(HashMap::new())
    }

    /// Using the node ID, retrieve the number of variables
    /// and location of this node in the circuit.
    pub fn get_location_num_vars_from_node_id(
        &self,
        node: &NodeId,
    ) -> Result<&(CircuitLocation, usize), DAGError> {
        self.0.get(node).ok_or(DAGError::DanglingNodeId(*node))
    }

    /// Add a node ID and its corresponding circuit location
    /// as well as number of variables to the hash map.
    pub fn add_node_id_and_location_num_vars(
        &mut self,
        node: NodeId,
        value: (CircuitLocation, usize),
    ) {
        self.0.insert(node, value);
    }
}

#[derive(Debug)]
/// A HashMap that maps layer ID to node ID, used specifically to associate
/// input layer nodes to input layer IDs for circuit creation.
pub struct InputNodeMap(pub(crate) HashMap<LayerId, NodeId>);

impl InputNodeMap {
    pub(crate) fn new() -> Self {
        Self(HashMap::new())
    }

    /// Get the node ID from a layer ID.
    pub fn get_node_id(&self, layer_id: LayerId) -> Option<&NodeId> {
        self.0.get(&layer_id)
    }

    /// Add a layer ID, node ID correspondence to the map.
    pub fn add_node_layer_id(&mut self, layer_id: LayerId, node_id: NodeId) {
        self.0.insert(layer_id, node_id);
    }
}

/// The location of a Node in the circuit
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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
    /// The DAG has a cycle.
    #[error("The DAG has a cycle")]
    DAGCycle,
    /// Node has no parent.
    #[error("Node has no parent")]
    NodeHasNoParent,
    /// A NodeId exists that references a node that is not present in the DAG.
    #[error("A NodeId exists that references a node that is not present in the DAG: Id = {0:?}")]
    DanglingNodeId(NodeId),
    /// We have gotten to a layer whose parts of the expression have not been generatede..
    #[error("This circuit location does not exist, or has not been compiled yet")]
    NoCircuitLocation,
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
        if let Some(children) = node.subnodes() {
            for child in children.into_iter() {
                children_to_parent_map.insert(child, node.id());
            }
        }
    }

    let mut subgraph_nodes: HashSet<NodeId> = HashSet::new();

    for node in nodes.iter() {
        subgraph_nodes.insert(node.id());
        for node in node.subnodes().iter().flatten() {
            subgraph_nodes.insert(*node);
        }
    }

    let mut edges_out: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
    let mut edges_in: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
    let mut starting_nodes = BTreeSet::new();

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

enum IntermediateNode<F: Field> {
    CompilableNode(Box<dyn CompilableNode<F>>),
    Sector(Sector<F>),
}

impl<F: Field> IntermediateNode<F> {
    fn new<N: CompilableNode<F> + 'static>(node: N) -> Self {
        Self::CompilableNode(Box::new(node) as Box<dyn CompilableNode<F>>)
    }
}

impl<F: Field> CircuitNode for IntermediateNode<F> {
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

    fn subnodes(&self) -> Option<Vec<NodeId>> {
        match self {
            IntermediateNode::CompilableNode(node) => node.subnodes(),
            IntermediateNode::Sector(node) => node.subnodes(),
        }
    }

    fn get_num_vars(&self) -> usize {
        match self {
            IntermediateNode::CompilableNode(node) => node.get_num_vars(),
            IntermediateNode::Sector(node) => node.get_num_vars(),
        }
    }
}

type LayouterNodes<F> = (
    Vec<InputLayerNode>,
    Vec<FiatShamirChallengeNode>,
    Vec<Box<dyn CompilableNode<F>>>,
    Vec<LookupTable>,
    Vec<OutputNode>,
);

/// Current Core Layouter
/// Assigns circuit nodes in the circuit to different layers based on their dependencies
/// this algorithm is greedy and assigns nodes to the first available layer that it can,
/// without breaking any dependencies, and any restrictions that are imposed by the node type
/// (such as matmal / gate nodes cannot be combined with other nodes in the same layer)
/// This algorithm currently minimizes the number of layers, and does not account for
/// specific layer size / its constituent nodes's numvars, etc.
/// Returns a vector of [CompilableNode] in which inputs are first, then intermediates
/// (topologically sorted), then lookups, then outputs.
pub fn layout<F: Field>(
    ctx: Context,
    nodes: Vec<NodeEnum<F>>,
) -> Result<LayouterNodes<F>, DAGError> {
    let mut dag = NodeEnumGroup::new(nodes);

    // Handle input layers
    let input_shreds: Vec<InputShred> = dag.get_nodes();
    let mut input_layer_nodes: Vec<InputLayerNode> = dag.get_nodes();
    let fiat_shamir_challenge_nodes: Vec<FiatShamirChallengeNode> = dag.get_nodes();

    let mut input_layer_map: HashMap<NodeId, &mut InputLayerNode> = HashMap::new();

    for layer in input_layer_nodes.iter_mut() {
        input_layer_map.insert(layer.id(), layer);
    }

    // Add InputShreds to specified parents
    for input_shred in input_shreds {
        let input_layer_id = input_shred.get_parent();
        let input_layer = input_layer_map
            .get_mut(&input_layer_id)
            .ok_or(DAGError::DanglingNodeId(input_layer_id))?;

        input_layer.add_shred(input_shred);
    }

    // handle intermediate layers
    let sector_groups: Vec<SectorGroup<F>> = dag.get_nodes();
    let sectors: Vec<Sector<F>> = dag.get_nodes();
    let gates: Vec<GateNode> = dag.get_nodes();
    let id_gates: Vec<IdentityGateNode> = dag.get_nodes();
    let splits: Vec<SplitNode> = dag.get_nodes();
    let matmults: Vec<MatMultNode> = dag.get_nodes();
    let other_layers = sector_groups
        .into_iter()
        .map(|node| IntermediateNode::new(node))
        .chain(gates.into_iter().map(|node| IntermediateNode::new(node)))
        .chain(id_gates.into_iter().map(|node| IntermediateNode::new(node)))
        .chain(splits.into_iter().map(|node| IntermediateNode::new(node)))
        .chain(matmults.into_iter().map(|node| IntermediateNode::new(node)));

    let intermediate_nodes = other_layers
        .chain(
            sectors
                .into_iter()
                .map(|sector| IntermediateNode::Sector(sector)),
        )
        .collect_vec();

    // topo_sort all the nodes which can be immediately compiled and the sectors that need to be
    // laid out before compilation
    let intermediate_nodes = topo_sort(intermediate_nodes)?;

    // collapse the sectors into sector_groups
    let (mut intermediate_nodes, final_sector_group) = intermediate_nodes.into_iter().fold(
        (vec![], None::<SectorGroup<F>>),
        |(mut layedout_nodes, curr_sector_group), node| {
            let curr_sector_group = match node {
                IntermediateNode::CompilableNode(node) => {
                    if let Some(curr_sector_group) = curr_sector_group {
                        layedout_nodes
                            .push(Box::new(curr_sector_group) as Box<dyn CompilableNode<F>>);
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

    // Handle lookup tables

    // Build a map node id -> LookupTable
    let mut lookup_table_map: HashMap<NodeId, &mut LookupTable> = HashMap::new();
    let mut lookup_tables: Vec<LookupTable> = dag.get_nodes();
    for lookup_table in lookup_tables.iter_mut() {
        lookup_table_map.insert(lookup_table.id(), lookup_table);
    }
    // Add LookupConstraints to their respective LookupTables
    let lookup_constraints: Vec<LookupConstraint> = dag.get_nodes();
    for lookup_constraint in lookup_constraints {
        let lookup_table_id = lookup_constraint.table_node_id;
        let lookup_table = lookup_table_map
            .get_mut(&lookup_table_id)
            .ok_or(DAGError::DanglingNodeId(lookup_table_id))?;
        lookup_table.add_lookup_constraint(lookup_constraint);
    }

    // handle output layers
    let output_layers: Vec<OutputNode> = dag.get_nodes();

    Ok((
        input_layer_nodes,
        fiat_shamir_challenge_nodes,
        intermediate_nodes,
        lookup_tables,
        output_layers,
    ))
}
