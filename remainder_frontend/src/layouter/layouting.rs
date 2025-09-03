//! Defines the utilities for taking a list of nodes and turning it into a
//! layedout circuit
#[cfg(test)]
mod tests;

use itertools::any;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

use itertools::Itertools;
use remainder_shared_types::Field;
use thiserror::Error;

use crate::layouter::nodes::sector::Sector;

use super::nodes::{
    circuit_inputs::{InputLayerNode, InputShred},
    circuit_outputs::OutputNode,
    fiat_shamir_challenge::FiatShamirChallengeNode,
    gate::GateNode,
    identity_gate::IdentityGateNode,
    lookup::{LookupConstraint, LookupTable},
    matmult::MatMultNode,
    split_node::SplitNode,
    CircuitNode, CompilableNode, NodeId,
};

use anyhow::{anyhow, Result};

/// Possible errors when topologically ordering the dependency graph that arises
/// from circuit creation and then categorizing them into layers.
#[derive(Error, Debug, Clone)]
pub enum LayoutingError {
    /// There is a cycle in the node dependencies.
    #[error("There exists a cycle in the dependency of the nodes, and therefore no topological ordering.")]
    CircularDependency,
    /// There exists a node which the circuit builder created, but no other node
    /// references or depends on.
    #[error("There exists a node which the circuit builder created, but no other node references or depends on: Id = {0:?}")]
    DanglingNodeId(NodeId),
    /// We have gotten to a layer whose parts of the expression have not been
    /// generated.
    #[error("This circuit location does not exist, or has not been compiled yet")]
    NoCircuitLocation,
}
/// A directed graph represented by a HashMap from a node in the graph to the
/// nodes that it depends on. I.e., there is a directed edge from the node in
/// the key to all the nodes in its values.
#[derive(Clone, Debug)]
pub struct Graph<N: Hash + Eq + Clone + Debug> {
    repr: HashMap<N, HashSet<N>>,
}

impl<N: Hash + Eq + Clone + Debug> Graph<N> {
    /// Constructor given the map of dependencies.
    fn new_from_map(map: HashMap<N, HashSet<N>>) -> Self {
        Self { repr: map }
    }
    /// Constructor specifically for a [Graph<NodeId>], which will convert an
    /// array of [CompilableNode], each of which reference their sources, and
    /// convert that into the graph representation.
    ///
    /// Note: we only topologically sort intermediate nodes, therefore we
    /// provide `input_shred_ids` to exclude them from the graph.
    fn new_from_circuit_nodes<F: Field>(
        intermediate_circuit_nodes: &[Box<dyn CompilableNode<F>>],
        input_shred_ids: &[NodeId],
    ) -> Graph<NodeId> {
        let mut children_to_parent_map: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        intermediate_circuit_nodes.iter().for_each(|circuit_node| {
            // Disregard the nodes which are input shreds.
            if !input_shred_ids.contains(&circuit_node.id()) {
                children_to_parent_map.insert(
                    circuit_node.id(),
                    circuit_node
                        .sources()
                        .iter()
                        .filter_map(|circuit_node_source| {
                            // If any of the sources are input shreds, we don't
                            // add them to the graph.
                            if input_shred_ids.contains(circuit_node_source) {
                                None
                            } else {
                                Some(circuit_node_source.clone())
                            }
                        })
                        .collect(),
                );
            }
        });
        Graph::<NodeId>::new_from_map(children_to_parent_map)
    }

    /// Given the graph, a starting node (`node`), the set of visited/"marked"
    /// nodes (`exploring_nodes`) and nodes that have been fully explored
    /// (`terminated_nodes`), traverse the graph recursively using the DFS
    /// algorithm from the starting node.
    ///
    /// Returns nothing, but mutates the `exploring_nodes` and
    /// `terminated_nodes` accordingly.
    fn visit_node_dfs(
        &self,
        node: &N,
        exploring_nodes: &mut HashSet<N>,
        terminated_nodes: &mut HashSet<N>,
        topological_order: &mut Vec<N>,
    ) -> Result<()> {
        if terminated_nodes.contains(node) {
            return Ok(());
        }
        if exploring_nodes.contains(node) {
            return Err(anyhow!(LayoutingError::CircularDependency));
        }
        let neighbors = self.repr.get(node).unwrap();
        if neighbors.len() == 0 {
            exploring_nodes.remove(node);
            terminated_nodes.insert(node.clone());
            topological_order.push(node.clone());
            return Ok(());
        } else {
            exploring_nodes.insert(node.clone());
            for neighbor_not_terminated in neighbors {
                self.visit_node_dfs(
                    neighbor_not_terminated,
                    exploring_nodes,
                    terminated_nodes,
                    topological_order,
                )?;
            }
        }
        exploring_nodes.remove(node);
        terminated_nodes.insert(node.clone());
        topological_order.push(node.clone());
        Ok(())
    }

    /// Returns whether `node_a` is dependent on any of the nodes in
    /// `node_list`. (I.e., there $\exists$ path from `node_a` to `node_b`
    /// $\forall `node_b` \in `node_list`$).
    fn check_path_from(&self, node_a: &N, node_list: &[N]) -> Result<bool> {
        let mut topological_order = Vec::new();
        let mut terminated_nodes = HashSet::new();
        let mut exploring_nodes = HashSet::new();
        self.visit_node_dfs(
            node_a,
            &mut exploring_nodes,
            &mut terminated_nodes,
            &mut topological_order,
        )?;
        Ok(any(node_list, |node_b| terminated_nodes.contains(node_b)))
    }

    /// Topologically orders the graph by visiting each node via DFS. Returns
    /// the terminated nodes in order of termination.
    ///
    /// Note: normally, the topological sort is the "reverse" order of
    /// termination via DFS. However, since we wish to order from sources ->
    /// sinks, and our dependency graph is structured as children -> parent, we
    /// do not reverse the termination order of the DFS search.
    fn topo_sort(&self) -> Result<Vec<N>> {
        let mut terminated_nodes: HashSet<N> = HashSet::new();
        let mut topological_order: Vec<N> = Vec::with_capacity(self.repr.len());
        let mut exploring_nodes: HashSet<N> = HashSet::new();

        while !(topological_order.len() == self.repr.len()) {
            for node in self.repr.keys() {
                self.visit_node_dfs(
                    node,
                    &mut exploring_nodes,
                    &mut terminated_nodes,
                    &mut topological_order,
                )?;
            }
        }
        Ok(topological_order)
    }
}

/// The type returned by the [layout] function. Categorizes the nodes into their
/// respective layers.
type LayouterNodes<F> = (
    Vec<InputLayerNode>,
    Vec<FiatShamirChallengeNode>,
    // The inner vector represents nodes to be combined into the same layer. The
    // outer vector is in the order of the layers to be compiled in terms of
    // dependency.
    Vec<Vec<Box<dyn CompilableNode<F>>>>,
    Vec<LookupTable>,
    Vec<OutputNode>,
);

/// Given the nodes provided by the circuit builder, this function returns a
/// tuple of type [LayouterNodes].
///
/// This function categorizes nodes into their respective layers, by doing the
/// following:
/// * Assigning `input_shred_nodes` to their respective `input_layer_nodes`
///   parent.
/// * The `fiat_shamir_challenge_nodes` are to be compiled next, so they are
///   returned as is.
/// * `sector_nodes`, `gate_nodes`, `identity_gate_nodes`, `matmult_nodes`, and
///    `split_nodes` are considered intermediate nodes. `sector_nodes` are the
///    only nodes which can be combined with each other via a selector.
///    Therefore, first we topologically sort the intermediate nodes by creating
///    a dependency graph using their specified sources. Then, we do a forward
///    pass through these sorted nodes, identify which ones are the sectors, and
///    combine them greedily (if there is no dependency between them, we
///    combine). We then return a [Vec<Vec<Box<dyn CompilableNode<F>>>>] for
///    which each inner vector represents nodes that can be combined into a
///    single layer.
/// * `lookup_constraint_nodes` are added to their respective
///    `lookup_table_nodes`. Because no nodes are dependent on lookups (their
///    results are always outputs), we compile them after the intermediate
///    nodes.
/// * `output_nodes` are compiled last, so they are returned as is.
///
/// The ordering in which the nodes are returned as [LayouterNodes] is the order
/// in which the nodes are expected to be compiled into layers.
pub fn layout<F: Field>(
    mut input_layer_nodes: Vec<InputLayerNode>,
    input_shred_nodes: Vec<InputShred>,
    fiat_shamir_challenge_nodes: Vec<FiatShamirChallengeNode>,
    output_nodes: Vec<OutputNode>,
    sector_nodes: Vec<Sector<F>>,
    gate_nodes: Vec<GateNode>,
    identity_gate_nodes: Vec<IdentityGateNode>,
    split_nodes: Vec<SplitNode>,
    matmult_nodes: Vec<MatMultNode>,
    lookup_constraint_nodes: Vec<LookupConstraint>,
    mut lookup_table_nodes: Vec<LookupTable>,
) -> Result<LayouterNodes<F>> {
    let mut input_layer_map: HashMap<NodeId, &mut InputLayerNode> = HashMap::new();
    let sector_node_ids: Vec<NodeId> = sector_nodes.iter().map(|sector| sector.id()).collect();

    for layer in input_layer_nodes.iter_mut() {
        input_layer_map.insert(layer.id(), layer);
    }

    // Step 1: Add `input_shred_nodes` to their specified `input_layer_nodes`
    // parent.
    let mut input_shred_ids = vec![];
    for input_shred in input_shred_nodes {
        let input_layer_id = input_shred.get_parent();
        input_shred_ids.push(input_shred.id());
        let input_layer = input_layer_map
            .get_mut(&input_layer_id)
            .ok_or(LayoutingError::DanglingNodeId(input_layer_id))?;
        input_layer.add_shred(input_shred);
    }
    input_shred_ids.extend(fiat_shamir_challenge_nodes.iter().map(|node| node.id()));

    // We cast all intermediate nodes into their generic trait implementation
    // type.
    let intermediate_nodes = gate_nodes
        .into_iter()
        .map(|node| Box::new(node) as Box<dyn CompilableNode<F>>)
        .chain(
            identity_gate_nodes
                .into_iter()
                .map(|node| Box::new(node) as Box<dyn CompilableNode<F>>),
        )
        .chain(
            split_nodes
                .into_iter()
                .map(|node| Box::new(node) as Box<dyn CompilableNode<F>>),
        )
        .chain(
            matmult_nodes
                .into_iter()
                .map(|node| Box::new(node) as Box<dyn CompilableNode<F>>),
        )
        .chain(
            sector_nodes
                .into_iter()
                .map(|node| Box::new(node) as Box<dyn CompilableNode<F>>),
        )
        .collect_vec();

    // Step 2a: Determine the topological ordering of intermediate nodes, given
    // their sources.
    let circuit_node_graph =
        Graph::<NodeId>::new_from_circuit_nodes(&intermediate_nodes, &input_shred_ids);

    let mut id_to_node_mapping: HashMap<NodeId, Box<dyn CompilableNode<F>>> = intermediate_nodes
        .into_iter()
        .map(|node| (node.id(), node))
        .collect();
    let topo_sorted_intermediate_node_ids = &circuit_node_graph.topo_sort()?;

    // We topologically sort via NodeId and map them back to their respective
    // node.
    let mut topo_sorted_nodes = topo_sorted_intermediate_node_ids
        .iter()
        .map(|node_id| id_to_node_mapping.remove(node_id).unwrap())
        .collect_vec();
    let mut intermediate_layers: Vec<Vec<Box<dyn CompilableNode<F>>>> = Vec::new();
    // Step 2b: Determine which nodes can be combined into one.
    if !(topo_sorted_nodes.is_empty()) {
        let first_node = topo_sorted_nodes.remove(0);
        let mut current_layer: Vec<Box<dyn CompilableNode<F>>> = vec![first_node];
        for node in topo_sorted_nodes.drain(..) {
            let prev_node_id = &current_layer.last().unwrap().id();
            let current_layer_ids = &current_layer.iter().map(|sector| sector.id()).collect_vec();
            // If all the nodes in a group are sectors, and there is no
            // dependency from the current group of sectors to the current node,
            // then we add it to the group.
            if sector_node_ids.contains(prev_node_id)
                && sector_node_ids.contains(&node.id())
                && !(circuit_node_graph.check_path_from(&node.id(), current_layer_ids)).unwrap()
            {
                current_layer.push(node);
            // Otherwise, we add the current group to the list of intermediate
            // layers, and create a new group with the current node.
            } else {
                intermediate_layers.push(current_layer);
                current_layer = vec![node]
            }
        }
        // Edge case for the last layer.
        if !current_layer.is_empty() {
            intermediate_layers.push(current_layer);
        }
    }

    // Step 3: Add LookupConstraints to their respective LookupTables. Build a
    // map node id -> LookupTable
    let mut lookup_table_map: HashMap<NodeId, &mut LookupTable> = HashMap::new();
    for lookup_table in lookup_table_nodes.iter_mut() {
        lookup_table_map.insert(lookup_table.id(), lookup_table);
    }
    for lookup_constraint in lookup_constraint_nodes {
        let lookup_table_id = lookup_constraint.table_node_id;
        let lookup_table = lookup_table_map
            .get_mut(&lookup_table_id)
            .ok_or(LayoutingError::DanglingNodeId(lookup_table_id))?;
        lookup_table.add_lookup_constraint(lookup_constraint);
    }

    Ok((
        input_layer_nodes,
        fiat_shamir_challenge_nodes,
        intermediate_layers,
        lookup_table_nodes,
        output_nodes,
    ))
}
