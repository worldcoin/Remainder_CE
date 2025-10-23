//! Defines the utilities for taking a list of nodes and turning it into a
//! layedout circuit
#[cfg(test)]
mod tests;

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
    repr: HashMap<N, Vec<N>>,
}

impl<N: Hash + Eq + Clone + Debug> Graph<N> {
    /// Constructor given the map of dependencies.
    fn new_from_map(map: HashMap<N, Vec<N>>) -> Self {
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
        // Turn the Input Shred IDs into a hash set for O(1) membership queries.
        let input_shred_ids: HashSet<NodeId> = input_shred_ids.into_iter().map(|x| *x).collect();

        let mut children_to_parent_map: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
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
                                Some(*circuit_node_source)
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

        exploring_nodes.insert(node.clone());
        for neighbor_not_terminated in neighbors {
            self.visit_node_dfs(
                neighbor_not_terminated,
                exploring_nodes,
                terminated_nodes,
                topological_order,
            )?;
        }
        exploring_nodes.remove(node);

        terminated_nodes.insert(node.clone());
        topological_order.push(node.clone());

        Ok(())
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

        for node in self.repr.keys() {
            self.visit_node_dfs(
                node,
                &mut exploring_nodes,
                &mut terminated_nodes,
                &mut topological_order,
            )?;
        }

        assert_eq!(topological_order.len(), self.repr.len());

        Ok(topological_order)
    }

    /// Given a list of nodes that are already topologically ordered, we check
    /// all nodes that come before it to return the index in the topological
    /// sort of the latest node that it was dependent on.
    fn get_index_of_latest_dependency(
        &self,
        topological_order: &[N],
        idx_to_check: usize,
    ) -> Option<usize> {
        let node_to_check = &topological_order[idx_to_check];
        let neighbors = self.repr.get(node_to_check).unwrap();
        let mut latest_dependency_idx: Option<usize> = None;

        for idx in (0..idx_to_check).rev() {
            let node = &topological_order[idx];
            if neighbors.contains(node) {
                latest_dependency_idx = Some(idx);
                break;
            }
        }
        latest_dependency_idx
    }

    fn gen_latest_dependecy_indices(&self, topological_order: &[N]) -> Vec<Option<usize>> {
        let mut node_order = HashMap::<N, usize>::new();
        topological_order
            .iter()
            .enumerate()
            .for_each(|(idx, node)| {
                node_order.insert(node.clone(), idx);
            });

        let latest_dependency = topological_order
            .iter()
            .map(|u| {
                let deps = self.repr.get(u).unwrap();

                deps.iter().map(|u| node_order[u]).max()
            })
            .collect_vec();

        latest_dependency
    }

    pub fn naive_gen_latest_dependecy_indices(
        &self,
        topological_order: &[N],
    ) -> Vec<Option<usize>> {
        let n = topological_order.len();
        (0..n)
            .map(|i| self.get_index_of_latest_dependency(topological_order, i))
            .collect()
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
///   `split_nodes` are considered intermediate nodes. `sector_nodes` are the
///   only nodes which can be combined with each other via a selector.
///   Therefore, first we topologically sort the intermediate nodes by creating
///   a dependency graph using their specified sources. Then, we do a forward
///   pass through these sorted nodes, identify which ones are the sectors, and
///   combine them greedily (if there is no dependency between them, we
///   combine). We then return a [Vec<Vec<Box<dyn CompilableNode<F>>>>] for
///   which each inner vector represents nodes that can be combined into a
///   single layer.
/// * `lookup_constraint_nodes` are added to their respective
///   `lookup_table_nodes`. Because no nodes are dependent on lookups (their
///   results are always outputs), we compile them after the intermediate
///   nodes.
/// * `output_nodes` are compiled last, so they are returned as is.
///
/// The ordering in which the nodes are returned as [LayouterNodes] is the order
/// in which the nodes are expected to be compiled into layers.
#[allow(clippy::too_many_arguments)]
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
    let sector_node_ids: HashSet<NodeId> = sector_nodes.iter().map(|sector| sector.id()).collect();

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

    // Step 2b: Determine which nodes can be combined into one.
    //
    // Re-order the topological order so that nodes that are not sector nodes
    // appear in their first valid position. This means that these nodes serve
    // as dividers for which sectors can be combined with each other. I.e.,
    // sector nodes before a certain non-sector node cannot be combined with
    // sector nodes after that same non-sector node.

    // OLD WAY:
    // -------------------------------------------------------------------------
    /*
    // We topologically sort via NodeId and map them back to their respective
    // node.
    let mut topo_sorted_nodes = topo_sorted_intermediate_node_ids
        .iter()
        .map(|node_id| id_to_node_mapping.remove(node_id).unwrap())
        .collect_vec();

    topo_sorted_intermediate_node_ids
        .iter()
        .enumerate()
        .for_each(|(top_idx, node_id)| {
            if !(sector_node_ids.contains(node_id)) {
                let maybe_first_idx_can_appear = circuit_node_graph.get_index_of_latest_dependency(
                    &(topo_sorted_nodes.iter().map(|node| node.id())).collect_vec(),
                    top_idx,
                );
                let node = topo_sorted_nodes.remove(top_idx);
                if let Some(first_idx_can_appear) = maybe_first_idx_can_appear {
                    topo_sorted_nodes.insert(first_idx_can_appear + 1, node);
                } else {
                    topo_sorted_nodes.insert(0, node);
                }
            }
        });
    */
    // -------------------------------------------------------------------------

    // NEW WAY:
    // -------------------------------------------------------------------------
    let latest_dependency_indices =
        circuit_node_graph.gen_latest_dependecy_indices(&topo_sorted_intermediate_node_ids);
    debug_assert_eq!(
        latest_dependency_indices,
        circuit_node_graph.naive_gen_latest_dependecy_indices(&topo_sorted_intermediate_node_ids)
    );

    let mut adjusted_priority = vec![0; topo_sorted_intermediate_node_ids.len()];

    let mut nodes_with_priority = topo_sorted_intermediate_node_ids
        .iter()
        .enumerate()
        .map(|(idx, node_id)| {
            let node = id_to_node_mapping.remove(node_id).unwrap();
            if sector_node_ids.contains(node_id) {
                // A sector node's priority is it's current index in the topological sorting.
                if idx > 0 {
                    adjusted_priority[idx] = adjusted_priority[idx - 1] + 1;
                }
                (node, (adjusted_priority[idx], 0))
            } else {
                // A non-sector node's priority is the index of it's latest dependency.
                if idx > 0 {
                    adjusted_priority[idx] = adjusted_priority[idx - 1];
                }
                (
                    node,
                    (
                        adjusted_priority[latest_dependency_indices[idx].unwrap_or(0)],
                        1,
                    ),
                )
            }
        })
        .collect_vec();

    nodes_with_priority.sort_by_key(|&(_, priority)| priority);

    let topo_sorted_nodes = nodes_with_priority
        .into_iter()
        .map(|(val, _)| val)
        .collect_vec();

    // -------------------------------------------------------------------------

    // Turn the input shred ID vector into a hash set for easier searching.
    // let input_shred_ids: HashSet<NodeId> = input_shred_ids.into_iter().collect();
    let mut intermediate_layers: Vec<Vec<Box<dyn CompilableNode<F>>>> = Vec::new();
    let mut node_to_layer_map: HashMap<NodeId, usize> = HashMap::new();
    // The first layer that stores sectors.
    let mut first_sector_layer_idx = 0;

    // For index i in the vector, keep track of the layer number corresponding to the next
    // sector layer, None if it does not exist.
    //
    // For example, if we have the layers [non-sector, sector, non-sector, non-sector, sector, non-sector],
    // the list would be [1, 4, 4, 4, 0, 0]
    let mut next_sector_layer_idx_list = Vec::new();

    topo_sorted_nodes.into_iter().for_each(|node| {
        // If it is a non-sector node, insert it as its own layer
        // Note that non-sector nodes are already re-ordered to the earliest possible location,
        // so their layers are also the earliest possible layer.
        let layer_idx = if !(sector_node_ids.contains(&node.id())) {
            // Insert a new layer.
            intermediate_layers.push(Vec::new());
            // Next sector layer is currently unknown:
            next_sector_layer_idx_list.push(0);
            let layer_idx = intermediate_layers.len() - 1;
            // If every layer so far are non-sector, then the first sector layer hasn't appeared.
            if layer_idx == first_sector_layer_idx {
                first_sector_layer_idx += 1;
            }
            layer_idx
        }
        // If it is a sector node:
        else {
            let maybe_latest_layer_dependency = node
                .sources()
                .iter()
                .filter_map(|node_source| {
                    if input_shred_ids.contains(node_source) {
                        None
                    } else {
                        Some(node_to_layer_map.get(node_source).unwrap())
                    }
                })
                .max();
            // If it is dependent on some previous node:
            if let Some(layer_of_node) = maybe_latest_layer_dependency {
                // There is no sector layer after the layer it is dependent on.
                if next_sector_layer_idx_list[*layer_of_node] == 0 {
                    // If the dependency is at the last sector layer, create a new layer.
                    intermediate_layers.push(Vec::new());
                    // The next sector layer is currently unknown.
                    next_sector_layer_idx_list.push(0);
                    let layer_to_insert = intermediate_layers.len() - 1;
                    // All the previous layers with unknown next sector must all be the
                    // newest layers. Find them and update their next sector layer.
                    for i in (0..intermediate_layers.len() - 1).rev() {
                        if next_sector_layer_idx_list[i] == 0 {
                            next_sector_layer_idx_list[i] = layer_to_insert;
                        } else {
                            break;
                        }
                    }
                    layer_to_insert
                }
                // There exists a sector layer after the layer it is dependent on.
                else {
                    next_sector_layer_idx_list[*layer_of_node]
                }
            }
            // Otherwise it is not dependent on any node:
            else {
                // Add it to the first layer.
                if intermediate_layers.len() == first_sector_layer_idx {
                    intermediate_layers.push(Vec::new());
                    // The next sector layer is currently unknown.
                    next_sector_layer_idx_list.push(0);
                    first_sector_layer_idx = intermediate_layers.len() - 1
                }
                first_sector_layer_idx
            }
        };
        node_to_layer_map.insert(node.id(), layer_idx);
        intermediate_layers[layer_idx].push(node);
    });

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
