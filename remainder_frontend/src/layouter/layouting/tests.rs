use std::collections::{HashMap, HashSet};

use crate::abstract_expr::AbstractExpression;
use crate::layouter::{
    layouting::{layout, Graph},
    nodes::{
        circuit_inputs::{InputLayerNode, InputShred},
        circuit_outputs::OutputNode,
        fiat_shamir_challenge::FiatShamirChallengeNode,
        gate::GateNode,
        identity_gate::IdentityGateNode,
        lookup::{LookupConstraint, LookupTable},
        matmult::MatMultNode,
        sector::Sector,
        split_node::SplitNode,
        CircuitNode, CompilableNode, NodeId,
    },
};
use itertools::all;
use quickcheck::{quickcheck, Arbitrary, TestResult};
use remainder::layer::gate::BinaryOperation;
use remainder::mle::evals::MultilinearExtension;
use remainder_shared_types::{Field, Fr};

/// Dependency graph wrapper for Quickcheck.
impl Arbitrary for Graph<usize> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        // Instance size.
        let n = 15;
        // Number of sources.
        let num_sources = n;
        // Number of intermediate nodes.
        let num_intermediates = 3 * n;
        // Number of sinks.
        let num_sinks = n;

        let mut id_counter: usize = 0;

        // Start with the input nodes.
        let mut graph: HashMap<usize, HashSet<usize>> = (0..num_sources)
            .map(|_| {
                let res = (id_counter, HashSet::new());
                id_counter += 1;
                res
            })
            .collect();

        for _ in 0..num_intermediates {
            let available_edges: Vec<usize> = graph.keys().cloned().collect();
            let inputs: HashSet<usize> = available_edges
                .into_iter()
                .filter(|_| *g.choose(&[true, false]).unwrap())
                .collect();
            graph.insert(id_counter, inputs);
            id_counter += 1;
        }

        let available_edges: Vec<usize> = graph.keys().cloned().collect();

        for _ in 0..num_sinks {
            let inputs: HashSet<usize> = available_edges
                .iter()
                .filter(|_| *g.choose(&[true, false]).unwrap())
                .cloned()
                .collect();
            graph.insert(id_counter, inputs);
            id_counter += 1;
        }

        Graph::new_from_map(graph)
    }
}

type LayouterNodes<F> = (
    Vec<InputLayerNode>,
    Vec<FiatShamirChallengeNode>,
    Vec<Vec<Box<dyn CompilableNode<F>>>>,
    Vec<LookupTable>,
    Vec<OutputNode>,
);

#[derive(Clone, Debug)]
struct NodesToLayout<F: Field> {
    input_layer_nodes: Vec<InputLayerNode>,
    input_shred_nodes: Vec<InputShred>,
    fiat_shamir_challenge_nodes: Vec<FiatShamirChallengeNode>,
    output_nodes: Vec<OutputNode>,
    sector_nodes: Vec<Sector<F>>,
    gate_nodes: Vec<GateNode>,
    identity_gate_nodes: Vec<IdentityGateNode>,
    split_nodes: Vec<SplitNode>,
    matmult_nodes: Vec<MatMultNode>,
    lookup_constraint_nodes: Vec<LookupConstraint>,
    lookup_table_nodes: Vec<LookupTable>,
}

/// Layouting nodes wrapper for Quickcheck.
impl<F: Field> NodesToLayout<F> {
    fn layout(&self) -> LayouterNodes<F> {
        layout(
            self.input_layer_nodes.clone(),
            self.input_shred_nodes.clone(),
            self.fiat_shamir_challenge_nodes.clone(),
            self.output_nodes.clone(),
            self.sector_nodes.clone(),
            self.gate_nodes.clone(),
            self.identity_gate_nodes.clone(),
            self.split_nodes.clone(),
            self.matmult_nodes.clone(),
            self.lookup_constraint_nodes.clone(),
            self.lookup_table_nodes.clone(),
        )
        .unwrap()
    }
}

impl Arbitrary for NodesToLayout<Fr> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        // Instance size.
        let n = 15;
        // Number of input nodes (all are sources).
        let num_input_shreds = n;
        // Number of FS nodes
        let num_fs_nodes = n / 2;
        // Number of intermediate nodes.
        let num_sectors = 3 * n;

        let dummy_data = MultilinearExtension::<Fr>::new_sized_zero(2);

        // Start with the input nodes.
        let (input_layer_nodes, input_shred_nodes): (Vec<InputLayerNode>, Vec<InputShred>) = (0
            ..num_input_shreds)
            .map(|_| {
                let input_node = InputLayerNode::new(None);
                let input_shred = InputShred::new(dummy_data.num_vars(), &input_node);
                (input_node, input_shred)
            })
            .unzip();
        let fs_challenge_nodes: Vec<FiatShamirChallengeNode> = (0..num_fs_nodes)
            .map(|_| FiatShamirChallengeNode::new(2))
            .collect();
        let mut nodes_so_far: Vec<Box<dyn CircuitNode>> = input_shred_nodes
            .iter()
            .map(|node| Box::new(node.clone()) as Box<dyn CircuitNode>)
            .chain(
                fs_challenge_nodes
                    .iter()
                    .map(|node| Box::new(node.clone()) as Box<dyn CircuitNode>),
            )
            .collect();

        let mut sector_nodes = vec![];
        let mut gate_nodes = vec![];
        let mut id_gate_nodes = vec![];
        let mut split_nodes = vec![];
        let mut matmult_nodes = vec![];

        for _ in 0..num_sectors {
            let available_edges: Vec<NodeId> = nodes_so_far.iter().map(|node| node.id()).collect();

            // Add a sector by randomly selecting its sources.
            let inputs: Vec<NodeId> = available_edges
                .into_iter()
                .filter(|_| *g.choose(&[true, false]).unwrap())
                .collect();

            let sector = Sector::new(AbstractExpression::products(inputs), 2);
            nodes_so_far.push(Box::new(sector.clone()) as Box<dyn CircuitNode>);
            sector_nodes.push(sector);

            // Choose if we add a [GateNode] via coin flip.
            if *g.choose(&[true, false]).unwrap() {
                let lhs = g.choose(&sector_nodes).unwrap();
                let rhs = g.choose(&sector_nodes).unwrap();
                let gate = GateNode::new(lhs, rhs, vec![], BinaryOperation::Add, Some(2));
                nodes_so_far.push(Box::new(gate.clone()) as Box<dyn CircuitNode>);
                gate_nodes.push(gate);
            }
            // Choose if we add a [IdentityGateNode] via coin flip.
            if *g.choose(&[true, false]).unwrap() {
                let source = g.choose(&sector_nodes).unwrap();
                let id_gate = IdentityGateNode::new(source, vec![], 2, Some(2));
                nodes_so_far.push(Box::new(id_gate.clone()) as Box<dyn CircuitNode>);
                id_gate_nodes.push(id_gate);
            }
            // Choose if we add a [SplitNode] via coin flip.
            if *g.choose(&[true, false]).unwrap() {
                let source = g.choose(&sector_nodes).unwrap();
                let split_node = SplitNode::new(source, 1);
                nodes_so_far.extend(
                    split_node
                        .iter()
                        .map(|node| Box::new(node.clone()) as Box<dyn CircuitNode>),
                );
                split_nodes.extend(split_node);
            }
            // Choose if we add a [MatMultNode] via coin flip.
            if *g.choose(&[true, false]).unwrap() {
                let lhs = g.choose(&sector_nodes).unwrap();
                let rhs = g.choose(&sector_nodes).unwrap();
                let matmult = MatMultNode::new(lhs, (1, 2), rhs, (2, 1));
                nodes_so_far.push(Box::new(matmult.clone()) as Box<dyn CircuitNode>);
                matmult_nodes.push(matmult);
            }
        }
        NodesToLayout {
            input_layer_nodes,
            input_shred_nodes,
            fiat_shamir_challenge_nodes: fs_challenge_nodes,
            output_nodes: vec![],
            sector_nodes,
            gate_nodes,
            identity_gate_nodes: id_gate_nodes,
            split_nodes,
            matmult_nodes,
            lookup_constraint_nodes: vec![],
            lookup_table_nodes: vec![],
        }
    }
}

quickcheck! {
    fn toposorted_property(graph: Graph<usize>) -> TestResult {
        let sorted_nodes = &graph.topo_sort().unwrap();
        let mut ids_seen = HashSet::new();

        // Every node's sources, in order of topo sort, must have already been seen.
        TestResult::from_bool(sorted_nodes.into_iter().all(|u| {
            let vs = graph.repr.get(&u).unwrap();
            ids_seen.insert(u);
            vs.into_iter().all(|v_id| ids_seen.contains(&v_id))
        }))
    }
}

quickcheck! {
    fn test_layout_nodes(nodes_to_layout: NodesToLayout<Fr>) -> TestResult {
        let (input_layer_nodes, fs_challenge_nodes, intermediate_nodes, _, _) =
            nodes_to_layout.layout();
        let mut ids_seen = HashSet::new();
        input_layer_nodes.iter().for_each(|input_layer| {
            input_layer.input_shreds.iter().for_each(|shred| {
                ids_seen.insert(shred.id());
            })
        });
        ids_seen.extend(fs_challenge_nodes.iter().map(|fs_node| fs_node.id()));
        TestResult::from_bool(all(intermediate_nodes, |group| {
            // Every source of a node in a group must have been seen before we add
            // them to the list of ids already seen.
            let group_sources_seen = all(&group, |node| {
                let vs = node.sources();
                all(vs, |v_node| ids_seen.contains(&v_node))
            });
            ids_seen.extend(group.iter().map(|node| node.id()));
            group_sources_seen
        }))
    }
}

#[test]
#[should_panic]
fn test_error_on_cycle() {
    let mut nodes_map: HashMap<usize, HashSet<usize>> = HashMap::new();
    nodes_map.insert(1, HashSet::<usize>::new());
    nodes_map.insert(2, HashSet::<usize>::new());
    nodes_map.insert(3, HashSet::<usize>::new());
    nodes_map.insert(4, HashSet::<usize>::new());
    nodes_map.get_mut(&1).unwrap().insert(2);
    nodes_map.get_mut(&1).unwrap().insert(3);
    nodes_map.get_mut(&4).unwrap().insert(1);
    nodes_map.get_mut(&4).unwrap().insert(2);
    nodes_map.get_mut(&3).unwrap().insert(4);
    // Given the above nodes, there is a dependency cycle 1 -> 3 -> 4 -> 1.
    let nodes_graph = Graph::<usize>::new_from_map(nodes_map);
    nodes_graph.topo_sort().unwrap();
}
