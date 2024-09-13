//! A default Enum for a representation of all possible DAG Nodes

use remainder_shared_types::Field;

use crate::node_enum;

use super::{
    circuit_inputs::{InputLayerNode, InputShred},
    circuit_outputs::OutputNode,
    debug::DebugNode,
    gate::GateNode,
    identity_gate::IdentityGateNode,
    lookup::{LookupTable, LookupConstraint},
    matmult::MatMultNode,
    sector::{Sector, SectorGroup},
    split_node::SplitNode,
    NodeGroup, YieldNode,
};

node_enum!(NodeEnum: Field,
    (InputShred: InputShred<F>),
    (InputLayer: InputLayerNode<F>),
    (Output: OutputNode<F>),
    (Debug: DebugNode),
    (Sector: Sector<F>),
    (SectorGroup: SectorGroup<F>),
    (GateNode: GateNode<F>),
    (IdentityGateNode: IdentityGateNode<F>),
    (SplitNode: SplitNode<F>),
    (MatMultNode: MatMultNode<F>),
    (LookupConstraint: LookupConstraint),
    (LookupTable: LookupTable)
);

/// Organizational wrapper for a vec of `NodeEnum`s
pub struct NodeEnumGroup<F: Field> {
    input_shreds: Option<Vec<InputShred<F>>>,
    input_layers: Option<Vec<InputLayerNode<F>>>,
    output: Option<Vec<OutputNode<F>>>,
    debugs: Option<Vec<DebugNode>>,
    sectors: Option<Vec<Sector<F>>>,
    sector_groups: Option<Vec<SectorGroup<F>>>,
    gate_nodes: Option<Vec<GateNode<F>>>,
    identity_gate_nodes: Option<Vec<IdentityGateNode<F>>>,
    split_nodes: Option<Vec<SplitNode<F>>>,
    matmult_nodes: Option<Vec<MatMultNode<F>>>,
    lookup_tables: Option<Vec<LookupTable>>,
    lookup_constraints: Option<Vec<LookupConstraint>>,
}

impl<F: Field> NodeGroup for NodeEnumGroup<F> {
    type NodeEnum = NodeEnum<F>;

    fn new(nodes: Vec<Self::NodeEnum>) -> Self {
        let mut out = Self {
            input_shreds: Some(vec![]),
            input_layers: Some(vec![]),
            output: Some(vec![]),
            debugs: Some(vec![]),
            sectors: Some(vec![]),
            sector_groups: Some(vec![]),
            gate_nodes: Some(vec![]),
            identity_gate_nodes: Some(vec![]),
            split_nodes: Some(vec![]),
            matmult_nodes: Some(vec![]),
            lookup_tables: Some(vec![]),
            lookup_constraints: Some(vec![]),
        };

        for node in nodes {
            match node {
                NodeEnum::InputShred(node) => out.input_shreds.as_mut().unwrap().push(node),
                NodeEnum::InputLayer(node) => out.input_layers.as_mut().unwrap().push(node),
                NodeEnum::Output(node) => out.output.as_mut().unwrap().push(node),
                NodeEnum::Debug(node) => out.debugs.as_mut().unwrap().push(node),
                NodeEnum::Sector(node) => out.sectors.as_mut().unwrap().push(node),
                NodeEnum::SectorGroup(node) => out.sector_groups.as_mut().unwrap().push(node),
                NodeEnum::GateNode(node) => out.gate_nodes.as_mut().unwrap().push(node),
                NodeEnum::IdentityGateNode(node) => {
                    out.identity_gate_nodes.as_mut().unwrap().push(node)
                }
                NodeEnum::SplitNode(node) => out.split_nodes.as_mut().unwrap().push(node),
                NodeEnum::MatMultNode(node) => out.matmult_nodes.as_mut().unwrap().push(node),
                NodeEnum::LookupTable(node) => out.lookup_tables.as_mut().unwrap().push(node),
                NodeEnum::LookupConstraint(node) => out.lookup_constraints.as_mut().unwrap().push(node),
            }
        }
        out
    }
}

impl<F: Field> YieldNode<InputShred<F>> for NodeEnumGroup<F> {
    fn get_nodes(&mut self) -> Vec<InputShred<F>> {
        self.input_shreds.take().unwrap_or_default()
    }
}

impl<F: Field> YieldNode<InputLayerNode<F>> for NodeEnumGroup<F> {
    fn get_nodes(&mut self) -> Vec<InputLayerNode<F>> {
        self.input_layers.take().unwrap_or_default()
    }
}
impl<F: Field> YieldNode<OutputNode<F>> for NodeEnumGroup<F> {
    fn get_nodes(&mut self) -> Vec<OutputNode<F>> {
        self.output.take().unwrap_or_default()
    }
}
impl<F: Field> YieldNode<DebugNode> for NodeEnumGroup<F> {
    fn get_nodes(&mut self) -> Vec<DebugNode> {
        self.debugs.take().unwrap_or_default()
    }
}
impl<F: Field> YieldNode<Sector<F>> for NodeEnumGroup<F> {
    fn get_nodes(&mut self) -> Vec<Sector<F>> {
        self.sectors.take().unwrap_or_default()
    }
}
impl<F: Field> YieldNode<SectorGroup<F>> for NodeEnumGroup<F> {
    fn get_nodes(&mut self) -> Vec<SectorGroup<F>> {
        self.sector_groups.take().unwrap_or_default()
    }
}

impl<F: Field> YieldNode<GateNode<F>> for NodeEnumGroup<F> {
    fn get_nodes(&mut self) -> Vec<GateNode<F>> {
        self.gate_nodes.take().unwrap_or_default()
    }
}

impl<F: Field> YieldNode<IdentityGateNode<F>> for NodeEnumGroup<F> {
    fn get_nodes(&mut self) -> Vec<IdentityGateNode<F>> {
        self.identity_gate_nodes.take().unwrap_or_default()
    }
}

impl<F: Field> YieldNode<SplitNode<F>> for NodeEnumGroup<F> {
    fn get_nodes(&mut self) -> Vec<SplitNode<F>> {
        self.split_nodes.take().unwrap_or_default()
    }
}

impl<F: Field> YieldNode<MatMultNode<F>> for NodeEnumGroup<F> {
    fn get_nodes(&mut self) -> Vec<MatMultNode<F>> {
        self.matmult_nodes.take().unwrap_or_default()
    }
}

impl<F: Field> YieldNode<LookupTable> for NodeEnumGroup<F> {
    fn get_nodes(&mut self) -> Vec<LookupTable> {
        self.lookup_tables.take().unwrap_or_default()
    }
}

impl<F: Field> YieldNode<LookupConstraint> for NodeEnumGroup<F> {
    fn get_nodes(&mut self) -> Vec<LookupConstraint> {
        self.lookup_constraints.take().unwrap_or_default()
    }
}