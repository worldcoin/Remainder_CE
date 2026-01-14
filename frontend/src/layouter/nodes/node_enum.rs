//! A default Enum for a representation of all possible DAG Nodes

use shared_types::Field;

use crate::node_enum;

use super::{
    circuit_inputs::{InputLayerNode, InputShred},
    circuit_outputs::OutputNode,
    fiat_shamir_challenge::FiatShamirChallengeNode,
    gate::GateNode,
    identity_gate::IdentityGateNode,
    lookup::{LookupConstraint, LookupTable},
    matmult::MatMultNode,
    sector::Sector,
    split_node::SplitNode,
};

node_enum!(NodeEnum: Field,
    (InputShred: InputShred),
    (InputLayer: InputLayerNode),
    (FiatShamirChallengeNode: FiatShamirChallengeNode),
    (Output: OutputNode),
    (Sector: Sector<F>),
    (GateNode: GateNode),
    (IdentityGateNode: IdentityGateNode),
    (SplitNode: SplitNode),
    (MatMultNode: MatMultNode),
    (LookupConstraint: LookupConstraint),
    (LookupTable: LookupTable)
);
