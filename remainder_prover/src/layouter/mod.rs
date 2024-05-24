//! Module for easily creating Circuits from re-usable components

use remainder_shared_types::FieldExt;

use crate::prover::{proof_system::ProofSystem, Witness};

use self::nodes::CircuitNode;

pub mod compiling;
pub mod component;
pub mod layouting;
pub mod nodes;

pub trait Layouter<F: FieldExt, N: CircuitNode, Pf: ProofSystem<F>> {
    fn layout(input_nodes: Vec<N>) -> Witness<F, Pf>;
}
