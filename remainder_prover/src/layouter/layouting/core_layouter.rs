use remainder_shared_types::FieldExt;

use crate::{
    layouter::{compiling::DAG, component::Component, nodes::CircuitNode},
    prover::proof_system::DefaultProofSystem,
};

use super::Layouter;

pub struct CoreLayouter;

impl<F: FieldExt, N: CircuitNode> Layouter<F, N> for CoreLayouter {
    type ProofSystem = DefaultProofSystem;

    fn layout<C: Component<N>>(pre_circuit: C) -> crate::prover::Witness<F, Self::ProofSystem> {
        let nodes = pre_circuit.yield_nodes();
        let dag = DAG::new(nodes);
        todo!()
    }
}
