//! Nodes that implement LogUp.
use std::marker::PhantomData;

use remainder_shared_types::FieldExt;

use crate::{mle::evals::MultilinearExtension, prover::proof_system::ProofSystem};

use super::{CircuitNode, ClaimableNode, CompilableNode, Context, NodeId};

/// Represents the use of a lookup into a particular table (represented by a LookupNode).
#[derive(Debug, Clone)]
pub struct LookupShred<F: FieldExt> {
    pub id: NodeId,
    /// The id of the LookupNode (lookup table) that we are a lookup up into.
    pub table: NodeId,
    /// The id of the node that is being constrained by this lookup.
    pub constrained_node_id: NodeId,
    /// The node that provides the multiplicities for the constrained data.
    pub multiplicities: &dyn ClaimableNode<F=F>,
    // FIXME needed?
    _data: PhantomData<F>
}

impl<F: FieldExt> LookupShred<F> {
    /// Creates a new LookupShred, constraining the data of `constrained` to form a subset of the
    /// data in `table` with multiplicities given by `multiplicities`.
    pub fn new(
        ctx: &Context,
        table: &LookupNode<F>,
        constrained: &dyn ClaimableNode<F = F>,
        multiplicities: &dyn ClaimableNode<F=F>,
    ) -> Self {
        let id = ctx.get_new_id();
        LookupShred {
            id,
            table: table.id(),
            constrained_node_id: constrained.id(),
            multiplicities,
            _data: PhantomData,
        }
    }
}

impl<F: FieldExt> CircuitNode for LookupShred<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![]
    }
}

#[derive(Debug, Clone)]
pub struct LookupNode<F: FieldExt> {
    id: NodeId,
    /// The lookups that are performed on this table (will be populated by calls to add_shred).
    shreds: Vec<LookupShred<F>>,
    /// The id of the node providing the table entries.
    table: NodeId,
}

impl<F: FieldExt> LookupNode<F> {
    /// Create a new table to use for subsequent lookups.
    /// (Perform a lookup in this table by creating a [LookupShred].)
    pub fn new(
        ctx: &Context,
        table: NodeId,
        table_data: MultilinearExtension<F>,
    ) -> Self {
        let id = ctx.get_new_id();

        LookupNode {
            id,
            shreds: vec![],
            table,
        }
    }

    /// Add a lookup shred to this node.
    /// (Will be called by the layouter when laying out the circuit.)
    pub fn add_shred(&mut self, shred: LookupShred<F>) {
        self.shreds.push(shred);
    }
}

impl<F: FieldExt> CircuitNode for LookupNode<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn children(&self) -> Option<Vec<NodeId>> {
        Some(self.shreds.iter().map(|shred| shred.id()).collect())
    }

    fn sources(&self) -> Vec<NodeId> {
        self.shreds.iter().map(|shred| shred.id()).collect()
    }
}

impl<F: FieldExt, Pf: ProofSystem<F>> CompilableNode<F, Pf> for LookupNode<F> {
    fn compile<'a>(
        &'a self,
        _: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
        circuit_map: &mut crate::layouter::layouting::CircuitMap<'a, F>,
    ) -> Result<(), crate::layouter::layouting::DAGError> {

        // todo: check the SectorGroup code

        // add the table data to an input layer?

        // get the MLEs of the constrained nodes and concatenate them

        todo!();
    }
}