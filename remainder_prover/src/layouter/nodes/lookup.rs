//! Nodes that implement the LogUp.
use std::marker::PhantomData;

use remainder_shared_types::FieldExt;

use crate::{mle::evals::MultilinearExtension, prover::proof_system::ProofSystem};

use super::{CircuitNode, ClaimableNode, CompilableNode, Context, NodeId};

/// Represents the use of a lookup into a particular table (represented by a LookupNode).
#[derive(Debug, Clone)]
pub struct LookupShred<F: FieldExt> {
    id: NodeId,
    /// The id of the LookupNode (lookup table) that we are lookup up into.
    pub table: NodeId,
    /// The id of the node that is being constrained by this lookup.
    pub constrained_node_id: NodeId,
    /// (Optionally) the id of the node that provides the multiplicities for the constrained data.
    pub multiciplicites: Option<NodeId>,
    // FIXME needed?
    _data: PhantomData<F>
}

impl<F: FieldExt> LookupShred<F> {
    /// Creates a new LookupShred, constraining the data of `constrained` to form a subset of the
    /// data in `table` with multiplicities given by `multiplicities` (these will be populated
    /// automatically and added to the input layer if not specified).
    pub fn new(
        ctx: &Context,
        table: &LookupNode<F>,
        constrained: &dyn ClaimableNode<F = F>,
        multiplicities: Option<&dyn ClaimableNode<F=F>>,
    ) -> Self {
        let id = ctx.get_new_id();
        let multiplicities_id = if let Some(multiplicities) = multiplicities {
            Some(multiplicities.id())
        } else { None };
        LookupShred {
            id,
            table: table.id(),
            constrained_node_id: constrained.id(),
            multiciplicites: multiplicities_id,
            _data: PhantomData,
        }
    }

    pub fn get_input_shreds(&self) -> Vec<InputShred<F>> {
        //FIXME is this the method that is meant to be returning the multiplicities?
        // if so, I guess I had better be storing the multiplicities in the LookupShred
        todo!();
    }
}

impl<F: FieldExt> CircuitNode for LookupShred<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    // FIXME What exactly is this method meant to do?
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
        // FIXME is this correct?
        Some(self.shreds.iter().map(|shred| shred.id()).collect())
    }

    fn sources(&self) -> Vec<NodeId> {
        let mut ids = self.shreds.iter().map(|shred| shred.table).collect::<Vec<_>>();
        ids.push(self.table);
        ids
    }
}

impl<F: FieldExt, Pf: ProofSystem<F>> CompilableNode<F, Pf> for LookupNode<F> {
    fn compile<'a>(
        &'a self,
        _: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
        circuit_map: &mut crate::layouter::layouting::CircuitMap<'a, F>,
    ) -> Result<(), crate::layouter::layouting::DAGError> {

        todo!();
    }
}