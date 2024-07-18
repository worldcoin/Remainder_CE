//! Nodes that implement LogUp.
use std::marker::PhantomData;

use remainder_shared_types::FieldExt;

use crate::{expression::abstract_expr::{self, ExprBuilder}, layer::regular_layer::RegularLayer, mle::evals::MultilinearExtension, prover::proof_system::ProofSystem};

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
        witness_builder: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
        circuit_map: &mut crate::layouter::layouting::CircuitMap<'a, F>,
    ) -> Result<(), crate::layouter::layouting::DAGError> {
        // FIXME: make this work for multiple shreds
        assert_eq!(self.shreds.len(), 1, "LookupNode should have exactly one shred (for now)");
        let shred = &self.shreds[0];
        let constrained = shred.constrained_node_id;

        // TODO (future) get the MLEs of the constrained nodes and concatenate them
        let (_, constrained_mle) = circuit_map.0[shred.constrained_node_id];

        // TODO (future) get the MLEs of the multiplicities and add them all together
        let multiplicities = shred.multiplicities.get_data();

        // TODO (future) Draw a random value from the transcript
        let r = F::from(1u64); // FIXME
        let r_mle = MultilinearExtension::<F>::new(vec![r]);
        let r_layer_id = witness_builder.next_input_layer();
        witness_builder.add_input_layer(PublicInputLayer::new(r_mle, r_layer_id).into());

        // Form r - constrained
        // How do we get the expr() of an inputlayer?
        let expr = (r_layer_id.expr() - constrained.expr()).build_prover_expr(circuit_map)?;
        let layer = RegularLayer::new_raw(witness_builder.next_layer(), expr);
        witness_builder.add_layer(layer.into());
        let r_minus_constrained = MultilinearExtension::new(
            contrained_mle
                .iter()
                .map(|val| r - val)
                .collect()
        );

        let witness_numerators = ExprBuilder<F>::constant(F::one());

        Ok(())
    }
}