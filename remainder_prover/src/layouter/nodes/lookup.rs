//! Nodes that implement LogUp.
use std::marker::PhantomData;

use remainder_shared_types::FieldExt;

use crate::{expression::{abstract_expr::{self, ExprBuilder}, generic_expr::Expression}, layer::regular_layer::RegularLayer, mle::{dense::DenseMle, evals::MultilinearExtension}, prover::proof_system::ProofSystem};

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
        let r_densemle = DenseMle::new_with_prefix_bits(r_mle, r_layer_id, vec![]);

        // Form r - constrained
        let expr = r_densemle.expression() - constrained.expr().build_prover_expr(circuit_map)?;
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_layer(layer.into());
        let denom_mle = MultilinearExtension::new(
            contrained_mle
                .iter()
                .map(|val| r - val)
                .collect()
        );
        // FIXME not sure if we actually need this guy
        //let denom_densemle = DenseMle::new_with_prefix_bits(denom_mle, layer.id(), vec![]);
        let (denom_left, denom_right) = build_split_dense_mles(&denom_mle, &layer_id);

        // Add the two split MLEs to one another
        let expr = denom_left.expression() + denom_right.expression();
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_layer(layer.into());
        let mle = MultilinearExtension::new(
            denom_left
                .get_evals_vector()
                .iter()
                .zip(denom_right.get_evals_vector())
                .map(|(l, r)| l + r)
                .collect()
        );
        
        // and so we continue!




        let witness_numerators = ExprBuilder<F>::constant(F::one());

        Ok(())
    }
}

/// Split a MultilinearExtension into two DenseMles, with the left half containing the even-indexed elements and the right half containing the odd-indexed elements, setting the prefix bits accordingly (and using the supplied LayerId).
pub fn build_split_dense_mles<F: FieldExt>(
    data: &MultilinearExtension<F>,
    layer_id: &LayerId
) -> (DenseMle<F>, DenseMle<F>) {
    let left: Vec<F> = data
        .get_evals_vector()
        .iter()
        .step_by(2)
        .cloned()
        .collect();
    let right: Vec<F> = data
        .get_evals_vector()
        .iter()
        .skip(1)
        .step_by(2)
        .cloned()
        .collect();
    let left_dense = DenseMle::new_with_prefix_bits(MultilinearExtension::new_from_evals(Evaluations::new(data.num_vars() - 1, left)), layer_id, vec![false]);
    let right_dense = DenseMle::new_with_prefix_bits(MultilinearExtension::new_from_evals(Evaluations::new(data.num_vars() - 1, right)), layer_id, vec![true]);
    (left_dense, right_dense)
}

/// Split a MultilinearExtension into two, with the left half containing the even-indexed elements and the right half containing the odd-indexed elements.
pub fn split_mle<F: FieldExt>(
    data: &MultilinearExtension<F>,
) -> (MultilinearExtension<F>, MultilinearExtension<F>) {
    let left: Vec<F> = data
        .get_evals_vector()
        .iter()
        .step_by(2)
        .cloned()
        .collect();
    let right: Vec<F> = data
        .get_evals_vector()
        .iter()
        .skip(1)
        .step_by(2)
        .cloned()
        .collect();
    (MultilinearExtension::new_from_evals(Evaluations::new(data.num_vars() - 1, left)), MultilinearExtension::new_from_evals(Evaluations::new(data.num_vars() - 1, right)))
}