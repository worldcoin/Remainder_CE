//! Nodes that implement LogUp.
use crate::mle::evals::Evaluations;
use crate::layer::LayerId;
use crate::expression::prover_expr::ProverExpr;
use crate::input_layer::public_input_layer::PublicInputLayer;
use crate::input_layer::MleInputLayer;
use crate::input_layer::InputLayer;
use crate::mle::Mle;
use std::marker::PhantomData;

use remainder_shared_types::FieldExt;
use serde::de;
use tracing_subscriber::field::debug;

use crate::{expression::{abstract_expr::ExprBuilder, generic_expr::Expression}, layer::regular_layer::RegularLayer, mle::{dense::DenseMle, evals::MultilinearExtension}, prover::proof_system::ProofSystem};

use super::{CircuitNode, ClaimableNode, CompilableNode, Context, NodeId};

/// Represents the use of a lookup into a particular table (represented by a LookupNode).
pub struct LookupShred<F: FieldExt> {
    pub id: NodeId,
    /// The id of the LookupNode (lookup table) that we are a lookup up into.
    pub table_node_id: NodeId,
    /// The id of the node that is being constrained by this lookup.
    pub constrained_node_id: NodeId,
    /// The node that provides the multiplicities for the constrained data.
    pub multiplicities_node_id: NodeId,
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
            table_node_id: table.id(),
            constrained_node_id: constrained.id(),
            multiplicities_node_id: multiplicities.id(),
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

pub struct LookupNode<F: FieldExt> {
    id: NodeId,
    /// The lookups that are performed on this table (will be populated by calls to add_shred).
    shreds: Vec<LookupShred<F>>,
    /// The id of the node providing the table entries.
    table_node_id: NodeId,
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
            table_node_id: table,
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

impl<F: FieldExt, Pf: ProofSystem<F, InputLayer = IL, Layer = L, OutputLayer = OL>, IL, L, OL> CompilableNode<F, Pf>
    for LookupNode<F> 
where
    IL: From<PublicInputLayer<F>>,
    L: From<RegularLayer<F>>,
    OL: From<RegularLayer<F>>,
{
    fn compile<'a>(
        &'a self,
        witness_builder: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
        circuit_map: &mut crate::layouter::layouting::CircuitMap<'a, F>,
    ) -> Result<(), crate::layouter::layouting::DAGError>
     {
        type PE<F> = Expression::<F, ProverExpr>;
        // FIXME: make this work for multiple shreds
        assert_eq!(self.shreds.len(), 1, "LookupNode should have exactly one shred (for now)");
        let shred = &self.shreds[0];
        let constrained = shred.constrained_node_id;

        // Build the LHS of the equation

        // TODO (future) get the MLEs of the constrained nodes and concatenate them
        let (_, constrained_mle) = circuit_map.0[&shred.constrained_node_id];

        // TODO (future) Draw a random value from the transcript
        let r = F::from(1u64); // FIXME
        let r_mle = MultilinearExtension::<F>::new(vec![r]);
        let r_layer_id = witness_builder.next_input_layer();
        witness_builder.add_input_layer(PublicInputLayer::new(r_mle.clone(), r_layer_id).into());
        let r_densemle = DenseMle::new_with_prefix_bits(r_mle, r_layer_id, vec![]);

        // Build the numerator: is all ones (create explicitly since don't want to pad with zeros)
        let expr = ExprBuilder::<F>::constant(F::from(1u64)).build_prover_expr(circuit_map)?;
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_layer(layer.into());
        let mle = MultilinearExtension::new(
            constrained_mle.get_evals_vector()
                .iter()
                .map(|_val| F::from(1u64))
                .collect()
        );
        let lhs_numerator = DenseMle::new_with_prefix_bits(mle, layer_id, vec![]);

        // Build the denominator r - constrained
        let expr = r_densemle.clone().expression() - constrained.expr().build_prover_expr(circuit_map)?;
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_layer(layer.into());
        let mle = MultilinearExtension::new(
            constrained_mle.get_evals_vector()
                .iter()
                .map(|val| r - val)
                .collect()
        );
        let lhs_denominator = DenseMle::new_with_prefix_bits(mle, layer_id, vec![]);

        // Build the numerator and denominator of the sum of the fractions
        let (lhs_numerator, lhs_denominator) = build_fractional_sum(lhs_numerator, lhs_denominator, witness_builder);

        // Build the RHS of the equation
        // TODO (future) get the MLEs of the multiplicities and add them all together
        let (multiplicities_location, multiplicities) = &circuit_map.0[&shred.multiplicities_node_id];
        let (_, table) = circuit_map.0[&self.table_node_id];

        // Build the numerator
        let rhs_numerator = DenseMle::new_with_prefix_bits((*multiplicities).clone(), multiplicities_location.layer_id, multiplicities_location.prefix_bits.clone());

        // Build the denominator r - table
        let expr = r_densemle.expression() - self.table_node_id.expr().build_prover_expr(circuit_map)?;
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_layer(layer.into());
        let mle = MultilinearExtension::new(
            table.get_evals_vector()
                .iter()
                .map(|val| r - val)
                .collect()
        );
        let rhs_denominator = DenseMle::new_with_prefix_bits(mle, layer_id, vec![]);

        // Build the numerator and denominator of the sum of the fractions
        let (rhs_numerator, rhs_denominator) = build_fractional_sum(rhs_numerator, rhs_denominator, witness_builder);

        // Add an input layer for the inverse of the denominators of the LHS
        let mle = MultilinearExtension::new(vec![lhs_denominator.current_mle.value().invert().unwrap()]);
        let layer_id = witness_builder.next_input_layer();
        witness_builder.add_input_layer(PublicInputLayer::new(mle.clone(), layer_id).into()); // FIXME change to Hyrax when it becomes available
        let lhs_inverse_densemle = DenseMle::new_with_prefix_bits(mle, layer_id, vec![]);

        // Same, but for the RHS
        let mle = MultilinearExtension::new(vec![rhs_denominator.current_mle.value().invert().unwrap()]);
        let layer_id = witness_builder.next_input_layer();
        witness_builder.add_input_layer(PublicInputLayer::new(mle.clone(), layer_id).into()); // FIXME change to Hyrax when it becomes available
        let rhs_inverse_densemle = DenseMle::new_with_prefix_bits(mle, layer_id, vec![]);

        // Add an output layer that checks that the product of the denominator and the inverse is 1 (for both LHS and RHS)
        let lhs_expr = PE::<F>::products(vec![lhs_denominator.clone(), lhs_inverse_densemle.clone()]);
        let rhs_expr = PE::<F>::products(vec![rhs_denominator.clone(), rhs_inverse_densemle.clone()]);
        let expr = lhs_expr.concat_expr(rhs_expr) - PE::<F>::constant(F::from(1u64));
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_output_layer(layer.into());


        Ok(())
    }
}

/// Split a DenseMle into two DenseMles, with the left half containing the even-indexed elements and the right half containing the odd-indexed elements, setting the prefix bits accordingly.
pub fn split_dense_mle<F: FieldExt>(
    mle: &DenseMle<F>,
) -> (DenseMle<F>, DenseMle<F>) {
    let data = mle.current_mle.clone();
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
    let left_dense = DenseMle::new_with_prefix_bits(MultilinearExtension::new_from_evals(Evaluations::new(data.num_vars() - 1, left)), mle.layer_id, vec![false]);
    let right_dense = DenseMle::new_with_prefix_bits(MultilinearExtension::new_from_evals(Evaluations::new(data.num_vars() - 1, right)), mle.layer_id, vec![true]);
    (left_dense, right_dense)
}

/// Given two Mles of the same length representing the numerators and denominators of a sequence of
/// fractions, add layers that perform a sum of the fractions, return a new pair of Mles
/// representing the numerator and denominator of the (unreduced) sum.
pub fn build_fractional_sum<F: FieldExt, Pf: ProofSystem<F, Layer = L>, L>(
    numerator: DenseMle<F>,
    denominator: DenseMle<F>,
   witness_builder: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
) -> (DenseMle<F>, DenseMle<F>) where
    L: From<RegularLayer<F>>
{
    type PE<F> = Expression::<F, ProverExpr>;
    assert_eq!(numerator.num_iterated_vars(), denominator.num_iterated_vars());
    let mut numerator = numerator;
    let mut denominator = denominator;

    for _ in 0..numerator.num_iterated_vars() {
        let numerators = split_dense_mle(&numerator);
        let denominators = split_dense_mle(&denominator);
        // TODO v2: presently we are creating two layers per loop - what is the correct way to concatenate them in Newmainder?

        // Calculate the new numerator
        let expr = PE::<F>::products(vec![numerators.0.clone(), denominators.1.clone()]) + PE::<F>::products(vec![numerators.1.clone(), denominators.0.clone()]);
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_layer(layer.into());
        let mle = MultilinearExtension::new(
            numerators.0.clone().into_iter().zip(numerators.1.clone().into_iter())
            .zip(denominators.0.clone().into_iter().zip(denominators.1.clone().into_iter()))
            .map(|((num1, num2), (denom1, denom2))| {
                num1 * denom2 + num2 * denom1
            }).collect()
        );
        numerator = DenseMle::new_with_prefix_bits(mle, layer_id, vec![]);

        // Calculate the new denominator
        let expr = PE::<F>::products(vec![denominators.0.clone(), denominators.1.clone()]);
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_layer(layer.into());
        let mle = MultilinearExtension::new(
            denominators.0.clone().into_iter()
            .zip(denominators.1.clone().into_iter())
            .map(|(denom1, denom2)| {
                denom1 * denom2
            }).collect()
        );
        denominator = DenseMle::new_with_prefix_bits(mle, layer_id, vec![]);
    }
    debug_assert_eq!(numerator.num_iterated_vars(), 0);
    debug_assert_eq!(denominator.num_iterated_vars(), 0);
    (numerator, denominator)
}