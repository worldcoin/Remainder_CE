//! Nodes that implement LogUp.
use crate::expression::abstract_expr::AbstractExpr;
use crate::expression::prover_expr::ProverExpr;
use crate::input_layer::public_input_layer::PublicInputLayer;
use crate::input_layer::{InputLayer, MleInputLayer};
use crate::mle::zero::ZeroMle;
use crate::mle::Mle;
use crate::mle::{evals::Evaluations, MleIndex};
use std::marker::PhantomData;

use remainder_shared_types::FieldExt;

use crate::{
    expression::{abstract_expr::ExprBuilder, generic_expr::Expression},
    layer::regular_layer::RegularLayer,
    mle::{dense::DenseMle, evals::MultilinearExtension},
    prover::proof_system::ProofSystem,
};

use super::{CircuitNode, ClaimableNode, CompilableNode, Context, NodeId};

/// Represents the use of a lookup into a particular table (represented by a LookupNode).
#[derive(Clone, Debug)]
pub struct LookupShred {
    /// The id of this LookupShred.
    pub id: NodeId,
    /// The id of the LookupNode (lookup table) that we are a lookup up into.
    pub table_node_id: NodeId,
    /// The id of the node that is being constrained by this lookup.
    pub constrained_node_id: NodeId,
    /// The node that provides the multiplicities for the constrained data.
    pub multiplicities_node_id: NodeId,
}

impl LookupShred {
    /// Creates a new LookupShred, constraining the data of `constrained` to form a subset of the
    /// data in `table` with multiplicities given by `multiplicities`. Caller is responsible for the
    /// yielding of all nodes (including `constrained` and `multiplicities`).
    pub fn new<F: FieldExt>(
        ctx: &Context,
        lookup_node: &LookupNode,
        constrained: &dyn ClaimableNode<F = F>,
        multiplicities: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let id = ctx.get_new_id();
        LookupShred {
            id,
            table_node_id: lookup_node.id(),
            constrained_node_id: constrained.id(),
            multiplicities_node_id: multiplicities.id(),
        }
    }
}

impl CircuitNode for LookupShred {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![]
    }
}

/// Represents a table of data that can be looked up into, e.g. for a range check.
#[derive(Clone, Debug)]
pub struct LookupNode {
    id: NodeId,
    /// The lookups that are performed on this table (will be populated by calls to add_shred).
    shreds: Vec<LookupShred>,
    /// The id of the node providing the table entries.
    table_node_id: NodeId,
    /// Whether the values constrained by this table should be considered secret
    /// (Determines which InputLayer type is used for the denominator inverses.)
    constrained_values_secret: bool
}

impl LookupNode {
    /// Create a new table to use for subsequent lookups.
    /// (Perform a lookup in this table by creating a [LookupShred].)
    pub fn new(ctx: &Context, table: NodeId) -> Self {
        let id = ctx.get_new_id();

        LookupNode {
            id,
            shreds: vec![],
            table_node_id: table,
            constrained_values_secret: false,
        }
    }

    /// Add a lookup shred to this node.
    /// (Will be called by the layouter when laying out the circuit.)
    pub fn add_shred(&mut self, shred: LookupShred) {
        self.shreds.push(shred);
    }
}

impl CircuitNode for LookupNode {
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

// FIXME this belongs elsewhere.
/// Companion function to [selectors] that calculates the resulting MLE from the MLEs of the
/// expressions that make up the selector tree.
pub fn calculate_selector_values<F: FieldExt>(mles: Vec<Vec<F>>) -> Vec<F> {
    use itertools::Itertools;
    let mut mles = mles;
    assert!(mles.len().is_power_of_two());

    while mles.len() > 1 {
        mles = mles
            .into_iter()
            .tuples()
            .map(|(mle1, mle2)| {
                mle1.into_iter()
                    .zip(mle2.into_iter())
                    .flat_map(|(a, b)| vec![a, b])
                    .collect()
            })
            .collect();
    }
    mles[0].clone()
}

impl<F: FieldExt, Pf: ProofSystem<F, InputLayer = IL, Layer = L, OutputLayer = OL>, IL, L, OL>
    CompilableNode<F, Pf> for LookupNode
where
    IL: From<PublicInputLayer<F>>,
    L: From<RegularLayer<F>>,
    OL: From<ZeroMle<F>>,
{
    fn compile<'a>(
        &'a self,
        witness_builder: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
        circuit_map: &mut crate::layouter::layouting::CircuitMap<'a, F>,
    ) -> Result<(), crate::layouter::layouting::DAGError> {
        type AE<F> = Expression<F, AbstractExpr>;
        type PE<F> = Expression<F, ProverExpr>;

        // Ensure that number of LookupShreds is a power of two (otherwise when we concat the
        // constrained nodes, there will be padding, and the padding value is potentially not in the
        // table
        assert_eq!(
            self.shreds.len().count_ones(),
            1,
            "Number of LookupShreds should be a power of two"
        );

        // Ensure that the table length is a power of two (otherwise 0 will be added implicitly,
        // which is potentially unwanted and moreover the padding of the denominators with zeros
        // will cause failure)
        let (_, table) = circuit_map.0[&self.table_node_id];
        assert_eq!(
            table.get_evals_vector().len().count_ones(),
            1,
            "Table length should be a power of two"
        );

        // Build the LHS of the equation (defined by the constrained values)
        println!("Build the LHS of the equation (defined by the constrained values)");

        // TODO (future) Draw a random value from the transcript
        let r = F::from(11111111u64); // FIXME
        let r_mle = MultilinearExtension::<F>::new(vec![r]);
        let r_layer_id = witness_builder.next_input_layer();
        witness_builder.add_input_layer(PublicInputLayer::new(r_mle.clone(), r_layer_id).into());
        let r_densemle = DenseMle::new_with_prefix_bits(r_mle, r_layer_id, vec![]);
        println!(
            "Input layer for the would-be random value r has layer id: {:?}",
            r_layer_id
        );

        // Build the denominator r - constrained
        // There may be more than one shred, so build a selector tree if necessary
        let constrained_expr = AE::<F>::selectors(
            self.shreds
                .iter()
                .map(|shred| shred.constrained_node_id.expr())
                .collect(),
        );
        let expr =
            r_densemle.clone().expression() - constrained_expr.build_prover_expr(circuit_map)?;
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_layer(layer.into());
        println!(
            "Layer that calcs r - constrained has layer id: {:?}",
            layer_id
        );
        // Create the MLE for the denominator
        let constrained_values = calculate_selector_values(
            self.shreds
                .iter()
                .map(|shred| {
                    let (_, constrained_mle) = circuit_map.0[&shred.constrained_node_id];
                    constrained_mle.get_evals_vector().clone()
                })
                .collect(),
        );
        let mle =
            MultilinearExtension::new(constrained_values.into_iter().map(|val| r - val).collect());
        let denominator_length = mle.get_evals_vector().len();
        let lhs_denominator = DenseMle::new_with_prefix_bits(mle, layer_id, vec![]);

        // Build the numerator: is all ones (create explicitly since don't want to pad with zeros)
        let expr = ExprBuilder::<F>::constant(F::from(1u64)).build_prover_expr(circuit_map)?;
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_layer(layer.into());
        println!(
            "Layer that sets the numerators to 1 has layer id: {:?}",
            layer_id
        );
        let mle = MultilinearExtension::new(vec![F::from(1u64); denominator_length]);
        let lhs_numerator = DenseMle::new_with_prefix_bits(mle, layer_id, vec![]);

        // Build the numerator and denominator of the sum of the fractions
        let (lhs_numerator, lhs_denominator) =
            build_fractional_sum(lhs_numerator, lhs_denominator, witness_builder);

        // Build the RHS of the equation (defined by the table values and multiplicities)
        println!("Build the RHS of the equation (defined by the table values and multiplicities)");

        // Build the numerator (the multiplicities, which we aggregate with an extra layer if there is more than one shred)
        let (multiplicities_location, multiplicities) =
            &circuit_map.0[&self.shreds[0].multiplicities_node_id];
        let mut rhs_numerator = DenseMle::new_with_prefix_bits(
            (*multiplicities).clone(),
            multiplicities_location.layer_id,
            multiplicities_location.prefix_bits.clone(),
        );
        if self.shreds.len() > 1 {
            // insert an extra layer that aggregates the multiplicities
            let expr = self
                .shreds
                .iter()
                .skip(1)
                .fold(rhs_numerator.expression(), |acc, shred| {
                    let (multiplicities_location, multiplicities) =
                        &circuit_map.0[&shred.multiplicities_node_id];
                    let mult_shred_mle = DenseMle::new_with_prefix_bits(
                        (*multiplicities).clone(),
                        multiplicities_location.layer_id,
                        multiplicities_location.prefix_bits.clone(),
                    );
                    acc + mult_shred_mle.expression()
                });
            let layer_id = witness_builder.next_layer();
            let layer = RegularLayer::new_raw(layer_id, expr);
            witness_builder.add_layer(layer.into());
            println!(
                "Layer that aggs the multiplicities has layer id: {:?}",
                layer_id
            );
            let eval_vecs: Vec<_> = self
                .shreds
                .iter()
                .map(|shred| {
                    let (_, multiplicities) = circuit_map.0[&shred.multiplicities_node_id];
                    multiplicities.get_evals_vector()
                })
                .collect();
            let agg_evals = eval_vecs
                .iter()
                .fold(vec![F::from(0u64); eval_vecs[0].len()], |acc, evals| {
                    acc.iter().zip(evals.iter()).map(|(a, b)| *a + *b).collect()
                });
            rhs_numerator = DenseMle::new_with_prefix_bits(
                MultilinearExtension::new(agg_evals),
                layer_id,
                vec![],
            );
        }

        // Build the denominator r - table
        let expr =
            r_densemle.expression() - self.table_node_id.expr().build_prover_expr(circuit_map)?;
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_layer(layer.into());
        println!(
            "Layer that calculates r - table has layer id: {:?}",
            layer_id
        );
        let mle =
            MultilinearExtension::new(table.get_evals_vector().iter().map(|val| r - val).collect());
        let rhs_denominator = DenseMle::new_with_prefix_bits(mle, layer_id, vec![]);

        // Build the numerator and denominator of the sum of the fractions
        let (rhs_numerator, rhs_denominator) =
            build_fractional_sum(rhs_numerator, rhs_denominator, witness_builder);

        // Add an input layer for the inverse of the denominators of the LHS This value holds
        // reveals some information about the constrained values, so we optionally use a
        // HyraxInputLayer.
        let mle =
            MultilinearExtension::new(vec![lhs_denominator.current_mle.value().invert().unwrap()]);
        let layer_id = witness_builder.next_input_layer();
        if self.constrained_values_secret {
            // TODO use HyraxInputLayer, once its implemented
            unimplemented!();
        } else {
            witness_builder.add_input_layer(PublicInputLayer::new(mle.clone(), layer_id).into());
        }
        let lhs_inverse_densemle = DenseMle::new_with_prefix_bits(mle, layer_id, vec![]);
        println!(
            "Input layer that for LHS denom prod inverse has layer id: {:?}",
            layer_id
        );
        // Same, but for the RHS - this doesn't reveal any information about the constrained values,
        // so OK to use PublicInputLayer
        let mle =
            MultilinearExtension::new(vec![rhs_denominator.current_mle.value().invert().unwrap()]);
        let layer_id = witness_builder.next_input_layer();
        witness_builder.add_input_layer(PublicInputLayer::new(mle.clone(), layer_id).into());
        let rhs_inverse_densemle = DenseMle::new_with_prefix_bits(mle, layer_id, vec![]);
        println!(
            "Input layer that for RHS denom prod inverse has layer id: {:?}",
            layer_id
        );

        // Add a layer that calculates the product of the denominator and the inverse and subtracts 1 (for both LHS and RHS)
        let lhs_expr =
            PE::<F>::products(vec![lhs_denominator.clone(), lhs_inverse_densemle.clone()]);
        let rhs_expr =
            PE::<F>::products(vec![rhs_denominator.clone(), rhs_inverse_densemle.clone()]);
        let expr = lhs_expr.concat_expr(rhs_expr) - PE::<F>::constant(F::from(1u64));
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_layer(layer.into());
        println!(
            "Layer calcs product of (product of denoms) and their inverses has layer id: {:?}",
            layer_id
        );
        // Add an output layer that checks that the result is zero
        let mle = ZeroMle::new(1, None, layer_id);
        witness_builder.add_output_layer(mle.into());

        // Add a layer that calculates the difference between the fractions on the LHS and RHS
        let expr = PE::<F>::products(vec![lhs_numerator.clone(), rhs_denominator.clone()])
            - PE::<F>::products(vec![rhs_numerator.clone(), lhs_denominator.clone()]);
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(layer_id, expr);
        witness_builder.add_layer(layer.into());
        println!(
            "Layer that checks that fractions are equal has layer id: {:?}",
            layer_id
        );
        // Add an output layer that checks that the result is zero
        let mle = ZeroMle::new(0, None, layer_id);
        witness_builder.add_output_layer(mle.into());

        Ok(())
    }
}

/// Extract the prefix bits from a DenseMle
pub fn extract_prefix_bits<F: FieldExt>(mle: &DenseMle<F>) -> Vec<bool> {
    mle.mle_indices()
        .iter()
        .map(|mle_index| match mle_index {
            MleIndex::Fixed(b) => Some(*b),
            _ => None,
        })
        .filter_map(|opt| opt)
        .collect()
}

/// Split a DenseMle into two DenseMles, with the left half containing the even-indexed elements and the right half containing the odd-indexed elements, setting the prefix bits accordingly.
pub fn split_dense_mle<F: FieldExt>(mle: &DenseMle<F>) -> (DenseMle<F>, DenseMle<F>) {
    let data = mle.current_mle.clone();
    let prefix_bits = extract_prefix_bits(mle);
    let left: Vec<F> = data.get_evals_vector().iter().step_by(2).cloned().collect();
    let right: Vec<F> = data
        .get_evals_vector()
        .iter()
        .skip(1)
        .step_by(2)
        .cloned()
        .collect();
    let left_dense = DenseMle::new_with_prefix_bits(
        MultilinearExtension::new_from_evals(Evaluations::new(data.num_vars() - 1, left)),
        mle.layer_id,
        prefix_bits.iter().cloned().chain(vec![false]).collect(),
    );
    let right_dense = DenseMle::new_with_prefix_bits(
        MultilinearExtension::new_from_evals(Evaluations::new(data.num_vars() - 1, right)),
        mle.layer_id,
        prefix_bits.iter().cloned().chain(vec![true]).collect(),
    );
    (left_dense, right_dense)
}

/// Given two Mles of the same length representing the numerators and denominators of a sequence of
/// fractions, add layers that perform a sum of the fractions, return a new pair of Mles
/// representing the numerator and denominator of the (unreduced) sum.
pub fn build_fractional_sum<F: FieldExt, Pf: ProofSystem<F, Layer = L>, L>(
    numerator: DenseMle<F>,
    denominator: DenseMle<F>,
    witness_builder: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
) -> (DenseMle<F>, DenseMle<F>)
where
    L: From<RegularLayer<F>>,
{
    type PE<F> = Expression<F, ProverExpr>;
    assert_eq!(
        numerator.num_iterated_vars(),
        denominator.num_iterated_vars()
    );
    let mut numerator = numerator;
    let mut denominator = denominator;

    for i in 0..numerator.num_iterated_vars() {
        let numerators = split_dense_mle(&numerator);
        let denominators = split_dense_mle(&denominator);

        // Calculate the new numerator
        let next_numerator_expr =
            PE::<F>::products(vec![numerators.0.clone(), denominators.1.clone()])
                + PE::<F>::products(vec![numerators.1.clone(), denominators.0.clone()]);
        let next_numerator_values = numerators
            .0
            .clone()
            .into_iter()
            .zip(numerators.1.clone().into_iter())
            .zip(
                denominators
                    .0
                    .clone()
                    .into_iter()
                    .zip(denominators.1.clone().into_iter()),
            )
            .map(|((num1, num2), (denom1, denom2))| num1 * denom2 + num2 * denom1)
            .collect();

        // Calculate the new denominator
        let next_denominator_expr =
            PE::<F>::products(vec![denominators.0.clone(), denominators.1.clone()]);
        let next_denominator_values = denominators
            .0
            .clone()
            .into_iter()
            .zip(denominators.1.clone().into_iter())
            .map(|(denom1, denom2)| denom1 * denom2)
            .collect();
        let layer_id = witness_builder.next_layer();
        let layer = RegularLayer::new_raw(
            layer_id,
            next_numerator_expr.concat_expr(next_denominator_expr),
        );
        witness_builder.add_layer(layer.into());
        println!(
            "Iteration {:?} of build_fractional_sumcheck has layer id: {:?}",
            i, layer_id
        );

        denominator = DenseMle::new_with_prefix_bits(
            MultilinearExtension::new(next_denominator_values),
            layer_id,
            vec![false],
        );
        numerator = DenseMle::new_with_prefix_bits(
            MultilinearExtension::new(next_numerator_values),
            layer_id,
            vec![true],
        );
    }
    debug_assert_eq!(numerator.num_iterated_vars(), 0);
    debug_assert_eq!(denominator.num_iterated_vars(), 0);
    (numerator, denominator)
}
