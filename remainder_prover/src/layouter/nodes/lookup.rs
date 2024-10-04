//! Nodes that implement LogUp.

use crate::expression::abstract_expr::AbstractExpr;
use crate::expression::circuit_expr::ExprDescription;
use crate::layer::layer_enum::LayerDescriptionEnum;
use crate::layer::regular_layer::RegularLayerDescription;
use crate::layer::LayerId;
use crate::layouter::layouting::{CircuitDescriptionMap, DAGError};
use crate::mle::mle_description::MleDescription;
use crate::mle::MleIndex;
use crate::output_layer::OutputLayerDescription;
use crate::utils::mle::get_total_mle_indices;

use itertools::{repeat_n, Itertools};
use remainder_shared_types::Field;

use crate::expression::generic_expr::Expression;

use super::fiat_shamir_challenge::FiatShamirChallengeNode;
use super::{CircuitNode, Context, NodeId};

/// Represents the use of a lookup into a particular table (represented by a LookupTable).
#[derive(Clone, Debug)]
pub struct LookupConstraint {
    id: NodeId,
    /// The id of the LookupTable (lookup table) that we are a lookup up into.
    pub table_node_id: NodeId,
    /// The id of the node that is being constrained by this lookup.
    constrained_node_id: NodeId,
    /// The id of the node that provides the multiplicities for the constrained data.
    multiplicities_node_id: NodeId,
}

impl LookupConstraint {
    /// Creates a new LookupConstraint, constraining the data of `constrained` to form a subset of
    /// the data in `lookup_table` with multiplicities given by `multiplicities`. Caller is
    /// responsible for the yielding of all nodes (including `constrained` and `multiplicities`).
    /// The adding of lookup specific input- and output layers is handled automatically by
    /// compile().
    ///
    /// # Requires:
    ///   if `constrained` has length not a power of two, then `multiplicitites` must also count the
    ///   implicit padding!
    pub fn new<F: Field>(
        ctx: &Context,
        lookup_table: &LookupTable,
        constrained: &dyn CircuitNode,
        multiplicities: &dyn CircuitNode,
    ) -> Self {
        let id = ctx.get_new_id();
        LookupConstraint {
            id,
            table_node_id: lookup_table.id(),
            constrained_node_id: constrained.id(),
            multiplicities_node_id: multiplicities.id(),
        }
    }
}

impl CircuitNode for LookupConstraint {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        // NB this function never gets called, since lookup tables and constraints are placed after
        // the intermediate nodes in the toposort
        vec![self.constrained_node_id, self.multiplicities_node_id]
    }

    fn get_num_vars(&self) -> usize {
        todo!()
    }
}

type LookupCircuitDescription<F> = (Vec<LayerDescriptionEnum<F>>, OutputLayerDescription<F>);
/// Represents a table of data that can be looked up into, e.g. for a range check.
/// Implements "Improving logarithmic derivative lookups using GKR" (2023) by Papini & Haböck. Note
/// that (as is usual e.g. in permutation checks) we do not check that the product of the
/// denominators is nonzero. This means the soundness of logUp is bounded by
///     `|F| / max{num_constrained_values, num_table_values}`.
/// To adapt this to a small field setting, consider using Fermat's Little Theorem.
#[derive(Clone, Debug)]
pub struct LookupTable {
    id: NodeId,
    /// The lookups that are performed on this table (will be automatically populated, via
    /// [add_lookup_constraint], during layout).
    constraints: Vec<LookupConstraint>,
    /// The id of the node providing the table entries.
    table_node_id: NodeId,
    /// The ID of the [FiatShamirChallengeNode] for the FS challenge.
    fiat_shamir_challenge_node_id: NodeId,
}

impl LookupTable {
    /// Create a new LookupTable to use for subsequent lookups. (To perform a lookup using this
    /// table, create a [LookupConstraint].)
    ///
    /// # Requires
    /// * The length of the table must be a power of two.
    pub fn new<F: Field>(
        ctx: &Context,
        table: &dyn CircuitNode,
        fiat_shamir_challenge_node: &FiatShamirChallengeNode,
    ) -> Self {
        LookupTable {
            id: ctx.get_new_id(),
            constraints: vec![],
            table_node_id: table.id(),
            fiat_shamir_challenge_node_id: fiat_shamir_challenge_node.id(),
        }
    }

    /// Add a lookup constraint to this node.
    /// (Will be called by the layouter when laying out the circuit.)
    pub fn add_lookup_constraint(&mut self, constraint: LookupConstraint) {
        self.constraints.push(constraint);
    }

    /// Create the circuit description of a lookup node by returning the corresponding circuit
    /// descriptions, and output circuit description needed in order to verify the lookup.
    pub fn generate_lookup_circuit_description<F: Field>(
        &self,
        intermediate_layer_id: &mut LayerId,
        circuit_description_map: &mut CircuitDescriptionMap,
    ) -> Result<LookupCircuitDescription<F>, DAGError> {
        type AE<F> = Expression<F, AbstractExpr>;
        type CE<F> = Expression<F, ExprDescription>;

        // Ensure that number of LookupConstraints is a power of two (otherwise when we concat the
        // constrained nodes, there will be padding, and the padding value is potentially not in the
        // table
        assert_eq!(
            self.constraints.len().count_ones(),
            1,
            "Number of LookupConstraints should be a power of two"
        );

        // Build the LHS of the equation (defined by the constrained values)
        println!("Build the LHS of the equation (defined by the constrained values)");

        let (fiat_shamir_challenge_location, fiat_shamir_challenge_node_vars) =
            circuit_description_map
                .get_location_num_vars_from_node_id(&self.fiat_shamir_challenge_node_id)?;

        let fiat_shamir_challenge_mle_indices = get_total_mle_indices(
            &fiat_shamir_challenge_location.prefix_bits,
            *fiat_shamir_challenge_node_vars,
        );
        let fiat_shamir_challenge_mle = MleDescription::new(
            fiat_shamir_challenge_location.layer_id,
            &fiat_shamir_challenge_mle_indices,
        );

        // Build the denominator r - constrained
        // There may be more than one constraint, so build a selector tree if necessary
        let constrained_expr = AE::<F>::selectors(
            self.constraints
                .iter()
                .map(|constraint| constraint.constrained_node_id.expr())
                .collect(),
        );
        let expr = CE::sum(
            CE::from_mle_desc(fiat_shamir_challenge_mle),
            CE::negated(constrained_expr.build_circuit_expr(circuit_description_map)?),
        );
        let expr_num_vars = expr.num_vars();

        let layer_id = intermediate_layer_id.get_and_inc();
        let layer = RegularLayerDescription::new_raw(layer_id, expr);
        let mut intermediate_layers = vec![LayerDescriptionEnum::Regular(layer)];
        println!(
            "Layer that calcs r - constrained has layer id: {:?}",
            layer_id
        );

        let lhs_denominator_vars = repeat_n(MleIndex::Free, expr_num_vars).collect_vec();
        let lhs_denominator_desc = MleDescription::new(layer_id, &lhs_denominator_vars);

        // Super special case: need to create a 0-variable MLE for the numerator which is JUST
        // derived from an expression producing the constant 1
        let maybe_lhs_numerator_desc = if lhs_denominator_vars.is_empty() {
            Some(MleDescription::new(layer_id, &[]))
        } else {
            None
        };

        // Build the numerator and denominator of the sum of the fractions
        let (lhs_numerator, lhs_denominator) = build_fractional_sum(
            maybe_lhs_numerator_desc,
            lhs_denominator_desc,
            &mut intermediate_layers,
            intermediate_layer_id,
        );

        // Build the RHS of the equation (defined by the table values and multiplicities)
        println!("Build the RHS of the equation (defined by the table values and multiplicities)");

        // Build the numerator (the multiplicities, which we aggregate with an extra layer if there is more than one constraint)
        let (multiplicities_location, multiplicities_num_vars) =
            &circuit_description_map.0[&self.constraints[0].multiplicities_node_id];
        let mut rhs_numerator_desc = MleDescription::new(
            multiplicities_location.layer_id,
            &get_total_mle_indices(
                &multiplicities_location.prefix_bits,
                *multiplicities_num_vars,
            ),
        );

        if self.constraints.len() > 1 {
            // Insert an extra layer that aggregates the multiplicities
            let expr = self.constraints.iter().skip(1).fold(
                CE::from_mle_desc(rhs_numerator_desc),
                |acc, constraint| {
                    let (multiplicities_location, multiplicities_num_vars) =
                        &circuit_description_map.0[&constraint.multiplicities_node_id];
                    let mult_constraint_mle_desc = MleDescription::new(
                        multiplicities_location.layer_id,
                        &get_total_mle_indices(
                            &multiplicities_location.prefix_bits,
                            *multiplicities_num_vars,
                        ),
                    );
                    acc + CE::from_mle_desc(mult_constraint_mle_desc)
                },
            );
            let layer_id = intermediate_layer_id.get_and_inc();
            let layer = RegularLayerDescription::new_raw(layer_id, expr);
            intermediate_layers.push(LayerDescriptionEnum::Regular(layer));
            println!(
                "Layer that aggs the multiplicities has layer id: {:?}",
                layer_id
            );

            // Note that this is the aggregated version!
            // It's just the element-wise sum of the elements within the bookkeeping tables
            // However, because we're only dealing with the circuit description, we can
            // just take the number of variables within the *first* constraint
            let (_first_self_constraint_loc, first_self_constraint_num_vars) =
                circuit_description_map.0[&self.constraints[0].multiplicities_node_id].clone();
            rhs_numerator_desc = MleDescription::new(
                layer_id,
                &get_total_mle_indices(&[], first_self_constraint_num_vars),
            )
        }

        // Build the denominator r - table

        // First grab `r` as a `CircuitMle` from the `circuit_description_map`
        let (fiat_shamir_challenge_loc, fiat_shamir_challenge_num_vars) =
            circuit_description_map.0[&self.fiat_shamir_challenge_node_id].clone();
        let fiat_shamir_challenge_circuit_mle = MleDescription::new(
            fiat_shamir_challenge_loc.layer_id,
            &get_total_mle_indices(
                &fiat_shamir_challenge_loc.prefix_bits,
                fiat_shamir_challenge_num_vars,
            ),
        );

        // Next grab `table` as a `MleDescription` from the `circuit_description_map`
        let (table_loc, table_num_vars) = circuit_description_map.0[&self.table_node_id].clone();
        let table_circuit_mle = MleDescription::new(
            table_loc.layer_id,
            &get_total_mle_indices(&table_loc.prefix_bits, table_num_vars),
        );

        let expr = CE::from_mle_desc(fiat_shamir_challenge_circuit_mle)
            - CE::from_mle_desc(table_circuit_mle);
        let r_minus_table_num_vars = expr.num_vars();
        let layer_id = intermediate_layer_id.get_and_inc();
        let layer = RegularLayerDescription::new_raw(layer_id, expr);
        intermediate_layers.push(LayerDescriptionEnum::Regular(layer));
        println!(
            "Layer that calculates r - table has layer id: {:?}",
            layer_id
        );

        let rhs_denominator_desc = MleDescription::new(
            layer_id,
            &repeat_n(MleIndex::Free, r_minus_table_num_vars).collect_vec(),
        );

        // Build the numerator and denominator of the sum of the fractions
        let (rhs_numerator, rhs_denominator) = build_fractional_sum(
            Some(rhs_numerator_desc),
            rhs_denominator_desc,
            &mut intermediate_layers,
            intermediate_layer_id,
        );

        // Add a layer that calculates the difference between the fractions on the LHS and RHS
        assert!(rhs_numerator.is_some());
        let rhs_numerator = rhs_numerator.unwrap();
        let expr = if lhs_numerator.is_none() {
            CE::<F>::products(vec![rhs_denominator.clone()])
                - CE::<F>::products(vec![rhs_numerator.clone(), lhs_denominator.clone()])
        } else {
            let lhs_numerator = lhs_numerator.unwrap();
            CE::<F>::products(vec![lhs_numerator.clone(), rhs_denominator.clone()])
                - CE::<F>::products(vec![rhs_numerator.clone(), lhs_denominator.clone()])
        };

        let layer_id = intermediate_layer_id.get_and_inc();
        let layer = RegularLayerDescription::new_raw(layer_id, expr);
        intermediate_layers.push(LayerDescriptionEnum::Regular(layer));
        println!(
            "Layer that checks that fractions are equal has layer id: {:?}",
            layer_id
        );

        // Add an output layer that checks that the result is zero
        let output_layer = OutputLayerDescription::new_zero(layer_id, &[]);

        Ok((intermediate_layers, output_layer))
    }
}

impl CircuitNode for LookupTable {
    fn id(&self) -> NodeId {
        self.id
    }

    fn subnodes(&self) -> Option<Vec<NodeId>> {
        Some(
            self.constraints
                .iter()
                .map(|constraint| constraint.id())
                .collect(),
        )
    }

    fn sources(&self) -> Vec<NodeId> {
        // NB this function never gets called, since lookup tables and constraints are placed after
        // the intermediate nodes in the toposort
        self.constraints
            .iter()
            .map(|constraint| constraint.id())
            .chain(
                self.constraints
                    .iter()
                    .flat_map(|constraint| constraint.sources()),
            )
            .collect()
    }

    fn get_num_vars(&self) -> usize {
        todo!()
    }
}

/// Extract the prefix bits from a DenseMle.
fn extract_prefix_num_free_bits<F: Field>(mle: &MleDescription<F>) -> (Vec<MleIndex<F>>, usize) {
    let mut num_free_bits = 0;
    let prefix_bits = mle
        .var_indices()
        .iter()
        .filter_map(|mle_index| match mle_index {
            MleIndex::Fixed(_) => Some(mle_index.clone()),
            MleIndex::Free => {
                num_free_bits += 1;
                None
            }
            _ => None,
        })
        .collect();
    (prefix_bits, num_free_bits)
}

/// Split a DenseMle into two DenseMles, with the left half containing the even-indexed elements and
/// the right half containing the odd-indexed elements, setting the prefix bits accordingly.
fn split_circuit_mle<F: Field>(
    mle_desc: &MleDescription<F>,
) -> (MleDescription<F>, MleDescription<F>) {
    let (prefix_bits, num_free_bits) = extract_prefix_num_free_bits(mle_desc);

    let left_mle_desc = MleDescription::new(
        mle_desc.layer_id(),
        &prefix_bits
            .iter()
            .cloned()
            .chain(vec![MleIndex::Fixed(false)])
            .chain(repeat_n(MleIndex::Free, num_free_bits - 1))
            .collect_vec(),
    );
    let right_mle_desc = MleDescription::new(
        mle_desc.layer_id(),
        &prefix_bits
            .iter()
            .cloned()
            .chain(vec![MleIndex::Fixed(true)])
            .chain(repeat_n(MleIndex::Free, num_free_bits - 1))
            .collect_vec(),
    );
    (left_mle_desc, right_mle_desc)
}

/// Given two Mles of the same length representing the numerators and denominators of a sequence of
/// fractions, add layers that perform a sum of the fractions, return a new pair of length-1 Mles
/// representing the numerator and denominator of the sum.
///
/// Setting `maybe_numerator_desc` to None indicates that the numerator has the same length as
/// `denominator_desc` and takes the constant value 1.
fn build_fractional_sum<F: Field>(
    maybe_numerator_desc: Option<MleDescription<F>>,
    denominator_desc: MleDescription<F>,
    layers: &mut Vec<LayerDescriptionEnum<F>>,
    current_layer_id: &mut LayerId,
) -> (Option<MleDescription<F>>, MleDescription<F>) {
    type CE<F> = Expression<F, ExprDescription>;

    // Sanitycheck number of vars in numerator == number of vars in denominator
    // EXCEPT when we're working with the fraction with constant 1 in the numerator
    if let Some(numerator_desc) = maybe_numerator_desc.as_ref() {
        assert_eq!(
            numerator_desc.num_free_vars(),
            denominator_desc.num_free_vars()
        );
    }

    let mut maybe_numerator_desc = maybe_numerator_desc;
    let mut denominator_desc = denominator_desc;

    for i in 0..denominator_desc.num_free_vars() {
        let denominators = split_circuit_mle(&denominator_desc);
        let next_numerator_expr = if let Some(numerator_desc) = maybe_numerator_desc {
            let numerators = split_circuit_mle(&numerator_desc);

            // Calculate the new numerator
            CE::products(vec![numerators.0.clone(), denominators.1.clone()])
                + CE::products(vec![numerators.1.clone(), denominators.0.clone()])
        } else {
            // If there is no numerator CircuitMLE,
            CE::from_mle_desc(denominators.1.clone()) + CE::from_mle_desc(denominators.0.clone())
        };

        // Calculate the new denominator
        let next_denominator_expr =
            CE::products(vec![denominators.0.clone(), denominators.1.clone()]);

        // Grab the size of each
        let next_numerator_num_vars = next_numerator_expr.num_vars();
        let next_denominator_num_vars = next_denominator_expr.num_vars();

        // Create the circuit layer by combining the two
        let layer_id = current_layer_id.get_and_inc();

        let layer = RegularLayerDescription::new_raw(
            layer_id,
            next_denominator_expr.select(next_numerator_expr),
        );

        layers.push(LayerDescriptionEnum::Regular(layer));

        println!(
            "Iteration {:?} of build_fractional_sumcheck has layer id: {:?}",
            i, layer_id
        );

        denominator_desc = MleDescription::new(
            layer_id,
            &std::iter::once(MleIndex::Fixed(false))
                .chain(repeat_n(MleIndex::Free, next_denominator_num_vars))
                .collect_vec(),
        );
        maybe_numerator_desc = Some(MleDescription::new(
            layer_id,
            &std::iter::once(MleIndex::Fixed(true))
                .chain(repeat_n(MleIndex::Free, next_numerator_num_vars))
                .collect_vec(),
        ));
    }
    if let Some(numerator_desc) = maybe_numerator_desc.as_ref() {
        assert_eq!(numerator_desc.num_free_vars(), 0);
    }
    assert_eq!(denominator_desc.num_free_vars(), 0);
    (maybe_numerator_desc, denominator_desc)
}
