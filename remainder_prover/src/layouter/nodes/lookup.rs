//! Nodes that implement LogUp.

use crate::expression::abstract_expr::AbstractExpr;
use crate::expression::circuit_expr::{CircuitExpr, CircuitMle};
use crate::input_layer::enum_input_layer::CircuitInputLayerEnum;
use crate::input_layer::hyrax_input_layer::CircuitHyraxInputLayer;
use crate::input_layer::public_input_layer::CircuitPublicInputLayer;
use crate::layer::layer_enum::CircuitLayerEnum;
use crate::layer::regular_layer::CircuitRegularLayer;
use crate::layer::LayerId;
use crate::layouter::layouting::{CircuitDescriptionMap, CircuitLocation, InputLayerHintMap};
use crate::mle::evals::MultilinearExtension;
use crate::mle::MleIndex;
use crate::output_layer::mle_output_layer::CircuitMleOutputLayer;
use crate::utils::mle::get_total_mle_indices;

use itertools::{repeat_n, Itertools};
use remainder_shared_types::Field;

use crate::expression::generic_expr::Expression;

use super::verifier_challenge::VerifierChallengeNode;
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

type LookupCircuitDescription<F> = (
    Vec<CircuitInputLayerEnum<F>>,
    Vec<CircuitLayerEnum<F>>,
    Vec<CircuitMleOutputLayer<F>>,
);
/// Represents a table of data that can be looked up into, e.g. for a range check.
#[derive(Clone, Debug)]
pub struct LookupTable {
    id: NodeId,
    /// The lookups that are performed on this table (will be automatically populated, via
    /// [add_lookup_constraint], during layout).
    constraints: Vec<LookupConstraint>,
    /// The id of the node providing the table entries.
    table_node_id: NodeId,
    /// The ID of the random input node for the FS challenge.
    random_node_id: NodeId,
    /// Whether any of the values to be constrained by this LookupTable should be considered secret
    /// (Determines which InputLayer type is used for the denominator inverses.)
    secret_constrained_values: bool,
}

impl LookupTable {
    /// Create a new LookupTable to use for subsequent lookups.
    /// (To perform a lookup using this table, create a [LookupConstraint].)
    /// `secret_constrained_values` controls whether a public or a hiding input layer is used for
    /// the denominator inverse, which is derived from the constrained values (note that LookupTable
    /// does not hide the constrained values themselves - that is up to the caller).
    ///
    /// # Requires
    /// The length of the table must be a power of two.
    pub fn new<F: Field>(
        ctx: &Context,
        table: &dyn CircuitNode,
        secret_constrained_values: bool,
        random_input_node: &VerifierChallengeNode,
    ) -> Self {
        LookupTable {
            id: ctx.get_new_id(),
            constraints: vec![],
            table_node_id: table.id(),
            random_node_id: random_input_node.id(),
            secret_constrained_values,
        }
    }

    /// Add a lookup constraint to this node.
    /// (Will be called by the layouter when laying out the circuit.)
    pub fn add_lookup_constraint(&mut self, constraint: LookupConstraint) {
        self.constraints.push(constraint);
    }

    /// Create the circuit description of a lookup node by returning
    /// the corresponding input circuit descriptions, intermediate
    /// circuit descriptions, and output circuit descriptions needed
    /// in order to verify the lookup.
    pub fn generate_lookup_circuit_description<F: Field>(
        &self,
        input_layer_id: &mut LayerId,
        intermediate_layer_id: &mut LayerId,
        circuit_description_map: &mut CircuitDescriptionMap,
        input_hint_map: &mut InputLayerHintMap<F>,
    ) -> Result<LookupCircuitDescription<F>, crate::layouter::layouting::DAGError> {
        type AE<F> = Expression<F, AbstractExpr>;
        type CE<F> = Expression<F, CircuitExpr>;

        // --- LogUp adds a few circuit "inputs" in the flavor of the denominator inverses ---
        let mut logup_additional_input_layers: Vec<CircuitInputLayerEnum<F>> = vec![];

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

        let (verifier_challenge_location, verifier_challenge_node_vars) =
            circuit_description_map.get_location_num_vars_from_node_id(&self.random_node_id)?;

        let verifier_challenge_mle_indices = get_total_mle_indices(
            &verifier_challenge_location.prefix_bits,
            *verifier_challenge_node_vars,
        );
        let verifier_challenge_mle = CircuitMle::new(
            verifier_challenge_location.layer_id,
            &verifier_challenge_mle_indices,
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
            verifier_challenge_mle.expression(),
            CE::negated(constrained_expr.build_circuit_expr(circuit_description_map)?),
        );
        let expr_num_vars = expr.num_vars();

        let layer_id = intermediate_layer_id.get_and_inc();
        let layer = CircuitRegularLayer::new_raw(layer_id, expr);
        let mut intermediate_layers = vec![CircuitLayerEnum::Regular(layer)];
        println!(
            "Layer that calcs r - constrained has layer id: {:?}",
            layer_id
        );

        let lhs_denominator_vars = repeat_n(MleIndex::Iterated, expr_num_vars).collect_vec();
        let lhs_denominator_desc = CircuitMle::new(layer_id, &lhs_denominator_vars);

        // --- Super special case: need to create a 0-variable MLE for the numerator which is JUST derived from an expression producing the constant 1 ---
        let maybe_lhs_numerator_desc = if lhs_denominator_vars.is_empty() {
            Some(CircuitMle::new(layer_id, &[]))
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
        let mut rhs_numerator_desc = CircuitMle::new(
            multiplicities_location.layer_id,
            &get_total_mle_indices(
                &multiplicities_location.prefix_bits,
                *multiplicities_num_vars,
            ),
        );

        if self.constraints.len() > 1 {
            // Insert an extra layer that aggregates the multiplicities
            let expr = self.constraints.iter().skip(1).fold(
                rhs_numerator_desc.expression(),
                |acc, constraint| {
                    let (multiplicities_location, multiplicities_num_vars) =
                        &circuit_description_map.0[&constraint.multiplicities_node_id];
                    let mult_constraint_mle_desc = CircuitMle::new(
                        multiplicities_location.layer_id,
                        &get_total_mle_indices(
                            &multiplicities_location.prefix_bits,
                            *multiplicities_num_vars,
                        ),
                    );
                    acc + mult_constraint_mle_desc.expression()
                },
            );
            let layer_id = intermediate_layer_id.get_and_inc();
            let layer = CircuitRegularLayer::new_raw(layer_id, expr);
            intermediate_layers.push(CircuitLayerEnum::Regular(layer));
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
            rhs_numerator_desc = CircuitMle::new(
                layer_id,
                &get_total_mle_indices(&[], first_self_constraint_num_vars),
            )
        }

        // Build the denominator r - table

        // --- First grab `r` as a `CircuitMle` from the `circuit_description_map` ---
        let (verifier_challenge_loc, verifier_challenge_num_vars) =
            circuit_description_map.0[&self.random_node_id].clone();
        let verifier_challenge_circuit_mle = CircuitMle::new(
            verifier_challenge_loc.layer_id,
            &get_total_mle_indices(
                &verifier_challenge_loc.prefix_bits,
                verifier_challenge_num_vars,
            ),
        );

        // --- Next grab `table` as a `CircuitMle` from the `circuit_description_map` ---
        let (table_loc, table_num_vars) = circuit_description_map.0[&self.table_node_id].clone();
        let table_circuit_mle = CircuitMle::new(
            table_loc.layer_id,
            &get_total_mle_indices(&table_loc.prefix_bits, table_num_vars),
        );

        let expr = verifier_challenge_circuit_mle.expression() - table_circuit_mle.expression();
        let r_minus_table_num_vars = expr.num_vars();
        let layer_id = intermediate_layer_id.get_and_inc();
        let layer = CircuitRegularLayer::new_raw(layer_id, expr);
        intermediate_layers.push(CircuitLayerEnum::Regular(layer));
        println!(
            "Layer that calculates r - table has layer id: {:?}",
            layer_id
        );

        let rhs_denominator_desc = CircuitMle::new(
            layer_id,
            &repeat_n(MleIndex::Iterated, r_minus_table_num_vars).collect_vec(),
        );

        // Build the numerator and denominator of the sum of the fractions
        let (rhs_numerator, rhs_denominator) = build_fractional_sum(
            Some(rhs_numerator_desc),
            rhs_denominator_desc,
            &mut intermediate_layers,
            intermediate_layer_id,
        );

        // Add an input layer for the inverse of the denominators of the LHS. This value holds
        // reveals some information about the constrained values, so we optionally use a
        // HyraxInputLayer.
        // --- Grab the layer ID for the new "input layer" to be added ---
        let lhs_denom_inverse_layer_id = input_layer_id.get_and_inc();
        let lhs_denom_circuit_location =
            CircuitLocation::new(lhs_denominator.layer_id(), lhs_denominator.prefix_bits());

        let inverse_function = |mle: &MultilinearExtension<F>| {
            assert_eq!(mle.get_evals_vector().len(), 1);
            MultilinearExtension::new(vec![mle.get_evals_vector()[0].invert().unwrap()])
        };
        input_hint_map.add_hint_function(
            &lhs_denom_inverse_layer_id,
            (lhs_denom_circuit_location, inverse_function),
        );
        let lhs_inverse_input_layer = if self.secret_constrained_values {
            let hyrax_input_layer_description =
                CircuitHyraxInputLayer::<F>::new(lhs_denom_inverse_layer_id.to_owned(), 0);
            CircuitInputLayerEnum::HyraxInputLayer(hyrax_input_layer_description)
        } else {
            let public_input_layer_description =
                CircuitPublicInputLayer::<F>::new(lhs_denom_inverse_layer_id.to_owned(), 0);
            CircuitInputLayerEnum::PublicInputLayer(public_input_layer_description)
        };
        logup_additional_input_layers.push(lhs_inverse_input_layer);

        let lhs_inverse_mle_desc = CircuitMle::new(lhs_denom_inverse_layer_id, &[]);
        println!(
            "Input layer that for LHS denom prod inverse has layer id: {:?}",
            lhs_denom_inverse_layer_id
        );

        // Add an input layer for the inverse of the denominators of the RHS. This doesn't reveal
        // any information about the constrained values, so it's OK to use PublicInputLayer.
        // let mle =
        //     MultilinearExtension::new(vec![rhs_denominator.current_mle.value().invert().unwrap()]);

        // --- Grab the layer ID for the new "input layer" to be added ---
        let rhs_denom_inverse_layer_id = input_layer_id.get_and_inc();
        let rhs_denom_circuit_location =
            CircuitLocation::new(rhs_denominator.layer_id(), rhs_denominator.prefix_bits());
        input_hint_map.add_hint_function(
            &rhs_denom_inverse_layer_id,
            (rhs_denom_circuit_location, inverse_function),
        );
        let rhs_inverse_mle =
            CircuitPublicInputLayer::<F>::new(rhs_denom_inverse_layer_id.to_owned(), 0);
        let rhs_inverse_input_layer = CircuitInputLayerEnum::PublicInputLayer(rhs_inverse_mle);
        logup_additional_input_layers.push(rhs_inverse_input_layer);

        let rhs_inverse_mle_desc = CircuitMle::new(rhs_denom_inverse_layer_id, &[]);
        println!(
            "Input layer that for RHS denom prod inverse has layer id: {:?}",
            rhs_denom_inverse_layer_id
        );

        // Add a layer that calculates the product of the denominator and the inverse and subtracts
        // 1 (for both LHS and RHS)
        let lhs_expr =
            CE::<F>::products(vec![lhs_denominator.clone(), lhs_inverse_mle_desc.clone()]);
        let rhs_expr =
            CE::<F>::products(vec![rhs_denominator.clone(), rhs_inverse_mle_desc.clone()]);
        let expr = lhs_expr.concat_expr(rhs_expr) - CE::<F>::constant(F::from(1u64));
        let layer_id = intermediate_layer_id.get_and_inc();
        let layer = CircuitRegularLayer::new_raw(layer_id, expr);
        intermediate_layers.push(CircuitLayerEnum::Regular(layer));
        println!(
            "Layer calcs product of (product of denoms) and their inverses has layer id: {:?}",
            layer_id
        );
        // Add an output layer that checks that the result is zero
        let output_layer = CircuitMleOutputLayer::new_zero(layer_id, &[MleIndex::Iterated]);
        let mut output_layers = vec![output_layer];

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
        let layer = CircuitRegularLayer::new_raw(layer_id, expr);
        intermediate_layers.push(CircuitLayerEnum::Regular(layer));
        println!(
            "Layer that checks that fractions are equal has layer id: {:?}",
            layer_id
        );

        // Add an output layer that checks that the result is zero
        let output_layer = CircuitMleOutputLayer::new_zero(layer_id, &[]);
        output_layers.push(output_layer);

        Ok((
            logup_additional_input_layers,
            intermediate_layers,
            output_layers,
        ))
    }
}

impl CircuitNode for LookupTable {
    fn id(&self) -> NodeId {
        self.id
    }

    fn children(&self) -> Option<Vec<NodeId>> {
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
fn extract_prefix_num_iterated_bits<F: Field>(mle: &CircuitMle<F>) -> (Vec<MleIndex<F>>, usize) {
    let mut num_iterated_bits = 0;
    let prefix_bits = mle
        .mle_indices()
        .iter()
        .filter_map(|mle_index| match mle_index {
            MleIndex::Fixed(_) => Some(mle_index.clone()),
            MleIndex::Iterated => {
                num_iterated_bits += 1;
                None
            }
            _ => None,
        })
        .collect();
    (prefix_bits, num_iterated_bits)
}

/// Split a DenseMle into two DenseMles, with the left half containing the even-indexed elements and
/// the right half containing the odd-indexed elements, setting the prefix bits accordingly.
fn split_circuit_mle<F: Field>(mle_desc: &CircuitMle<F>) -> (CircuitMle<F>, CircuitMle<F>) {
    let (prefix_bits, num_iterated_bits) = extract_prefix_num_iterated_bits(mle_desc);

    let left_mle_desc = CircuitMle::new(
        mle_desc.layer_id(),
        &prefix_bits
            .iter()
            .cloned()
            .chain(vec![MleIndex::Fixed(false)])
            .chain(repeat_n(MleIndex::Iterated, num_iterated_bits - 1))
            .collect_vec(),
    );
    let right_mle_desc = CircuitMle::new(
        mle_desc.layer_id(),
        &prefix_bits
            .iter()
            .cloned()
            .chain(vec![MleIndex::Fixed(true)])
            .chain(repeat_n(MleIndex::Iterated, num_iterated_bits - 1))
            .collect_vec(),
    );
    (left_mle_desc, right_mle_desc)
}

/// Given two Mles of the same length representing the numerators and denominators of a sequence of
/// fractions, add layers that perform a sum of the fractions, return a new pair of length-1 Mles
/// representing the numerator and denominator of the sum.
fn build_fractional_sum<F: Field>(
    maybe_numerator_desc: Option<CircuitMle<F>>,
    denominator_desc: CircuitMle<F>,
    layers: &mut Vec<CircuitLayerEnum<F>>,
    current_layer_id: &mut LayerId,
) -> (Option<CircuitMle<F>>, CircuitMle<F>) {
    type CE<F> = Expression<F, CircuitExpr>;

    // --- Sanitycheck number of vars in numerator == number of vars in denominator ---
    // --- EXCEPT when we're working with the fraction with constant 1 in the numerator ---
    if let Some(numerator_desc) = maybe_numerator_desc.as_ref() {
        assert_eq!(
            numerator_desc.num_iterated_vars(),
            denominator_desc.num_iterated_vars()
        );
    }

    let mut maybe_numerator_desc = maybe_numerator_desc;
    let mut denominator_desc = denominator_desc;

    for i in 0..denominator_desc.num_iterated_vars() {
        let denominators = split_circuit_mle(&denominator_desc);
        let next_numerator_expr = if let Some(numerator_desc) = maybe_numerator_desc {
            let numerators = split_circuit_mle(&numerator_desc);

            // Calculate the new numerator
            CE::products(vec![numerators.0.clone(), denominators.1.clone()])
                + CE::products(vec![numerators.1.clone(), denominators.0.clone()])
        } else {
            // TODO(ryancao): Technically it should be this
            // CE::scaled(denominators.1.expression(), F::ONE)
            //     + CE::scaled(denominators.0.expression(), F::ONE)
            denominators.1.clone().expression() + denominators.0.clone().expression()
        };

        // Calculate the new denominator
        let next_denominator_expr =
            CE::products(vec![denominators.0.clone(), denominators.1.clone()]);

        // Grab the size of each
        let next_numerator_num_vars = next_numerator_expr.num_vars();
        let next_denominator_num_vars = next_denominator_expr.num_vars();

        // Create the circuit layer by combining the two
        let layer_id = current_layer_id.get_and_inc();

        let layer = CircuitRegularLayer::new_raw(
            layer_id,
            next_numerator_expr.concat_expr(next_denominator_expr),
        );

        layers.push(CircuitLayerEnum::Regular(layer));

        println!(
            "Iteration {:?} of build_fractional_sumcheck has layer id: {:?}",
            i, layer_id
        );

        denominator_desc = CircuitMle::new(
            layer_id,
            &std::iter::once(MleIndex::Fixed(false))
                .chain(repeat_n(MleIndex::Iterated, next_denominator_num_vars))
                .collect_vec(),
        );
        maybe_numerator_desc = Some(CircuitMle::new(
            layer_id,
            &std::iter::once(MleIndex::Fixed(true))
                .chain(repeat_n(MleIndex::Iterated, next_numerator_num_vars))
                .collect_vec(),
        ));
    }
    if let Some(numerator_desc) = maybe_numerator_desc.as_ref() {
        assert_eq!(numerator_desc.num_iterated_vars(), 0);
    }
    assert_eq!(denominator_desc.num_iterated_vars(), 0);
    (maybe_numerator_desc, denominator_desc)
}
