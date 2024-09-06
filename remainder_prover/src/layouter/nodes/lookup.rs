//! Nodes that implement LogUp.

use crate::expression::abstract_expr::AbstractExpr;
use crate::expression::circuit_expr::{CircuitExpr, CircuitMle};
use crate::input_layer::enum_input_layer::CircuitInputLayerEnum;
use crate::input_layer::public_input_layer::CircuitPublicInputLayer;
use crate::layer::layer_enum::CircuitLayerEnum;
use crate::layer::regular_layer::CircuitRegularLayer;
use crate::layer::LayerId;
use crate::layouter::layouting::CircuitDescriptionMap;
use crate::mle::MleIndex;
use crate::output_layer::mle_output_layer::CircuitMleOutputLayer;
use crate::utils::get_total_mle_indices;

use itertools::{repeat_n, Itertools};
use remainder_shared_types::FieldExt;

use crate::expression::{abstract_expr::ExprBuilder, generic_expr::Expression};

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
    pub fn new<F: FieldExt>(
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
    ///     - `table` must have length a power of two.
    pub fn new<F: FieldExt>(
        ctx: &Context,
        table: &dyn CircuitNode,
        secret_constrained_values: bool,
        random_input_node: VerifierChallengeNode,
    ) -> Self {
        if secret_constrained_values {
            unimplemented!(
                "Secret constrained values not yet supported (requires HyraxInputLayer)"
            );
        }
        LookupTable {
            id: ctx.get_new_id(),
            constraints: vec![],
            table_node_id: table.id(),
            random_node_id: random_input_node.id(),
            secret_constrained_values: false,
        }
    }

    /// Add a lookup constraint to this node.
    /// (Will be called by the layouter when laying out the circuit.)
    pub fn add_lookup_constraint(&mut self, constraint: LookupConstraint) {
        self.constraints.push(constraint);
    }

    pub fn generate_lookup_circuit_description<F: FieldExt>(
        &self,
        input_layer_id: &mut LayerId,
        intermediate_layer_id: &mut LayerId,
        circuit_description_map: &mut CircuitDescriptionMap,
    ) -> Result<
        (
            Vec<CircuitInputLayerEnum<F>>,
            Vec<CircuitLayerEnum<F>>,
            Vec<CircuitMleOutputLayer<F>>,
        ),
        crate::layouter::layouting::DAGError,
    > {
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
            circuit_description_map.get_node(&self.random_node_id)?;
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

        // Build the numerator: is all ones (create explicitly since don't want to pad with zeros)
        let expr = ExprBuilder::<F>::constant(F::from(1u64))
            .build_circuit_expr(circuit_description_map)?;
        let layer_id = intermediate_layer_id.get_and_inc();
        let layer = CircuitRegularLayer::new_raw(layer_id, expr);
        println!(
            "Layer that sets the numerators to 1 has layer id: {:?}",
            layer_id
        );
        intermediate_layers.push(CircuitLayerEnum::Regular(layer));
        let lhs_numerator_desc = CircuitMle::new(layer_id, &lhs_denominator_vars);

        // Build the numerator and denominator of the sum of the fractions
        let (lhs_numerator, lhs_denominator) = build_fractional_sum(
            lhs_numerator_desc,
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
            &multiplicities_location
                .prefix_bits
                .iter()
                .map(|prefix_bit_bool| MleIndex::Fixed(*prefix_bit_bool))
                .chain(repeat_n(MleIndex::Iterated, *multiplicities_num_vars))
                .collect_vec(),
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
                        &multiplicities_location
                            .prefix_bits
                            .iter()
                            .map(|prefix_bit_bool| MleIndex::Fixed(*prefix_bit_bool))
                            .chain(repeat_n(MleIndex::Iterated, *multiplicities_num_vars))
                            .collect_vec(),
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
            let first_self_constraint_circuit_info =
                circuit_description_map.0[&self.constraints[0].multiplicities_node_id].clone();
            rhs_numerator_desc = CircuitMle::new(
                layer_id,
                &repeat_n(MleIndex::Iterated, first_self_constraint_circuit_info.1).collect_vec(),
            )
        }

        // Build the denominator r - table

        // --- First grab `r` as a `CircuitMle` from the `circuit_description_map` ---
        let (verifier_challenge_loc, verifier_challenge_num_vars) =
            circuit_description_map.0[&self.random_node_id].clone();
        let verifier_challenge_circuit_mle = CircuitMle::new(
            verifier_challenge_loc.layer_id,
            &repeat_n(MleIndex::Iterated, verifier_challenge_num_vars).collect_vec(),
        );

        // --- Next grab `table` as a `CircuitMle` from the `circuit_description_map` ---
        let (table_loc, table_num_vars) = circuit_description_map.0[&self.table_node_id].clone();
        let table_circuit_mle = CircuitMle::new(
            table_loc.layer_id,
            &repeat_n(MleIndex::Iterated, table_num_vars).collect_vec(),
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
            rhs_numerator_desc,
            rhs_denominator_desc,
            &mut intermediate_layers,
            intermediate_layer_id,
        );

        // Add an input layer for the inverse of the denominators of the LHS. This value holds
        // reveals some information about the constrained values, so we optionally use a
        // HyraxInputLayer.
        // let mle =
        //     MultilinearExtension::new(vec![lhs_denominator.current_mle.value().invert().unwrap()]);

        // --- Grab the layer ID for the new "input layer" to be added ---
        let lhs_denom_inverse_layer_id = input_layer_id.get_and_inc();

        logup_additional_input_layers.push(if self.secret_constrained_values {
            // TODO use HyraxInputLayer, once it's implemented
            unimplemented!();
        } else {
            let public_input_layer_description =
                CircuitPublicInputLayer::<F>::new(lhs_denom_inverse_layer_id.to_owned(), 0);
            CircuitInputLayerEnum::PublicInputLayer(public_input_layer_description)
        });

        let lhs_inverse_mle_desc = CircuitMle::new(lhs_denom_inverse_layer_id, &vec![]);
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
        logup_additional_input_layers.push({
            let public_input_layer_description =
                CircuitPublicInputLayer::<F>::new(rhs_denom_inverse_layer_id.to_owned(), 0);
            CircuitInputLayerEnum::PublicInputLayer(public_input_layer_description)
        });
        let rhs_inverse_mle_desc = CircuitMle::new(rhs_denom_inverse_layer_id, &vec![]);
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
        let output_layer = CircuitMleOutputLayer::new_zero(layer_id, &vec![MleIndex::Iterated]);
        let mut output_layers = vec![output_layer];

        // Add a layer that calculates the difference between the fractions on the LHS and RHS
        let expr = CE::<F>::products(vec![lhs_numerator.clone(), rhs_denominator.clone()])
            - CE::<F>::products(vec![rhs_numerator.clone(), lhs_denominator.clone()]);
        let layer_id = intermediate_layer_id.get_and_inc();
        let layer = CircuitRegularLayer::new_raw(layer_id, expr);
        intermediate_layers.push(CircuitLayerEnum::Regular(layer));
        println!(
            "Layer that checks that fractions are equal has layer id: {:?}",
            layer_id
        );

        // Add an output layer that checks that the result is zero
        let output_layer = CircuitMleOutputLayer::new_zero(layer_id, &vec![]);
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
                    .map(|constraint| constraint.sources())
                    .flatten(),
            )
            .collect()
    }

    fn get_num_vars(&self) -> usize {
        todo!()
    }
}

/// Extract the prefix bits from a DenseMle.
fn extract_prefix_num_iterated_bits<F: FieldExt>(mle: &CircuitMle<F>) -> (Vec<MleIndex<F>>, usize) {
    let mut num_iterated_bits = 0;
    let prefix_bits = mle
        .mle_indices()
        .iter()
        .map(|mle_index| match mle_index {
            MleIndex::Fixed(_) => Some(mle_index.clone()),
            MleIndex::Iterated => {
                num_iterated_bits += 1;
                None
            }
            _ => None,
        })
        .filter_map(|opt| opt)
        .collect();
    (prefix_bits, num_iterated_bits)
}

/// Split a DenseMle into two DenseMles, with the left half containing the even-indexed elements and
/// the right half containing the odd-indexed elements, setting the prefix bits accordingly.
fn split_circuit_mle<F: FieldExt>(mle_desc: &CircuitMle<F>) -> (CircuitMle<F>, CircuitMle<F>) {
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
fn build_fractional_sum<F: FieldExt>(
    numerator_desc: CircuitMle<F>,
    denominator_desc: CircuitMle<F>,
    layers: &mut Vec<CircuitLayerEnum<F>>,
    current_layer_id: &mut LayerId,
) -> (CircuitMle<F>, CircuitMle<F>) {
    type CE<F> = Expression<F, CircuitExpr>;
    assert_eq!(
        numerator_desc.num_iterated_vars(),
        denominator_desc.num_iterated_vars()
    );
    let mut numerator_desc = numerator_desc;
    let mut denominator_desc = denominator_desc;

    for i in 0..numerator_desc.num_iterated_vars() {
        let numerators = split_circuit_mle(&numerator_desc);
        let denominators = split_circuit_mle(&denominator_desc);

        // Calculate the new numerator
        let next_numerator_expr = CE::products(vec![numerators.0.clone(), denominators.1.clone()])
            + CE::products(vec![numerators.1.clone(), denominators.0.clone()]);

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
            i, current_layer_id
        );

        denominator_desc = CircuitMle::new(
            layer_id,
            &std::iter::once(MleIndex::Fixed(false))
                .chain(repeat_n(MleIndex::Iterated, next_denominator_num_vars))
                .collect_vec(),
        );
        numerator_desc = CircuitMle::new(
            layer_id,
            &std::iter::once(MleIndex::Fixed(true))
                .chain(repeat_n(MleIndex::Iterated, next_numerator_num_vars))
                .collect_vec(),
        );
    }
    debug_assert_eq!(numerator_desc.num_iterated_vars(), 0);
    debug_assert_eq!(denominator_desc.num_iterated_vars(), 0);
    (numerator_desc, denominator_desc)
}
