//! The basic building block of a regular gkr circuit. The Sector node

use std::collections::{BTreeSet, HashMap};

use remainder_shared_types::Field;

use remainder::{
    circuit_layout::CircuitLocation,
    layer::{layer_enum::LayerDescriptionEnum, regular_layer::RegularLayerDescription, LayerId},
    utils::arithmetic::log2_ceil,
};

use crate::{
    abstract_expr::AbstractExpression,
    layouter::{builder::CircuitMap, nodes::CompilableNode},
};

use super::{CircuitNode, NodeId};

use anyhow::Result;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
/// A sector node in the circuit DAG, can have multiple inputs, and a single
/// output
pub struct Sector<F: Field> {
    id: NodeId,
    expr: AbstractExpression<F>,
    num_vars: usize,
}

impl<F: Field> Sector<F> {
    /// creates a new sector node
    pub fn new(expr: AbstractExpression<F>, num_vars: usize) -> Self {
        Self {
            id: NodeId::new(),
            expr,
            num_vars,
        }
    }
}

impl<F: Field> CircuitNode for Sector<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        self.expr.get_sources()
    }

    fn get_num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field> CompilableNode<F> for Sector<F> {
    fn generate_circuit_description(
        &self,
        circuit_map: &mut CircuitMap,
    ) -> Result<Vec<LayerDescriptionEnum<F>>> {
        Ok(generate_sector_circuit_description(
            &[self],
            circuit_map,
            None,
        ))
    }
}

/// Generate a circuit description for a vector of sectors that are to be
/// combined. I.e., the sectors passed into this function do not have any
/// dependencies between each other. The expected behavior of this function is
/// to return a single layer which has merged the expressions of each of the
/// individual sectors into a single expression.
pub fn generate_sector_circuit_description<F: Field>(
    sectors: &[&Sector<F>],
    circuit_map: &mut CircuitMap,
    maybe_maximum_log_layer_size: Option<usize>,
) -> Vec<LayerDescriptionEnum<F>> {
    compile_sectors_into_layer_descriptions(sectors, circuit_map, maybe_maximum_log_layer_size)
        .unwrap()
        .into_iter()
        .map(|regular_layer| LayerDescriptionEnum::Regular(regular_layer))
        .collect()
}

/// Takes some sectors that all belong in a single layer and builds the
/// layer/adds their locations to the circuit map
fn compile_sectors_into_layer_descriptions<F: Field>(
    children: &[&Sector<F>],
    circuit_map: &mut CircuitMap,
    maybe_maximum_log_layer_size: Option<usize>,
) -> Result<Vec<RegularLayerDescription<F>>> {
    // Compute the total number of coefficients required to fully merge this
    // expression.
    let mut total_num_coeff: usize = 0;
    // This will store all the expression yet to be merged, along with the
    // sector's ID as well as the number of variables.
    let mut expression_vec = children
        .iter()
        .map(|sector| {
            total_num_coeff += 1 << sector.get_num_vars();
            Ok((
                vec![sector.id()],
                sector.expr.clone(),
                sector.get_num_vars(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    // If the max is specified by the circuit builder, we set it to that.
    // Otherwise, we allow all of the expressions to be combined so we set it to
    // what would be the total number of coefficients of the fully merged
    // expression.
    let maximum_log_layer_size = maybe_maximum_log_layer_size
        .unwrap_or(log2_ceil(total_num_coeff.next_power_of_two()) as usize);

    // These prefix bits will be stored in reverse order for each NodeID. Store
    // the number of variables existing in the sector.
    let mut prefix_bits_map: HashMap<NodeId, (Vec<bool>, usize)> = HashMap::new();
    for sector in children.iter() {
        prefix_bits_map.insert(sector.id(), (vec![], sector.get_num_vars()));
    }

    // We loop until all the expressions are merged, or the smallest merged
    // expression exceeds the maximum layer size specified. This means that we
    // cannot merge any more expressions, and we should compile the rest of the
    // expressions as is.
    let new_expr_vec = loop {
        // Either we have merged all the expressions into one, in this case we
        // are done combining the expressions.
        if expression_vec.len() == 1 {
            break expression_vec;
        }
        // We merge the two smallest expressions first.
        expression_vec.sort_by(|rhs, lhs| rhs.2.cmp(&lhs.2).reverse());

        // Or the two smallest expressions exceed the maximum layer size,
        // meaning none of the other expressions can be combined. We are done
        // combining if this is true.
        //
        // The total number of coefficients of the merged expression is one
        // power of two more than the second smallest expression, as we are
        // combining the two smallest expressions.
        let total_num_coeff_merged_expr = 1 << (expression_vec[expression_vec.len() - 2].2 + 1);
        if total_num_coeff_merged_expr > (1 << maximum_log_layer_size) {
            break expression_vec;
        }

        let (smallest_ids, mut smallest, smallest_num_vars) = expression_vec.pop().unwrap();
        let (next_ids, next, next_num_vars) = expression_vec.pop().unwrap();

        // The number of selector variables that need to be added to the smaller
        // expression to make both expressions the same size.
        let padding_selector_vars = next_num_vars - smallest_num_vars;
        // Add any new selector nodes that are needed for padding.
        for _ in 0..padding_selector_vars {
            // This results in padding the MLE of the smallest expression with
            // 0s.
            smallest = AbstractExpression::constant(F::ZERO).select(smallest);
            for node_id in &smallest_ids {
                let (prefix_bits, _) = prefix_bits_map.get_mut(node_id).unwrap();
                prefix_bits.push(true);
            }
        }

        // Merge the two expressions. Now the smallest expression and next
        // expression are the same size.
        smallest = next.select(smallest);

        // Track the prefix bits we're creating so they can be added to the
        // circuit_map; each concat operation pushes a new prefix_bit.
        for node_id in &smallest_ids {
            let (prefix_bits, _) = prefix_bits_map.get_mut(node_id).unwrap();
            prefix_bits.push(true);
        }

        for node_id in &next_ids {
            let (prefix_bits, _) = prefix_bits_map.get_mut(node_id).unwrap();
            prefix_bits.push(false);
        }

        expression_vec.push((
            [smallest_ids, next_ids].concat(),
            smallest,
            next_num_vars + 1,
        ));
    };

    // Keep track of which node IDs are getting added to the circuit map.
    let mut node_ids_added_to_circuit_map: BTreeSet<NodeId> = BTreeSet::new();
    // Go through all of the expressions in the vector, which have either been
    // merged or stayed the same, and compile them into
    // [RegularLayerDescription]s.
    let layer_vec: Vec<RegularLayerDescription<F>> = new_expr_vec
        .into_iter()
        .map(|(sector_nodes, expression, _expr_num_vars)| {
            let expr = expression.build_circuit_expr(circuit_map).unwrap();
            let regular_layer_id = LayerId::next_layer_id();
            let layer = RegularLayerDescription::new_raw(regular_layer_id, expr);
            prefix_bits_map
                .iter_mut()
                .for_each(|(node_id, (prefix_bits, num_vars))| {
                    if sector_nodes.contains(node_id) {
                        node_ids_added_to_circuit_map.insert(*node_id);
                        prefix_bits.reverse();
                        circuit_map.add_node_id_and_location_num_vars(
                            *node_id,
                            (
                                CircuitLocation::new(regular_layer_id, prefix_bits.to_vec()),
                                *num_vars,
                            ),
                        );
                    }
                });
            layer
        })
        .collect();
    // Assert that all of the node ids have been added to the circuit map from
    // those originally populated into the `prefix_bits_map`.
    assert_eq!(node_ids_added_to_circuit_map.len(), prefix_bits_map.len());

    Ok(layer_vec)
}
