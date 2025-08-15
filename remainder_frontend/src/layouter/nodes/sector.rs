//! The basic building block of a regular gkr circuit. The Sector node

use std::collections::HashMap;

use itertools::Itertools;
use remainder_shared_types::Field;

<<<<<<< HEAD:remainder_prover/src/layouter/nodes/sector.rs
use crate::{
    abstract_expr::AbstractExpression,
=======
use remainder::{
    circuit_layout::CircuitLocation,
>>>>>>> benny/extract_frontend:remainder_frontend/src/layouter/nodes/sector.rs
    layer::{layer_enum::LayerDescriptionEnum, regular_layer::RegularLayerDescription, LayerId},
};

use crate::{
    abstract_expr::AbstractExpression,
    layouter::{builder::CircuitMap, nodes::CompilableNode},
};

use super::{CircuitNode, NodeId};

use anyhow::Result;

#[derive(Debug, Clone)]
/// A sector node in the circuit DAG, can have multiple inputs, and a single output
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
        Ok(generate_sector_circuit_description(&[self], circuit_map))
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
) -> Vec<LayerDescriptionEnum<F>> {
    let compiled_layer =
        LayerDescriptionEnum::Regular(compile_layer(sectors, circuit_map).unwrap());
    vec![compiled_layer]
}

/// Takes some sectors that all belong in a single layer and
/// builds the layer/adds their locations to the circuit map
fn compile_layer<F: Field>(
    children: &[&Sector<F>],
    circuit_map: &mut CircuitMap,
) -> Result<RegularLayerDescription<F>> {
    // This will store all the expression yet to be merged
    let mut expression = children
        .iter()
        .map(|sector| Ok((vec![sector.id], sector.expr.clone(), sector.get_num_vars())))
        .collect::<Result<Vec<_>>>()?;

    // These prefix bits will be stored in reverse order!
    let mut prefix_bits: HashMap<NodeId, Vec<bool>> = HashMap::new();
    for child in children.iter().map(|child| child.id()) {
        prefix_bits.insert(child, vec![]);
    }
    // let mut children_to_

    // We loop until all the expressions are merged
    let new_expr = loop {
        if expression.len() == 1 {
            break expression.pop().unwrap();
        }
        // We merge smallest first
        expression.sort_by(|rhs, lhs| rhs.2.cmp(&lhs.2).reverse());

        let (smallest_ids, mut smallest, smallest_num_vars) = expression.pop().unwrap();
        let (next_ids, next, next_num_vars) = expression.pop().unwrap();

        let padding_bits = next_num_vars - smallest_num_vars;

        // Add any new selector nodes that are needed for padding
        for _ in 0..padding_bits {
            smallest = AbstractExpression::constant(F::ZERO).select(smallest);
            for node_id in &smallest_ids {
                let prefix_bits = prefix_bits.get_mut(node_id).unwrap();
                prefix_bits.push(true);
            }
        }

        // Merge the two expressions
        smallest = next.select(smallest);

        // Track the prefix bits we're creating so they can be added to the circuit_map; each concat operation pushes a new prefix_bit
        for node_id in &smallest_ids {
            let prefix_bits = prefix_bits.get_mut(node_id).unwrap();
            prefix_bits.push(true);
        }

        for node_id in &next_ids {
            let prefix_bits = prefix_bits.get_mut(node_id).unwrap();
            prefix_bits.push(false);
        }

        expression.push((
            [smallest_ids, next_ids].concat(),
            smallest,
            next_num_vars + 1,
        ));
    }
    .1;

    let expr = new_expr.build_circuit_expr(circuit_map)?;

    let regular_layer_id = LayerId::next_layer_id();
    let layer = RegularLayerDescription::new_raw(regular_layer_id, expr);

    // Add the new sectors to the circuit map
    for (node_id, mut prefix_bits) in prefix_bits {
        prefix_bits.reverse();
        //todo! make this less crap
        let num_vars_sector = &children
            .iter()
            .filter(|item| item.id == node_id)
            .collect_vec()[0]
            .get_num_vars();
        circuit_map.add_node_id_and_location_num_vars(
            node_id,
            (
                CircuitLocation::new(regular_layer_id, prefix_bits),
                *num_vars_sector,
            ),
        );
    }

    Ok(layer)
}
