//! The basic building block of a regular gkr circuit. The Sector node

use std::collections::HashMap;

use itertools::Itertools;
use remainder_shared_types::Field;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layer::{layer_enum::LayerDescriptionEnum, regular_layer::RegularLayerDescription, LayerId},
    layouter::layouting::{topo_sort, CircuitDescriptionMap, CircuitLocation, DAGError},
};

use super::{CircuitNode, CompilableNode, Context, NodeId};

#[derive(Debug, Clone)]
/// A sector node in the circuit DAG, can have multiple inputs, and a single output
pub struct Sector<F: Field> {
    id: NodeId,
    expr: Expression<F, AbstractExpr>,
    num_vars: usize,
}

impl<F: Field> Sector<F> {
    /// creates a new sector node
    pub fn new(
        ctx: &Context,
        inputs: &[&dyn CircuitNode],
        expr_builder: impl FnOnce(Vec<NodeId>) -> Expression<F, AbstractExpr>,
    ) -> Self {
        let node_ids = inputs.iter().map(|node| node.id()).collect();
        let expr = expr_builder(node_ids);
        let num_vars_map = inputs
            .iter()
            .map(|node| (node.id(), node.get_num_vars()))
            .collect();
        let expr_num_vars = expr.num_vars(&num_vars_map).unwrap();

        Self {
            id: ctx.get_new_id(),
            expr,
            num_vars: expr_num_vars,
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

//todo remove this super jank workaround
impl<'a, F: Field> CircuitNode for &'a Sector<F> {
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

/// A grouping of Sectors that are compiled all at once
///
/// Creating a SectorGroup as part of a Component before layouting will
/// prevent the layouter from joining any more Sectors to the SectorGroup
#[derive(Clone, Debug)]
pub struct SectorGroup<F: Field> {
    children: Vec<Sector<F>>,
    id: NodeId,
}

impl<F: Field> SectorGroup<F> {
    /// Creates a new SectorGroup
    pub fn new(ctx: &Context, children: Vec<Sector<F>>) -> Self {
        Self {
            children,
            id: ctx.get_new_id(),
        }
    }

    /// Add a sector to the SectorGroup
    pub fn add_sector(&mut self, new_sector: Sector<F>) {
        self.children.push(new_sector);
    }
}

impl<F: Field> CircuitNode for SectorGroup<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        self.children
            .iter()
            .flat_map(|sector| sector.sources())
            .collect()
    }

    fn subnodes(&self) -> Option<Vec<NodeId>> {
        let children = self.children.iter().map(|sector| sector.id()).collect();
        Some(children)
    }

    fn get_num_vars(&self) -> usize {
        todo!()
    }
}

impl<F: Field> CompilableNode<F> for SectorGroup<F> {
    fn generate_circuit_description(
        &self,
        layer_id: &mut LayerId,
        circuit_description_map: &mut CircuitDescriptionMap,
    ) -> Result<Vec<LayerDescriptionEnum<F>>, DAGError> {
        //topo sort the children
        let children = self.children.iter().collect_vec();
        let children = topo_sort(children)?;

        //assign layers
        let mut node_to_layer_offset_map: HashMap<NodeId, usize> = HashMap::new();
        let mut layers: Vec<Vec<&Sector<F>>> = vec![];

        for child in children {
            let max_source_layer_offset = child
                .sources()
                .iter()
                .map(|id| node_to_layer_offset_map.get(id).copied())
                .max()
                .unwrap_or(None);

            let layer_offset = max_source_layer_offset
                .map(|offset| offset + 1)
                .unwrap_or(0);

            node_to_layer_offset_map.insert(child.id(), layer_offset);

            if let Some(sectors) = layers.get_mut(layer_offset) {
                sectors.push(child);
            } else {
                layers.push(vec![child]);
            }
        }

        // compile and add layers
        let compiled_layers = layers
            .into_iter()
            .map(|children| {
                LayerDescriptionEnum::Regular(
                    compile_layer(children.as_slice(), layer_id, circuit_description_map).unwrap(),
                )
            })
            .collect_vec();
        Ok(compiled_layers)
    }
}

/// Takes some sectors that all belong in a single layer and
/// builds the layer/adds their locations to the circuit map
fn compile_layer<F: Field>(
    children: &[&Sector<F>],
    layer_id: &mut LayerId,
    circuit_description_map: &mut CircuitDescriptionMap,
) -> Result<RegularLayerDescription<F>, DAGError> {
    // This will store all the expression yet to be merged
    let mut expression = children
        .iter()
        .map(|sector| Ok((vec![sector.id], sector.expr.clone(), sector.get_num_vars())))
        .collect::<Result<Vec<_>, _>>()?;

    // These prefix bits will be stored in reverse order!
    let mut prefix_bits: HashMap<NodeId, Vec<bool>> = HashMap::new();
    for child in children.iter().map(|child| child.id()) {
        prefix_bits.insert(child, vec![]);
    }

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
            smallest = Expression::<_, AbstractExpr>::constant(F::ZERO).select(smallest);
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

    let expr = new_expr.build_circuit_expr(circuit_description_map)?;

    let regular_layer_id = layer_id.get_and_inc();
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
        circuit_description_map.add_node_id_and_location_num_vars(
            node_id,
            (
                CircuitLocation::new(regular_layer_id, prefix_bits),
                *num_vars_sector,
            ),
        );
    }

    Ok(layer)
}

#[cfg(test)]
mod tests {
    use remainder_shared_types::Fr;

    use crate::{
        expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
        layer::LayerId,
        layouter::{
            layouting::{CircuitDescriptionMap, CircuitLocation},
            nodes::{
                circuit_inputs::{InputLayerNode, InputShred},
                CircuitNode, CompilableNode, Context,
            },
        },
    };

    use super::{Sector, SectorGroup};

    #[test]
    fn test_sector_group_compile() {
        let ctx = Context::new();
        let input_node = InputLayerNode::new(&ctx, None);
        let input_shred_1: InputShred = InputShred::new(&ctx, 0, &input_node);
        let input_shred_2 = InputShred::new(&ctx, 0, &input_node);

        let sector_1: Sector<Fr> = Sector::new(&ctx, &[&input_shred_1, &input_shred_2], |inputs| {
            Expression::<_, AbstractExpr>::mle(inputs[0])
                + Expression::<_, AbstractExpr>::mle(inputs[1])
        });
        let sector_2 = Sector::new(&ctx, &[&input_shred_1, &input_shred_2], |inputs| {
            Expression::<_, AbstractExpr>::mle(inputs[0])
                - Expression::<_, AbstractExpr>::mle(inputs[1])
        });

        let sector_out = Sector::new(&ctx, &[&sector_1, &&sector_2], |inputs| {
            Expression::<_, AbstractExpr>::products(vec![inputs[0], inputs[1]])
        });

        let sector_group = SectorGroup::new(&ctx, vec![sector_1, sector_2, sector_out]);
        let mut circuit_description_map = CircuitDescriptionMap::new();
        circuit_description_map.add_node_id_and_location_num_vars(
            input_shred_1.id(),
            (
                CircuitLocation::new(LayerId::Input(0), vec![]),
                input_shred_1.get_num_vars(),
            ),
        );
        circuit_description_map.add_node_id_and_location_num_vars(
            input_shred_2.id(),
            (
                CircuitLocation::new(LayerId::Input(1), vec![]),
                input_shred_2.get_num_vars(),
            ),
        );
        let _circuit_description = sector_group
            .generate_circuit_description(&mut LayerId::Input(1), &mut circuit_description_map)
            .unwrap();
    }
}
