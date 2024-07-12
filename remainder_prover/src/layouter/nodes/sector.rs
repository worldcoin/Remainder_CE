//! The basic building block of a regular gkr circuit. The Sector node

use std::collections::HashMap;

use itertools::Itertools;
use remainder_shared_types::FieldExt;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layer::regular_layer::RegularLayer,
    layouter::{
        compiling::WitnessBuilder,
        layouting::{topo_sort, CircuitLocation, CircuitMap, DAGError},
    },
    mle::evals::MultilinearExtension,
    prover::proof_system::ProofSystem,
};

use super::{CircuitNode, ClaimableNode, CompilableNode, Context, NodeId};

#[derive(Debug, Clone)]
/// A sector node in the circuit DAG, can have multiple inputs, and a single output
pub struct Sector<F: FieldExt> {
    id: NodeId,
    expr: Expression<F, AbstractExpr>,
    data: MultilinearExtension<F>,
}

impl<F: FieldExt> Sector<F> {
    /// creates a new sector node
    pub fn new(
        ctx: &Context,
        inputs: &[&dyn ClaimableNode<F = F>],
        expr_builder: impl FnOnce(Vec<NodeId>) -> Expression<F, AbstractExpr>,
        data_builder: impl FnOnce(Vec<&MultilinearExtension<F>>) -> MultilinearExtension<F>,
    ) -> Self {
        let node_ids = inputs.iter().map(|node| node.id()).collect();
        let expr = expr_builder(node_ids);
        let input_data = inputs.iter().map(|node| node.get_data()).collect();
        let data = data_builder(input_data);

        Self {
            id: ctx.get_new_id(),
            expr,
            data,
        }
    }
}

impl<F: FieldExt> CircuitNode for Sector<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        self.expr.get_sources()
    }
}

//todo remove this super jank workaround
impl<'a, F: FieldExt> CircuitNode for &'a Sector<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        self.expr.get_sources()
    }
}

impl<F: FieldExt> ClaimableNode for Sector<F> {
    type F = F;

    fn get_data(&self) -> &MultilinearExtension<Self::F> {
        &self.data
    }

    fn get_expr(&self) -> Expression<Self::F, AbstractExpr> {
        Expression::<F, AbstractExpr>::mle(self.id)
    }
}

/// A grouping of Sectors that are compiled all at once
///
/// Creating a SectorGroup as part of a Component before layouting will
/// prevent the layouter from joining any more Sectors to the SectorGroup
#[derive(Clone, Debug)]
pub struct SectorGroup<F: FieldExt> {
    children: Vec<Sector<F>>,
    id: NodeId,
}

impl<F: FieldExt> SectorGroup<F> {
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

impl<F: FieldExt> CircuitNode for SectorGroup<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        self.children
            .iter()
            .flat_map(|sector| sector.sources())
            .collect()
    }
    fn children(&self) -> Option<Vec<NodeId>> {
        let children = self.children.iter().map(|sector| sector.id()).collect();
        Some(children)
    }
}

impl<F: FieldExt, Pf: ProofSystem<F, Layer = L>, L> CompilableNode<F, Pf> for SectorGroup<F>
where
    L: From<RegularLayer<F>>,
{
    fn compile<'a>(
        &'a self,
        witness_builder: &mut WitnessBuilder<F, Pf>,
        circuit_map: &mut CircuitMap<'a, F>,
    ) -> Result<(), DAGError> {
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

        //compile and add layers
        for children in layers {
            compile_layer(children.as_slice(), witness_builder, circuit_map)?;
        }
        Ok(())
    }
}

/// Takes some sectors that all belong in a single layer and
/// builds the layer/adds their locations to the circuit map
fn compile_layer<'a, F: FieldExt, Pf: ProofSystem<F, Layer = L>, L: From<RegularLayer<F>>>(
    children: &[&'a Sector<F>],
    witness_builder: &mut WitnessBuilder<F, Pf>,
    circuit_map: &mut CircuitMap<'a, F>,
) -> Result<(), DAGError> {
    let layer_id = witness_builder.next_layer();
    let mut expression = children
        .iter()
        .map(|sector| {
            Ok((
                vec![sector.id],
                sector.expr.clone(),
                sector.expr.num_vars(circuit_map)?,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // These prefix bits will be stored in reverse order!
    let mut prefix_bits: HashMap<NodeId, Vec<bool>> = HashMap::new();
    for child in children.iter().map(|child| child.id()) {
        prefix_bits.insert(child, vec![]);
    }

    let new_expr = loop {
        if expression.len() == 1 {
            break expression.pop().unwrap();
        }
        expression.sort_by(|rhs, lhs| rhs.2.cmp(&lhs.2).reverse());

        let (smallest_ids, mut smallest, smallest_num_vars) = expression.pop().unwrap();
        let (next_ids, next, next_num_vars) = expression.pop().unwrap();

        let padding_bits = next_num_vars - smallest_num_vars;

        for _ in 0..padding_bits {
            smallest = smallest.concat_expr(Expression::<_, AbstractExpr>::constant(F::ZERO));
            for node_id in &smallest_ids {
                let prefix_bits = prefix_bits.get_mut(node_id).unwrap();
                prefix_bits.push(true);
            }
        }

        smallest = smallest.concat_expr(next);

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

    let expr = new_expr.build_prover_expr(circuit_map)?;

    let layer = RegularLayer::new_raw(layer_id, expr);

    witness_builder.add_layer(layer.into());

    // Add the new sectors to the circuit map
    for (node_id, mut prefix_bits) in prefix_bits {
        prefix_bits.reverse();
        //todo! make this less crap
        let data = &children
            .iter()
            .filter(|item| item.id == node_id)
            .collect_vec()[0]
            .data;
        circuit_map
            .0
            .insert(node_id, (CircuitLocation::new(layer_id, prefix_bits), data));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use remainder_shared_types::{layer::LayerId, Fr};

    use crate::{
        expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
        layouter::{
            compiling::WitnessBuilder,
            layouting::{CircuitLocation, CircuitMap},
            nodes::{
                circuit_inputs::InputShred, CircuitNode, ClaimableNode, CompilableNode, Context,
            },
        },
        mle::evals::MultilinearExtension,
        prover::proof_system::DefaultProofSystem,
    };

    use super::{Sector, SectorGroup};

    #[test]
    fn test_sector_group_compile() {
        let ctx = Context::new();
        let input_shred_1: InputShred<Fr> =
            InputShred::new(&ctx, MultilinearExtension::new_zero(), None);
        let input_shred_2 = InputShred::new(&ctx, MultilinearExtension::new_zero(), None);

        let sector_1 = Sector::new(
            &ctx,
            &[&input_shred_1, &input_shred_2],
            |inputs| {
                Expression::<_, AbstractExpr>::mle(inputs[0])
                    + Expression::<_, AbstractExpr>::mle(inputs[1])
            },
            |_| MultilinearExtension::new_zero(),
        );
        let sector_2 = Sector::new(
            &ctx,
            &[&input_shred_1, &input_shred_2],
            |inputs| {
                Expression::<_, AbstractExpr>::mle(inputs[0])
                    - Expression::<_, AbstractExpr>::mle(inputs[1])
            },
            |_| MultilinearExtension::new_zero(),
        );

        let sector_out = Sector::new(
            &ctx,
            &[&sector_1, &sector_2],
            |inputs| Expression::<_, AbstractExpr>::products(vec![inputs[0], inputs[1]]),
            |_| MultilinearExtension::new_zero(),
        );

        let sector_group = SectorGroup::new(&ctx, vec![sector_1, sector_2, sector_out]);
        let mut witness_builder: WitnessBuilder<Fr, DefaultProofSystem> = WitnessBuilder::new();
        let mut circuit_map = CircuitMap::new();
        circuit_map.0.insert(
            input_shred_1.id(),
            (
                CircuitLocation::new(LayerId::Input(0), vec![]),
                input_shred_1.get_data(),
            ),
        );
        circuit_map.0.insert(
            input_shred_2.id(),
            (
                CircuitLocation::new(LayerId::Input(1), vec![]),
                input_shred_2.get_data(),
            ),
        );
        sector_group
            .compile(&mut witness_builder, &mut circuit_map)
            .unwrap();
    }
}
