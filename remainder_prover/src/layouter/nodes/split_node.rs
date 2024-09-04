//! A node that can alter the claims made on it's source `ClaimableNode`

use std::{iter, marker::PhantomData};

use itertools::{repeat_n, Itertools};
use remainder_shared_types::FieldExt;

use crate::{
    expression::{
        abstract_expr::AbstractExpr,
        circuit_expr::{CircuitExpr, CircuitMle},
        generic_expr::Expression,
    },
    layer::{layer_enum::LayerEnum, regular_layer::RegularLayer, Layer, LayerId},
    layouter::layouting::{CircuitDescriptionMap, CircuitLocation, DAGError},
    mle::evals::{Evaluations, MultilinearExtension},
    prover::proof_system::ProofSystem,
};

use super::{CircuitNode, CompilableNode, Context, NodeId};

/// A Node that derives new `ClaimableNode`s from a single
/// `ClaimableNode`.
///
/// The new nodes represent the input node split
/// by a selector bit.
#[derive(Clone, Debug)]
pub struct SplitNode<F: FieldExt> {
    id: NodeId,
    num_vars: usize,
    source: NodeId,
    prefix_bits: Vec<bool>,
    _marker: PhantomData<F>,
}

impl<F: FieldExt> SplitNode<F> {
    /// Creates 2^num_vars `SplitNodes` from a single ClaimableNode
    pub fn new(ctx: &Context, node: &impl CircuitNode, num_vars: usize) -> Vec<Self> {
        let num_vars_node = node.get_num_vars();
        let source = node.id();
        let max_num_vars = num_vars_node - num_vars;
        (0..(1 << num_vars))
            .zip(bits_iter(num_vars))
            .map(|(_, prefix_bits)| Self {
                id: ctx.get_new_id(),
                source,
                num_vars: max_num_vars,
                prefix_bits,
                _marker: PhantomData,
            })
            .collect()
    }
}

impl<F: FieldExt> CircuitNode for SplitNode<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.source]
    }

    fn get_num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: FieldExt> CompilableNode<F> for SplitNode<F> {
    fn generate_circuit_description<'a>(
        &'a self,
        layer_id: &mut LayerId,
        circuit_description_map: &mut CircuitDescriptionMap<'a, F>,
    ) -> Result<Vec<LayerEnum<F>>, DAGError> {
        let (source_location, _) = circuit_description_map.get_node(&self.source)?;

        let prefix_bits = source_location
            .prefix_bits
            .iter()
            .chain(self.prefix_bits.iter())
            .copied()
            .collect();

        let location = CircuitLocation::new(source_location.layer_id, prefix_bits);

        circuit_description_map.add_node(self.id, location);
        Ok(vec![])
    }
}

///returns an iterator that wil give permutations of binary bits of size
/// num_bits
///
/// 0,0,0 -> 0,0,1 -> 0,1,0 -> 0,1,1 -> 1,0,0 -> 1,0,1 -> 1,1,0 -> 1,1,1
fn bits_iter(num_bits: usize) -> impl Iterator<Item = Vec<bool>> {
    std::iter::successors(Some(vec![false; num_bits]), move |prev| {
        let mut prev = prev.clone();
        let mut removed_bits = 0;
        for index in (0..num_bits).rev() {
            let curr = prev.remove(index);
            if !curr {
                prev.push(true);
                break;
            } else {
                removed_bits += 1;
            }
        }
        if removed_bits == num_bits {
            None
        } else {
            Some(
                prev.into_iter()
                    .chain(repeat_n(false, removed_bits))
                    .collect_vec(),
            )
        }
    })
}

#[cfg(test)]
mod test {

    use itertools::Itertools;
    use remainder_shared_types::Fr;

    use crate::{
        expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
        layouter::{
            compiling::LayouterCircuit,
            component::ComponentSet,
            nodes::{
                circuit_inputs::{InputLayerNode, InputLayerType, InputShred},
                circuit_outputs::OutputNode,
                node_enum::NodeEnum,
                sector::Sector,
            },
        },
        mle::evals::MultilinearExtension,
        prover::helpers::test_circuit,
    };

    use super::SplitNode;

    #[test]
    fn test_split_node_in_circuit() {
        let circuit = LayouterCircuit::new(|ctx| {
            let mle = MultilinearExtension::<Fr>::interlace_mles(vec![
                MultilinearExtension::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]),
                MultilinearExtension::new(vec![Fr::from(5), Fr::from(6), Fr::from(7), Fr::from(8)]),
            ]);
            // the mle_out = [1*5, 2*6, 3*7, 4*8], the product between the two split nodes
            let mle_out = MultilinearExtension::new(vec![
                Fr::from(1 * 5),
                Fr::from(2 * 6),
                Fr::from(3 * 7),
                Fr::from(4 * 8),
            ]);

            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);

            let input_shred = InputShred::new(ctx, mle, &input_layer);
            let input_shred_out = InputShred::new(ctx, mle_out, &input_layer);

            let split_sectors = SplitNode::new(ctx, &input_shred, 1);
            let sector_prod = Sector::new(ctx, &[&split_sectors[0], &split_sectors[1]], |inputs| {
                Expression::<_, AbstractExpr>::products(vec![inputs[0], inputs[1]])
            });

            let final_sector = Sector::new(ctx, &[&&sector_prod, &input_shred_out], |inputs| {
                Expression::<Fr, AbstractExpr>::mle(inputs[0])
                    - Expression::<Fr, AbstractExpr>::mle(inputs[1])
            });

            let output = OutputNode::new_zero(ctx, &final_sector);

            ComponentSet::<NodeEnum<Fr>>::new_raw(
                vec![
                    input_layer.into(),
                    input_shred.into(),
                    input_shred_out.into(),
                    sector_prod.into(),
                    final_sector.into(),
                    output.into(),
                ]
                .into_iter()
                .chain(
                    split_sectors
                        .into_iter()
                        .map(|split_node| split_node.into())
                        .collect_vec(),
                )
                .collect_vec(),
            )
        });

        test_circuit(circuit, None);
    }
}
