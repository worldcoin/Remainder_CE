//!Utilities for combining sub-circuits

use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder_shared_types::{transcript::Transcript, FieldExt};
use thiserror::Error;

use crate::{
    expression::ExpressionStandard,
    layer::{layer_enum::LayerEnum, Layer, LayerId},
    mle::{MleIndex, MleRef},
};

use super::Layers;

#[derive(Error, Debug)]
#[error("Layers can't be combined!")]
pub struct CombineError;

///Utility for combining sub-circuits into a single circuit
/// DOES NOT WORK FOR GATE MLE
pub fn combine_layers<F: FieldExt, Tr: Transcript<F>>(
    mut layers: Vec<Layers<F, Tr>>,
) -> Result<Layers<F, Tr>, CombineError> {
    let layer_count = layers.iter().map(|layers| layers.0.len()).max().unwrap();

    let bits_iter = |num_bits| {
        std::iter::successors(
            Some(vec![MleIndex::<F>::Fixed(false); num_bits]),
            move |prev| {
                let mut prev = prev.clone();
                let mut removed_bits = 0;
                for index in (0..num_bits).rev() {
                    let curr = prev.remove(index);
                    if curr == MleIndex::Fixed(false) {
                        prev.push(MleIndex::Fixed(true));
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
                            .chain(repeat_n(MleIndex::Fixed(false), removed_bits))
                            .collect_vec(),
                    )
                }
            },
        )
    };

    let mut layer_new_bits = vec![0; layer_count];

    let layers: Vec<LayerEnum<F, Tr>> = (0..layer_count)
        .map(|layer_idx| {
            layers
                .iter()
                .filter_map(|layers| layers.0.get(layer_idx).cloned())
                .collect_vec()
        })
        .zip(layer_new_bits.iter_mut())
        .map(|(layers, layer_new_bits)| todo!())
        .collect_vec();

    todo!()
}

fn add_bits_to_layer_refs<F: FieldExt, Tr: Transcript<F>>(
    layer: &mut LayerEnum<F, Tr>,
    new_bits: Vec<MleIndex<F>>,
    effected_layer: LayerId,
) {
    let expression = match layer {
        LayerEnum::Gkr(layer) => &mut layer.expression,
        LayerEnum::EmptyLayer(layer) => &mut layer.expr,
        _ => unimplemented!(),
    };

    let mut closure = for<'a> |expr: &'a mut ExpressionStandard<F>| -> Result<(), ()> {
        match expr {
            ExpressionStandard::Mle(mle) => {
                if mle.layer_id == effected_layer {
                    mle.mle_indices = new_bits
                        .iter()
                        .chain(mle.mle_indices.iter())
                        .cloned()
                        .collect();
                }
                Ok(())
            }
            ExpressionStandard::Product(mles) => {
                for mle in mles {
                    if mle.layer_id == effected_layer {
                        mle.mle_indices = new_bits
                            .iter()
                            .chain(mle.mle_indices.iter())
                            .cloned()
                            .collect();
                    }
                }
                Ok(())
            }
            ExpressionStandard::Constant(_)
            | ExpressionStandard::Scaled(_, _)
            | ExpressionStandard::Sum(_, _)
            | ExpressionStandard::Negated(_)
            | ExpressionStandard::Selector(_, _, _) => Ok(()),
        }
    };

    expression.traverse_mut(&mut closure).unwrap();
}
