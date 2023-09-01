//!Utilities for combining sub-circuits

use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder_shared_types::{transcript::Transcript, FieldExt};
use thiserror::Error;

use crate::{
    expression::{Expression, ExpressionStandard},
    layer::{from_mle, layer_enum::LayerEnum, GKRLayer, Layer, LayerId},
    mle::{mle_enum::MleEnum, MleIndex, MleRef},
    utils::{argsort, bits_iter},
};

use super::Layers;

#[derive(Error, Debug)]
#[error("Layers can't be combined!")]
pub struct CombineError;

///Utility for combining sub-circuits into a single circuit
/// DOES NOT WORK FOR GATE MLE
pub fn combine_layers<F: FieldExt, Tr: Transcript<F>>(
    mut layers: Vec<Layers<F, Tr>>,
    mut output_layers: Vec<Vec<MleEnum<F>>>,
) -> Result<(Layers<F, Tr>, Vec<MleEnum<F>>), CombineError> {
    let layer_count = layers.iter().map(|layers| layers.0.len()).max().unwrap();

    let interpolated_layers = (0..layer_count).map(|layer_idx| {
        layers
            .iter()
            .filter_map(|layers| layers.0.get(layer_idx))
            .collect_vec()
    });

    //The variants of the layer to be combined is the inner vec
    let bit_counts: Vec<Vec<Vec<MleIndex<F>>>> = interpolated_layers
        .clone()
        .map(|layers| {
            // bits_iter(log2(layers.len()) as usize).collect_vec()
            let layer_sizes = layers.iter().map(|layer| layer.layer_size());
            let total_size = log2(layer_sizes.clone().map(|size| 1 << size).sum()) as usize;

            let extra_bits = layer_sizes
                .clone()
                .map(|size| total_size - size)
                .collect_vec();
            let max_extra_bits = extra_bits.iter().max().unwrap();
            let sorted_indices = argsort(&layer_sizes.collect_vec(), true);
            let mut bit_indices = bits_iter::<F>(*max_extra_bits);
            //Go through the list of layers from largest to smallest
            //When a layer is added it comsumes a possible permutation of bits from the iterator
            //If the layer is larger than the smallest layer, then it will consume all the permutations of the bits it's using in it's sumcheck
            sorted_indices
                .into_iter()
                .map(|index| {
                    let bits = bit_indices.next().ok_or(CombineError).unwrap();
                    if bits.len() != extra_bits[index] {
                        let diff = bits.len() - extra_bits[index];
                        for _ in 0..((1 << diff) - 1) {
                            let _ = bit_indices.next();
                        }
                    }
                    (index, bits[0..extra_bits[index]].to_vec())
                })
                //resort them in thier original order so that the zip later works
                .sorted_by(|(index_first, _), (index_second, _)| index_first.cmp(index_second))
                .map(|(_, bits)| bits)
                .collect_vec()
        })
        .filter(|item: &Vec<Vec<MleIndex<F>>>| item.len() > 1)
        .collect_vec();

    //The layers of the circuit are the inner vec
    let layer_bits = (0..layers.len())
        .map(|index| {
            bit_counts
                .iter()
                .map(|bit_counts| bit_counts[index].clone())
                .collect_vec()
        })
        .collect_vec();

    layers
        .iter_mut()
        .zip(output_layers.iter_mut())
        .zip(layer_bits)
        .map(|((layers, output_layers), new_bits)| {
            for (layer_idx, new_bits) in new_bits.into_iter().enumerate() {
                if let Some(&effected_layer) = layers.0.get(layer_idx).map(|layer| layer.id()) {
                    add_bits_to_layer_refs(
                        &mut layers.0[layer_idx..],
                        output_layers,
                        new_bits,
                        effected_layer,
                    )?;
                }
            }
            Ok(())
        })
        .try_collect()?;

    let layers: Vec<LayerEnum<F, Tr>> = (0..layer_count)
        .map(|layer_idx| {
            layers
                .iter()
                .filter_map(|layers| layers.0.get(layer_idx).cloned())
                .collect_vec()
        })
        .map(|layers| {
            // let new_bits = log2(layers.len()) as usize;
            let layer_id = *layers[0].id();

            // let combine_expressions = |exprs: Vec<ExpressionStandard<F>>| {
            //     exprs.chunks(2).map(|exprs| {
            //         exprs.get(1).cloned().unwrap_or(ExpressionStandard::Constant(F::zero())).concat(exprs[0].clone())
            //     }).collect_vec()
            // };
            let expressions = layers
                .into_iter()
                .map(|layer| match layer {
                    LayerEnum::Gkr(layer) => Ok(layer.expression),
                    LayerEnum::EmptyLayer(layer) => Ok(layer.expr),
                    _ => Err(CombineError),
                })
                .try_collect()?;

            let expression = combine_expressions(expressions);

            Ok(GKRLayer::new_raw(layer_id, expression).get_enum())
        })
        .try_collect()?;

    Ok((
        Layers(layers),
        output_layers.into_iter().flatten().collect(),
    ))
}

fn add_bits_to_layer_refs<F: FieldExt, Tr: Transcript<F>>(
    layers: &mut [LayerEnum<F, Tr>],
    output_layers: &mut Vec<MleEnum<F>>,
    new_bits: Vec<MleIndex<F>>,
    effected_layer: LayerId,
) -> Result<(), CombineError> {
    for layer in layers {
        let expression = match layer {
            LayerEnum::Gkr(layer) => Ok(&mut layer.expression),
            LayerEnum::EmptyLayer(layer) => Ok(&mut layer.expr),
            _ => Err(CombineError),
        }?;

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
    for mle in output_layers {
        match mle {
            MleEnum::Dense(mle) => {
                if mle.layer_id == effected_layer {
                    mle.mle_indices = new_bits
                        .iter()
                        .chain(mle.mle_indices.iter())
                        .cloned()
                        .collect();
                }
            }
            MleEnum::Zero(mle) => {
                if mle.layer_id == effected_layer {
                    mle.mle_indices = new_bits
                        .iter()
                        .chain(mle.mle_indices.iter())
                        .cloned()
                        .collect();
                }
            }
        }
    }
    Ok(())
}

fn combine_expressions<F: FieldExt>(
    mut exprs: Vec<ExpressionStandard<F>>,
) -> ExpressionStandard<F> {
    loop {
        if exprs.len() == 1 {
            break exprs.remove(0);
        }

        exprs.sort_by(|first, second| {
            first
                .get_expression_size(0)
                .cmp(&second.get_expression_size(0))
        });

        let first = exprs.remove(0);
        let second = exprs.remove(0);

        let diff = second.get_expression_size(0) - first.get_expression_size(0);

        let expr = if diff == 0 {
            second.concat_expr(first)
        } else {
            let first = add_padding(first, diff);

            first.concat_expr(second)
        };

        exprs.push(expr);
    }
}

///Function that adds padding to a layer with a selector, left aligned, pads with zero
///
/// Basically turns V(b_1) = \[1, 2\] to V(b_1, b_2) = \[1, 2, 0, 0\] but with expressions
fn add_padding<F: FieldExt>(
    mut expr: ExpressionStandard<F>,
    num_padding: usize,
) -> ExpressionStandard<F> {
    for _ in 0..num_padding {
        expr = ExpressionStandard::Constant(F::zero()).concat_expr(expr);
    }
    expr
}
