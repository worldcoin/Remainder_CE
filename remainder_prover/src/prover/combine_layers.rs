//!Utilities for combining sub-circuits

use std::cmp::min;

use ark_std::log2;
use itertools::Itertools;
use remainder_shared_types::{transcript::Transcript, FieldExt};
use thiserror::Error;

use crate::{
    expression::{generic_expr::Expression, prover_expr::ProverExpression},
    layer::{empty_layer::EmptyLayer, layer_enum::LayerEnum, GKRLayer, Layer, LayerId},
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
    //We're going to add multiple selectors to merge the sub-circuits, and then
    //the future layers need to take those selectors into account in thier claims.
    let layer_count = layers.iter().map(|layers| layers.0.len()).max().unwrap();
    let subcircuit_count = layers.len();

    // --- Grabbing the "columns" of layers (with associated circuit index) ---
    let interpolated_layers = (0..layer_count).map(|layer_idx| {
        layers
            .iter()
            .enumerate()
            .filter_map(|(subcircuit_idx, layers)| {
                layers.0.get(layer_idx).map(|layer| (subcircuit_idx, layer))
            })
            .collect_vec()
    });

    //The variants of the layer to be combined is the inner vec
    //This list is the list of extra bits that need to be added for each layer
    //and each sub-circuit
    let bit_counts: Vec<Vec<Vec<MleIndex<F>>>> = interpolated_layers
        .map(|layers_at_combined_index| {
            // --- Global layer ID for this column ---
            let _layer_id = layers_at_combined_index[0].1.id();
            let layer_sizes = layers_at_combined_index
                .iter()
                .map(|layer| layer.1.layer_size());
            // let layer_sizes_concrete = layer_sizes.clone().collect_vec();
            // dbg!(layer_sizes_concrete);

            // --- Getting the total combined layer size ---
            let total_size = log2(layer_sizes.clone().map(|size| 1 << size).sum()) as usize;

            // --- Diffs between total size and individual layer size (to pad) ---
            let extra_bits = layer_sizes
                .clone()
                .map(|size| total_size - size)
                .collect_vec();
            let max_extra_bits = extra_bits.iter().max().unwrap();
            let sorted_indices = argsort(&layer_sizes.collect_vec(), false);
            let mut bit_indices = bits_iter::<F>(*max_extra_bits);

            let mut sorted_and_padded_bits = vec![vec![]; subcircuit_count];
            //Go through the list of layers from largest to smallest When a
            //layer is added it comsumes a possible permutation of bits from the
            //iterator If the layer is larger than the smallest layer, then it
            //will consume all the permutations of the bits it's using in it's
            //sumcheck
            sorted_indices
                .into_iter()
                .map(|index| {
                    if *max_extra_bits != extra_bits[index] {
                        let diff = max_extra_bits - extra_bits[index];
                        for _ in 0..((1 << diff) - 1) {
                            let _ = bit_indices.next();
                        }
                    }
                    // --- `bits` now starts from the 2^{diff} bitstring ---
                    let bits = bit_indices.next().ok_or(CombineError).unwrap();
                    // --- This return thing goes from 2^{diff}, ..., 2^{max_extra_bits} ---
                    (index, bits[0..extra_bits[index]].to_vec())
                })
                // --- Associate the layers within this column with their bitstrings from above ---
                .for_each(|(index, bits)| {
                    sorted_and_padded_bits[layers_at_combined_index[index].0] = bits;
                });

            sorted_and_padded_bits
        })
        .filter(|item: &Vec<Vec<MleIndex<F>>>| item.len() > 1)
        .collect_vec();

    // dbg!(&bit_counts);

    // --- Iterates the above `sorted_and_padded_bits` in subcircuit-major (row-major) order ---
    let layer_bits = (0..layers.len())
        .map(|index| {
            bit_counts
                .iter()
                .map(|bit_counts| bit_counts.get(index).cloned().unwrap_or_default())
                .collect_vec()
        })
        .collect_vec();

    // --- First zip combines the subcircuit layers with all of its output layers AND all of its subcircuit-major "layer bits" ---
    layers
        .iter_mut()
        .zip(output_layers.iter_mut())
        .zip(layer_bits)
        // --- For each subcircuit... ---
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

    //Combine all the sub-circuits expressions in such a way that it matches the
    //extra bits we calculated
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

            let expressions = layers
                .into_iter()
                .map(|layer| match layer {
                    LayerEnum::Gkr(layer) => Ok(layer.expression),
                    LayerEnum::EmptyLayer(layer) => Ok(layer.expr),
                    _ => Err(CombineError),
                })
                .try_collect()?;

            let expression = combine_expressions(expressions);

            if expression.get_expression_size(0) == 0 {
                Ok(EmptyLayer::new_raw(layer_id, expression).get_enum())
            } else {
                Ok(GKRLayer::new_raw(layer_id, expression).get_enum())
            }
        })
        .try_collect()?;

    Ok((
        Layers(layers),
        output_layers.into_iter().flatten().collect(),
    ))
}

///Add all the extra bits that represent selectors between the sub-circuits to
///the future DenseMleRefs that refer to the modified layer
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

        let mut closure = for<'a> |expr: &'a mut Expression<F, ProverExpression>| -> Result<(), ()> {
            match expr {
                Expression::Mle(mle) => {
                    if mle.layer_id == effected_layer {
                        mle.mle_indices = new_bits
                            .iter()
                            .chain(mle.mle_indices.iter())
                            .cloned()
                            .collect();
                        mle.original_mle_indices = new_bits
                            .iter()
                            .chain(mle.original_mle_indices.iter())
                            .cloned()
                            .collect();
                    }
                    Ok(())
                }
                Expression::Product(mles) => {
                    for mle in mles {
                        if mle.layer_id == effected_layer {
                            mle.mle_indices = new_bits
                                .iter()
                                .chain(mle.mle_indices.iter())
                                .cloned()
                                .collect();
                            mle.original_mle_indices = new_bits
                                .iter()
                                .chain(mle.original_mle_indices.iter())
                                .cloned()
                                .collect();
                        }
                    }
                    Ok(())
                }
                Expression::Constant(_)
                | Expression::Scaled(_, _)
                | Expression::Sum(_, _)
                | Expression::Negated(_)
                | Expression::Selector(_, _, _) => Ok(()),
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
                    mle.original_mle_indices = new_bits
                        .iter()
                        .chain(mle.original_mle_indices.iter())
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
                    mle.original_mle_indices = new_bits
                        .iter()
                        .chain(mle.original_mle_indices.iter())
                        .cloned()
                        .collect();
                }
            }
        }
    }
    Ok(())
}

///Combine expression w/ padding using selectors
fn combine_expressions<F: FieldExt>(
    mut exprs: Vec<Expression<F, ProverExpression>>,
) -> Expression<F, ProverExpression> {
    let _floor_size = exprs
        .iter()
        .map(|expr| expr.get_expression_size(0))
        .min()
        .unwrap();

    exprs.sort_by(|first, second| {
        first
            .get_expression_size(0)
            .cmp(&second.get_expression_size(0))
    });

    let mut exprs = exprs.into_iter().enumerate().collect_vec();

    loop {
        if exprs.len() == 1 {
            break exprs.remove(0).1;
        }

        exprs.sort_by(|first, second| {
            first
                .1
                .get_expression_size(0)
                .cmp(&second.1.get_expression_size(0))
        });

        let (first_index, first) = exprs.remove(0);
        let _first_size = first.get_expression_size(0);
        let (second_index, second) = exprs.remove(0);

        let diff = second.get_expression_size(0) - first.get_expression_size(0);

        let first = add_padding(first, diff);

        let expr = if first_index < second_index {
            second.concat_expr(first)
        } else {
            first.concat_expr(second)
        };
        exprs.insert(0, (min(first_index, second_index), expr));
    }
}

///Function that adds padding to a layer with a selector, left aligned, pads with zero
///
/// Basically turns V(b_1) = \[1, 2\] to V(b_1, b_2) = \[1, 2, 0, 0\] but with expressions
fn add_padding<F: FieldExt>(
    mut expr: Expression<F, ProverExpression>,
    num_padding: usize,
) -> Expression<F, ProverExpression> {
    for _ in 0..num_padding {
        expr = Expression::Constant(F::zero()).concat_expr(expr);
    }
    expr
}
