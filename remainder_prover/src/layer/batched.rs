use std::marker::PhantomData;
use ark_std::log2;
use itertools::{repeat_n, Itertools};
use thiserror::Error;

use crate::{
    expression::ExpressionStandard,
    mle::{dense::{DenseMleRef, DenseMle}, zero::ZeroMleRef, MleIndex, MleRef, MleAble, Mle},
};
use remainder_shared_types::FieldExt;

use super::{LayerBuilder, LayerId};

#[derive(Error, Debug, Clone)]
#[error("Expressions that are being combined do not have the same shape")]
///An error for when combining expressions
pub struct CombineExpressionError();

pub struct BatchedLayer<F: FieldExt, A: LayerBuilder<F>> {
    layers: Vec<A>,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, A: LayerBuilder<F>> BatchedLayer<F, A> {
    pub fn new(layers: Vec<A>) -> Self {
        Self {
            layers,
            _marker: PhantomData,
        }
    }
}

///Helper function for "unbatching" when required by the output layer
pub fn combine_zero_mle_ref<F: FieldExt>(mle_refs: Vec<ZeroMleRef<F>>) -> ZeroMleRef<F> {
    let new_bits = 0;
    let num_vars = mle_refs[0].mle_indices().len();
    let layer_id = mle_refs[0].get_layer_id();
    ZeroMleRef::new(num_vars + new_bits, None, layer_id)
}

///Helper function for "unbatching" when required by circuit design
pub fn unbatch_mles<F: FieldExt>(mles: Vec<DenseMle<F, F>>) -> DenseMle<F, F> {
    let old_layer_id = mles[0].layer_id;
    let new_bits = log2(mles.len()) as usize;
    let old_prefix_bits = mles[0].prefix_bits.clone().map(|old_prefix_bits| old_prefix_bits[0..old_prefix_bits.len() - new_bits].to_vec());
    DenseMle::new_from_raw(combine_mles(mles.into_iter().map(|mle| mle.mle_ref()).collect_vec(), new_bits).bookkeeping_table, old_layer_id, old_prefix_bits)
}

/// convert a flattened batch mle to a vector of mles
pub fn unflatten_mle<F: FieldExt>(flattened_mle: DenseMle<F, F>, num_copy_bits: usize) -> Vec<DenseMle<F, F>> {
    let num_copies = 1 << num_copy_bits;
    let individual_mle_len = 1 << (flattened_mle.num_iterated_vars() - num_copy_bits);
    
    (0..num_copies).map(
        |idx| {
            let zero = &F::zero();
            let copy_idx = idx;
            let individual_mle_table = (0..individual_mle_len).map(
                |mle_idx| {
                    let flat_mle_ref = flattened_mle.mle_ref();
                    let val = flat_mle_ref.bookkeeping_table.get(copy_idx + (mle_idx * num_copies)).unwrap_or(zero);
                    *val
                }
            ).collect_vec();
            let individual_mle: DenseMle<F, F> = DenseMle::new_from_raw(individual_mle_table, flattened_mle.layer_id, Some(repeat_n(MleIndex::Iterated, num_copy_bits).collect_vec()));
            individual_mle
        }
    ).collect_vec()
}

///Helper function for batchedlayer that takes in m expressions of size n, and
///turns it into a single expression o size n*m
fn combine_expressions<F: FieldExt>(
    exprs: Vec<ExpressionStandard<F>>,
) -> Result<ExpressionStandard<F>, CombineExpressionError> {
    let new_bits = log2(exprs.len());

    combine_expressions_helper(exprs, new_bits as usize)
}

fn combine_expressions_helper<F: FieldExt>(
    exprs: Vec<ExpressionStandard<F>>,
    new_bits: usize,
) -> Result<ExpressionStandard<F>, CombineExpressionError> {
    //Check if all expressions have the same structure, and if they do, combine
    //their parts. 
    //Combination is done through either recursion or simple methods, except for
    //Mle and Products; which use a helper function `combine_mles`
    match &exprs[0] {
        ExpressionStandard::Selector(index, _, _) => {
            let index = index.clone();
            let out: Vec<(ExpressionStandard<F>, ExpressionStandard<F>)> = exprs
                .into_iter()
                .map(|expr| {
                    if let ExpressionStandard::Selector(_, first, second) = expr {
                        Ok((*first, *second))
                    } else {
                        Err(CombineExpressionError())
                    }
                })
                .try_collect()?;

            let (first, second): (Vec<_>, Vec<_>) = out.into_iter().unzip();

            Ok(ExpressionStandard::Selector(
                index,
                Box::new(combine_expressions_helper(first, new_bits)?),
                Box::new(combine_expressions_helper(second, new_bits)?),
            ))
        }
        ExpressionStandard::Mle(_) => {
            let mles: Vec<DenseMleRef<F>> = exprs
                .into_iter()
                .map(|expr| {
                    if let ExpressionStandard::Mle(mle) = expr {
                        Ok(mle)
                    } else {
                        Err(CombineExpressionError())
                    }
                })
                .try_collect()?;

            Ok(ExpressionStandard::Mle(combine_mles(mles, new_bits)))
        }
        ExpressionStandard::Sum(_, _) => {
            let out: Vec<(ExpressionStandard<F>, ExpressionStandard<F>)> = exprs
                .into_iter()
                .map(|expr| {
                    if let ExpressionStandard::Sum(first, second) = expr {
                        Ok((*first, *second))
                    } else {
                        Err(CombineExpressionError())
                    }
                })
                .try_collect()?;

            let (first, second): (Vec<_>, Vec<_>) = out.into_iter().unzip();

            Ok(ExpressionStandard::Sum(
                Box::new(combine_expressions_helper(first, new_bits)?),
                Box::new(combine_expressions_helper(second, new_bits)?),
            ))
        }
        ExpressionStandard::Product(_) => {
            let mles: Vec<Vec<DenseMleRef<F>>> = exprs
                .into_iter()
                .map(|expr| {
                    if let ExpressionStandard::Product(mles) = expr {
                        Ok(mles)
                    } else {
                        Err(CombineExpressionError())
                    }
                })
                .try_collect()?;

            let out = (0..mles[0].len())
                .map(|index| mles.iter().map(|mle| mle[index].clone()).collect_vec())
                .collect_vec();

            Ok(ExpressionStandard::Product(
                out.into_iter()
                    .map(|mles| combine_mles(mles, new_bits))
                    .collect_vec(),
            ))
        }
        ExpressionStandard::Scaled(_, coeff) => {
            let coeff = *coeff;
            let out: Vec<_> = exprs
                .into_iter()
                .map(|expr| {
                    if let ExpressionStandard::Scaled(expr, _) = expr {
                        Ok(*expr)
                    } else {
                        Err(CombineExpressionError())
                    }
                })
                .try_collect()?;

            Ok(ExpressionStandard::Scaled(
                Box::new(combine_expressions_helper(out, new_bits)?),
                coeff,
            ))
        }
        ExpressionStandard::Negated(_) => {
            let out: Vec<_> = exprs
                .into_iter()
                .map(|expr| {
                    if let ExpressionStandard::Negated(expr) = expr {
                        Ok(*expr)
                    } else {
                        Err(CombineExpressionError())
                    }
                })
                .try_collect()?;

            Ok(ExpressionStandard::Negated(Box::new(
                combine_expressions_helper(out, new_bits)?,
            )))
        }
        ExpressionStandard::Constant(_) => Ok(exprs[0].clone()),
    }
}

/// for batching. Taking m DenseMleRefs of size n and turning them into a single
/// DenseMleRef of size n*m
pub fn combine_mles<F: FieldExt>(mles: Vec<DenseMleRef<F>>, new_bits: usize) -> DenseMleRef<F> {
    let old_indices = mles[0].mle_indices();
    let old_num_vars = mles[0].num_vars();
    let layer_id = mles[0].get_layer_id();

    // --- TODO!(ryancao): SUPER hacky fix for the random packing constants ---
    // --- Basically if all the MLEs are exactly the same, we don't combine at all ---
    if matches!(layer_id, LayerId::Input(_)) && old_num_vars == 0 {
        let all_same = (0..mles[0].bookkeeping_table().len()).all(|idx| mles.iter().skip(1).all(|mle| (mles[0].bookkeeping_table()[idx] == mle.bookkeeping_table()[idx])));
        if all_same {
            return mles[0].clone();
        }
    }

    let out = (0..mles[0].bookkeeping_table.len())
        .flat_map(|index| {
            mles.iter()
                .map(|mle| mle.bookkeeping_table()[index])
                .collect_vec()
        })
        .collect_vec();

    DenseMleRef {
        bookkeeping_table: out,
        mle_indices: old_indices.to_vec(),
        num_vars: old_num_vars + new_bits,
        layer_id,
        indexed: false,
    }
}

impl<F: FieldExt, A: LayerBuilder<F>> LayerBuilder<F> for BatchedLayer<F, A> {
    type Successor = Vec<A::Successor>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        let exprs = self
            .layers
            .iter()
            .map(|layer| layer.build_expression())
            .collect_vec();

        
        combine_expressions(exprs)
            .expect("Expressions fed into BatchedLayer don't have the same structure!")
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let new_bits = log2(self.layers.len()) as usize;

        //Mles yielded by this BatchedLayer have the batched bits taken into
        //account so that they are ordered correctly compared to all other bits,
        //even though there is some semantic incorrectness to having the batch
        //bits be part of the individual mles

        self.layers
            .iter()
            // .zip(bits)
            .map(|layer| {
                layer.next_layer(
                    id,
                    Some(
                        prefix_bits
                            .clone()
                            .into_iter()
                            .flatten()
                            .chain(repeat_n(MleIndex::Iterated, new_bits))
                            .collect_vec(),
                    ),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ark_std::test_rng;
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use itertools::Itertools;

    use crate::{
        expression::ExpressionStandard,
        layer::{from_mle, LayerBuilder, LayerId},
        mle::{dense::DenseMle, Mle, MleIndex},
        sumcheck::tests::{dummy_sumcheck, get_dummy_claim, verify_sumcheck_messages},
    };

    use super::BatchedLayer;

    #[test]
    fn test_batched_layer() {
        let mut rng = test_rng();
        let expression_builder =
            |(mle1, mle2): &(DenseMle<Fr, Fr>, DenseMle<Fr, Fr>)| -> ExpressionStandard<Fr> {
                mle1.mle_ref().expression() + mle2.mle_ref().expression()
            };
        let layer_builder = |(mle1, mle2): &(DenseMle<Fr, Fr>, DenseMle<Fr, Fr>),
                             layer_id,
                             prefix_bits|
         -> DenseMle<Fr, Fr> {
            DenseMle::new_from_iter(
                mle1.clone()
                    .into_iter()
                    .zip(mle2.clone().into_iter())
                    .map(|(first, second)| first + second),
                layer_id,
                prefix_bits,
            )
        };
        let output: (DenseMle<Fr, Fr>, DenseMle<Fr, Fr>) = {
            let mut first = DenseMle::new_from_raw(
                vec![Fr::from(3), Fr::from(7), Fr::from(8), Fr::from(10)],
                LayerId::Input(0),
                Some(vec![MleIndex::Iterated]),
            );
            let mut second = DenseMle::new_from_raw(
                vec![Fr::from(4), Fr::from(11), Fr::from(5), Fr::from(6)],
                LayerId::Input(0),
                Some(vec![MleIndex::Iterated]),
            );
            (first, second)
        };
        let builder = from_mle(output, expression_builder, layer_builder);

        let output_2: (DenseMle<Fr, Fr>, DenseMle<Fr, Fr>) = {
            let mut first = DenseMle::new_from_raw(
                vec![Fr::from(2), Fr::from(0), Fr::from(4), Fr::from(9)],
                LayerId::Input(0),
                Some(vec![MleIndex::Iterated]),
            );
            let mut second = DenseMle::new_from_raw(
                vec![Fr::from(5), Fr::from(8), Fr::from(5), Fr::from(6)],
                LayerId::Input(0),
                Some(vec![MleIndex::Iterated]),
            );
            (first, second)
        };

        let builder_2 = from_mle(output_2, expression_builder, layer_builder);

        let layer_builder = BatchedLayer::new(vec![builder, builder_2]);

        let mut expr = layer_builder.build_expression();

        let output = layer_builder.next_layer(LayerId::Layer(0), None);

        let output_real = DenseMle::new_from_iter(
            output[0]
                .clone()
                .into_iter()
                .interleave(output[1].clone().into_iter()),
            LayerId::Layer(0),
            None,
        );

        let layer_claims = get_dummy_claim(output_real.mle_ref(), &mut rng, None);

        let sumcheck = dummy_sumcheck(&mut expr, &mut rng, layer_claims.clone());
        verify_sumcheck_messages(sumcheck, expr, layer_claims, &mut rng).unwrap();
    }
}
