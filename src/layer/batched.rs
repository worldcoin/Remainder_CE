use std::marker::PhantomData;

use ark_std::log2;
use itertools::{repeat_n, Itertools};
use thiserror::Error;

use crate::{
    expression::ExpressionStandard,
    mle::{dense::DenseMleRef, MleIndex, MleRef},
    FieldExt,
};

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
            let coeff = coeff.clone();
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

fn combine_mles<F: FieldExt>(mles: Vec<DenseMleRef<F>>, new_bits: usize) -> DenseMleRef<F> {
    let old_indices = mles[0].mle_indices();
    let old_num_vars = mles[0].num_vars();
    let layer_id = Some(mles[0].get_layer_id());

    let out = (0..mles[0].bookkeeping_table.len())
        .flat_map(|index| {
            mles.iter()
                .map(|mle| mle.bookkeeping_table()[index])
                .collect_vec()
        })
        .collect_vec();

    DenseMleRef {
        bookkeeping_table: out,
        mle_indices: repeat_n(MleIndex::Iterated, new_bits)
            .chain(old_indices.into_iter().cloned())
            .collect_vec(),
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
        let num_bits = log2(self.layers.len()) as usize;
        let bits = std::iter::successors(
            Some(vec![MleIndex::Fixed(false); num_bits]),
            |prev| {
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
                    Some(prev.into_iter().chain(repeat_n(MleIndex::Fixed(false), removed_bits)).collect_vec())
                }
            },
        );

        self.layers
            .iter()
            .zip(bits)
            .map(|(layer, bits)| {
                layer.next_layer(
                    id.clone(),
                    Some(
                        bits.into_iter()
                            .chain(prefix_bits.clone().into_iter().flatten())
                            .collect_vec(),
                    ),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use itertools::Itertools;

    use crate::{layer::{from_mle, LayerBuilder, LayerId}, mle::{Mle, dense::DenseMle}, sumcheck::tests::{get_dummy_claim, dummy_sumcheck, verify_sumcheck_messages}};

    use super::BatchedLayer;

    
    #[test]
    fn test_batched_layer() {
        let mut rng = test_rng();
        let expression_builder = |(mle1, mle2): &(DenseMle<Fr, Fr>, DenseMle<Fr, Fr>)| mle1.mle_ref().expression() + mle2.mle_ref().expression();
        let layer_builder = |(mle1, mle2): &(DenseMle<Fr, Fr>, DenseMle<Fr, Fr>), layer_id, prefix_bits| {
            let mut new_mle = mle1
                .clone()
                .into_iter()
                .zip(mle2.clone().into_iter())
                .map(|(first, second)| first + second)
                .collect::<DenseMle<Fr, Fr>>();
            new_mle.define_layer_id(layer_id);
            new_mle.add_prefix_bits(prefix_bits);
            new_mle
        };        
        let output: (DenseMle<Fr, Fr>, DenseMle<Fr, Fr>) = {
            let mut first = DenseMle::new(vec![Fr::from(3), Fr::from(7), Fr::from(8), Fr::from(10)]);
            first.define_layer_id(LayerId::Input);
            let mut second = DenseMle::new(vec![Fr::from(4), Fr::from(11), Fr::from(5), Fr::from(6)]);
            second.define_layer_id(LayerId::Input);
            (first, second)
        };
        let builder = from_mle(
            output,
            expression_builder,
            layer_builder,
        );

        let output_2: (DenseMle<Fr, Fr>, DenseMle<Fr, Fr>) = {
            let mut first = DenseMle::new(vec![Fr::from(2), Fr::from(0), Fr::from(4), Fr::from(9)]);
            first.define_layer_id(LayerId::Input);
            let mut second = DenseMle::new(vec![Fr::from(5), Fr::from(8), Fr::from(5), Fr::from(6)]);
            second.define_layer_id(LayerId::Input);
            (first, second)
        };

        let builder_2 = from_mle(
            output_2,
            expression_builder,
            layer_builder,
        );

        let layer_builder = BatchedLayer::new(vec![builder, builder_2]);

        let mut expr = layer_builder.build_expression();

        let output = layer_builder.next_layer(LayerId::Layer(0), None);

        let output_real = output[0].clone().into_iter().interleave(output[1].clone().into_iter()).collect::<DenseMle<Fr, Fr>>();

        let layer_claims = get_dummy_claim(output_real.mle_ref(), &mut rng, None);

        let sumcheck = dummy_sumcheck(&mut expr, &mut rng, layer_claims.clone());
        verify_sumcheck_messages(sumcheck, expr, layer_claims, &mut rng).unwrap();
    }
}