use std::marker::PhantomData;

use ark_std::log2;
use itertools::{Itertools, repeat_n};
use thiserror::Error;

use crate::{FieldExt, mle::{MleIndex, dense::DenseMleRef, MleRef}, expression::ExpressionStandard};

use super::{LayerBuilder, LayerId};

#[derive(Error, Debug, Clone)]
#[error("Expressions that are being combined do not have the same shape")]
///An error for when combining expressions
pub struct CombineExpressionError();

pub struct BatchedLayer<F: FieldExt, A: LayerBuilder<F>> {
    layers: Vec<A>,
    _marker: PhantomData<F>
}

fn combine_expressions<F: FieldExt>(exprs: Vec<ExpressionStandard<F>>) -> Result<ExpressionStandard<F>, CombineExpressionError> {
    let new_bits = log2(exprs.len());

    combine_expressions_helper(exprs, new_bits as usize)
}

fn combine_expressions_helper<F: FieldExt>(exprs: Vec<ExpressionStandard<F>>, new_bits: usize) -> Result<ExpressionStandard<F>, CombineExpressionError> {
    match &exprs[0] {
        ExpressionStandard::Selector(index, _, _) => {
            let index = index.clone();
            let out: Vec<(ExpressionStandard<F>, ExpressionStandard<F>)> = exprs.into_iter().map(|expr| {
                if let ExpressionStandard::Selector(_, first, second) = expr {
                    Ok((*first, *second))
                } else {
                    Err(CombineExpressionError())
                }
            }).try_collect()?;

            let (first, second): (Vec<_>, Vec<_>) = out.into_iter().unzip();

            Ok(ExpressionStandard::Selector(index, Box::new(combine_expressions_helper(first, new_bits)?), Box::new(combine_expressions_helper(second, new_bits)?)))
        },
        ExpressionStandard::Mle(_) => {
            let new_indices = vec![MleIndex::Iterated; new_bits];
            let mles: Vec<DenseMleRef<F>> = exprs.into_iter().map(|expr| {
                if let ExpressionStandard::Mle(mut mle) = expr {
                    mle.push_mle_indices(&new_indices);
                    Ok(mle)
                } else {
                    Err(CombineExpressionError())
                }
            }).try_collect()?;

            todo!()
        },
        ExpressionStandard::Sum(_, _) => todo!(),
        ExpressionStandard::Product(_) => todo!(),
        ExpressionStandard::Scaled(_, _) => todo!(),
        ExpressionStandard::Constant(_) | ExpressionStandard::Negated(_) => Ok(exprs[0].clone()),
    }
}

impl<F: FieldExt, A: LayerBuilder<F>> LayerBuilder<F> for BatchedLayer<F, A> {
    type Successor = Vec<A::Successor>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        self.layers.iter().map(|layer| layer.build_expression()).reduce(|acc, item| {
            todo!()
        }).unwrap_or(ExpressionStandard::Constant(F::zero()))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor  {
        self.layers.iter().map(|layer| layer.next_layer(id.clone(), prefix_bits.clone())).collect()
    }
}