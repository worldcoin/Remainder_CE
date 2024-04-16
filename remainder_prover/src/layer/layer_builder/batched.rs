//! A LayerBuilder combinator that takes in many LayerBuilders and combines them
//! into a batched version, that proves a constraint over all of them at once.

use ark_std::log2;
use itertools::{repeat_n, Itertools};
use std::marker::PhantomData;
use thiserror::Error;

use crate::{
    expression::{
        generic_expr::{Expression, ExpressionNode, ExpressionType},
        prover_expr::ProverExpr,
    },
    mle::{
        dense::{DenseMle, DenseMleRef},
        evals::{Evaluations, MultilinearExtension},
        zero::ZeroMleRef,
        Mle, MleIndex, MleRef,
    },
};
use remainder_shared_types::FieldExt;

use super::{LayerBuilder, LayerId};

#[derive(Error, Debug, Clone)]
#[error("Expressions that are being combined do not have the same shape")]
///An error for when combining expressions
pub struct CombineExpressionError();

/// A LayerBuilder combinator that takes in many LayerBuilders and combines them
/// into a batched version, that proves a constraint over all of them at once.
pub struct BatchedLayer<F: FieldExt, A: LayerBuilder<F>> {
    layers: Vec<A>,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, A: LayerBuilder<F>> BatchedLayer<F, A> {
    /// Creates a new `BatchedLayer` from a homogenous set of
    /// sub `LayerBuilder`s
    pub fn new(layers: Vec<A>) -> Self {
        Self {
            layers,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt, A: LayerBuilder<F>> LayerBuilder<F> for BatchedLayer<F, A> {
    type Successor = Vec<A::Successor>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
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
    let old_prefix_bits = mles[0]
        .prefix_bits
        .clone()
        .map(|old_prefix_bits| old_prefix_bits[0..old_prefix_bits.len() - new_bits].to_vec());
    DenseMle::new_from_raw(
        combine_mles(
            mles.into_iter().map(|mle| mle.mle_ref()).collect_vec(),
            new_bits,
        )
        .current_mle
        .get_evals_vector()
        .clone(),
        old_layer_id,
        old_prefix_bits,
    )
}

/// convert a flattened batch mle to a vector of mles
pub fn unflatten_mle<F: FieldExt>(
    flattened_mle: DenseMle<F, F>,
    num_dataparallel_bits: usize,
) -> Vec<DenseMle<F, F>> {
    let num_copies = 1 << num_dataparallel_bits;
    let individual_mle_len = 1 << (flattened_mle.num_iterated_vars() - num_dataparallel_bits);

    (0..num_copies)
        .map(|idx| {
            let _zero = &F::zero();
            let copy_idx = idx;
            let individual_mle_table = (0..individual_mle_len)
                .map(|mle_idx| {
                    let flat_mle_ref = flattened_mle.mle_ref();

                    flat_mle_ref.current_mle.f[copy_idx + (mle_idx * num_copies)]
                })
                .collect_vec();
            let individual_mle: DenseMle<F, F> = DenseMle::new_from_raw(
                individual_mle_table,
                flattened_mle.layer_id,
                Some(
                    flattened_mle
                        .get_prefix_bits()
                        .unwrap()
                        .into_iter()
                        .chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits))
                        .collect_vec(),
                ),
            );
            individual_mle
        })
        .collect_vec()
}

///Helper function for batchedlayer that takes in m expressions of size n, and
///turns it into a single expression o size n*m
fn combine_expressions<F: FieldExt>(
    exprs: Vec<Expression<F, ProverExpr>>,
) -> Result<Expression<F, ProverExpr>, CombineExpressionError> {
    let new_bits = log2(exprs.len());

    let mut new_mle_vec: Vec<Option<DenseMleRef<F>>> = vec![None; exprs[0].num_mle_ref()];
    let (expression_nodes, mle_vecs): (Vec<_>, Vec<_>) =
        exprs.into_iter().map(|expr| expr.deconstruct()).unzip();

    let out_expression_node = expression_nodes[0].clone();

    combine_expressions_helper(
        expression_nodes,
        &mle_vecs,
        &mut new_mle_vec,
        new_bits as usize,
    );

    let out_mle_vec = new_mle_vec
        .into_iter()
        .map(|mle| mle.unwrap())
        .collect_vec();

    Ok(Expression::new(out_expression_node, out_mle_vec))
}

fn combine_expressions_helper<F: FieldExt>(
    expression_nodes: Vec<ExpressionNode<F, ProverExpr>>,
    mle_vecs: &Vec<<ProverExpr as ExpressionType<F>>::MleVec>,
    new_mle_vec: &mut Vec<Option<DenseMleRef<F>>>,
    new_bits: usize,
) {
    //Check if all expressions have the same structure, and if they do, combine
    //their parts.
    //Combination is done through either recursion or simple methods, except for
    //Mle and Products; which use a helper function `combine_mles`
    match &expression_nodes[0] {
        ExpressionNode::Selector(_index, _, _) => {
            let out: Vec<(ExpressionNode<F, ProverExpr>, ExpressionNode<F, ProverExpr>)> =
                expression_nodes
                    .into_iter()
                    .map(|expr| {
                        if let ExpressionNode::Selector(_, first, second) = expr {
                            Ok((*first, *second))
                        } else {
                            Err(CombineExpressionError())
                        }
                    })
                    .try_collect()
                    .unwrap();

            let (first, second): (Vec<_>, Vec<_>) = out.into_iter().unzip();

            combine_expressions_helper(first, mle_vecs, new_mle_vec, new_bits);
            combine_expressions_helper(second, mle_vecs, new_mle_vec, new_bits);
        }
        ExpressionNode::Mle(_) => {
            let mut mle_vec_index = 0;
            let mles: Vec<DenseMleRef<F>> = expression_nodes
                .into_iter()
                .enumerate()
                .map(|(idx, expr)| {
                    if let ExpressionNode::Mle(mle_vec_idx) = expr {
                        mle_vec_index = mle_vec_idx.index();
                        let mle_ref = mle_vec_idx.get_mle(&mle_vecs[idx]);
                        Ok(mle_ref.clone())
                    } else {
                        Err(CombineExpressionError())
                    }
                })
                .try_collect()
                .unwrap();

            let new_mle = combine_mles(mles, new_bits);
            new_mle_vec[mle_vec_index] = Some(new_mle);
        }
        ExpressionNode::Sum(_, _) => {
            let out: Vec<(ExpressionNode<F, ProverExpr>, ExpressionNode<F, ProverExpr>)> =
                expression_nodes
                    .into_iter()
                    .map(|expr| {
                        if let ExpressionNode::Sum(first, second) = expr {
                            Ok((*first, *second))
                        } else {
                            Err(CombineExpressionError())
                        }
                    })
                    .try_collect()
                    .unwrap();

            let (first, second): (Vec<_>, Vec<_>) = out.into_iter().unzip();

            combine_expressions_helper(first, mle_vecs, new_mle_vec, new_bits);
            combine_expressions_helper(second, mle_vecs, new_mle_vec, new_bits);
        }
        ExpressionNode::Product(_) => {
            let mut mle_vec_index = vec![];
            let mles: Vec<Vec<DenseMleRef<F>>> = expression_nodes
                .into_iter()
                .enumerate()
                .map(|(idx, expr)| {
                    if let ExpressionNode::Product(mle_vec_indices) = expr {
                        if mle_vec_index.is_empty() {
                            mle_vec_index = mle_vec_indices
                                .iter()
                                .map(|mle_vec_index| mle_vec_index.index())
                                .collect_vec();
                        }

                        // get the mle_refs
                        let mle_refs = mle_vec_indices
                            .into_iter()
                            .map(|mle_vec_index| mle_vec_index.get_mle(&mle_vecs[idx]).clone())
                            .collect_vec();

                        Ok(mle_refs)
                    } else {
                        Err(CombineExpressionError())
                    }
                })
                .try_collect()
                .unwrap();

            let out = (0..mles[0].len())
                .map(|index| mles.iter().map(|mle| mle[index].clone()).collect_vec())
                .collect_vec();

            let out = out
                .into_iter()
                .map(|mles| combine_mles(mles, new_bits))
                .collect_vec();

            mle_vec_index.into_iter().zip(out).for_each(|(idx, mle)| {
                new_mle_vec[idx] = Some(mle);
            });
        }
        ExpressionNode::Scaled(_, _coeff) => {
            let out: Vec<_> = expression_nodes
                .into_iter()
                .map(|expr| {
                    if let ExpressionNode::Scaled(expr, _) = expr {
                        Ok(*expr)
                    } else {
                        Err(CombineExpressionError())
                    }
                })
                .try_collect()
                .unwrap();

            combine_expressions_helper(out, mle_vecs, new_mle_vec, new_bits);
        }
        ExpressionNode::Negated(_) => {
            let out: Vec<_> = expression_nodes
                .into_iter()
                .map(|expr| {
                    if let ExpressionNode::Negated(expr) = expr {
                        Ok(*expr)
                    } else {
                        Err(CombineExpressionError())
                    }
                })
                .try_collect()
                .unwrap();

            combine_expressions_helper(out, mle_vecs, new_mle_vec, new_bits);
        }
        ExpressionNode::Constant(_) => (),
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
    if matches!(layer_id, LayerId::RandomInput(_)) && old_num_vars == 0 {
        let all_same = (0..mles[0].bookkeeping_table().len()).all(|idx| {
            mles.iter()
                .skip(1)
                .all(|mle| (mles[0].bookkeeping_table()[idx] == mle.bookkeeping_table()[idx]))
        });
        if all_same {
            return mles[0].clone();
        }
    }

    let out = (0..mles[0].current_mle.get_evals_vector().len())
        .flat_map(|index| {
            mles.iter()
                .map(|mle| mle.bookkeeping_table()[index])
                .collect_vec()
        })
        .collect_vec();

    let mle = MultilinearExtension::new(Evaluations::new(old_num_vars + new_bits, out));

    DenseMleRef {
        current_mle: mle.clone(),
        original_mle: mle,
        mle_indices: old_indices.to_vec(),
        original_mle_indices: old_indices.to_vec(),
        layer_id,
        indexed: false,
    }
}

#[cfg(test)]
mod tests {
    use ark_std::test_rng;
    use itertools::Itertools;
    use remainder_shared_types::Fr;

    use crate::{
        expression::{generic_expr::Expression, prover_expr::ProverExpr},
        layer::{
            layer_builder::{from_mle, LayerBuilder},
            LayerId,
        },
        mle::{dense::DenseMle, MleIndex},
        sumcheck::tests::{dummy_sumcheck, get_dummy_claim, verify_sumcheck_messages},
    };

    use super::BatchedLayer;

    #[test]
    fn test_batched_layer() {
        let mut rng = test_rng();
        let expression_builder =
            |(mle1, mle2): &(DenseMle<Fr, Fr>, DenseMle<Fr, Fr>)| -> Expression<Fr, ProverExpr> {
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
            let first = DenseMle::new_from_raw(
                vec![Fr::from(3), Fr::from(7), Fr::from(8), Fr::from(10)],
                LayerId::Input(0),
                Some(vec![MleIndex::Iterated]),
            );
            let second = DenseMle::new_from_raw(
                vec![Fr::from(4), Fr::from(11), Fr::from(5), Fr::from(6)],
                LayerId::Input(0),
                Some(vec![MleIndex::Iterated]),
            );
            (first, second)
        };
        let builder = from_mle(output, expression_builder, layer_builder);

        let output_2: (DenseMle<Fr, Fr>, DenseMle<Fr, Fr>) = {
            let first = DenseMle::new_from_raw(
                vec![Fr::from(2), Fr::from(0), Fr::from(4), Fr::from(9)],
                LayerId::Input(0),
                Some(vec![MleIndex::Iterated]),
            );
            let second = DenseMle::new_from_raw(
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
