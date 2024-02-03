use std::{
    cmp::max, fmt::Debug, marker::PhantomData
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use crate::mle::{beta::*, dense::DenseMleRef, MleIndex, MleRef};
use remainder_shared_types::FieldExt;
use super::{expr_errors::ExpressionError, generic_expr::{ExpressionNode, ExpressionType}, verifier_expr::{VerifierExpression}};

/// mid-term solution for deduplication of DenseMleRefs
/// basically a wrapper around usize, which denotes the index
/// of the MleRef in an expression's MleRef list
pub struct MleRefIndex<F>(usize, PhantomData<F>);

impl<F: FieldExt> MleRefIndex<F> {

    /// create a new MleRefIndex
    pub fn new(index: usize) -> Self {
        MleRefIndex(index, PhantomData)
    }

    /// return the actual mle_ref in the vec within the prover expression
    pub fn get_mle<'a>(
        &self,
        mle_ref_vec: &'a Vec<DenseMleRef<F>>
    ) -> &'a DenseMleRef<F> {
        &mle_ref_vec[self.0]
    }

    /// return the actual mle_ref in the vec within the prover expression
    pub fn get_mle_mut<'a>(
        &self,
        mle_ref_vec: &'a mut Vec<DenseMleRef<F>>
    ) -> &'a mut DenseMleRef<F> {
        &mut mle_ref_vec[self.0]
    }
}

/// Prover Expression
/// the leaf nodes of the expression tree are DenseMleRefs
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProverExpression;
impl<F: FieldExt> ExpressionType<F> for ProverExpression {
    type Container = DenseMleRef<F>;
}


/// this is what the prover manipulates to prove the correctness of the computation.
/// Methods here include ones to fix bits, evaluate sumcheck messages, etc.
impl<F: FieldExt> ExpressionNode<F, ProverExpression> {

    /// transforms the expression to a verifier expression
    /// should only be called when the entire expression is fully bound
    /// traverses the expression and changes the DenseMleRef to F,
    /// by grabbing their bookkeeping table's 1st and only element,
    /// if the bookkeeping table has more than 1 element, it 
    /// throws an ExpressionError::EvaluateNotFullyBoundError
    pub fn transform_to_verifier_expression(&mut self) -> Result<ExpressionNode<F, VerifierExpression>, ExpressionError>{

        match self {
            ExpressionNode::Constant(scalar) => Ok(ExpressionNode::Constant(*scalar)),
            ExpressionNode::Selector(index, a, b) => {
                Ok(
                    ExpressionNode::Selector(
                        index.clone(),
                        Box::new(a.transform_to_verifier_expression()?),
                        Box::new(b.transform_to_verifier_expression()?)
                    )
                )
            }
            ExpressionNode::Mle(mle_ref) => {

                if mle_ref.bookkeeping_table().len() != 1 {
                    return Err(ExpressionError::EvaluateNotFullyBoundError);
                }
                Ok(ExpressionNode::Mle(mle_ref.bookkeeping_table()[0]))
            }
            ExpressionNode::Negated(a) => Ok(
                ExpressionNode::Negated(Box::new(a.transform_to_verifier_expression()?))
            ),
            ExpressionNode::Sum(a, b) => {

                Ok(
                    ExpressionNode::Sum(
                        Box::new(a.transform_to_verifier_expression()?),
                        Box::new(b.transform_to_verifier_expression()?)
                    )
                )
            }
            ExpressionNode::Product(mle_refs) => {
                for mle_ref in mle_refs.iter() {
                    if mle_ref.bookkeeping_table().len() != 1 {
                        return Err(ExpressionError::EvaluateNotFullyBoundError);
                    }
                }
                Ok(
                    ExpressionNode::Product(
                        mle_refs.into_iter().map(|mle_ref| mle_ref.bookkeeping_table()[0].clone()).collect_vec()
                    )
                )
                
            }
            ExpressionNode::Scaled(mle, scalar) => Ok(
                ExpressionNode::Scaled(Box::new(mle.transform_to_verifier_expression()?), *scalar)
            ),
        }
    }

    /// fix the variable at a certain round index
    pub fn fix_variable(&mut self, round_index: usize, challenge: F) {
        match self {
            ExpressionNode::Selector(index, a, b) => {
                if *index == MleIndex::IndexedBit(round_index) {
                    index.bind_index(challenge);
                } else {
                    a.fix_variable(round_index, challenge);
                    b.fix_variable(round_index, challenge);
                }
            }
            ExpressionNode::Mle(mle_ref) => {
                if mle_ref
                    .mle_indices()
                    .contains(&MleIndex::IndexedBit(round_index))
                {
                    mle_ref.fix_variable(round_index, challenge);
                }
            }
            ExpressionNode::Negated(a) => a.fix_variable(round_index, challenge),
            ExpressionNode::Sum(a, b) => {
                a.fix_variable(round_index, challenge);
                b.fix_variable(round_index, challenge);
            }
            ExpressionNode::Product(mle_refs) => {
                for mle_ref in mle_refs {
                    if mle_ref
                        .mle_indices()
                        .contains(&MleIndex::IndexedBit(round_index))
                    {
                        mle_ref.fix_variable(round_index, challenge);
                    }
                }
            }
            ExpressionNode::Scaled(a, _) => {
                a.fix_variable(round_index, challenge);
            }
            ExpressionNode::Constant(_) => (),
        }
    }

    /// evaluates an expression on the given challenges points, by fixing the variables
    pub fn evaluate_expr(&mut self, challenges: Vec<F>) -> Result<F, ExpressionError> {
        // --- It's as simple as fixing all variables ---
        challenges
            .iter()
            .enumerate()
            .for_each(|(round_idx, &challenge)| {
                self.fix_variable(round_idx, challenge);
            });

        // ----- this is literally a check -----
        let mut observer_fn = |exp: &ExpressionNode<F, ProverExpression>| -> Result<(), ExpressionError> {
            match exp {
                ExpressionNode::Mle(mle_ref) => {
                    let indices = mle_ref
                        .mle_indices()
                        .iter()
                        .filter_map(|index| match index {
                            MleIndex::Bound(chal, index) => Some((*chal, index)),
                            _ => None,
                        })
                        .collect_vec();

                    let start = *indices[0].1;
                    let end = *indices[indices.len() - 1].1;

                    let (indices, _): (Vec<_>, Vec<usize>) = indices.into_iter().unzip();

                    if indices.as_slice() == &challenges[start..=end] {
                        Ok(())
                    } else {
                        Err(ExpressionError::EvaluateBoundIndicesDontMatch)
                    }
                }
                ExpressionNode::Product(mle_refs) => mle_refs
                    .iter()
                    .map(|mle_ref| {
                        let indices = mle_ref
                            .mle_indices()
                            .iter()
                            .filter_map(|index| match index {
                                MleIndex::Bound(chal, index) => Some((*chal, index)),
                                _ => None,
                            })
                            .collect_vec();

                        let start = *indices[0].1;
                        let end = *indices[indices.len() - 1].1;

                        let (indices, _): (Vec<_>, Vec<usize>) = indices.into_iter().unzip();

                        if indices.as_slice() == &challenges[start..=end] {
                            Ok(())
                        } else {
                            Err(ExpressionError::EvaluateBoundIndicesDontMatch)
                        }
                    })
                    .try_collect(),

                _ => Ok(()),
            }
        };
        self.traverse(&mut observer_fn)?;
        // ----- this is literally a check ends -----

        // --- Traverse the expression and pick up all the evals ---
        self.transform_to_verifier_expression().unwrap().gather_combine_all_evals()
    }


    /// computes the sumcheck message for the given round index, and beta mle
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_sumcheck<T>(
        &self,
        constant: &impl Fn(F, &DenseMleRef<F>) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&DenseMleRef<F>, &DenseMleRef<F>) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[DenseMleRef<F>], &DenseMleRef<F>) -> T,
        scaled: &impl Fn(T, F) -> T,
        beta_mle_ref: &DenseMleRef<F>,
        round_index: usize,
    ) -> T {
        match self {
            ExpressionNode::Constant(scalar) => constant(*scalar, beta_mle_ref),
            ExpressionNode::Selector(index, a, b) => {
                // need to check whether the selector bit is the current independent variable
                if let MleIndex::IndexedBit(idx) = index {
                    match Ord::cmp(&round_index, idx) {
                        // if not, need to split the beta table according to the beta_split function
                        std::cmp::Ordering::Less => {
                            let (beta_mle_first, beta_mle_second) = beta_split(beta_mle_ref);
                            selector_column(
                                index,
                                a.evaluate_sumcheck(
                                    constant,
                                    selector_column,
                                    mle_eval,
                                    negated,
                                    sum,
                                    product,
                                    scaled,
                                    &beta_mle_first,
                                    round_index,
                                ),
                                b.evaluate_sumcheck(
                                    constant,
                                    selector_column,
                                    mle_eval,
                                    negated,
                                    sum,
                                    product,
                                    scaled,
                                    &beta_mle_second,
                                    round_index,
                                ),
                            )
                        }
                        // otherwise, proceed normally
                        std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => selector_column(
                            index,
                            a.evaluate_sumcheck(
                                constant,
                                selector_column,
                                mle_eval,
                                negated,
                                sum,
                                product,
                                scaled,
                                beta_mle_ref,
                                round_index,
                            ),
                            b.evaluate_sumcheck(
                                constant,
                                selector_column,
                                mle_eval,
                                negated,
                                sum,
                                product,
                                scaled,
                                beta_mle_ref,
                                round_index,
                            ),
                        ),
                    }
                } else {
                    selector_column(
                        index,
                        a.evaluate_sumcheck(
                            constant,
                            selector_column,
                            mle_eval,
                            negated,
                            sum,
                            product,
                            scaled,
                            beta_mle_ref,
                            round_index,
                        ),
                        b.evaluate_sumcheck(
                            constant,
                            selector_column,
                            mle_eval,
                            negated,
                            sum,
                            product,
                            scaled,
                            beta_mle_ref,
                            round_index,
                        ),
                    )
                }
            }
            ExpressionNode::Mle(query) => mle_eval(query, beta_mle_ref),
            ExpressionNode::Negated(a) => {
                let a = a.evaluate_sumcheck(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta_mle_ref,
                    round_index,
                );
                negated(a)
            }
            ExpressionNode::Sum(a, b) => {
                let a = a.evaluate_sumcheck(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta_mle_ref,
                    round_index,
                );
                let b = b.evaluate_sumcheck(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta_mle_ref,
                    round_index,
                );
                sum(a, b)
            }
            ExpressionNode::Product(queries) => product(queries, beta_mle_ref),
            ExpressionNode::Scaled(a, f) => {
                let a = a.evaluate_sumcheck(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta_mle_ref,
                    round_index,
                );
                scaled(a, *f)
            }
        }
    }

    /// Mutate the MleIndices that are Iterated in the expression and turn them into IndexedBit
    /// Returns the max number of bits that are indexed
    pub fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                *mle_index = MleIndex::IndexedBit(curr_index);
                let a_bits = a.index_mle_indices(curr_index + 1);
                let b_bits = b.index_mle_indices(curr_index + 1);
                max(a_bits, b_bits)
            }
            ExpressionNode::Mle(mle_ref) => mle_ref.index_mle_indices(curr_index),
            ExpressionNode::Sum(a, b) => {
                let a_bits = a.index_mle_indices(curr_index);
                let b_bits = b.index_mle_indices(curr_index);
                max(a_bits, b_bits)
            }
            ExpressionNode::Product(mle_refs) => mle_refs
                .iter_mut()
                .map(|mle_ref| mle_ref.index_mle_indices(curr_index))
                .reduce(max)
                .unwrap_or(curr_index),
            ExpressionNode::Scaled(a, _) => a.index_mle_indices(curr_index),
            ExpressionNode::Negated(a) => a.index_mle_indices(curr_index),
            ExpressionNode::Constant(_) => curr_index,
        }
    }

    /// Gets the size of an expression in terms of the number of rounds of sumcheck
    pub fn get_expression_size(&self, curr_size: usize) -> usize {
        match self {
            ExpressionNode::Selector(_mle_index, a, b) => {
                let a_bits = a.get_expression_size(curr_size + 1);
                let b_bits = b.get_expression_size(curr_size + 1);
                max(a_bits, b_bits)
            }
            ExpressionNode::Mle(mle_ref) => {
                mle_ref
                    .mle_indices()
                    .iter()
                    .filter(|item| {
                        matches!(
                            item,
                            &&MleIndex::Iterated
                                | &&MleIndex::IndexedBit(_)
                                | &&MleIndex::Bound(_, _)
                        )
                    })
                    .collect_vec()
                    .len()
                    + curr_size
            }
            ExpressionNode::Sum(a, b) => {
                let a_bits = a.get_expression_size(curr_size);
                let b_bits = b.get_expression_size(curr_size);
                max(a_bits, b_bits)
            }
            ExpressionNode::Product(mle_refs) => {
                mle_refs
                    .iter()
                    .map(|mle_ref| {
                        mle_ref
                            .mle_indices()
                            .iter()
                            .filter(|item| {
                                matches!(
                                    item,
                                    &&MleIndex::Iterated
                                        | &&MleIndex::IndexedBit(_)
                                        | &&MleIndex::Bound(_, _)
                                )
                            })
                            .collect_vec()
                            .len()
                    })
                    .max()
                    .unwrap_or(0)
                    + curr_size
            }
            ExpressionNode::Scaled(a, _) => a.get_expression_size(curr_size),
            ExpressionNode::Negated(a) => a.get_expression_size(curr_size),
            ExpressionNode::Constant(_) => curr_size,
        }
    }
}


/// describes the circuit given the expression (includes all the info of the data that the expression is instantiated with)
impl<F: std::fmt::Debug + FieldExt> ExpressionNode<F, ProverExpression> {
    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> impl std::fmt::Display + 'a {
        struct CircuitDesc<'a, F: FieldExt>(&'a ExpressionNode<F, ProverExpression>);
        impl<'a, F: std::fmt::Debug + FieldExt> std::fmt::Display for CircuitDesc<'a, F> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    ExpressionNode::Constant(scalar) => {
                        f.debug_tuple("const").field(scalar).finish()
                    }
                    ExpressionNode::Selector(index, a, b) => f.write_fmt(format_args!("sel {index:?}; {}; {}", CircuitDesc(a), CircuitDesc(b))),
                    // Skip enum variant and print query struct directly to maintain backwards compatibility.
                    ExpressionNode::Mle(mle_ref) => {
                        f.debug_struct("mle").field("layer", &mle_ref.get_layer_id()).field("indices", &mle_ref.mle_indices()).finish()
                    }
                    ExpressionNode::Negated(poly) => f.write_fmt(format_args!("-{}", CircuitDesc(poly))),
                    ExpressionNode::Sum(a, b) => f.write_fmt(format_args!("+ {}; {}", CircuitDesc(a), CircuitDesc(b))),
                    ExpressionNode::Product(a) => {
                        let str = a.iter().map(|mle| {
                            format!("{:?}; {:?}", mle.get_layer_id(), mle.mle_indices())
                        }).reduce(|acc, str| acc + &str).unwrap();
                        f.write_str(&str)
                    },
                    ExpressionNode::Scaled(poly, scalar) => {
                        f.write_fmt(format_args!("* {}; {:?}", CircuitDesc(poly), scalar))
                    }
                }
            }
        }

        CircuitDesc(self)
    }
}