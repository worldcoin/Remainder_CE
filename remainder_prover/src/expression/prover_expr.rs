use std::{
    cmp::max, fmt::Debug, marker::PhantomData, ops::{Add, Mul, Neg, Sub}
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use crate::mle::{beta::*, dense::DenseMleRef, MleIndex, MleRef};
use remainder_shared_types::FieldExt;
use super::{expr_errors::ExpressionError, generic_expr::{Expression, ExpressionNode, ExpressionType}, verifier_expr::{VerifierExpression}};

/// mid-term solution for deduplication of DenseMleRefs
/// basically a wrapper around usize, which denotes the index
/// of the MleRef in an expression's MleRef list/// Generic Expressions
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct MleVecIndex<F>(usize, PhantomData<F>);

impl<F: FieldExt> MleVecIndex<F> {

    /// create a new MleRefIndex
    pub fn new(index: usize) -> Self {
        MleVecIndex(index, PhantomData)
    }

    /// returns the index
    pub fn index(&self) -> usize {
        self.0
    }

    /// add the index with an increment amount
    pub fn increment(&mut self, offset: usize) {
        self.0 += offset;
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

/// this also needs to be prover / verifier / abstract specific
// defines how the Expressions are printed and displayed
impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for MleVecIndex<F> {
    /// needs to change ^above^ to be Expression
    /// and then change how to display ExpressionNode::Mle by indexing using MleIndex
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

        f.debug_tuple("MleVecIndex")
        .field(&self.0)
        .finish()

    }
}

/// Prover Expression
/// the leaf nodes of the expression tree are DenseMleRefs
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProverExpressionMle;
impl<F: FieldExt> ExpressionType<F> for ProverExpressionMle {
    type Container = MleVecIndex<F>; /// this also needs to change, perhaps duplicate a struct, rename, and then change
    type MleVec = Vec<DenseMleRef<F>>;
}

/// this is what the prover manipulates to prove the correctness of the computation.
/// Methods here include ones to fix bits, evaluate sumcheck messages, etc.
impl<F: FieldExt> Expression<F, ProverExpressionMle> {

    /// Concatenates two expressions together
    pub fn concat_expr(mut self, lhs: Expression<F, ProverExpressionMle>) -> Self {

        let offset = lhs.num_mle_ref(); 
        self.increment_mle_vec_indices(offset);

        let (lhs_node, lhs_mle_vec) = lhs.deconstruct();
        let (rhs_node, rhs_mle_vec) = self.deconstruct();

        let concat_node = ExpressionNode::Selector(MleIndex::Iterated, Box::new(lhs_node), Box::new(rhs_node));
        let concat_mle_vec = lhs_mle_vec.into_iter().chain(rhs_mle_vec.into_iter()).collect_vec();

        Expression::new(concat_node, concat_mle_vec)
    }

    /// Create a product Expression that multiplies many MLEs together
    pub fn products(product_list: <ProverExpressionMle as ExpressionType<F>>::MleVec) -> Self {

        let mle_vec_indices = (0..product_list.len()).map(
            |index| MleVecIndex::new(index)
        ).collect_vec();

        let product_node = ExpressionNode::Product(mle_vec_indices);
        
        Expression::new(product_node, product_list)
        
    }

    /// Create a mle Expression that contains one MLE
    pub fn mle(mle: DenseMleRef<F>) -> Self {

        let mle_node = ExpressionNode::Mle(MleVecIndex::new(0));
        
        Expression::new(mle_node, [mle].to_vec())
        
    }

    /// Create a constant Expression that contains one field element
    pub fn constant(constant: F) -> Self {

        let mle_node = ExpressionNode::Constant(constant);
        
        Expression::new(mle_node, [].to_vec())
        
    }

    /// negates an Expression
    pub fn negated(expression: Box<Expression<F, ProverExpressionMle>>) -> Self {

        let (node, mle_vec) = expression.deconstruct();

        let mle_node = ExpressionNode::Negated(Box::new(node));
        
        Expression::new(mle_node, mle_vec)
        
    }

    /// Create a Sum Expression that contains two MLEs
    pub fn sum(
        lhs: Box<Expression<F, ProverExpressionMle>>,
        mut rhs: Box<Expression<F, ProverExpressionMle>>
    ) -> Self {

        let offset = lhs.num_mle_ref(); 
        rhs.increment_mle_vec_indices(offset);

        let (lhs_node, lhs_mle_vec) = lhs.deconstruct();
        let (rhs_node, rhs_mle_vec) = rhs.deconstruct();

        let sum_node = ExpressionNode::Sum(Box::new(lhs_node), Box::new(rhs_node));
        let sum_mle_vec = lhs_mle_vec.into_iter().chain(rhs_mle_vec.into_iter()).collect_vec();

        Expression::new(sum_node, sum_mle_vec)
        
    }

    /// scales an Expression by a field element
    pub fn scaled(expression: Box<Expression<F, ProverExpressionMle>>, scale: F) -> Self {

        let (node, mle_vec) = expression.deconstruct();

        Expression::new(ExpressionNode::Scaled(Box::new(node), scale), mle_vec)
        
    }

    /// returns the number of MleRefs in the expression
    pub fn num_mle_ref(&self) -> usize {
        self.mle_vec().len()
    }

    /// which increments all the MleVecIndex in the expression by *param* amount
    pub fn increment_mle_vec_indices(&mut self, offset: usize) {
        // define a closure that increments the MleVecIndex by the given amount
        // use traverse_mut
        let mut increment_closure = for<'a, 'b> |expr: &'a mut ExpressionNode<F, ProverExpressionMle>, _mle_vec: &'b mut Vec<DenseMleRef<F>>| -> Result<(), ()> {
            match expr {
                ExpressionNode::Mle(mle_vec_index) => {
                    mle_vec_index.increment(offset);
                    Ok(())
                }
                ExpressionNode::Product(mle_indices) => {
                    for mle_vec_index in mle_indices {
                        mle_vec_index.increment(offset);
                    }
                    Ok(())
                }
                ExpressionNode::Constant(_)
                | ExpressionNode::Scaled(_, _)
                | ExpressionNode::Sum(_, _)
                | ExpressionNode::Negated(_)
                | ExpressionNode::Selector(_, _, _) => Ok(()),
            }
        };

        self.traverse_mut(&mut increment_closure).unwrap();
    }


    /// transforms the expression to a verifier expression
    /// should only be called when the entire expression is fully bound
    /// traverses the expression and changes the DenseMleRef to F,
    /// by grabbing their bookkeeping table's 1st and only element,
    /// if the bookkeeping table has more than 1 element, it 
    /// throws an ExpressionError::EvaluateNotFullyBoundError
    pub fn transform_to_verifier_expression(self) -> Result<Expression<F, VerifierExpression>, ExpressionError>{
        let (
            mut expression_node,
            mle_vec
        ) = self.deconstruct();
        Ok(
            Expression::new(
                expression_node.transform_to_verifier_expression_node(&mle_vec).unwrap(),
                ()
            )
        )
    }

    
    /// fix the variable at a certain round index
    pub fn fix_variable(&mut self, round_index: usize, challenge: F) {

        let (
            expression_node,
            mle_vec
        ) = self.deconstruct_mut();

        expression_node.fix_variable_node(round_index, challenge, mle_vec)
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
        let mut observer_fn = |exp: &ExpressionNode<F, ProverExpressionMle>, mle_vec: &<ProverExpressionMle as ExpressionType<F>>::MleVec| -> Result<(), ExpressionError> {
            match exp {
                ExpressionNode::Mle(mle_vec_idx) => {

                    let mle_ref = mle_vec_idx.get_mle(mle_vec);

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
                ExpressionNode::Product(mle_vec_indices) => {
                    let mle_refs = mle_vec_indices.into_iter().map(
                        |mle_vec_index|
                            mle_vec_index.get_mle(mle_vec)
                    ).collect_vec();

                    mle_refs
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
                    .try_collect()
                },

                _ => Ok(()),
            }
        };
        self.traverse(&mut observer_fn)?;
        // ----- this is literally a check ends -----

        // --- Traverse the expression and pick up all the evals ---
        self.clone().transform_to_verifier_expression().unwrap().gather_combine_all_evals()
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
        product: &impl Fn(&[&DenseMleRef<F>], &DenseMleRef<F>) -> T, // changed signature here, note to modify caller's calling code
        scaled: &impl Fn(T, F) -> T,
        beta_mle_ref: &DenseMleRef<F>,
        round_index: usize,
    ) -> T {
        self.expression_node().evaluate_sumcheck_node(
            constant,
            selector_column,
            mle_eval,
            negated,
            sum,
            product,
            scaled,
            beta_mle_ref,
            round_index,
            self.mle_vec()
        )
    }


    /// Mutate the MleIndices that are Iterated in the expression and turn them into IndexedBit
    /// Returns the max number of bits that are indexed
    pub fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        let (
            expression_node,
            mle_vec
        ) = self.deconstruct_mut();

        expression_node.index_mle_indices_node(curr_index, mle_vec)
    }


    /// Gets the size of an expression in terms of the number of rounds of sumcheck
    pub fn get_expression_size(&self, curr_size: usize) -> usize {
        self.expression_node().get_expression_size_node(curr_size, self.mle_vec())
    }

}

impl<F: FieldExt> ExpressionNode<F, ProverExpressionMle> {

    /// transforms the expression to a verifier expression
    /// should only be called when the entire expression is fully bound
    /// traverses the expression and changes the DenseMleRef to F,
    /// by grabbing their bookkeeping table's 1st and only element,
    /// if the bookkeeping table has more than 1 element, it 
    /// throws an ExpressionError::EvaluateNotFullyBoundError
    pub fn transform_to_verifier_expression_node(
        &mut self,
        mle_vec: &<ProverExpressionMle as ExpressionType<F>>::MleVec
    ) -> Result<ExpressionNode<F, VerifierExpression>, ExpressionError>{

        match self {
            ExpressionNode::Constant(scalar) => Ok(ExpressionNode::Constant(*scalar)),
            ExpressionNode::Selector(index, a, b) => {
                Ok(
                    ExpressionNode::Selector(
                        index.clone(),
                        Box::new(a.transform_to_verifier_expression_node(mle_vec)?),
                        Box::new(b.transform_to_verifier_expression_node(mle_vec)?)
                    )
                )
            }
            ExpressionNode::Mle(mle_vec_idx) => {

                let mle_ref = mle_vec_idx.get_mle(mle_vec);

                if mle_ref.bookkeeping_table().len() != 1 {
                    return Err(ExpressionError::EvaluateNotFullyBoundError);
                }
                Ok(ExpressionNode::Mle(mle_ref.bookkeeping_table()[0]))
            }
            ExpressionNode::Negated(a) => Ok(
                ExpressionNode::Negated(Box::new(a.transform_to_verifier_expression_node(mle_vec)?))
            ),
            ExpressionNode::Sum(a, b) => {

                Ok(
                    ExpressionNode::Sum(
                        Box::new(a.transform_to_verifier_expression_node(mle_vec)?),
                        Box::new(b.transform_to_verifier_expression_node(mle_vec)?)
                    )
                )
            }
            ExpressionNode::Product(mle_vec_indices) => {
                let mle_refs = mle_vec_indices.into_iter().map(
                    |mle_vec_index|
                        mle_vec_index.get_mle(mle_vec)
                ).collect_vec();

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
                ExpressionNode::Scaled(Box::new(mle.transform_to_verifier_expression_node(mle_vec)?), *scalar)
            ),
        }
    }


    /// fix the variable at a certain round index
    pub fn fix_variable_node(
        &mut self,
        round_index: usize,
        challenge: F,
        mle_vec: &mut <ProverExpressionMle as ExpressionType<F>>::MleVec
    ) {
        match self {
            ExpressionNode::Selector(index, a, b) => {
                if *index == MleIndex::IndexedBit(round_index) {
                    index.bind_index(challenge);
                } else {
                    a.fix_variable_node(round_index, challenge, mle_vec);
                    b.fix_variable_node(round_index, challenge, mle_vec);
                }
            }
            ExpressionNode::Mle(mle_vec_idx) => {

                let mle_ref = mle_vec_idx.get_mle_mut(mle_vec);

                if mle_ref
                    .mle_indices()
                    .contains(&MleIndex::IndexedBit(round_index))
                {
                    mle_ref.fix_variable(round_index, challenge);
                }
            }
            ExpressionNode::Negated(a) => a.fix_variable_node(round_index, challenge, mle_vec),
            ExpressionNode::Sum(a, b) => {
                a.fix_variable_node(round_index, challenge, mle_vec);
                b.fix_variable_node(round_index, challenge, mle_vec);
            }
            ExpressionNode::Product(mle_vec_indices) => {

                mle_vec_indices.into_iter().map(
                    |mle_vec_index| {
                        let mle_ref = mle_vec_index.get_mle_mut(mle_vec);

                        if mle_ref.mle_indices().contains(&MleIndex::IndexedBit(round_index)){
                            mle_ref.fix_variable(round_index, challenge);
                        }
                    }
                ).collect_vec();
            }
            ExpressionNode::Scaled(a, _) => {
                a.fix_variable_node(round_index, challenge, mle_vec);
            }
            ExpressionNode::Constant(_) => (),
        }
    }


    /// computes the sumcheck message for the given round index, and beta mle
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_sumcheck_node<T>(
        &self,
        constant: &impl Fn(F, &DenseMleRef<F>) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&DenseMleRef<F>, &DenseMleRef<F>) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[&DenseMleRef<F>], &DenseMleRef<F>) -> T,     // changed the signature here, note to change the caller of this function to match this
        scaled: &impl Fn(T, F) -> T,
        beta_mle_ref: &DenseMleRef<F>,
        round_index: usize,
        mle_vec: &<ProverExpressionMle as ExpressionType<F>>::MleVec
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
                                a.evaluate_sumcheck_node(
                                    constant,
                                    selector_column,
                                    mle_eval,
                                    negated,
                                    sum,
                                    product,
                                    scaled,
                                    &beta_mle_first,
                                    round_index,
                                    mle_vec,
                                ),
                                b.evaluate_sumcheck_node(
                                    constant,
                                    selector_column,
                                    mle_eval,
                                    negated,
                                    sum,
                                    product,
                                    scaled,
                                    &beta_mle_second,
                                    round_index,
                                    mle_vec,
                                ),
                            )
                        }
                        // otherwise, proceed normally
                        std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => selector_column(
                            index,
                            a.evaluate_sumcheck_node(
                                constant,
                                selector_column,
                                mle_eval,
                                negated,
                                sum,
                                product,
                                scaled,
                                beta_mle_ref,
                                round_index,
                                mle_vec,
                            ),
                            b.evaluate_sumcheck_node(
                                constant,
                                selector_column,
                                mle_eval,
                                negated,
                                sum,
                                product,
                                scaled,
                                beta_mle_ref,
                                round_index,
                                mle_vec,
                            ),
                        ),
                    }
                } else {
                    selector_column(
                        index,
                        a.evaluate_sumcheck_node(
                            constant,
                            selector_column,
                            mle_eval,
                            negated,
                            sum,
                            product,
                            scaled,
                            beta_mle_ref,
                            round_index,
                            mle_vec,
                        ),
                        b.evaluate_sumcheck_node(
                            constant,
                            selector_column,
                            mle_eval,
                            negated,
                            sum,
                            product,
                            scaled,
                            beta_mle_ref,
                            round_index,
                            mle_vec,
                        ),
                    )
                }
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle_ref = mle_vec_idx.get_mle(mle_vec);
                mle_eval(mle_ref, beta_mle_ref)
            },
            ExpressionNode::Negated(a) => {
                let a = a.evaluate_sumcheck_node(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta_mle_ref,
                    round_index,
                    mle_vec,
                );
                negated(a)
            }
            ExpressionNode::Sum(a, b) => {
                let a = a.evaluate_sumcheck_node(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta_mle_ref,
                    round_index,
                    mle_vec,
                );
                let b = b.evaluate_sumcheck_node(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta_mle_ref,
                    round_index,
                    mle_vec,
                );
                sum(a, b)
            }
            ExpressionNode::Product(mle_vec_indices) => {

                let mle_refs = mle_vec_indices.into_iter().map(
                    |mle_vec_index|
                        mle_vec_index.get_mle(mle_vec)
                ).collect_vec();

                product(&mle_refs, beta_mle_ref)
            },
            ExpressionNode::Scaled(a, f) => {
                let a = a.evaluate_sumcheck_node(
                    constant,
                    selector_column,
                    mle_eval,
                    negated,
                    sum,
                    product,
                    scaled,
                    beta_mle_ref,
                    round_index,
                    mle_vec,
                );
                scaled(a, *f)
            }
        }
    }


    /// Mutate the MleIndices that are Iterated in the expression and turn them into IndexedBit
    /// Returns the max number of bits that are indexed
    pub fn index_mle_indices_node(
        &mut self,
        curr_index: usize,
        mle_vec: &mut <ProverExpressionMle as ExpressionType<F>>::MleVec
    ) -> usize {
        match self {
            ExpressionNode::Selector(mle_index, a, b) => {
                *mle_index = MleIndex::IndexedBit(curr_index);
                let a_bits = a.index_mle_indices_node(curr_index + 1, mle_vec);
                let b_bits = b.index_mle_indices_node(curr_index + 1, mle_vec);
                max(a_bits, b_bits)
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle_ref = mle_vec_idx.get_mle_mut(mle_vec);
                if!mle_ref.indexed() {mle_ref.index_mle_indices(curr_index)} else {0} // if it's already indexed, then return 0 (the max number of indexed bit will get propogated up)
            },
            ExpressionNode::Sum(a, b) => {
                let a_bits = a.index_mle_indices_node(curr_index, mle_vec);
                let b_bits = b.index_mle_indices_node(curr_index, mle_vec);
                max(a_bits, b_bits)
            }
            ExpressionNode::Product(mle_vec_indices) => {

                mle_vec_indices.into_iter().map(
                    |mle_vec_index| {
                        let mle_ref = mle_vec_index.get_mle_mut(mle_vec);
                        if!mle_ref.indexed() {mle_ref.index_mle_indices(curr_index)} else {0}
                    }
                ).reduce(max)
                .unwrap_or(curr_index)

            },
            ExpressionNode::Scaled(a, _) => a.index_mle_indices_node(curr_index, mle_vec),
            ExpressionNode::Negated(a) => a.index_mle_indices_node(curr_index, mle_vec),
            ExpressionNode::Constant(_) => curr_index,
        }
    }


    /// Gets the size of an expression in terms of the number of rounds of sumcheck
    pub fn get_expression_size_node(
        &self,
        curr_size: usize,
        mle_vec: &<ProverExpressionMle as ExpressionType<F>>::MleVec
    ) -> usize {
        match self {
            ExpressionNode::Selector(_mle_index, a, b) => {
                let a_bits = a.get_expression_size_node(curr_size + 1, mle_vec);
                let b_bits = b.get_expression_size_node(curr_size + 1, mle_vec);
                max(a_bits, b_bits)
            }
            ExpressionNode::Mle(mle_vec_idx) => {

                let mle_ref = mle_vec_idx.get_mle(mle_vec);

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
                let a_bits = a.get_expression_size_node(curr_size, mle_vec);
                let b_bits = b.get_expression_size_node(curr_size, mle_vec);
                max(a_bits, b_bits)
            }
            ExpressionNode::Product(mle_vec_indices) => {

                let mle_refs = mle_vec_indices.into_iter().map(
                    |mle_vec_index|
                        mle_vec_index.get_mle(mle_vec)
                ).collect_vec();

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
            ExpressionNode::Scaled(a, _) => a.get_expression_size_node(curr_size, mle_vec),
            ExpressionNode::Negated(a) => a.get_expression_size_node(curr_size, mle_vec),
            ExpressionNode::Constant(_) => curr_size,
        }
    }


    /// get the number of MleVecIndexs in the expression
    /// assumes that it starts with 0
    pub fn get_mle_vec_len(
        &self,
    ) -> usize {
        let mut mle_exists = false;
        let largest_mle_vec_idx = match self {
            ExpressionNode::Selector(_mle_index, a, b) => {
                let a_bits = a.get_mle_vec_len();
                let b_bits = b.get_mle_vec_len();
                max(a_bits, b_bits)
            }
            ExpressionNode::Mle(mle_vec_idx) => {
                mle_exists = true;
                mle_vec_idx.0
            }
            ExpressionNode::Sum(a, b) => {
                let a_bits = a.get_mle_vec_len();
                let b_bits = b.get_mle_vec_len();
                max(a_bits, b_bits)
            }
            ExpressionNode::Product(mle_vec_indices) => {
                mle_exists = true;
                mle_vec_indices.into_iter().map(
                    |mle_vec_index|
                        mle_vec_index.0
                ).max()
                .unwrap_or(0)

            }
            ExpressionNode::Scaled(a, _) => a.get_mle_vec_len(),
            ExpressionNode::Negated(a) => a.get_mle_vec_len(),
            ExpressionNode::Constant(_) => 0,
        };

        
        if mle_exists {
            largest_mle_vec_idx + 1

        // if there isn't any mle, then the expression's mle_vec's length is 0
        } else {0}
    }
}



impl<F: FieldExt> Neg for Expression<F, ProverExpressionMle> {
    type Output = Expression<F, ProverExpressionMle>;
    fn neg(self) -> Self::Output {

        let (node, mle_vec) = self.deconstruct();
        Expression::new(ExpressionNode::Negated(Box::new(node)), mle_vec)
    }
}


/// implement the Add, Sub, and Mul traits for the Expression
impl<F: FieldExt> Add for Expression<F, ProverExpressionMle> {
    type Output = Expression<F, ProverExpressionMle>;
    fn add(self, mut rhs: Expression<F, ProverExpressionMle>) -> Expression<F, ProverExpressionMle> {

        let offset = self.num_mle_ref(); 
        rhs.increment_mle_vec_indices(offset);

        let (lhs_node, lhs_mle_vec) = self.deconstruct();
        let (rhs_node, rhs_mle_vec) = rhs.deconstruct();

        let sum_node = ExpressionNode::Sum(Box::new(lhs_node), Box::new(rhs_node));
        let sum_mle_vec = lhs_mle_vec.into_iter().chain(rhs_mle_vec.into_iter()).collect_vec();

        Expression::new(sum_node, sum_mle_vec)
    }
}

impl<F: FieldExt> Sub for Expression<F, ProverExpressionMle> {
    type Output = Expression<F, ProverExpressionMle>;
    fn sub(self, rhs: Expression<F, ProverExpressionMle>) -> Expression<F, ProverExpressionMle> {

        self.add(rhs.neg())
    }
}

impl<F: FieldExt> Mul<F> for Expression<F, ProverExpressionMle> {
    type Output = Expression<F, ProverExpressionMle>;
    fn mul(self, rhs: F) -> Self::Output {

        let (node, mle_vec) = self.deconstruct();
        Expression::new(ExpressionNode::Scaled(Box::new(node), rhs), mle_vec)
    }
}

// defines how the Expressions are printed and displayed
impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for Expression<F, ProverExpressionMle> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

        f.debug_struct("Expression")
        .field("Expression_Node", &self.expression_node())
        .field("MleRef_Vec", &self.mle_vec())
        .finish()

    }
}

// defines how the ExpressionNodes are printed and displayed
impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for ExpressionNode<F, ProverExpressionMle> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

        match self {
            ExpressionNode::Constant(scalar) => {
                f.debug_tuple("Constant").field(scalar).finish()
            }
            ExpressionNode::Selector(index, a, b) => f
                .debug_tuple("Selector")
                .field(index)
                .field(a)
                .field(b)
                .finish(),
            // Skip enum variant and print query struct directly to maintain backwards compatibility.
            ExpressionNode::Mle(_mle_ref) => {
                f.debug_struct("Mle").field("mle_ref", _mle_ref).finish()
            }
            ExpressionNode::Negated(poly) => f.debug_tuple("Negated").field(poly).finish(),
            ExpressionNode::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            ExpressionNode::Product(a) => f.debug_tuple("Product").field(a).finish(),
            ExpressionNode::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }

    }
}

/// describes the circuit given the expression (includes all the info of the data that the expression is instantiated with)
impl<F: std::fmt::Debug + FieldExt> Expression<F, ProverExpressionMle> {
    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> impl std::fmt::Display + 'a {

        struct CircuitDesc<'a, F: FieldExt>(&'a ExpressionNode<F, ProverExpressionMle>, &'a <ProverExpressionMle as ExpressionType<F>>::MleVec);
        
        impl<'a, F: std::fmt::Debug + FieldExt> std::fmt::Display for CircuitDesc<'a, F> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    ExpressionNode::Constant(scalar) => {
                        f.debug_tuple("const").field(scalar).finish()
                    }
                    ExpressionNode::Selector(index, a, b) => f.write_fmt(format_args!("sel {index:?}; {}; {}", CircuitDesc(a, self.1), CircuitDesc(b, self.1))),
                    // Skip enum variant and print query struct directly to maintain backwards compatibility.
                    ExpressionNode::Mle(mle_vec_idx) => {

                        let mle_ref = mle_vec_idx.get_mle(self.1);

                        f.debug_struct("mle").field("layer", &mle_ref.get_layer_id()).field("indices", &mle_ref.mle_indices()).finish()
                    }
                    ExpressionNode::Negated(poly) => f.write_fmt(format_args!("-{}", CircuitDesc(poly, self.1))),
                    ExpressionNode::Sum(a, b) => f.write_fmt(format_args!("+ {}; {}", CircuitDesc(a, self.1), CircuitDesc(b, self.1))),
                    ExpressionNode::Product(a) => {
                        let str = a.iter().map(|mle_vec_idx| {

                            let mle = mle_vec_idx.get_mle(self.1);

                            format!("{:?}; {:?}", mle.get_layer_id(), mle.mle_indices())

                        }).reduce(|acc, str| acc + &str).unwrap();
                        f.write_str(&str)
                    },
                    ExpressionNode::Scaled(poly, scalar) => {
                        f.write_fmt(format_args!("* {}; {:?}", CircuitDesc(poly, self.1), scalar))
                    }
                }
            }
        }

        CircuitDesc(self.expression_node(), self.mle_vec())
    }
}