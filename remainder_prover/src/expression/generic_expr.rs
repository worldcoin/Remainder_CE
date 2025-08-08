//! Functionality which is common to all "expression"s (see documentation within
//! [crate::expression]). See documentation in [Expression] for high-level
//! summary.

use crate::mle::{AbstractMle, MleIndex};
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};
use std::{cmp::max, hash::Hash, ops::{Add, Mul, Neg, Sub}};

use anyhow::{Ok, Result};

/// mid-term solution for deduplication of MleRefs
/// basically a wrapper around usize, which denotes the index
/// of the MleRef in an expression's MleRef list/// Generic Expressions
///
/// TODO(ryancao): We should deprecate this and instead just have
/// references to the `DenseMLE<F>`s which are stored in the circuit_map.
#[derive(Serialize, Deserialize, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MleVecIndex(usize);

impl MleVecIndex {
    /// create a new MleRefIndex
    pub fn new(index: usize) -> Self {
        MleVecIndex(index)
    }

    /// returns the index
    pub fn index(&self) -> usize {
        self.0
    }

    /// add the index with an increment amount
    pub fn increment(&mut self, offset: usize) {
        self.0 += offset;
    }

    /// return the actual mle in the vec within the prover expression
    pub fn get_mle<'a, F: Field, M: AbstractMle<F>>(&self, mle_vec: &'a [M]) -> &'a M {
        &mle_vec[self.0]
    }

    /// return the actual mle in the vec within the prover expression
    pub fn get_mle_mut<'a, F: Field, M: AbstractMle<F>>(&self, mle_vec: &'a mut [M]) -> &'a mut M {
        &mut mle_vec[self.0]
    }
}

/// [ExpressionNode] can be made up of the following:
/// * [ExpressionNode::Constant], i.e. + c for c \in \mathbb{F}
/// * [ExpressionNode::Mle], i.e. \widetilde{V}_{j > i}(b_1, ..., b_{m \leq n})
/// * [ExpressionNode::Product], i.e. \prod_j \widetilde{V}_{j > i}(b_1, ..., b_{m \leq n})
/// * [ExpressionNode::Selector], i.e. (1 - b_0) * Expr(b_1, ..., b_{m \leq n}) + b_0 * Expr(b_1, ..., b_{m \leq n})
/// * [ExpressionNode::Sum], i.e. \widetilde{V}_{j_1 > i}(b_1, ..., b_{m_1 \leq n}) + \widetilde{V}_{j_2 > i}(b_1, ..., b_{m_2 \leq n})
/// * [ExpressionNode::Scaled], i.e. c * Expr(b_1, ..., b_{m \leq n}) for c \in mathbb{F}
#[derive(Serialize, Deserialize, Clone, PartialEq, Hash, Eq)]
#[serde(bound = "F: Field")]
pub enum ExpressionNode<F: Field> {
    /// See documentation for [ExpressionNode]. Note that
    /// [ExpressionNode::Constant] can be an expression tree's leaf.
    Constant(F),
    /// See documentation for [ExpressionNode].
    Selector(
        MleIndex<F>,
        Box<ExpressionNode<F>>,
        Box<ExpressionNode<F>>,
    ),
    /// An [ExpressionNode] representing the leaf of an expression tree which
    /// points to a multilinear extension.
    Mle(MleVecIndex),
    /// See documentation for [ExpressionNode].
    Sum(Box<ExpressionNode<F>>, Box<ExpressionNode<F>>),
    /// The product of several multilinear extension functions. This is also
    /// an expression tree's leaf.
    Product(Vec<MleVecIndex>),
    /// See documentation for [ExpressionNode].
    Scaled(Box<ExpressionNode<F>>, F),
}

/// The high-level idea is that an [Expression] is generic over an MLE type
/// , and contains within it a single parent [ExpressionNode] as well as an
/// [Vec<AbstractMle<F>>] containing the unique leaf representations for the
/// leaves of the [ExpressionNode] tree.
#[derive(Serialize, Deserialize, Clone, Hash)]
#[serde(bound = "F: Field")]
pub struct Expression<F: Field, M: AbstractMle<F>> {
    /// The root of the expression "tree".
    pub expression_node: ExpressionNode<F>,
    /// The unique owned copies of all MLEs which are "leaves" within the
    /// expression "tree".
    pub mle_vec: Vec<M>,
}

/// generic methods shared across all types of expressions
impl<F: Field, M: AbstractMle<F>> Expression<F, M> {
    /// Create a new expression
    pub fn new(expression_node: ExpressionNode<F>, mle_vec: Vec<M>) -> Self {
        Self {
            expression_node,
            mle_vec,
        }
    }

    /// Returns a reference to the internal `expression_node` and `mle_vec` fields.
    pub fn deconstruct_ref(&self) -> (&ExpressionNode<F>, &[M]) {
        (&self.expression_node, &self.mle_vec)
    }

    /// Returns a mutable reference to the `expression_node` and `mle_vec`
    /// present within the given [Expression].
    pub fn deconstruct_mut(&mut self) -> (&mut ExpressionNode<F>, &mut [M]) {
        (&mut self.expression_node, &mut self.mle_vec)
    }

    /// Takes ownership of the [Expression] and returns the owned values to its
    /// internal `expression_node` and `mle_vec`.
    pub fn deconstruct(self) -> (ExpressionNode<F>, Vec<M>) {
        (self.expression_node, self.mle_vec)
    }

    /// traverse the expression tree, and applies the observer_fn to all child node
    /// because the expression node has the recursive structure, the traverse_node
    /// helper function is implemented on it, with the mle_vec reference passed in
    pub fn traverse(
        &self,
        observer_fn: &mut impl FnMut(&ExpressionNode<F>, &[M]) -> Result<()>,
    ) -> Result<()> {
        self.expression_node
            .traverse_node(observer_fn, &self.mle_vec)
    }

    /// similar to traverse, but allows mutation of self (expression node and mle_vec)
    pub fn traverse_mut(
        &mut self,
        observer_fn: &mut impl FnMut(&mut ExpressionNode<F>) -> Result<()>,
    ) -> Result<()> {
        self.expression_node
            .traverse_node_mut(observer_fn)
    }

    /// returns the number of MleRefs in the expression
    pub fn num_mle(&self) -> usize {
        self.mle_vec.len()
    }

    /// increments all the MleVecIndex in the expression by *param* amount
    pub fn increment_mle_vec_indices(&mut self, offset: usize) {
        // define a closure that increments the MleVecIndex by the given amount
        // use traverse_mut
        let mut increment_closure = |expr: &mut ExpressionNode<F>|
         -> Result<()> {
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
                | ExpressionNode::Selector(_, _, _) => Ok(()),
            }
        };

        self.traverse_mut(&mut increment_closure).unwrap();
    }

    /// Returns the total number of variables (i.e. number of rounds of sumcheck)
    /// within the MLE representing the output "data" of this particular expression.
    ///
    /// Note that unlike within the AbstractExpr case, we don't need to return
    /// a `Result` since all MLEs within a `ExprDescription` are instantiated with their appropriate number of variables.
    pub fn num_vars(&self) -> usize {
        self.expression_node.get_num_vars(&self.mle_vec)
    }

    /// Traverses the expression tree to return all indices within the
    /// expression. Can only be used after indexing the expression.
    pub fn get_all_rounds(&self) -> Vec<usize> {
        let (expression_node, mle_vec) = self.deconstruct_ref();
        let mut all_rounds = expression_node.get_all_rounds(mle_vec);
        all_rounds.sort();
        all_rounds
    }

    /// this traverses the expression tree to get all of the nonlinear rounds. can only be used after indexing the expression.
    /// returns the indices sorted.
    pub fn get_all_nonlinear_rounds(&self) -> Vec<usize> {
        let (expression_node, mle_vec) = self.deconstruct_ref();
        let mut nonlinear_rounds = expression_node.get_all_nonlinear_rounds(mle_vec);
        nonlinear_rounds.sort();
        nonlinear_rounds
    }

    /// this traverses the expression tree to get all of the linear rounds. can only be used after indexing the expression.
    /// returns the indices sorted.
    pub fn get_all_linear_rounds(&self) -> Vec<usize> {
        let (expression_node, mle_vec) = self.deconstruct_ref();
        let mut linear_rounds = expression_node.get_all_linear_rounds(mle_vec);
        linear_rounds.sort();
        linear_rounds
    }

    /// Get the maximum degree of any variable in htis expression
    pub fn get_max_degree(&self) -> usize {
        self.expression_node.get_max_degree(&self.mle_vec)
    }

    /// Returns the maximum degree of b_{curr_round} within an expression
    /// (and therefore the number of prover messages we need to send)
    pub fn get_round_degree(&self, curr_round: usize) -> usize {
        let (expression_node, mle_vec) = self.deconstruct_ref();
        let max_var_degree = *expression_node.get_rounds_helper(mle_vec).get(curr_round).unwrap_or(&1);
        max_var_degree + 1 // for eq
    }
}

/// Generic helper methods shared across all types of [ExpressionNode]s.
impl<F: Field> ExpressionNode<F> {
    /// traverse the expression tree, and applies the observer_fn to all child node / the mle_vec reference
    pub fn traverse_node<M: AbstractMle<F>>(
        &self,
        observer_fn: &mut impl FnMut(&ExpressionNode<F>, &[M]) -> Result<()>,
        mle_vec: &[M],
    ) -> Result<()> {
        observer_fn(self, mle_vec)?;
        match self {
            ExpressionNode::Constant(_) | ExpressionNode::Mle(_) | ExpressionNode::Product(_) => {
                Ok(())
            }
            ExpressionNode::Scaled(exp, _) => exp.traverse_node(observer_fn, mle_vec),
            ExpressionNode::Selector(_, lhs, rhs) => {
                lhs.traverse_node(observer_fn, mle_vec)?;
                rhs.traverse_node(observer_fn, mle_vec)
            }
            ExpressionNode::Sum(lhs, rhs) => {
                lhs.traverse_node(observer_fn, mle_vec)?;
                rhs.traverse_node(observer_fn, mle_vec)
            }
        }
    }

    /// similar to traverse, but allows mutation of self (expression node and mle_vec)
    pub fn traverse_node_mut(
        &mut self,
        observer_fn: &mut impl FnMut(&mut ExpressionNode<F>) -> Result<()>
    ) -> Result<()> {
        observer_fn(self)?;
        match self {
            ExpressionNode::Constant(_) | ExpressionNode::Mle(_) | ExpressionNode::Product(_) => {
                Ok(())
            }
            ExpressionNode::Scaled(exp, _) => exp.traverse_node_mut(observer_fn),
            ExpressionNode::Selector(_, lhs, rhs) => {
                lhs.traverse_node_mut(observer_fn)?;
                rhs.traverse_node_mut(observer_fn)
            }
            ExpressionNode::Sum(lhs, rhs) => {
                lhs.traverse_node_mut(observer_fn)?;
                rhs.traverse_node_mut(observer_fn)
            }
        }
    }

    /// Evaluate the polynomial using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce<M: AbstractMle<F>, T>(
        &self,
        mle_vec: &[M],
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(&MleIndex<F>, T, T) -> T,
        mle_eval: &impl Fn(&M) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(&[&M]) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T {
        match self {
            ExpressionNode::Constant(scalar) => constant(*scalar),
            ExpressionNode::Selector(index, a, b) => {
                let lhs = a.reduce(mle_vec, constant, selector_column, mle_eval, sum, product, scaled);
                let rhs = b.reduce(mle_vec, constant, selector_column, mle_eval, sum, product, scaled);
                selector_column(index, lhs, rhs)
            }
            ExpressionNode::Mle(query) => mle_eval(query.get_mle(mle_vec)),
            ExpressionNode::Sum(a, b) => {
                let a = a.reduce(mle_vec, constant, selector_column, mle_eval, sum, product, scaled);
                let b = b.reduce(mle_vec, constant, selector_column, mle_eval, sum, product, scaled);
                sum(a, b)
            }
            ExpressionNode::Product(queries) => product(&queries.iter().map(|q| q.get_mle(mle_vec)).collect::<Vec<_>>()),
            ExpressionNode::Scaled(a, f) => {
                let a = a.reduce(mle_vec, constant, selector_column, mle_eval, sum, product, scaled);
                scaled(a, *f)
            }
        }
    }

    /// this traverses the expression to get all of the rounds, in total. requires going through each of the nodes
    /// and collecting the leaf node indices.
    pub(crate) fn get_all_rounds<M: AbstractMle<F>>(
        &self,
        mle_vec: &[M],
    ) -> Vec<usize> {
        let degree_per_index = self.get_rounds_helper(mle_vec);
        (0..degree_per_index.len())
            .filter(|&i| degree_per_index[i] > 0)
            .collect()
    }

    /// traverse an expression tree in order to get all of the nonlinear rounds in an expression.
    pub fn get_all_nonlinear_rounds<M: AbstractMle<F>>(
        &self,
        mle_vec: &[M],
    ) -> Vec<usize> {
        let degree_per_index = self.get_rounds_helper(mle_vec);
        (0..degree_per_index.len())
            .filter(|&i| degree_per_index[i] > 1)
            .collect()
    }

    /// get all of the linear rounds from an expression tree
    pub fn get_all_linear_rounds<M: AbstractMle<F>>(
        &self,
        mle_vec: &[M],
    ) -> Vec<usize> {
        let degree_per_index = self.get_rounds_helper(mle_vec);
        (0..degree_per_index.len())
            .filter(|&i| degree_per_index[i] == 1)
            .collect()
    }

    /// Get the maximum degree of an ExpressionNode, recursively.
    pub fn get_max_degree<M: AbstractMle<F>>(&self, mle_vec: &[M]) -> usize {
        self.get_rounds_helper(mle_vec).into_iter().max().unwrap()
    }

    // a recursive helper for get_all_rounds, get_all_nonlinear_rounds, and get_all_linear_rounds
    fn get_rounds_helper<M: AbstractMle<F>>(&self, mle_vec: &[M]) -> Vec<usize> {
        // degree of each index
        let mut degree_per_index = Vec::new();
        // set the degree of the corresponding index to max(OLD_DEGREE, NEW_DEGREE)
        let max_degree = |degree_per_index: &mut Vec<usize>, index: usize, new_degree: usize| {
            if degree_per_index.len() <= index {
                degree_per_index.extend(vec![0; index + 1 - degree_per_index.len()]);
            }
            if degree_per_index[index] < new_degree {
                degree_per_index[index] = new_degree;
            }
        };
        // set the degree of the corresponding index to OLD_DEGREE + NEW_DEGREE
        let add_degree = |degree_per_index: &mut Vec<usize>, index: usize, new_degree: usize| {
            if degree_per_index.len() <= index {
                degree_per_index.extend(vec![0; index + 1 - degree_per_index.len()]);
            }
            degree_per_index[index] += new_degree;
        };

        match self {
            // in a product, we need the union of all the indices in each of the individual mle refs.
            ExpressionNode::Product(mle_vec_indices) => {
                let mles = mle_vec_indices
                    .iter()
                    .map(|mle_vec_index| mle_vec_index.get_mle(mle_vec))
                    .collect::<Vec<_>>();
                mles.into_iter().for_each(|mle| {
                    mle.mle_indices().iter().for_each(|mle_index| {
                        if let MleIndex::Indexed(i) = mle_index {
                            add_degree(&mut degree_per_index, *i, 1);
                        }
                    })
                });
            }
            // in an mle, we need all of the mle indices in the mle.
            ExpressionNode::Mle(mle_vec_idx) => {
                let mle = mle_vec_idx.get_mle(mle_vec);
                mle.mle_indices().iter().for_each(|mle_index| {
                    if let MleIndex::Indexed(i) = mle_index {
                        max_degree(&mut degree_per_index, *i, 1);
                    }
                });
            }
            // in selector, take the max degree of each children, and add 1 degree to the selector itself
            ExpressionNode::Selector(sel_index, a, b) => {
                if let MleIndex::Indexed(i) = sel_index {
                    add_degree(&mut degree_per_index, *i, 1);
                };
                let a_degree_per_index = a.get_rounds_helper(mle_vec);
                let b_degree_per_index = b.get_rounds_helper(mle_vec);
                // linear operator -- take the max degree
                for i in 0..max(a_degree_per_index.len(), b_degree_per_index.len()) {
                    if let Some(a_degree) = a_degree_per_index.get(i) {
                        max_degree(&mut degree_per_index, i, *a_degree);
                    }
                    if let Some(b_degree) = b_degree_per_index.get(i) {
                        max_degree(&mut degree_per_index, i, *b_degree);
                    }
                }
            }
            // in sum, take the max degree of each children
            ExpressionNode::Sum(a, b) => {
                let a_degree_per_index = a.get_rounds_helper(mle_vec);
                let b_degree_per_index = b.get_rounds_helper(mle_vec);
                // linear operator -- take the max degree
                for i in 0..max(a_degree_per_index.len(), b_degree_per_index.len()) {
                    if let Some(a_degree) = a_degree_per_index.get(i) {
                        max_degree(&mut degree_per_index, i, *a_degree);
                    }
                    if let Some(b_degree) = b_degree_per_index.get(i) {
                        max_degree(&mut degree_per_index, i, *b_degree);
                    }
                }
            }
            // scaled and negated, does not affect degree
            ExpressionNode::Scaled(a, _) => {
                degree_per_index = a.get_rounds_helper(mle_vec);
            }
            // for a constant there are no new indices.
            ExpressionNode::Constant(_) => {}
        }
        degree_per_index
    }

    /// Returns the total number of variables (i.e. number of rounds of sumcheck) within
    /// the MLE representing the output "data" of this particular expression.
    ///
    /// Note that unlike within the `AbstractExpression` case, we don't need to return
    /// a `Result` since all MLEs extentiating `AbstractMle` are instantiated with their
    /// appropriate number of variables.
    pub fn get_num_vars<M: AbstractMle<F>>(&self, mle_vec: &[M]) -> usize {
        match self {
            ExpressionNode::Constant(_) => 0,
            ExpressionNode::Selector(_, lhs, rhs) => {
                max(lhs.get_num_vars(mle_vec) + 1, rhs.get_num_vars(mle_vec) + 1)
            }
            ExpressionNode::Mle(circuit_mle_desc) => circuit_mle_desc.get_mle(mle_vec).num_free_vars(),
            ExpressionNode::Sum(lhs, rhs) => max(lhs.get_num_vars(mle_vec), rhs.get_num_vars(mle_vec)),
            ExpressionNode::Product(nodes) => nodes.iter().fold(0, |cur_max, circuit_mle_desc| {
                max(cur_max, circuit_mle_desc.get_mle(mle_vec).num_free_vars())
            }),
            ExpressionNode::Scaled(expr, _) => expr.get_num_vars(mle_vec),
        }
    }
}

// defines how the Expressions are printed and displayed
impl<F: std::fmt::Debug + Field, M: std::fmt::Debug + AbstractMle<F>> std::fmt::Debug for Expression<F, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Expression")
            .field("Expression_Node", &self.expression_node)
            .field("MleRef_Vec", &self.mle_vec)
            .finish()
    }
}

// defines how the ExpressionNodes are printed and displayed
impl<F: std::fmt::Debug + Field> std::fmt::Debug for ExpressionNode<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpressionNode::Constant(scalar) => f.debug_tuple("Constant").field(scalar).finish(),
            ExpressionNode::Selector(index, a, b) => f
                .debug_tuple("Selector")
                .field(index)
                .field(a)
                .field(b)
                .finish(),
            // Skip enum variant and print query struct directly to maintain backwards compatibility.
            ExpressionNode::Mle(_mle) => f.debug_struct("Mle").field("mle", _mle).finish(),
            ExpressionNode::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            ExpressionNode::Product(a) => f.debug_tuple("Product").field(a).finish(),
            ExpressionNode::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}

// constructors and operators on generic MLEs
impl<F: Field, M: AbstractMle<F>> Expression<F, M> {
    /// create a constant Expression that contains one field element
    pub fn constant(constant: F) -> Self {
        let mle_node = ExpressionNode::Constant(constant);
        Expression::new(mle_node, [].to_vec())
    }

    /// create an MLE expression that contains one MLE
    pub fn mle(mle: M) -> Self {
        let mle_node = ExpressionNode::Mle(MleVecIndex::new(0));
        Expression::new(mle_node, [mle].to_vec())
    }

    /// negates an expression
    pub fn negated(expression: Self) -> Self {
        let (node, mle_vec) = expression.deconstruct();

        let mle_node = ExpressionNode::Scaled(Box::new(node), F::from(1).neg());

        Expression::new(mle_node, mle_vec)
    }

    /// scales an expression by a field element
    pub fn scaled(expression: Self, scale: F) -> Self {
        let (node, mle_vec) = expression.deconstruct();

        Expression::new(ExpressionNode::Scaled(Box::new(node), scale), mle_vec)
    }

    /// create a sum expression that contains two MLEs
    pub fn sum(lhs: Self, mut rhs: Self) -> Self {
        let offset = lhs.num_mle();
        rhs.increment_mle_vec_indices(offset);

        let (lhs_node, lhs_mle_vec) = lhs.deconstruct();
        let (rhs_node, rhs_mle_vec) = rhs.deconstruct();

        let sum_node = ExpressionNode::Sum(Box::new(lhs_node), Box::new(rhs_node));
        let sum_mle_vec = lhs_mle_vec.into_iter().chain(rhs_mle_vec).collect::<Vec<_>>();

        Expression::new(sum_node, sum_mle_vec)
    }

    /// create a product expression that multiplies many MLEs together
    pub fn products(product_list: Vec<M>) -> Self {
        let mle_vec_indices = (0..product_list.len()).map(MleVecIndex::new).collect::<Vec<_>>();

        let product_node = ExpressionNode::Product(mle_vec_indices);

        Expression::new(product_node, product_list)
    }

    /// create a select expression without reason about index changes
    pub fn select_with_index(index: MleIndex<F>, lhs: Self, mut rhs: Self) -> Self {
        let offset = lhs.num_mle();
        rhs.increment_mle_vec_indices(offset);
        let (lhs_node, lhs_mle_vec) = lhs.deconstruct();
        let (rhs_node, rhs_mle_vec) = rhs.deconstruct();

        let concat_node =
            ExpressionNode::Selector(index, Box::new(lhs_node), Box::new(rhs_node));

        let concat_mle_vec = lhs_mle_vec.into_iter().chain(rhs_mle_vec).collect::<Vec<_>>();

        Expression::new(concat_node, concat_mle_vec)
    }
}

impl<F: Field, M: AbstractMle<F>> From<F> for Expression<F, M> {
    fn from(f: F) -> Self {
        Expression::<F, M>::constant(f)
    }
}

impl<F: Field, M: AbstractMle<F>> Neg for Expression<F, M> {
    type Output = Expression<F, M>;
    fn neg(self) -> Self::Output {
        Expression::<F, M>::negated(self)
    }
}

/// implement the Add, Sub, and Mul traits for the Expression
impl<F: Field, M: AbstractMle<F>> Add for Expression<F, M> {
    type Output = Expression<F, M>;
    fn add(self, rhs: Expression<F, M>) -> Expression<F, M> {
        Expression::<F, M>::sum(self, rhs)
    }
}
impl<F: Field, M: AbstractMle<F>> Add<F> for Expression<F, M> {
    type Output = Expression<F, M>;
    fn add(self, rhs: F) -> Expression<F, M> {
        let rhs_expr = Expression::<F, M>::constant(rhs);
        Expression::<F, M>::sum(self, rhs_expr)
    }
}

impl<F: Field, M: AbstractMle<F>> Sub for Expression<F, M> {
    type Output = Expression<F, M>;
    fn sub(self, rhs: Expression<F, M>) -> Expression<F, M> {
        self.add(rhs.neg())
    }
}
impl<F: Field, M: AbstractMle<F>> Sub<F> for Expression<F, M> {
    type Output = Expression<F, M>;
    fn sub(self, rhs: F) -> Expression<F, M> {
        let rhs_expr = Expression::<F, M>::constant(rhs);
        Expression::<F, M>::sum(self, rhs_expr.neg())
    }
}

impl<F: Field, M: AbstractMle<F>> Mul<F> for Expression<F, M> {
    type Output = Expression<F, M>;
    fn mul(self, rhs: F) -> Self::Output {
        Expression::<F, M>::scaled(self, rhs)
    }
}