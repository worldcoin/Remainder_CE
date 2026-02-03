//! The "circuit-builder's" view of an expression. In particular, it represents
//! a template of polynomial relationships between an output computational
//! graph node and outputs from source computational graph nodes.
//!
//! WARNING: because [AbstractExpression] does *not* contain any semblance of MLE
//! sizes nor indices, it can thus represent an entire class of polynomial
//! relationships, depending on its circuit-time instantiation. For example,
//! the simple relationship of
//!
//! ```ignore
//!     node_id_1 + node_id_2
//! ```
//!
//! can refer to \widetilde{V}_{i}(x_1, ..., x_m) + \widetilde{V}_{j}(x_1, ..., x_n)
//! where:
//! * m > n, i.e. the second MLE's "data" is "wrapped around" via repetition
//! * m = n, i.e. the resulting bookkeeping table is the element-wise sum of the two
//! * m < n, i.e. the first MLE's "data" is "wrapped around" via repetition

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    collections::HashMap,
    ops::{Add, AddAssign, BitXor, Mul, MulAssign, Neg, Sub, SubAssign},
};

use shared_types::Field;

use crate::layouter::{builder::CircuitMap, layouting::LayoutingError, nodes::NodeId};
use remainder::{circuit_layout::CircuitLocation, expression::generic_expr::ExpressionNode};

use remainder::{
    expression::{circuit_expr::ExprDescription, generic_expr::Expression},
    mle::{mle_description::MleDescription, MleIndex},
    utils::mle::get_total_mle_indices,
};

use anyhow::{Ok, Result};

/// See [ExpressionNode] for more details. Note that this implementation is
/// somewhat redundant with [Expression] and [ExpressionNode], but the
/// separation allows for more flexibility with respect to this particular
/// frontend being able to create polynomial relationships in any way it
/// chooses, so long as those representations are compile-able into [Expression].
#[derive(Serialize, Deserialize, Clone, PartialEq, Hash, Eq)]
#[serde(bound = "F: Field")]
pub enum AbstractExpression<F: Field> {
    Constant(F),
    Selector(
        MleIndex<F>,
        Box<AbstractExpression<F>>,
        Box<AbstractExpression<F>>,
    ),
    Mle(NodeId),
    Sum(Box<AbstractExpression<F>>, Box<AbstractExpression<F>>),
    Product(Vec<NodeId>),
    Scaled(Box<AbstractExpression<F>>, F),
}

//  comments for Phase II:
//  This will be the the circuit "pre-data" stage
//  will take care of building a prover expression
//  building the most memory efficient denseMleRefs dictionaries, etc.
impl<F: Field> AbstractExpression<F> {
    /// Traverses the expression and applies the observer function to all nodes.
    pub fn traverse(
        &self,
        observer_fn: &mut impl FnMut(&AbstractExpression<F>) -> Result<()>,
    ) -> Result<()> {
        observer_fn(self)?;
        match self {
            AbstractExpression::Constant(_)
            | AbstractExpression::Mle(_)
            | AbstractExpression::Product(_) => Ok(()),
            AbstractExpression::Scaled(exp, _) => exp.traverse(observer_fn),
            AbstractExpression::Selector(_, lhs, rhs) => {
                lhs.traverse(observer_fn)?;
                rhs.traverse(observer_fn)
            }
            AbstractExpression::Sum(lhs, rhs) => {
                lhs.traverse(observer_fn)?;
                rhs.traverse(observer_fn)
            }
        }
    }

    /// find all the sources this expression depend on
    pub fn get_sources(&self) -> Vec<NodeId> {
        let mut sources = vec![];
        let mut get_sources_closure = |expr_node: &AbstractExpression<F>| -> Result<()> {
            if let AbstractExpression::Product(node_id_vec) = expr_node {
                sources.extend(node_id_vec.iter());
            } else if let AbstractExpression::Mle(node_id) = expr_node {
                sources.push(*node_id);
            }
            Ok(())
        };
        self.traverse(&mut get_sources_closure).unwrap();
        sources
    }

    /// Computes the num_vars of this expression (how many rounds of sumcheck it would take to prove)
    pub fn get_num_vars(&self, num_vars_map: &HashMap<NodeId, usize>) -> Result<usize> {
        match self {
            AbstractExpression::Constant(_) => Ok(0),
            AbstractExpression::Selector(_, lhs, rhs) => Ok(max(
                lhs.get_num_vars(num_vars_map)? + 1,
                rhs.get_num_vars(num_vars_map)? + 1,
            )),
            AbstractExpression::Mle(node_id) => Ok(*num_vars_map.get(node_id).unwrap()),
            AbstractExpression::Sum(lhs, rhs) => Ok(max(
                lhs.get_num_vars(num_vars_map)?,
                rhs.get_num_vars(num_vars_map)?,
            )),
            AbstractExpression::Product(nodes) => Ok(nodes
                .iter()
                .map(|node_id| Ok(Some(*num_vars_map.get(node_id).unwrap())))
                .fold_ok(None, max)?
                .unwrap_or(0)),
            AbstractExpression::Scaled(expr, _) => expr.get_num_vars(num_vars_map),
        }
    }

    /// Convert the abstract expression into a circuit expression, which
    /// stores information on the shape of the expression, using the
    /// [CircuitMap].
    pub fn build_circuit_expr(
        self,
        circuit_map: &CircuitMap,
    ) -> Result<Expression<F, ExprDescription>> {
        // First we get all the mles that this expression will need to store
        let mut nodes = self.get_node_ids(vec![]);
        nodes.sort();
        nodes.dedup();

        let mut node_map = HashMap::<NodeId, (usize, &CircuitLocation)>::new();

        nodes.into_iter().for_each(|node_id| {
            let (location, num_vars) = circuit_map
                .get_location_num_vars_from_node_id(&node_id)
                .unwrap();
            node_map.insert(node_id, (*num_vars, location));
        });

        // Then we replace the NodeIds in the AbstractExpr w/ indices of our stored MLEs

        let expression_node = self.build_circuit_node(&node_map)?;

        Ok(Expression::new(expression_node, ()))
    }

    /// See documentation for `select()` function within [remainder::expression::circuit_expr::ExprDescription]
    pub fn select(self, rhs: Self) -> Self {
        Self::Selector(MleIndex::Free, Box::new(self), Box::new(rhs))
    }

    /// Call [Self::select] sequentially
    pub fn select_seq<E: Clone + Into<AbstractExpression<F>>>(expressions: Vec<E>) -> Self {
        let mut base = expressions[0].clone().into();
        for e in expressions.into_iter().skip(1) {
            base = Self::select(base, e.into());
        }
        base
    }

    /// Create a nested selector Expression that selects between 2^k Expressions
    /// by creating a binary tree of Selector Expressions.
    /// The order of the leaves is the order of the input expressions.
    /// (Note that this is very different from [Self::select_seq].)
    pub fn binary_tree_selector<E: Into<AbstractExpression<F>>>(expressions: Vec<E>) -> Self {
        // Ensure length is a power of two
        assert!(expressions.len().is_power_of_two());
        let mut expressions = expressions
            .into_iter()
            .map(|e| e.into())
            .collect::<Vec<_>>();
        while expressions.len() > 1 {
            // Iterate over consecutive pairs of expressions, creating a new expression that selects between them
            expressions = expressions
                .into_iter()
                .tuples()
                .map(|(lhs, rhs)| Self::Selector(MleIndex::Free, Box::new(lhs), Box::new(rhs)))
                .collect();
        }
        expressions[0].clone()
    }

    /// Create a product Expression that raises one expression to a given power
    pub fn pow(pow: usize, node_id: Self) -> Self {
        // lazily construct a linear-depth expression tree
        let base = node_id;
        let mut result = base.clone();
        for _ in 1..pow {
            result *= base.clone();
        }
        result
    }

    /// Create a product Expression that multiplies many MLEs together
    pub fn products(node_ids: Vec<NodeId>) -> Self {
        Self::Product(node_ids)
    }

    /// Multiplication for expressions, DO NOT USE ON SELECTORS
    pub fn mult(lhs: Self, rhs: Self) -> Self {
        let switch = |lhs, rhs| Self::mult(rhs, lhs);

        // Simplify the expression into scaled and products
        match (&lhs, &rhs) {
            // Case 1: const() * X => scaled(X, const()) 
            (AbstractExpression::Constant(f), _) => AbstractExpression::Scaled(Box::new(rhs), *f),
            (_, AbstractExpression::Constant(_)) => switch(lhs, rhs),

            // Case 2: sel() * X => KILL (not allowed)
            (AbstractExpression::Selector(..), _) => panic!("Multiplying a non-constant with a selector is not allowed! Create a separate sector or fold the operand into each branch!"),
            (_, AbstractExpression::Selector(..)) => switch(lhs, rhs),

            // Case 3: add(X, Y) * Z => add(X * Z, Y * Z)
            (AbstractExpression::Sum(x, y), _) => {
                let xr = Self::mult(*x.clone(), rhs.clone());
                let yr = Self::mult(*y.clone(), rhs);
                AbstractExpression::Sum(Box::new(xr), Box::new(yr))
            }
            (_, AbstractExpression::Sum(..)) => switch(lhs, rhs),

            // Case 4: scaled(X, c1) * scaled(Y, c2) => scaled(X * Y, c1 * c2); scaled(X, c) * Z => scaled(X * Z, c)
            (AbstractExpression::Scaled(x, c1), AbstractExpression::Scaled(y, c2)) => {
                let xy = Self::mult(*x.clone(), *y.clone());
                let c = *c1 * *c2;
                AbstractExpression::Scaled(Box::new(xy), c)
            }
            (AbstractExpression::Scaled(x, c), _) => {
                let xz = Self::mult(*x.clone(), rhs);
                AbstractExpression::Scaled(Box::new(xz), *c)
            }
            (_, AbstractExpression::Scaled(..)) => switch(lhs, rhs),

            // Case 5: mle() * mle(); prod() * prod()
            (l, r) => {
                let l_ids = match l {
                    AbstractExpression::Mle(id) => vec![*id],
                    AbstractExpression::Product(ids) => ids.clone(),
                    _ => unreachable!()
                };
                let r_ids = match r {
                    AbstractExpression::Mle(id) => vec![*id],
                    AbstractExpression::Product(ids) => ids.clone(),
                    _ => unreachable!()
                };
                let ids = [l_ids, r_ids].concat();
                AbstractExpression::Product(ids)
            }
        }
    }

    /// Create a mle Expression that contains one MLE
    pub fn mle(node_id: NodeId) -> Self {
        AbstractExpression::Mle(node_id)
    }

    /// Create a constant Expression that contains one field element
    pub fn constant(constant: F) -> Self {
        AbstractExpression::Constant(constant)
    }

    /// negates an Expression
    pub fn negated(expression: Self) -> Self {
        AbstractExpression::Scaled(Box::new(expression), F::from(1).neg())
    }

    /// Create a Sum Expression that contains two MLEs
    pub fn sum(lhs: Self, rhs: Self) -> Self {
        AbstractExpression::Sum(Box::new(lhs), Box::new(rhs))
    }

    /// scales an Expression by a field element
    pub fn scaled(expression: AbstractExpression<F>, scale: F) -> Self {
        AbstractExpression::Scaled(Box::new(expression), scale)
    }
}

impl<F: Field> AbstractExpression<F> {
    fn build_circuit_node(
        self,
        node_map: &HashMap<NodeId, (usize, &CircuitLocation)>,
    ) -> Result<ExpressionNode<F, ExprDescription>> {
        // Note that the node_map is the map of node_ids to the internal vec of MLEs, not the circuit_map
        match self {
            AbstractExpression::Constant(val) => Ok(ExpressionNode::Constant(val)),
            AbstractExpression::Selector(mle_index, lhs, rhs) => {
                let lhs = lhs.build_circuit_node(node_map)?;
                let rhs = rhs.build_circuit_node(node_map)?;
                Ok(ExpressionNode::Selector(
                    mle_index,
                    Box::new(lhs),
                    Box::new(rhs),
                ))
            }
            AbstractExpression::Mle(node_id) => {
                let (
                    num_vars,
                    CircuitLocation {
                        prefix_bits,
                        layer_id,
                    },
                ) = node_map
                    .get(&node_id)
                    .ok_or(LayoutingError::DanglingNodeId(node_id))?;
                let total_indices = get_total_mle_indices(prefix_bits, *num_vars);
                let circuit_mle = MleDescription::new(*layer_id, &total_indices);
                Ok(ExpressionNode::Mle(circuit_mle))
            }
            AbstractExpression::Sum(lhs, rhs) => {
                let lhs = lhs.build_circuit_node(node_map)?;
                let rhs = rhs.build_circuit_node(node_map)?;
                Ok(ExpressionNode::Sum(Box::new(lhs), Box::new(rhs)))
            }
            AbstractExpression::Product(nodes) => {
                let circuit_mles = nodes
                    .into_iter()
                    .map(|node_id| {
                        let (
                            num_vars,
                            CircuitLocation {
                                prefix_bits,
                                layer_id,
                            },
                        ) = node_map
                            .get(&node_id)
                            .ok_or(LayoutingError::DanglingNodeId(node_id))
                            .unwrap();
                        let total_indices = get_total_mle_indices::<F>(prefix_bits, *num_vars);
                        MleDescription::new(*layer_id, &total_indices)
                    })
                    .collect::<Vec<MleDescription<F>>>();
                Ok(ExpressionNode::Product(circuit_mles))
            }
            AbstractExpression::Scaled(expr, scalar) => {
                let expr = expr.build_circuit_node(node_map)?;
                Ok(ExpressionNode::Scaled(Box::new(expr), scalar))
            }
        }
    }

    fn get_node_ids(&self, mut node_ids: Vec<NodeId>) -> Vec<NodeId> {
        match self {
            AbstractExpression::Constant(_) => node_ids,
            AbstractExpression::Selector(_, lhs, rhs) => {
                let node_ids = rhs.get_node_ids(node_ids);
                lhs.get_node_ids(node_ids)
            }
            AbstractExpression::Mle(node_id) => {
                node_ids.push(*node_id);
                node_ids
            }
            AbstractExpression::Sum(lhs, rhs) => {
                let node_ids = lhs.get_node_ids(node_ids);
                rhs.get_node_ids(node_ids)
            }
            AbstractExpression::Product(nodes) => {
                node_ids.extend(nodes.iter());
                node_ids
            }
            AbstractExpression::Scaled(expr, _) => expr.get_node_ids(node_ids),
        }
    }
}

// Additional operators
impl<F: Field> Neg for AbstractExpression<F> {
    type Output = AbstractExpression<F>;
    fn neg(self) -> Self::Output {
        AbstractExpression::<F>::negated(self)
    }
}

impl<F: Field> Neg for &AbstractExpression<F> {
    type Output = AbstractExpression<F>;
    fn neg(self) -> Self::Output {
        AbstractExpression::<F>::negated(self.clone())
    }
}

impl<F: Field> From<F> for AbstractExpression<F> {
    fn from(f: F) -> Self {
        AbstractExpression::<F>::constant(f)
    }
}

/// implement the Add, Sub, and Mul traits for the Expression
impl<F: Field, Rhs: Into<AbstractExpression<F>>> Add<Rhs> for AbstractExpression<F> {
    type Output = AbstractExpression<F>;
    fn add(self, rhs: Rhs) -> Self::Output {
        AbstractExpression::sum(self, rhs.into())
    }
}
impl<F: Field, Rhs: Into<AbstractExpression<F>>> Add<Rhs> for &AbstractExpression<F> {
    type Output = AbstractExpression<F>;
    fn add(self, rhs: Rhs) -> Self::Output {
        AbstractExpression::sum(self.clone(), rhs.into())
    }
}

impl<F: Field, Rhs: Into<AbstractExpression<F>>> AddAssign<Rhs> for AbstractExpression<F> {
    fn add_assign(&mut self, rhs: Rhs) {
        *self = self.clone() + rhs;
    }
}

impl<F: Field, Rhs: Into<AbstractExpression<F>>> Sub<Rhs> for AbstractExpression<F> {
    type Output = AbstractExpression<F>;
    fn sub(self, rhs: Rhs) -> Self::Output {
        AbstractExpression::sum(self, rhs.into().neg())
    }
}
impl<F: Field, Rhs: Into<AbstractExpression<F>>> Sub<Rhs> for &AbstractExpression<F> {
    type Output = AbstractExpression<F>;
    fn sub(self, rhs: Rhs) -> Self::Output {
        AbstractExpression::sum(self.clone(), rhs.into().neg())
    }
}
impl<F: Field, Rhs: Into<AbstractExpression<F>>> SubAssign<Rhs> for AbstractExpression<F> {
    fn sub_assign(&mut self, rhs: Rhs) {
        *self = self.clone() - rhs;
    }
}

impl<F: Field, Rhs: Into<AbstractExpression<F>>> Mul<Rhs> for AbstractExpression<F> {
    type Output = AbstractExpression<F>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        AbstractExpression::mult(self, rhs.into())
    }
}
impl<F: Field, Rhs: Into<AbstractExpression<F>>> Mul<Rhs> for &AbstractExpression<F> {
    type Output = AbstractExpression<F>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        AbstractExpression::mult(self.clone(), rhs.into())
    }
}
impl<F: Field, Rhs: Into<AbstractExpression<F>>> MulAssign<Rhs> for AbstractExpression<F> {
    fn mul_assign(&mut self, rhs: Rhs) {
        *self = self.clone() * rhs;
    }
}

impl<F: Field, Rhs: Into<AbstractExpression<F>>> BitXor<Rhs> for AbstractExpression<F> {
    type Output = AbstractExpression<F>;
    fn bitxor(self, rhs: Rhs) -> Self::Output {
        let rhs_expr: AbstractExpression<F> = rhs.into();
        self.clone() + rhs_expr.clone() - self * rhs_expr * F::from(2)
    }
}
impl<F: Field, Rhs: Into<AbstractExpression<F>>> BitXor<Rhs> for &AbstractExpression<F> {
    type Output = AbstractExpression<F>;
    fn bitxor(self, rhs: Rhs) -> Self::Output {
        let rhs_expr: &AbstractExpression<F> = &rhs.into();
        self.clone() + rhs_expr.clone() - self.clone() * rhs_expr * F::from(2)
    }
}

impl<F: Field> From<&AbstractExpression<F>> for AbstractExpression<F> {
    fn from(val: &AbstractExpression<F>) -> Self {
        val.clone()
    }
}

/// constant
#[macro_export]
macro_rules! const_expr {
    ($val:expr) => {{
        use frontend::abstract_expr::AbstractExpression;
        AbstractExpression::Constant($val)
    }};
}

/// selector
/// equivalent of calling `AbstractExpression::<F>::select_seq(vec![<INPUTS>])`
/// but allows the entries to be of different type
#[macro_export]
macro_rules! sel_expr {
    ($($expr:expr),+ $(,)?) => {{
        use frontend::abstract_expr::{AbstractExpression};
        let v = vec![$(Into::<AbstractExpression<F>>::into($expr)),+];
        AbstractExpression::<F>::select_seq(v)
    }};
}

// defines how the AbstractExpression are printed and displayed
impl<F: std::fmt::Debug + Field> std::fmt::Debug for AbstractExpression<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AbstractExpression::Constant(scalar) => {
                f.debug_tuple("Constant").field(scalar).finish()
            }
            AbstractExpression::Selector(index, a, b) => f
                .debug_tuple("Selector")
                .field(index)
                .field(a)
                .field(b)
                .finish(),
            // Skip enum variant and print query struct directly to maintain backwards compatibility.
            AbstractExpression::Mle(mle) => f.debug_struct("Mle").field("mle", mle).finish(),
            AbstractExpression::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            AbstractExpression::Product(a) => f.debug_tuple("Product").field(a).finish(),
            AbstractExpression::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}
