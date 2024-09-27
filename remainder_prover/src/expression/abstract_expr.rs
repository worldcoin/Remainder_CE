//! The "circuit-builder's" view of an expression. In particular, it represents
//! a template of polynomial relationships between an output computational
//! graph node and outputs from source computational graph nodes (see
//! documentation within [crate::expression] for more details).
//!
//! WARNING: because [AbstractExpr] does *not* contain any semblance of MLE
//! sizes nor indices, it can thus represent an entire class of polynomial
//! relationships, depending on its circuit-time instantiation. For example,
//! the simple relationship of
//!
//! ```ignore
//!     node_id_1.expr() + node_id_2.expr()
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
    fmt::Debug,
    iter::repeat,
    ops::{Add, Mul, Neg, Sub},
};

use remainder_shared_types::Field;

use crate::{
    layouter::{
        layouting::{CircuitDescriptionMap, CircuitLocation, DAGError},
        nodes::NodeId,
    },
    mle::MleIndex,
    utils::mle::get_total_mle_indices,
};

use super::{
    circuit_expr::{CircuitExpr, CircuitMle},
    generic_expr::{Expression, ExpressionNode, ExpressionType},
};

/// Abstract Expression
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AbstractExpr;
impl<F: Field> ExpressionType<F> for AbstractExpr {
    type MLENodeRepr = NodeId;
    type MleVec = ();
}

/// alias for circuit building
pub type ExprBuilder<F> = Expression<F, AbstractExpr>;

//  comments for Phase II:
//  This will be the the circuit "pre-data" stage
//  will take care of building a prover expression
//  building the most memory efficient denseMleRefs dictionaries, etc.
impl<F: Field> Expression<F, AbstractExpr> {
    /// find all the sources this expression depend on
    pub fn get_sources(&self) -> Vec<NodeId> {
        let mut sources = vec![];
        let mut get_sources_closure = |expr_node: &ExpressionNode<F, AbstractExpr>,
                                       _mle_vec: &<AbstractExpr as ExpressionType<F>>::MleVec|
         -> Result<(), ()> {
            if let ExpressionNode::Product(node_id_vec) = expr_node {
                sources.extend(node_id_vec.iter());
            } else if let ExpressionNode::Mle(node_id) = expr_node {
                sources.push(*node_id);
            }
            Ok(())
        };
        self.traverse(&mut get_sources_closure).unwrap();
        sources
    }

    /// Computes the num_vars of this expression (how many rounds of sumcheck it would take to prove)
    pub fn num_vars(&self, num_vars_map: &HashMap<NodeId, usize>) -> Result<usize, DAGError> {
        self.expression_node.get_num_vars(num_vars_map)
    }

    /// Convert the abstract expression into a circuit expression, which
    /// stores information on the shape of the expression, using the
    /// [CircuitDescriptionMap].
    pub fn build_circuit_expr(
        self,
        circuit_description_map: &CircuitDescriptionMap,
    ) -> Result<Expression<F, CircuitExpr>, DAGError> {
        // First we get all the mles that this expression will need to store
        let mut nodes = self.expression_node.get_node_ids(vec![]);
        nodes.sort();
        nodes.dedup();

        let mut node_map = HashMap::<NodeId, (usize, &CircuitLocation)>::new();

        nodes.into_iter().for_each(|node_id| {
            let (location, num_vars) = circuit_description_map
                .get_location_num_vars_from_node_id(&node_id)
                .unwrap();
            node_map.insert(node_id, (*num_vars, location));
        });

        // Then we replace the NodeIds in the AbstractExpr w/ indices of our stored MLEs

        let expression_node = self.expression_node.build_circuit_node(&node_map)?;

        Ok(Expression::new(expression_node, ()))
    }

    /// Concatenates two expressions together
    pub fn concat_expr(self, lhs: Expression<F, AbstractExpr>) -> Self {
        let (lhs_node, _) = lhs.deconstruct();
        let (rhs_node, _) = self.deconstruct();

        let concat_node =
            ExpressionNode::Selector(MleIndex::Free, Box::new(lhs_node), Box::new(rhs_node));

        Expression::new(concat_node, ())
    }

    /// Create a nested selector Expression that selects between 2^k Expressions
    /// by creating a binary tree of Selector Expressions.
    /// The order of the leaves is the order of the input expressions.
    /// (Note that this is very different from calling concat_expr consecutively.)
    /// See also [calculate_selector_values].
    pub fn selectors(expressions: Vec<Self>) -> Self {
        // Ensure length is a power of two
        assert!(expressions.len().is_power_of_two());
        let mut expressions = expressions;
        while expressions.len() > 1 {
            // Iterate over consecutive pairs of expressions, creating a new expression that selects between them
            expressions = expressions
                .into_iter()
                .tuples()
                .map(|(lhs, rhs)| {
                    let (lhs_node, _) = lhs.deconstruct();
                    let (rhs_node, _) = rhs.deconstruct();

                    let selector_node = ExpressionNode::Selector(
                        MleIndex::Free,
                        Box::new(lhs_node),
                        Box::new(rhs_node),
                    );

                    Expression::new(selector_node, ())
                })
                .collect();
        }
        expressions[0].clone()
    }

    /// Create a product Expression that raises one MLE to a given power
    pub fn pow(pow: usize, node_id: NodeId) -> Self {
        let node_ids = repeat(node_id).take(pow).collect();

        let product_node = ExpressionNode::Product(node_ids);

        Expression::new(product_node, ())
    }

    /// Create a product Expression that multiplies many MLEs together
    pub fn products(node_ids: Vec<NodeId>) -> Self {
        let product_node = ExpressionNode::Product(node_ids);

        Expression::new(product_node, ())
    }

    /// Create a mle Expression that contains one MLE
    pub fn mle(node_id: NodeId) -> Self {
        let mle_node = ExpressionNode::Mle(node_id);

        Expression::new(mle_node, ())
    }

    /// Create a constant Expression that contains one field element
    pub fn constant(constant: F) -> Self {
        let mle_node = ExpressionNode::Constant(constant);

        Expression::new(mle_node, ())
    }

    /// negates an Expression
    pub fn negated(expression: Self) -> Self {
        let (node, _) = expression.deconstruct();

        let mle_node = ExpressionNode::Negated(Box::new(node));

        Expression::new(mle_node, ())
    }

    /// Create a Sum Expression that contains two MLEs
    pub fn sum(lhs: Self, rhs: Self) -> Self {
        let (lhs_node, _) = lhs.deconstruct();
        let (rhs_node, _) = rhs.deconstruct();

        let sum_node = ExpressionNode::Sum(Box::new(lhs_node), Box::new(rhs_node));

        Expression::new(sum_node, ())
    }

    /// scales an Expression by a field element
    pub fn scaled(expression: Expression<F, AbstractExpr>, scale: F) -> Self {
        let (node, _) = expression.deconstruct();

        Expression::new(ExpressionNode::Scaled(Box::new(node), scale), ())
    }
}

impl<F: Field> ExpressionNode<F, AbstractExpr> {
    fn build_circuit_node(
        self,
        node_map: &HashMap<NodeId, (usize, &CircuitLocation)>,
    ) -> Result<ExpressionNode<F, CircuitExpr>, DAGError> {
        // Note that the node_map is the map of node_ids to the internal vec of MLEs, not the circuit_map
        match self {
            ExpressionNode::Constant(val) => Ok(ExpressionNode::Constant(val)),
            ExpressionNode::Selector(mle_index, lhs, rhs) => {
                let lhs = lhs.build_circuit_node(node_map)?;
                let rhs = rhs.build_circuit_node(node_map)?;
                Ok(ExpressionNode::Selector(
                    mle_index,
                    Box::new(lhs),
                    Box::new(rhs),
                ))
            }
            ExpressionNode::Mle(node_id) => {
                let (
                    num_vars,
                    CircuitLocation {
                        prefix_bits,
                        layer_id,
                    },
                ) = node_map
                    .get(&node_id)
                    .ok_or(DAGError::DanglingNodeId(node_id))?;
                let total_indices = get_total_mle_indices(prefix_bits, *num_vars);
                let circuit_mle = CircuitMle::new(*layer_id, &total_indices);
                Ok(ExpressionNode::Mle(circuit_mle))
            }
            ExpressionNode::Negated(expr) => Ok(ExpressionNode::Negated(Box::new(
                expr.build_circuit_node(node_map)?,
            ))),
            ExpressionNode::Sum(lhs, rhs) => {
                let lhs = lhs.build_circuit_node(node_map)?;
                let rhs = rhs.build_circuit_node(node_map)?;
                Ok(ExpressionNode::Sum(Box::new(lhs), Box::new(rhs)))
            }
            ExpressionNode::Product(nodes) => {
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
                            .ok_or(DAGError::DanglingNodeId(node_id))
                            .unwrap();
                        let total_indices = get_total_mle_indices::<F>(prefix_bits, *num_vars);
                        CircuitMle::new(*layer_id, &total_indices)
                    })
                    .collect::<Vec<CircuitMle<F>>>();
                Ok(ExpressionNode::Product(circuit_mles))
            }
            ExpressionNode::Scaled(expr, scalar) => {
                let expr = expr.build_circuit_node(node_map)?;
                Ok(ExpressionNode::Scaled(Box::new(expr), scalar))
            }
        }
    }

    fn get_node_ids(&self, mut node_ids: Vec<NodeId>) -> Vec<NodeId> {
        match self {
            ExpressionNode::Constant(_) => node_ids,
            ExpressionNode::Selector(_, lhs, rhs) => {
                let node_ids = rhs.get_node_ids(node_ids);
                lhs.get_node_ids(node_ids)
            }
            ExpressionNode::Mle(node_id) => {
                node_ids.push(*node_id);
                node_ids
            }
            ExpressionNode::Negated(expr) => expr.get_node_ids(node_ids),
            ExpressionNode::Sum(lhs, rhs) => {
                let node_ids = lhs.get_node_ids(node_ids);
                rhs.get_node_ids(node_ids)
            }
            ExpressionNode::Product(nodes) => {
                node_ids.extend(nodes.iter());
                node_ids
            }
            ExpressionNode::Scaled(expr, _) => expr.get_node_ids(node_ids),
        }
    }

    fn get_num_vars(&self, num_vars_map: &HashMap<NodeId, usize>) -> Result<usize, DAGError> {
        match self {
            ExpressionNode::Constant(_) => Ok(0),
            ExpressionNode::Selector(_, lhs, rhs) => Ok(max(
                lhs.get_num_vars(num_vars_map)? + 1,
                rhs.get_num_vars(num_vars_map)? + 1,
            )),
            ExpressionNode::Mle(node_id) => Ok(*num_vars_map.get(node_id).unwrap()),
            ExpressionNode::Negated(expr) => expr.get_num_vars(num_vars_map),
            ExpressionNode::Sum(lhs, rhs) => Ok(max(
                lhs.get_num_vars(num_vars_map)?,
                rhs.get_num_vars(num_vars_map)?,
            )),
            ExpressionNode::Product(nodes) => Ok(nodes
                .iter()
                .map(|node_id| Ok(Some(*num_vars_map.get(node_id).unwrap())))
                .fold_ok(None, max)?
                .unwrap_or(0)),
            ExpressionNode::Scaled(expr, _) => expr.get_num_vars(num_vars_map),
        }
    }
}

impl<F: Field> Neg for Expression<F, AbstractExpr> {
    type Output = Expression<F, AbstractExpr>;
    fn neg(self) -> Self::Output {
        Expression::<F, AbstractExpr>::negated(self)
    }
}

/// implement the Add, Sub, and Mul traits for the Expression
impl<F: Field> Add for Expression<F, AbstractExpr> {
    type Output = Expression<F, AbstractExpr>;
    fn add(self, rhs: Expression<F, AbstractExpr>) -> Expression<F, AbstractExpr> {
        Expression::<F, AbstractExpr>::sum(self, rhs)
    }
}

impl<F: Field> Sub for Expression<F, AbstractExpr> {
    type Output = Expression<F, AbstractExpr>;
    fn sub(self, rhs: Expression<F, AbstractExpr>) -> Expression<F, AbstractExpr> {
        self.add(rhs.neg())
    }
}

impl<F: Field> Mul<F> for Expression<F, AbstractExpr> {
    type Output = Expression<F, AbstractExpr>;
    fn mul(self, rhs: F) -> Self::Output {
        Expression::<F, AbstractExpr>::scaled(self, rhs)
    }
}

// defines how the Expressions are printed and displayed
impl<F: std::fmt::Debug + Field> std::fmt::Debug for Expression<F, AbstractExpr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Expression")
            .field("Expression_Node", &self.expression_node)
            .finish()
    }
}

// defines how the ExpressionNodes are printed and displayed
impl<F: std::fmt::Debug + Field> std::fmt::Debug for ExpressionNode<F, AbstractExpr> {
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
            ExpressionNode::Mle(mle_ref) => {
                f.debug_struct("Mle").field("mle_ref", mle_ref).finish()
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
impl<F: std::fmt::Debug + Field> Expression<F, AbstractExpr> {
    #[allow(dead_code)]
    pub(crate) fn circuit_description_fmt(&self) -> impl std::fmt::Display + '_ {
        struct CircuitDesc<'a, F: Field>(
            &'a ExpressionNode<F, AbstractExpr>,
            &'a <AbstractExpr as ExpressionType<F>>::MleVec,
        );

        impl<'a, F: std::fmt::Debug + Field> std::fmt::Display for CircuitDesc<'a, F> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    ExpressionNode::Constant(scalar) => {
                        f.debug_tuple("const").field(scalar).finish()
                    }
                    ExpressionNode::Selector(index, a, b) => f.write_fmt(format_args!(
                        "sel {index:?}; {}; {}",
                        CircuitDesc(a, self.1),
                        CircuitDesc(b, self.1)
                    )),
                    // Skip enum variant and print query struct directly to maintain backwards compatibility.
                    ExpressionNode::Mle(node_id) => {
                        f.debug_struct("node").field("id", node_id).finish()
                    }
                    ExpressionNode::Negated(poly) => {
                        f.write_fmt(format_args!("-{}", CircuitDesc(poly, self.1)))
                    }
                    ExpressionNode::Sum(a, b) => f.write_fmt(format_args!(
                        "+ {}; {}",
                        CircuitDesc(a, self.1),
                        CircuitDesc(b, self.1)
                    )),
                    ExpressionNode::Product(a) => {
                        let str = a
                            .iter()
                            .map(|node_id| format!("node id: {:?}", node_id))
                            .reduce(|acc, str| acc + &str)
                            .unwrap();
                        f.write_str(&str)
                    }
                    ExpressionNode::Scaled(poly, scalar) => f.write_fmt(format_args!(
                        "* {}; {:?}",
                        CircuitDesc(poly, self.1),
                        scalar
                    )),
                }
            }
        }

        CircuitDesc(&self.expression_node, &self.mle_vec)
    }
}

/// Companion function to [selectors] that calculates the resulting MLE from the MLEs of the
/// expressions that make up the selector tree.
pub fn calculate_selector_values<F: Field>(mles: Vec<Vec<F>>) -> Vec<F> {
    let mut mles = mles;
    assert!(mles.len().is_power_of_two());

    while mles.len() > 1 {
        mles = mles
            .into_iter()
            .tuples()
            .map(|(mle1, mle2)| {
                mle1.into_iter()
                    .zip(mle2.into_iter())
                    .flat_map(|(a, b)| vec![a, b])
                    .collect()
            })
            .collect();
    }
    mles[0].clone()
}
