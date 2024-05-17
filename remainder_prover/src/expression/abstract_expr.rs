use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    iter::repeat,
    ops::{Add, Mul, Neg, Sub},
};

use remainder_shared_types::FieldExt;

use crate::{
    layouter::nodes::NodeId,
    mle::{dense::DenseMle, MleIndex},
};

use super::generic_expr::{Expression, ExpressionNode, ExpressionType};

/// Abstract Expression
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AbstractExpr;
impl<F: FieldExt> ExpressionType<F> for AbstractExpr {
    type MLENodeRepr = NodeId;
    type MleVec = ();
}

//  comments for Phase II:
//  This will be the the circuit "pre-data" stage
//  will take care of building a prover expression
//  building the most memory efficient denseMleRefs dictionaries, etc.
impl<F: FieldExt> Expression<F, AbstractExpr> {
    pub fn get_sources(&self) -> Vec<NodeId> {
        let mut sources = vec![];
        let mut get_sources_closure = |expr_node: &ExpressionNode<F, AbstractExpr>,
                                       mle_vec: &<AbstractExpr as ExpressionType<F>>::MleVec|
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

    /// Concatenates two expressions together
    pub fn concat_expr(mut self, lhs: Expression<F, AbstractExpr>) -> Self {
        let (lhs_node, _) = lhs.deconstruct();
        let (rhs_node, _) = self.deconstruct();

        let concat_node =
            ExpressionNode::Selector(MleIndex::Iterated, Box::new(lhs_node), Box::new(rhs_node));

        Expression::new(concat_node, ())
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
    pub fn sum(lhs: Self, mut rhs: Self) -> Self {
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

impl<F: FieldExt> Neg for Expression<F, AbstractExpr> {
    type Output = Expression<F, AbstractExpr>;
    fn neg(self) -> Self::Output {
        Expression::<F, AbstractExpr>::negated(self)
    }
}

/// implement the Add, Sub, and Mul traits for the Expression
impl<F: FieldExt> Add for Expression<F, AbstractExpr> {
    type Output = Expression<F, AbstractExpr>;
    fn add(self, rhs: Expression<F, AbstractExpr>) -> Expression<F, AbstractExpr> {
        Expression::<F, AbstractExpr>::sum(self, rhs)
    }
}

impl<F: FieldExt> Sub for Expression<F, AbstractExpr> {
    type Output = Expression<F, AbstractExpr>;
    fn sub(self, rhs: Expression<F, AbstractExpr>) -> Expression<F, AbstractExpr> {
        self.add(rhs.neg())
    }
}

impl<F: FieldExt> Mul<F> for Expression<F, AbstractExpr> {
    type Output = Expression<F, AbstractExpr>;
    fn mul(self, rhs: F) -> Self::Output {
        Expression::<F, AbstractExpr>::scaled(self, rhs)
    }
}

// defines how the Expressions are printed and displayed
impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for Expression<F, AbstractExpr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Expression")
            .field("Expression_Node", &self.expression_node)
            .finish()
    }
}

// defines how the ExpressionNodes are printed and displayed
impl<F: std::fmt::Debug + FieldExt> std::fmt::Debug for ExpressionNode<F, AbstractExpr> {
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
impl<F: std::fmt::Debug + FieldExt> Expression<F, AbstractExpr> {
    pub(crate) fn circuit_description_fmt(&self) -> impl std::fmt::Display + '_ {
        struct CircuitDesc<'a, F: FieldExt>(
            &'a ExpressionNode<F, AbstractExpr>,
            &'a <AbstractExpr as ExpressionType<F>>::MleVec,
        );

        impl<'a, F: std::fmt::Debug + FieldExt> std::fmt::Display for CircuitDesc<'a, F> {
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
