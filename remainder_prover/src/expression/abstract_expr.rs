use serde::{Deserialize, Serialize};
use std::{fmt::Debug, iter::repeat};

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
