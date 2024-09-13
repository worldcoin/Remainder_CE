//! This module describes the "regular"-style polynomial relationships described
//! within [Tha13](https://eprint.iacr.org/2013/351.pdf), Theorem 1 on page 25.
//!
//! Specifically, polynomial relationships between GKR circuit layers which look
//! like the following:
//!
//! \widetilde{V}_i(x_1, ..., x_n) = \sum_{b_1, ..., b_n} \widetilde{\beta}(b, x) Expr(b_1, ..., b_n)
//!
//! Where $Expr(b_1, ..., b_n)$ is comprised of one or more of the following (read as a CFG):
//! - [generic_expr::ExpressionNode::Constant], i.e. + c for c \in \mathbb{F}
//! - [generic_expr::ExpressionNode::Mle], i.e. \widetilde{V}_{j > i}(b_1, ..., b_{m \leq n})
//! - [generic_expr::ExpressionNode::Product], i.e. \prod_j \widetilde{V}_{j > i}(b_1, ..., b_{m \leq n})
//! - [generic_expr::ExpressionNode::Selector], i.e. (1 - b_0) * Expr(b_1, ..., b_{m \leq n}) + b_0 * Expr(b_1, ..., b_{m \leq n})
//! - [generic_expr::ExpressionNode::Sum], i.e. \widetilde{V}_{j_1 > i}(b_1, ..., b_{m_1 \leq n}) + \widetilde{V}_{j_2 > i}(b_1, ..., b_{m_2 \leq n})
//! - [generic_expr::ExpressionNode::Negated], i.e. -1 * Expr(b_1, ..., b_{m \leq n})
//! - [generic_expr::ExpressionNode::Scaled], i.e. c * Expr(b_1, ..., b_{m \leq n}) for c \in mathbb{F}
//!
//! Note that all expressions are written in "expanded" form, such that all MLEs
//! which are multiplied together appear next to one another in the expression,
//! i.e. all distribution over the sum of other MLEs has already been performed.
//!
//! As an example, the "data" relationship described by
//! V_{i + 1}(b_1, ..., b_n) * (V_{i + 1}(b_1, ..., b_n) + V_{i + 2}(b_1, ..., b_n))
//! is inexpressible using the functionality within [crate::expression]. Rather,
//! it would be written as the (mathematically equivalent)
//! V_{i + 1}(b_1, ..., b_n) * V_{i + 1}(b_1, ..., b_n) + V_{i + 1}(b_1, ..., b_n) * V_{i + 2}(b_1, ..., b_n)
//!
//! Expressions are constructed as a tree of [generic_expr::ExpressionNode]s.
//! Note that there are four "phases" to the "life cycle" of an expression, i.e.
//! the changes it goes through during the circuit-building process (see documentation
//! for this under under TODO(vishady): write documentation for the compilation process):
//! - [circuit_expr::CircuitExpr]: The "circuit description" view of an
//! expression. In particular, it describes the polynomial relationship between
//! the MLE representing the data output of a particular layer against MLEs
//! representing data "slices" within other layers, and contains information about
//! the expression "structure" (in the form of the expansion of the CFG description
//! from above), with the leaves of the expression tree always being
//! [generic_expr::ExpressionNode::Constant]s or [generic_expr::ExpressionNode::Mle]s
//! or [generic_expr::ExpressionNode::Product]s.
//! - [abstract_expr::AbstractExpr]: The circuit-builder's view of an expression
//! (again, see documentation under TODO(vishady)). It describes the structural
//! relationship between the current graph node and the nodes which that node
//! depends on, without specifying the sizes of any MLEs which are represented
//! by those nodes.
//! - [prover_expr::ProverExpr]: The prover's view of an expression. It contains
//! all the information within [circuit_expr::CircuitExpr], but contains
//! [crate::mle::dense::DenseMle]s rather than [circuit_expr::CircuitMle]s.
//! - [verifier_expr::VerifierExpr]: The verifier's view of an expression. It
//! contains all the information within [circuit_expr::CircuitExpr], but contains
//! [verifier_expr::VerifierMle]s rather than [circuit_expr::CircuitMle]s.

///
pub mod generic_expr;

/// the abstract expression type, in the life cycle of an expression,
/// this is the starting point, where these abstract expressions construct layers,
/// make up circuits, etc. Methods here include ones to describe the circuit, etc.
pub mod abstract_expr;

/// the prover expression type, in the life cycle of an expression,
/// this is what the prover manipulates to prove the correctness of the computation.
/// Methods here include ones to fix bits, evaluate sumcheck messages, etc.
pub mod prover_expr;

/// the verifier expression type, in the life cycle of an expression,
/// this is what the verifier manipulates to verify the correctness of the computation.
/// Methods here include ones to gather and combine (claimed) evaluations of Mles.
/// The verifier will verfity that these evaluations are correct using the GKR protocol.
pub mod circuit_expr;

pub mod verifier_expr;

/// where the errors for the expression module are defined
pub mod expr_errors;

/// tests for the expression module
#[cfg(test)]
pub mod tests;
