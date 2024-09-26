//! This module describes the "regular"-style polynomial relationships described
//! within [Tha13](https://eprint.iacr.org/2013/351.pdf), Theorem 1 on page 25.
//!
//! Specifically, polynomial relationships between GKR circuit layers which look
//! like the following:
//!
//! \widetilde{V}_i(x_1, ..., x_n) = \sum_{b_1, ..., b_n} \widetilde{\beta}(b, x) Expr(b_1, ..., b_n)
//!
//! Where $Expr(b_1, ..., b_n)$ is comprised of one or more of the following:
//! (read as a CFG; see documentation within [generic_expr::ExpressionNode])
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
//! * [circuit_expr::CircuitExpr]: The "circuit description" view of an
//!   expression. In particular, it describes the polynomial relationship between
//!   the MLE representing the data output of a particular layer against MLEs
//!   representing data "slices" within other layers, and contains information about
//!   the expression "structure" (in the form of the expansion of the CFG description
//!   from above), with the leaves of the expression tree always being
//!   [generic_expr::ExpressionNode::Constant]s or [generic_expr::ExpressionNode::Mle]s
//!   or [generic_expr::ExpressionNode::Product]s.
//! * [abstract_expr::AbstractExpr]: The circuit-builder's view of an expression
//!   (again, see documentation under TODO(vishady)). It describes the structural
//!   relationship between the current graph node and the nodes which that node
//!   depends on, without specifying the sizes of any MLEs which are represented
//!   by those nodes.
//! * [prover_expr::ProverExpr]: The prover's view of an expression. It contains
//!   all the information within [circuit_expr::CircuitExpr], but contains
//!   [crate::mle::dense::DenseMle]s rather than [circuit_expr::CircuitMle]s.
//! * [verifier_expr::VerifierExpr]: The verifier's view of an expression. It
//!   contains all the information within [circuit_expr::CircuitExpr], but contains
//!   [verifier_expr::VerifierMle]s rather than [circuit_expr::CircuitMle]s.

#![allow(missing_docs)]
pub mod abstract_expr;
pub mod circuit_expr;
pub mod expr_errors;
pub mod generic_expr;
pub mod prover_expr;
pub mod verifier_expr;

#[cfg(test)]
pub mod tests;
