//! An expression is a type which allows for expressing the definition of a GKR layer
//! Expression can be constructed with constants, Mle, a product of Mles, selectors, etc.
//! In sumcheck protocol, the prover manipulates the expression on different layers, (by
//! fixing bits and producing evaluations, etc.) to prove certain operations are done correctly.
//! Later verifier checks that the prover's computation is correct by receiving claims (in
//! the form of VerifierExpressions), and interacting with the prover to reuce and verify the claims.

/// the generic expression type, consists of
///     - abstract expression,
///     - prover expression,
///     - verifier expression
/// Common methods shared by these expressions are implemented here.
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
