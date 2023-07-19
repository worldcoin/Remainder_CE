//! A layer is a combination of multiple MLEs with an expression

mod claims;
// mod gkr_layer;

use thiserror::Error;

use crate::{expression::{Expression, ExpressionError, ExpressionStandard}, FieldExt, sumcheck::{SumOrEvals, evaluate_expr, get_round_degree}};

type Claim<F> = (Vec<F>, F);

#[derive(Error, Debug, Clone)]
pub enum LayerError {
    #[error("Layer isn't ready to prove")]
    LayerNotReady,
    #[error("Error with underlying expression {0}")]
    ExpressionError(ExpressionError),
    #[error("No challenge when round_index != 0")]
    NoChallenge
}

///A layer is what you perform sumcheck over, it is made up of an expression and MLEs that contribute evaluations to that expression
pub trait Layer<F: FieldExt> {
    ///The Mle(s) that this Layer contains
    type Mle;

    ///The Expression type that this Layer is defined by
    // type Expression: Expression<F>;

    ///Creates an empty layer
    fn new(id: usize) -> Self;

    ///Computes a round of the sumcheck protocol on this Layer
    fn prove_round(&mut self, challenge: Option<F>) -> Result<Vec<F>, LayerError> {
        let round_index = self.round_index().ok_or(LayerError::LayerNotReady)?;
        let expression = self.get_expression_mut();
        if let Some(challenge) = challenge {
            expression.fix_variable(round_index - 1, challenge)
        } else if round_index != 0 {
            return Err(LayerError::NoChallenge)
        }

        // --- Grabs the degree of univariate polynomial we are sending over ---
        let degree = get_round_degree(expression, round_index);

        let eval = evaluate_expr(expression, round_index, degree).map_err(LayerError::ExpressionError)?;

        if let SumOrEvals::Evals(evals) = eval {
            self.increment_round_index();
            Ok(evals)
        } else {
            Err(LayerError::ExpressionError(ExpressionError::EvaluationError("Received a sum variant from evaluate expression before the final round")))
        }
    }

    fn round_index(&self) -> Option<usize>;

    fn increment_round_index(&mut self);

    // ///Concatonates an MLE and Expression to this Layer
    // fn add_component(&mut self, mle: Self::Mle, expression: Self::Expression);

    // ///Concatonates a Layer to this Layer
    // fn add_layer(&mut self, new_layer: impl Layer<F>);

    ///Get the claim that this layer makes on the next layer
    fn get_claim(&self) -> Option<Claim<F>>;

    ///Get the claims that this layer makes on other layers
    fn get_all_claims(&self) -> Option<Vec<(usize, Claim<F>)>>;

    ///Get the master expression associated with this Layer
    fn get_expression(&self) -> &ExpressionStandard<F>;

    fn get_expression_mut(&mut self) -> &mut ExpressionStandard<F>;

    ///Get the Mles associated with this layer
    fn get_mles(&self) -> Self::Mle;

    ///Gets the unique id of this layer
    fn get_id(&self) -> usize;
}
