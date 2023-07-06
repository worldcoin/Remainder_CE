//! A layer is a combination of multiple MLEs with an expression

mod claims;

use crate::{mle::MleIndex, FieldExt, expression::Expression};

type Claim<'a, F> = (&'a [MleIndex<F>], F);

///A layer is what you perform sumcheck over, it is made up of an expression and MLEs that contribute evaluations to that expression
pub trait Layer<F: FieldExt> {
    ///The Mle that this Layer contains
    type Mle;

    ///The Expression type that this Layer is defined by
    type Expression: Expression<F>;

    ///The Message that this Layer "sends" to the verifier
    type ProverMessage;

    ///Creates an empty layer
    fn new(id: usize) -> Self;

    ///Computes a round of the sumcheck protocol on this Layer
    fn prove_round(&mut self, challenge: F) -> Self::ProverMessage;

    ///Concatonates an MLE and Expression to this Layer
    fn add_component(&mut self, mle: Self::Mle, expression: Self::Expression);

    ///Concatonates a Layer to this Layer
    fn add_layer(&mut self, new_layer: impl Layer<F>);

    ///Get the claim that this layer makes on the next layer
    fn get_claim(&self) -> Option<Claim<F>>;

    ///Get the claims that this layer makes on other layers
    fn get_all_claims(&self) -> Option<Vec<(usize, Claim<F>)>>;

    ///Get the master expression associated with this Layer
    fn get_expression(&self) -> Self::Expression;

    ///Get the Mles associated with this layer
    fn get_mles(&self) -> Vec<Self::Mle>;

    ///Gets the unique id of this layer
    fn get_id(&self) -> usize;
}
