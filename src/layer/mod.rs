//! A layer is a combination of multiple MLEs with an expression

pub mod claims;
// mod gkr_layer;

use std::marker::PhantomData;

use thiserror::Error;

use crate::{
    expression::{Expression, ExpressionError, ExpressionStandard},
    mle::{Mle, MleAble, MleIndex, MleRef},
    sumcheck::{compute_sumcheck_message, get_round_degree, SumOrEvals},
    FieldExt,
};

pub type Claim<F> = (Vec<F>, F);

#[derive(Error, Debug, Clone)]
pub enum LayerError {
    #[error("Layer isn't ready to prove")]
    LayerNotReady,
    #[error("Error with underlying expression {0}")]
    ExpressionError(ExpressionError),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
///The location of a layer within the GKR circuit
pub enum LayerId {
    ///An Mle located in the input layer
    Input,
    ///A layer within the GKR protocol, indexed by it's layer id
    Layer(usize),
}

///A layer is what you perform sumcheck over, it is made up of an expression and MLEs that contribute evaluations to that expression
pub trait Layer<F: FieldExt> {
    ///The Expression type that this Layer is defined by
    // type Expression: Expression<F>;

    ///Injest a claim, initialize beta tables, and do any other bookeeping that needs to be done before the sumcheck starts
    fn start_sumcheck(&mut self, claim: Claim<F>) -> Result<(Vec<F>, usize), LayerError> {
        todo!()
    }

    ///Computes a round of the sumcheck protocol on this Layer
    fn prove_round(&mut self, round_index: usize, challenge: F) -> Result<Vec<F>, LayerError> {
        let expression = self.get_expression_mut();
        expression.fix_variable(round_index - 1, challenge);

        // --- Grabs the degree of univariate polynomial we are sending over ---
        let degree = get_round_degree(expression, round_index);

        let eval = compute_sumcheck_message(expression, round_index, degree)
            .map_err(LayerError::ExpressionError)?;

        if let SumOrEvals::Evals(evals) = eval {
            Ok(evals)
        } else {
            Err(LayerError::ExpressionError(
                ExpressionError::EvaluationError(
                    "Received a sum variant from evaluate expression before the final round",
                ),
            ))
        }
    }

    ///Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<(LayerId, Claim<F>)>, LayerError> {
        todo!()
    }

    ///Gets the unique id of this layer
    fn get_id(&self) -> &LayerId;

    ///Get the master expression associated with this Layer
    fn get_expression(&self) -> &ExpressionStandard<F>;

    ///Get the master expression associated with this Layer mutably
    fn get_expression_mut(&mut self) -> &mut ExpressionStandard<F>;

    ///Create new ConcreteLayer from a LayerBuilder
    fn new<L: LayerBuilder<F>>(builder: L, id: LayerId) -> Self
    where
        Self: Sized;
}

///Default Layer abstraction
pub struct GKRLayer<F: FieldExt> {
    id: LayerId,
    expression: ExpressionStandard<F>,
}

impl<F: FieldExt> Layer<F> for GKRLayer<F> {
    fn new<L: LayerBuilder<F>>(builder: L, id: LayerId) -> Self {
        Self {
            id,
            expression: builder.build_expression(),
        }
    }

    fn get_expression(&self) -> &ExpressionStandard<F> {
        &self.expression
    }

    fn get_expression_mut(&mut self) -> &mut ExpressionStandard<F> {
        &mut self.expression
    }

    fn get_id(&self) -> &LayerId {
        &self.id
    }
}

///The builder type for a Layer
pub trait LayerBuilder<F: FieldExt> {
    ///The layer that makes claims on this layer in the GKR protocol. The next layer in the GKR protocol
    type Successor;

    ///Build the expression that will be sumchecked
    fn build_expression(&self) -> ExpressionStandard<F>;

    ///Generate the next layer
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor;

    ///Concatonate two layers together
    fn concat<Other: LayerBuilder<F>>(self, rhs: Other) -> ConcatLayer<F, Self, Other>
    where
        Self: Sized,
    {
        ConcatLayer {
            first: self,
            second: rhs,
            _marker: PhantomData,
        }
    }

    ///Creates a simple layer from an mle, with closures for defining how the mle turns into an expression and a next layer
    fn from_mle<
        T: Send + Sync + MleAble<F>,
        M: Mle<F, T>,
        EFn: Fn(&M) -> ExpressionStandard<F>,
        S,
        LFn: Fn(&M, usize) -> S,
    >(
        mle: M,
        expression_builder: EFn,
        layer_builder: LFn,
    ) -> SimpleLayer<F, T, M, EFn, LFn> {
        SimpleLayer {
            mle,
            expression_builder,
            layer_builder,
            _marker: PhantomData,
        }
    }
}

///The layerbuilder that represents two layers concatonated together
pub struct ConcatLayer<F: FieldExt, A: LayerBuilder<F>, B: LayerBuilder<F>> {
    first: A,
    second: B,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, A: LayerBuilder<F>, B: LayerBuilder<F>> LayerBuilder<F> for ConcatLayer<F, A, B> {
    type Successor = (A::Successor, B::Successor);

    fn build_expression(&self) -> ExpressionStandard<F> {
        let first = self.first.build_expression();
        let second = self.second.build_expression();
        first.concat(second)
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        (
            self.first.next_layer(
                id.clone(),
                Some(
                    prefix_bits
                        .clone()
                        .into_iter()
                        .flatten()
                        .chain(std::iter::once(MleIndex::Fixed(true)))
                        .collect(),
                ),
            ),
            self.second.next_layer(
                id,
                Some(
                    prefix_bits
                        .into_iter()
                        .flatten()
                        .chain(std::iter::once(MleIndex::Fixed(false)))
                        .collect(),
                ),
            ),
        )
    }
}

///A simple layer defined ad-hoc with two closures
pub struct SimpleLayer<F: FieldExt, T: Send + Sync + MleAble<F>, M: Mle<F, T>, EFn, LFn> {
    mle: M,
    expression_builder: EFn,
    layer_builder: LFn,
    _marker: PhantomData<(F, T)>,
}

impl<
        F: FieldExt,
        T: Send + Sync + MleAble<F>,
        M: Mle<F, T>,
        EFn: Fn(&M) -> ExpressionStandard<F>,
        S,
        LFn: Fn(&M, LayerId, Option<Vec<MleIndex<F>>>) -> S,
    > LayerBuilder<F> for SimpleLayer<F, T, M, EFn, LFn>
{
    type Successor = S;

    fn build_expression(&self) -> ExpressionStandard<F> {
        (self.expression_builder)(&self.mle)
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        (self.layer_builder)(&self.mle, id, prefix_bits)
    }
}
