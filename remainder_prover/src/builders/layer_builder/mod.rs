//! A tool for circuit builders to create and combine Layer sub-parts

pub mod simple_builders;
use std::marker::PhantomData;

use itertools::repeat_n;
use remainder_shared_types::FieldExt;

use crate::{
    expression::{generic_expr::Expression, prover_expr::ProverExpr},
    mle::MleIndex,
};

use crate::layer::LayerId;

/// The builder type for a Layer
///
/// A intermediate trait for defining components that can be combined/modified
/// before being 'build' into an `Expression`
pub trait LayerBuilder<F: FieldExt> {
    /// The layer that makes claims on this layer in the GKR protocol. The next layer in the GKR protocol
    type Successor;

    /// Build the expression that will be sumchecked
    fn build_expression(&self) -> Expression<F, ProverExpr>;

    /// Generate the next layer
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor;

    /// Concatonate two layers together
    fn concat<Other: LayerBuilder<F>>(self, rhs: Other) -> ConcatLayer<F, Self, Other>
    where
        Self: Sized,
    {
        ConcatLayer {
            first: self,
            second: rhs,
            padding: Padding::None,
            _marker: PhantomData,
        }
    }

    ///Concatonate two layers together with some padding
    fn concat_with_padding<Other: LayerBuilder<F>>(
        self,
        rhs: Other,
        padding: Padding,
    ) -> ConcatLayer<F, Self, Other>
    where
        Self: Sized,
    {
        ConcatLayer {
            first: self,
            second: rhs,
            padding,
            _marker: PhantomData,
        }
    }
}

/// Creates a simple layer from an mle, with closures for defining how the mle turns into an expression and a next layer
pub fn from_mle<
    F: FieldExt,
    M,
    EFn: Fn(&M) -> Expression<F, ProverExpr>,
    S,
    LFn: Fn(&M, LayerId, Option<Vec<MleIndex<F>>>) -> S,
>(
    mle: M,
    expression_builder: EFn,
    layer_builder: LFn,
) -> SimpleLayer<M, EFn, LFn> {
    SimpleLayer {
        mle,
        expression_builder,
        layer_builder,
    }
}

///A description of how much padding should be added when
/// concatonating two `LayerBuilder`s together
pub enum Padding {
    /// Indicates that the 'Right' side is larger by
    /// the wrapped num_vars
    Right(usize),
    /// Indicates that the 'Left' side is larger by
    /// the wrapped num_vars
    Left(usize),
    /// Indicates that the two sides are of equal size
    None,
}

/// The layerbuilder that represents two layers concatonated together
pub struct ConcatLayer<F: FieldExt, A: LayerBuilder<F>, B: LayerBuilder<F>> {
    first: A,
    second: B,
    padding: Padding,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, A: LayerBuilder<F>, B: LayerBuilder<F>> LayerBuilder<F> for ConcatLayer<F, A, B> {
    type Successor = (A::Successor, B::Successor);

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        let first = self.first.build_expression();
        let second = self.second.build_expression();

        let zero_expression: Expression<F, ProverExpr> =
            Expression::<F, ProverExpr>::constant(F::ZERO);

        let first_padded = if let Padding::Left(padding) = self.padding {
            let mut left = first;
            for _ in 0..padding {
                left = zero_expression.clone().concat_expr(left);
            }
            left
        } else {
            first
        };

        let second_padded = if let Padding::Right(padding) = self.padding {
            let mut right = second;
            for _ in 0..padding {
                right = zero_expression.clone().concat_expr(right);
            }
            right
        } else {
            second
        };

        first_padded.concat_expr(second_padded)
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let first_padding = if let Padding::Left(padding) = self.padding {
            repeat_n(MleIndex::Fixed(false), padding)
        } else {
            repeat_n(MleIndex::Fixed(false), 0)
        };
        let second_padding = if let Padding::Right(padding) = self.padding {
            repeat_n(MleIndex::Fixed(false), padding)
        } else {
            repeat_n(MleIndex::Fixed(false), 0)
        };
        (
            self.first.next_layer(
                id,
                Some(
                    prefix_bits
                        .clone()
                        .into_iter()
                        .flatten()
                        .chain(first_padding)
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
                        .chain(second_padding)
                        .chain(std::iter::once(MleIndex::Fixed(false)))
                        .collect(),
                ),
            ),
        )
    }
}

/// A simple layer defined ad-hoc with two closures
pub struct SimpleLayer<M, EFn, LFn> {
    mle: M,
    expression_builder: EFn,
    layer_builder: LFn,
}

impl<
        F: FieldExt,
        M,
        EFn: Fn(&M) -> Expression<F, ProverExpr>,
        S,
        LFn: Fn(&M, LayerId, Option<Vec<MleIndex<F>>>) -> S,
    > LayerBuilder<F> for SimpleLayer<M, EFn, LFn>
{
    type Successor = S;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        (self.expression_builder)(&self.mle)
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        (self.layer_builder)(&self.mle, id, prefix_bits)
    }
}
