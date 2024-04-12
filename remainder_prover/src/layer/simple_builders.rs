use crate::expression::generic_expr::Expression;
use crate::expression::prover_expr::ProverExpr;
use crate::layer::batched::BatchedLayer;
use crate::layer::{LayerBuilder, LayerId};
use crate::mle::dense::DenseMle;
use crate::mle::{zero::ZeroMleRef, Mle, MleIndex};
use remainder_shared_types::FieldExt;
use std::cmp::max;

/// takes a densemleref that is all zeros and returns a zeromleref as the successor
pub struct ZeroBuilder<F: FieldExt> {
    mle: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for ZeroBuilder<F> {
    type Successor = ZeroMleRef<F>;
    fn build_expression(&self) -> Expression<F, ProverExpr> {
        Expression::mle(self.mle.mle_ref())
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let mle_num_vars = self.mle.num_iterated_vars();
        ZeroMleRef::new(mle_num_vars, prefix_bits, id)
    }
}

impl<F: FieldExt> ZeroBuilder<F> {
    /// create new leaf node packed
    pub fn new(mle: DenseMle<F, F>) -> Self {
        Self { mle }
    }
}

/// calculates the difference between two mles
/// effectively means checking they are equal
/// should spit out ZeroMleRef
pub struct EqualityCheck<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for EqualityCheck<F> {
    type Successor = ZeroMleRef<F>;
    // the difference between two mles, should be zero valued
    fn build_expression(&self) -> Expression<F, ProverExpr> {
        Expression::mle(self.mle_1.mle_ref()) - Expression::mle(self.mle_2.mle_ref())
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let num_vars = max(
            self.mle_1.num_iterated_vars(),
            self.mle_2.num_iterated_vars(),
        );
        ZeroMleRef::new(num_vars, prefix_bits, id)
    }
}

impl<F: FieldExt> EqualityCheck<F> {
    /// creates new difference mle
    pub fn new(mle_1: DenseMle<F, F>, mle_2: DenseMle<F, F>) -> Self {
        Self { mle_1, mle_2 }
    }

    /// creates a batched layer for equality check
    pub fn new_batched(
        mle_1: Vec<DenseMle<F, F>>,
        mle_2: Vec<DenseMle<F, F>>,
    ) -> BatchedLayer<F, Self> {
        BatchedLayer::new(
            mle_1
                .into_iter()
                .zip(mle_2.into_iter())
                .map(|(mle_1, mle_2)| Self { mle_1, mle_2 })
                .collect(),
        )
    }
}
