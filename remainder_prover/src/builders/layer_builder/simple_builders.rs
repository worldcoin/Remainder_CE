//! A set of simple re-usable `LayerBuilder`s

use crate::builders::layer_builder::LayerBuilder;
use crate::expression::generic_expr::Expression;
use crate::expression::prover_expr::ProverExpr;
use crate::layer::LayerId;
use crate::mle::dense::DenseMle;
use crate::mle::{zero::ZeroMle, Mle, MleIndex};
use remainder_shared_types::FieldExt;
use std::cmp::max;

/// takes a DenseMle and subtracts it from itself to get all zeroes.
pub struct ZeroBuilder<F: FieldExt> {
    mle: DenseMle<F>,
}

impl<F: FieldExt> LayerBuilder<F> for ZeroBuilder<F> {
    type Successor = ZeroMle<F>;
    fn build_expression(&self) -> Expression<F, ProverExpr> {
        Expression::<F, ProverExpr>::mle(self.mle.clone())
            - Expression::<F, ProverExpr>::mle(self.mle.clone())
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let mle_num_vars = self.mle.num_iterated_vars();
        ZeroMle::new(mle_num_vars, prefix_bits, id)
    }
}

impl<F: FieldExt> ZeroBuilder<F> {
    /// create new leaf node packed
    pub fn new(mle: DenseMle<F>) -> Self {
        Self { mle }
    }
}

/// calculates the difference between two mles
/// and contrains it to be a `ZeroMle`
pub struct EqualityCheck<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
}

impl<F: FieldExt> LayerBuilder<F> for EqualityCheck<F> {
    type Successor = ZeroMle<F>;
    // the difference between two mles, should be zero valued
    fn build_expression(&self) -> Expression<F, ProverExpr> {
        Expression::<F, ProverExpr>::mle(self.mle_1.clone())
            - Expression::<F, ProverExpr>::mle(self.mle_2.clone())
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let num_vars = max(
            self.mle_1.num_iterated_vars(),
            self.mle_2.num_iterated_vars(),
        );
        ZeroMle::new(num_vars, prefix_bits, id)
    }
}

impl<F: FieldExt> EqualityCheck<F> {
    /// creates new difference mle
    pub fn new(mle_1: DenseMle<F>, mle_2: DenseMle<F>) -> Self {
        Self { mle_1, mle_2 }
    }
}
