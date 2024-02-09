//! Extra builders for debugging/circuit testing
use crate::expression::generic_expr::Expression;
use crate::expression::prover_expr::ProverExpr;
use crate::layer::{LayerBuilder, LayerId};
use crate::mle::MleRef;
use crate::mle::dense::{DenseMle};
use crate::mle::{zero::ZeroMleRef, MleIndex};
use remainder_shared_types::FieldExt;


/// Grabs a single value from a `RandomInputLayer` and subtracts it from
/// everything within `mle`.
pub struct FSRandomBuilder<F: FieldExt> {
    mle: DenseMle<F, F>,
    val_mle: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for FSRandomBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        Expression::mle(self.mle.mle_ref()) - Expression::mle(self.val_mle.mle_ref())
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let val = self.val_mle.mle_ref().bookkeeping_table()[0];
        DenseMle::new_from_iter(self.mle.into_iter().map(|mle_val| mle_val - val), id, prefix_bits)

    }
}

impl<F: FieldExt> FSRandomBuilder<F> {
    /// Constructor
    pub(crate) fn new(
        mle: DenseMle<F, F>,
        val_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            mle,
            val_mle
        }
    }
}

/// Builder to just zero things out by subtracting an MLE
/// literally from itself
pub struct SelfMinusSelfBuilder<F: FieldExt> {
    mle: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for SelfMinusSelfBuilder<F> {
    type Successor = ZeroMleRef<F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        Expression::mle(self.mle.mle_ref()) - Expression::mle(self.mle.mle_ref())
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        ZeroMleRef::new(self.mle.mle_ref().num_vars, prefix_bits, id)

    }
}

impl<F: FieldExt> SelfMinusSelfBuilder<F> {
    /// Constructor
    pub(crate) fn new(
        mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            mle,
        }
    }
}