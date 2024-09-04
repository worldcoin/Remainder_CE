//! A wrapper `enum` type around various implementations of [MleRef]s.

use serde::{Deserialize, Serialize};

use remainder_shared_types::{
    transcript::{TranscriptSponge, TranscriptWriter},
    FieldExt,
};

use crate::{
    claims::{wlx_eval::ClaimMle, YieldClaim},
    expression::circuit_expr::CircuitMle,
    layer::{LayerError, LayerId},
    mle::Mle,
};

use super::{dense::DenseMle, zero::ZeroMle, MleIndex};

/// A wrapper type for various kinds of [MleRef]s.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(bound = "F: FieldExt")]
pub enum MleEnum<F: FieldExt> {
    /// A [DenseMle] variant.
    Dense(DenseMle<F>),
    /// A [ZeroMle] variant.
    Zero(ZeroMle<F>),
}

impl<F: FieldExt> Mle<F> for MleEnum<F> {
    fn bookkeeping_table(&self) -> &[F] {
        match self {
            MleEnum::Dense(item) => item.bookkeeping_table(),
            MleEnum::Zero(item) => item.bookkeeping_table(),
        }
    }

    fn original_bookkeeping_table(&self) -> &[F] {
        match self {
            MleEnum::Dense(item) => item.original_bookkeeping_table(),
            MleEnum::Zero(item) => item.original_bookkeeping_table(),
        }
    }

    fn mle_indices(&self) -> &[super::MleIndex<F>] {
        match self {
            MleEnum::Dense(item) => item.mle_indices(),
            MleEnum::Zero(item) => item.mle_indices(),
        }
    }

    fn original_mle_indices(&self) -> &Vec<super::MleIndex<F>> {
        match self {
            MleEnum::Dense(item) => item.original_mle_indices(),
            MleEnum::Zero(item) => item.original_mle_indices(),
        }
    }

    fn num_iterated_vars(&self) -> usize {
        match self {
            MleEnum::Dense(item) => item.num_iterated_vars(),
            MleEnum::Zero(item) => item.num_iterated_vars(),
        }
    }

    fn original_num_vars(&self) -> usize {
        match self {
            MleEnum::Dense(item) => item.original_num_vars(),
            MleEnum::Zero(item) => item.original_num_vars(),
        }
    }

    fn fix_variable(
        &mut self,
        round_index: usize,
        challenge: F,
    ) -> Option<crate::claims::Claim<F>> {
        match self {
            MleEnum::Dense(item) => item.fix_variable(round_index, challenge),
            MleEnum::Zero(item) => item.fix_variable(round_index, challenge),
        }
    }

    fn fix_variable_at_index(
        &mut self,
        indexed_bit_index: usize,
        point: F,
    ) -> Option<crate::claims::Claim<F>> {
        match self {
            MleEnum::Dense(item) => item.fix_variable_at_index(indexed_bit_index, point),
            MleEnum::Zero(item) => item.fix_variable_at_index(indexed_bit_index, point),
        }
    }

    fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        match self {
            MleEnum::Dense(item) => item.index_mle_indices(curr_index),
            MleEnum::Zero(item) => item.index_mle_indices(curr_index),
        }
    }

    fn get_layer_id(&self) -> LayerId {
        match self {
            MleEnum::Dense(item) => item.get_layer_id(),
            MleEnum::Zero(item) => item.get_layer_id(),
        }
    }

    fn get_enum(self) -> MleEnum<F> {
        self
    }

    fn get_padded_evaluations(&self) -> Vec<F> {
        todo!()
    }

    fn add_prefix_bits(&mut self, _new_bits: Vec<MleIndex<F>>) {
        todo!()
    }

    fn layer_id(&self) -> crate::layer::LayerId {
        match self {
            MleEnum::Dense(dense_mle) => dense_mle.layer_id(),
            MleEnum::Zero(zero_mle) => zero_mle.layer_id(),
        }
    }
}

impl<F: FieldExt> YieldClaim<ClaimMle<F>> for MleEnum<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        match self {
            MleEnum::Dense(layer) => layer.get_claims(),
            MleEnum::Zero(layer) => layer.get_claims(),
        }
    }
}

impl<F: FieldExt> From<DenseMle<F>> for MleEnum<F> {
    fn from(value: DenseMle<F>) -> Self {
        Self::Dense(value)
    }
}

impl<F: FieldExt> From<ZeroMle<F>> for MleEnum<F> {
    fn from(value: ZeroMle<F>) -> Self {
        Self::Zero(value)
    }
}
