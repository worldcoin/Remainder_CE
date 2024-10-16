//! A wrapper `enum` type around various implementations of [MleRef]s.

use serde::{Deserialize, Serialize};

use remainder_shared_types::Field;

use crate::{layer::LayerId, mle::Mle};

use super::{dense::DenseMle, evals::EvaluationsIterator, zero::ZeroMle, MleIndex};

/// A wrapper type for various kinds of [MleRef]s.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(bound = "F: Field")]
pub enum MleEnum<F: Field> {
    /// A [DenseMle] variant.
    Dense(DenseMle<F>),
    /// A [ZeroMle] variant.
    Zero(ZeroMle<F>),
}

impl<F: Field> Mle<F> for MleEnum<F> {
    fn len(&self) -> usize {
        match self {
            MleEnum::Dense(item) => item.len(),
            MleEnum::Zero(item) => item.len(),
        }
    }

    fn iter(&self) -> EvaluationsIterator<F> {
        match self {
            MleEnum::Dense(item) => item.iter(),
            MleEnum::Zero(item) => item.iter(),
        }
    }

    fn first(&self) -> F {
        match self {
            MleEnum::Dense(item) => item.first(),
            MleEnum::Zero(item) => item.first(),
        }
    }

    fn get(&self, index: usize) -> Option<F> {
        match self {
            MleEnum::Dense(item) => item.get(index),
            MleEnum::Zero(item) => item.get(index),
        }
    }

    fn mle_indices(&self) -> &[super::MleIndex<F>] {
        match self {
            MleEnum::Dense(item) => item.mle_indices(),
            MleEnum::Zero(item) => item.mle_indices(),
        }
    }

    fn num_free_vars(&self) -> usize {
        match self {
            MleEnum::Dense(item) => item.num_free_vars(),
            MleEnum::Zero(item) => item.num_free_vars(),
        }
    }

    fn fix_variable(
        &mut self,
        round_index: usize,
        challenge: F,
    ) -> Option<crate::claims::RawClaim<F>> {
        match self {
            MleEnum::Dense(item) => item.fix_variable(round_index, challenge),
            MleEnum::Zero(item) => item.fix_variable(round_index, challenge),
        }
    }

    fn fix_variable_at_index(
        &mut self,
        indexed_bit_index: usize,
        point: F,
    ) -> Option<crate::claims::RawClaim<F>> {
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

    fn layer_id(&self) -> LayerId {
        match self {
            MleEnum::Dense(item) => item.layer_id(),
            MleEnum::Zero(item) => item.layer_id(),
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
}

/*
impl<F: Field> YieldClaim<ClaimMle<F>> for MleEnum<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        match self {
            MleEnum::Dense(layer) => layer.get_claims(),
            MleEnum::Zero(layer) => layer.get_claims(),
        }
    }
}
*/

impl<F: Field> From<DenseMle<F>> for MleEnum<F> {
    fn from(value: DenseMle<F>) -> Self {
        Self::Dense(value)
    }
}

impl<F: Field> From<ZeroMle<F>> for MleEnum<F> {
    fn from(value: ZeroMle<F>) -> Self {
        Self::Zero(value)
    }
}
