use serde::{Serialize, Deserialize};

use crate::FieldExt;

use super::{zero::ZeroMleRef, dense::DenseMleRef, MleRef, MleIndex};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MleEnum<F: FieldExt> {
    Dense(DenseMleRef<F>),
    Zero(ZeroMleRef<F>)
}

impl<F: FieldExt> MleRef for MleEnum<F> {
    type F = F;

    fn bookkeeping_table(&self) -> &[Self::F] {
        match self {
            MleEnum::Dense(item) => item.bookkeeping_table(),
            MleEnum::Zero(item) => item.bookkeeping_table(),
        }
    }

    fn mle_indices(&self) -> &[super::MleIndex<Self::F>] {
        match self {
            MleEnum::Dense(item) => item.mle_indices(),
            MleEnum::Zero(item) => item.mle_indices(),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            MleEnum::Dense(item) => item.num_vars(),
            MleEnum::Zero(item) => item.num_vars(),
        }
    }

    fn fix_variable(&mut self, round_index: usize, challenge: Self::F) -> Option<crate::layer::Claim<Self::F>> {
        match self {
            MleEnum::Dense(item) => item.fix_variable(round_index, challenge),
            MleEnum::Zero(item) => item.fix_variable(round_index, challenge),
        }
    }

    fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        match self {
            MleEnum::Dense(item) => item.index_mle_indices(curr_index),
            MleEnum::Zero(item) => item.index_mle_indices(curr_index),
        }
    }

    fn get_layer_id(&self) -> crate::layer::LayerId {
        match self {
            MleEnum::Dense(item) => item.get_layer_id(),
            MleEnum::Zero(item) => item.get_layer_id(),
        }
    }

    fn indexed(&self) -> bool {
        match self {
            MleEnum::Dense(item) => item.indexed(),
            MleEnum::Zero(item) => item.indexed(),
        }
    }

    fn get_enum(self) -> MleEnum<Self::F> {
        self
    }

    fn push_mle_indices(&mut self, new_indices: &[MleIndex<Self::F>]) {
        todo!()
    }

}