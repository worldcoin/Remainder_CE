//! A wrapper `enum` type around various implementations of [MleRef]s.

use serde::{Deserialize, Serialize};

use remainder_shared_types::FieldExt;

use crate::{
    claims::{wlx_eval::ClaimMle, YieldClaim},
    layer::LayerError,
    mle::Mle,
};

use super::{dense::DenseMle, zero::ZeroMleRef, MleIndex, MleRef};

/// A wrapper type for various kinds of [MleRef]s.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub enum MleEnum<F: FieldExt> {
    /// A [DenseMle] variant.
    Dense(DenseMle<F>),
    /// A [ZeroMleRef] variant.
    Zero(ZeroMleRef<F>),
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

    fn num_vars(&self) -> usize {
        match self {
            MleEnum::Dense(item) => item.num_vars(),
            MleEnum::Zero(item) => item.num_vars(),
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

    fn get_enum(self) -> MleEnum<F> {
        self
    }

    fn push_mle_indices(&mut self, new_indices: &[MleIndex<F>]) {
        match self {
            MleEnum::Dense(item) => item.push_mle_indices(new_indices),
            MleEnum::Zero(item) => item.push_mle_indices(new_indices),
        }
    }
    
    fn num_iterated_vars(&self) -> usize {
        todo!()
    }
    
    fn get_padded_evaluations(&self) -> Vec<F> {
        todo!()
    }
    
    fn set_prefix_bits(&mut self, new_bits: Option<Vec<MleIndex<F>>>) {
        todo!()
    }
    
    fn get_prefix_bits(&self) -> Option<Vec<MleIndex<F>>> {
        todo!()
    }
    
    fn layer_id(&self) -> crate::layer::LayerId {
        todo!()
    }
}

impl<F: FieldExt> YieldClaim<F, ClaimMle<F>> for MleEnum<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        match self {
            MleEnum::Dense(layer) => layer.get_claims(),
            MleEnum::Zero(layer) => layer.get_claims(),
        }
    }
}
