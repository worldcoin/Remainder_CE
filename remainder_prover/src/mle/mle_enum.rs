//! A wrapper `enum` type around various implementations of MLEs.

use super::{dense::DenseMle, evals::EvaluationsIterator, zero::ZeroMle, MleIndex};
use crate::{
    layer::LayerId,
    mle::{
        evals::{Evaluations, MultilinearExtension},
        Mle,
    },
};
use itertools::{repeat_n, Itertools};
use remainder_shared_types::{extension_field::ExtensionField, Field};
use serde::{Deserialize, Serialize};

use crate::mle::AbstractMle;

/// A wrapper type for various kinds of [MleRef]s.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(bound = "F: Field")]
pub enum MleEnum<F: Field> {
    /// A [DenseMle] variant.
    Dense(DenseMle<F>),
    /// A [ZeroMle] variant.
    Zero(ZeroMle<F>),
}

impl<F: Field> AbstractMle for MleEnum<F> {
    fn mle_indices(&self) -> &[super::MleIndex] {
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

    fn layer_id(&self) -> LayerId {
        match self {
            MleEnum::Dense(item) => item.layer_id(),
            MleEnum::Zero(item) => item.layer_id(),
        }
    }
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

    fn value(&self) -> F {
        match self {
            MleEnum::Dense(item) => item.value(),
            MleEnum::Zero(item) => item.value(),
        }
    }

    fn get(&self, index: usize) -> Option<F> {
        match self {
            MleEnum::Dense(item) => item.get(index),
            MleEnum::Zero(item) => item.get(index),
        }
    }

    fn fix_variable(
        &mut self,
        round_index: usize,
        challenge: F,
        bind_list: &mut Vec<Option<F>>,
    ) -> Option<crate::claims::RawClaim<F>> {
        match self {
            MleEnum::Dense(item) => item.fix_variable(round_index, challenge, bind_list),
            MleEnum::Zero(item) => item.fix_variable(round_index, challenge, bind_list),
        }
    }

    fn fix_variable_at_index(
        &mut self,
        indexed_bit_index: usize,
        point: F,
        bind_list: &mut Vec<Option<F>>,
    ) -> Option<crate::claims::RawClaim<F>> {
        match self {
            MleEnum::Dense(item) => item.fix_variable_at_index(indexed_bit_index, point, bind_list),
            MleEnum::Zero(item) => item.fix_variable_at_index(indexed_bit_index, point, bind_list),
        }
    }

    fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        match self {
            MleEnum::Dense(item) => item.index_mle_indices(curr_index),
            MleEnum::Zero(item) => item.index_mle_indices(curr_index),
        }
    }

    fn get_enum(self) -> MleEnum<F> {
        self
    }

    fn get_padded_evaluations(&self) -> Vec<F> {
        match self {
            MleEnum::Dense(dense_mle) => dense_mle.mle.f.iter().collect_vec(),
            MleEnum::Zero(zero_mle) => repeat_n(F::ZERO, 1 << zero_mle.num_vars).collect_vec(),
        }
    }

    fn add_prefix_bits(&mut self, _new_bits: Vec<MleIndex>) {
        todo!()
    }
}

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

/// Lift from [MleEnum<F>] to [MleEnum<E>].
impl<F: Field, E> LiftTo<MleEnum<E>> for MleEnum<F>
where
    E: ExtensionField<BaseField = F>,
{
    fn lift(self) -> MleEnum<E> {
        match self {
            MleEnum::Dense(dense_mle) => MleEnum::Dense(dense_mle.lift()),
            MleEnum::Zero(zero_mle) => MleEnum::Zero(zero_mle.lift()),
        }
    }
}

/// Lift from [ZeroMle<F>] to [ZeroMle<E>].
impl<F: Field, E> LiftTo<ZeroMle<E>> for ZeroMle<F>
where
    E: ExtensionField<BaseField = F>,
{
    fn lift(self) -> ZeroMle<E> {
        ZeroMle {
            layer_id: self.layer_id,
            mle_indices: self.mle_indices,
            num_vars: self.num_vars,
            zero: [E::ZERO],
            zero_eval: self.zero_eval.lift(),
            indexed: self.indexed,
        }
    }
}

/// Lift from [DenseMle<F>] to [DenseMle<E>].
impl<F: Field, E> LiftTo<DenseMle<E>> for DenseMle<F>
where
    E: ExtensionField<BaseField = F>,
{
    fn lift(self) -> DenseMle<E> {
        DenseMle {
            layer_id: self.layer_id,
            mle: self.mle.lift(),
            mle_indices: self.mle_indices,
        }
    }
}

/// The lift trait allows us to "lift" a data struct over base
/// field elements into one over extension field elements.
pub trait LiftTo<T> {
    /// Lifts a type from base field to extension field
    fn lift(self) -> T;
}

/// Lift from [MultilinearExtension<F>] to [MultilinearExtension<E>].
impl<F: Field, E> LiftTo<MultilinearExtension<E>> for MultilinearExtension<F>
where
    E: ExtensionField<BaseField = F>,
{
    fn lift(self) -> MultilinearExtension<E> {
        let new_evaluations: Evaluations<E> = self.f.lift();
        MultilinearExtension { f: new_evaluations }
    }
}
