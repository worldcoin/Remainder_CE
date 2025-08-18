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

impl<F: Field> AbstractMle<F> for MleEnum<F> {
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

    fn get_enum(self) -> MleEnum<F> {
        self
    }

    fn get_padded_evaluations(&self) -> Vec<F> {
        match self {
            MleEnum::Dense(dense_mle) => dense_mle.mle.f.iter().collect_vec(),
            MleEnum::Zero(zero_mle) => repeat_n(F::ZERO, 1 << zero_mle.num_vars).collect_vec(),
        }
    }

    fn add_prefix_bits(&mut self, _new_bits: Vec<MleIndex<F>>) {
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

/// Lift from [MleEnum<F>] to [MleEnum<E>] in the wrapper way.
impl<E: ExtensionField> LiftTo<MleEnum<E>> for MleEnum<E::BaseField> {
    fn lift(self) -> MleEnum<E> {
        match self {
            MleEnum::Dense(dense_mle) => MleEnum::Dense(dense_mle.lift()),
            MleEnum::Zero(zero_mle) => MleEnum::Zero(zero_mle.lift()),
        }
    }
}

/// Lift from [ZeroMle<F>] to [ZeroMle<E>] in the trivial way.
impl<E: ExtensionField> LiftTo<ZeroMle<E>> for ZeroMle<E::BaseField> {
    fn lift(self) -> ZeroMle<E> {
        let new_mle_indices: Vec<MleIndex<E>> = self
            .mle_indices
            .into_iter()
            .map(|mle_var| mle_var.lift())
            .collect();
        ZeroMle {
            layer_id: self.layer_id,
            mle_indices: new_mle_indices,
            num_vars: self.num_vars,
            zero: [E::ZERO],
            zero_eval: self.zero_eval.lift(),
            indexed: self.indexed,
        }
    }
}

/// Lift from [DenseMle<F>] to [DenseMle<E>] in the trivial way.
impl<E: ExtensionField> LiftTo<DenseMle<E>> for DenseMle<E::BaseField> {
    fn lift(self) -> DenseMle<E> {
        let new_mle = self.mle.lift();
        let new_mle_indices: Vec<MleIndex<E>> = self
            .mle_indices
            .into_iter()
            .map(|mle_var| mle_var.lift())
            .collect();
        DenseMle {
            layer_id: self.layer_id,
            mle: new_mle,
            mle_indices: new_mle_indices,
        }
    }
}

/// ChatGPT-inspired trait which allows us to "lift" a data struct over base
/// field elements, e.g., into one over extension field elements.
pub trait LiftTo<T> {
    fn lift(self) -> T;
}

/// Lift from [MultilinearExtension<F>] to [MultilinearExtension<E>] in the
/// trivial way.
impl<E: ExtensionField> LiftTo<MultilinearExtension<E>> for MultilinearExtension<E::BaseField> {
    fn lift(self) -> MultilinearExtension<E> {
        let new_evaluations: Evaluations<E> = self.f.lift();
        MultilinearExtension { f: new_evaluations }
    }
}
