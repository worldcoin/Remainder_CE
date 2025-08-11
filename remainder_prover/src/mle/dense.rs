#[cfg(test)]
mod tests;

use std::fmt::Debug;

use ark_std::log2;
use itertools::{repeat_n, Itertools};

use serde::{Deserialize, Serialize};

use super::{evals::EvaluationsIterator, mle_enum::MleEnum, Mle, MleIndex};
use crate::{
    claims::RawClaim,
    mle::{
        evals::{
            fix_variable_at_index_to_bookkeeping_table_copy, Evaluations, MultilinearExtension,
        },
        mle_enum::LiftTo,
    },
};
use crate::{
    expression::{generic_expr::Expression, prover_expr::ProverExpr},
    layer::LayerId,
};
use remainder_shared_types::{field::ExtensionField, Field};

/// An implementation of an [Mle] using a dense representation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(bound = "F: Field")]
pub struct DenseMle<F: Field> {
    /// The ID of the layer this data belongs to.
    pub layer_id: LayerId,
    /// A representation of the MLE on its current state.
    pub mle: MultilinearExtension<F>,
    /// The MleIndices `current_mle`.
    pub mle_indices: Vec<MleIndex<F>>,
}

impl<F: Field> Mle<F> for DenseMle<F> {
    fn num_free_vars(&self) -> usize {
        self.mle.num_vars()
    }

    fn get_padded_evaluations(&self) -> Vec<F> {
        let size: usize = 1 << self.mle.num_vars();
        let padding = size - self.mle.len();

        self.mle.iter().chain(repeat_n(F::ZERO, padding)).collect()
    }

    fn add_prefix_bits(&mut self, mut new_bits: Vec<MleIndex<F>>) {
        new_bits.extend(self.mle_indices.clone());
        self.mle_indices.clone_from(&new_bits);
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn len(&self) -> usize {
        self.mle.len()
    }

    fn iter(&self) -> EvaluationsIterator<F> {
        self.mle.iter()
    }

    fn mle_indices(&self) -> &[MleIndex<F>] {
        &self.mle_indices
    }

    fn fix_variable_at_index(&mut self, indexed_bit_index: usize, point: F) -> Option<RawClaim<F>> {
        // Bind the `MleIndex::IndexedBit(index)` to the challenge `point`.

        // First, find the bit corresponding to `index` and compute its absolute
        // index. For example, if `mle_indices` is equal to
        // `[MleIndex::Fixed(0), MleIndex::Bound(42, 0), MleIndex::IndexedBit(1),
        // MleIndex::Bound(17, 2) MleIndex::IndexedBit(3))]`
        // then `fix_variable_at_index(3, r)` will fix `IndexedBit(3)`, which is
        // the 2nd indexed bit, to `r`

        // Count of the bit we're fixing. In the above example
        // `bit_count == 2`.
        let (index_found, bit_count) =
            self.mle_indices
                .iter_mut()
                .fold((false, 0), |state, mle_index| {
                    if state.0 {
                        // Index already found; do nothing.
                        state
                    } else if let MleIndex::Indexed(current_bit_index) = *mle_index {
                        if current_bit_index == indexed_bit_index {
                            // Found the indexed bit in the current index;
                            // bind it and increment the bit count.
                            mle_index.bind_index(point);
                            (true, state.1 + 1)
                        } else {
                            // Index not yet found but this is an indexed
                            // bit; increasing bit count.
                            (false, state.1 + 1)
                        }
                    } else {
                        // Index not yet found but the current bit is not an
                        // indexed bit; do nothing.
                        state
                    }
                });

        assert!(index_found);
        debug_assert!(1 <= bit_count && bit_count <= self.num_free_vars());

        self.mle.fix_variable_at_index(bit_count - 1, point);

        if self.is_fully_bounded() {
            let fixed_claim_return = RawClaim::new(
                self.mle_indices
                    .iter()
                    .map(|index| index.val().unwrap())
                    .collect_vec(),
                self.mle.value(),
            );
            Some(fixed_claim_return)
        } else {
            None
        }
    }

    /// Bind the bit `index` to the value `binding`.
    /// If this was the last unbound variable, then return a Claim object giving the fully specified
    /// evaluation point and the (single) value of the bookkeeping table.  Otherwise, return None.
    fn fix_variable(&mut self, index: usize, binding: F) -> Option<RawClaim<F>> {
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::Indexed(index) {
                mle_index.bind_index(binding);
            }
        }
        // Update the bookkeeping table.
        self.mle.fix_variable(binding);

        if self.is_fully_bounded() {
            let fixed_claim_return = RawClaim::new(
                self.mle_indices
                    .iter()
                    .map(|index| index.val().unwrap())
                    .collect_vec(),
                self.mle.value(),
            );
            Some(fixed_claim_return)
        } else {
            None
        }
    }

    fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        let mut new_indices = 0;
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::Free {
                *mle_index = MleIndex::Indexed(curr_index + new_indices);
                new_indices += 1;
            }
        }

        curr_index + new_indices
    }

    fn get_enum(self) -> MleEnum<F> {
        MleEnum::Dense(self)
    }

    fn get(&self, index: usize) -> Option<F> {
        self.mle.get(index)
    }

    fn first(&self) -> F {
        self.mle.first()
    }

    fn value(&self) -> F {
        self.mle.value()
    }
}

impl<F: Field> DenseMle<F> {
    /// Constructs a new `DenseMle` with specified prefix_bits
    /// todo: change this to create a DenseMle with already specified IndexedBits
    pub fn new_with_prefix_bits(
        data: MultilinearExtension<F>,
        layer_id: LayerId,
        prefix_bits: Vec<bool>,
    ) -> Self {
        let free_bits = data.num_vars();

        let mle_indices: Vec<MleIndex<F>> = prefix_bits
            .into_iter()
            .map(|bit| MleIndex::Fixed(bit))
            .chain((0..free_bits).map(|_| MleIndex::Free))
            .collect();
        Self {
            layer_id,
            mle: data,
            mle_indices,
        }
    }

    /// Constructs a new `DenseMle` with specified MLE indices, normally when we are
    /// trying to construct a new MLE based off of a previous MLE, such as in
    /// [crate::layer::matmult::MatMult], but want to preserve the "prefix vars."
    ///
    /// The MLE should not have ever been mutated if this function is ever called, so none of the
    /// indices should ever be Indexed here.
    pub fn new_with_indices(data: &[F], layer_id: LayerId, mle_indices: &[MleIndex<F>]) -> Self {
        let mut mle = DenseMle::new_from_raw(data.to_vec(), layer_id);

        let all_indices_free_or_fixed = mle_indices.iter().all(|index| {
            index == &MleIndex::Free
                || index == &MleIndex::Fixed(true)
                || index == &MleIndex::Fixed(false)
        });
        assert!(all_indices_free_or_fixed);

        mle.mle_indices = mle_indices.to_vec();
        mle
    }

    /// Constructs a new `DenseMle` from a bookkeeping table represented by
    /// [`Iterator<Item = F>`] and [LayerId].
    ///
    /// # Example
    /// ```
    ///     use remainder::layer::LayerId;
    ///     use remainder_shared_types::Fr;
    ///     use remainder::mle::dense::DenseMle;
    ///
    ///     DenseMle::<Fr>::new_from_iter(vec![Fr::one()].into_iter(), LayerId::Input(0));
    /// ```
    pub fn new_from_iter(iter: impl Iterator<Item = F>, layer_id: LayerId) -> Self {
        let items = iter.collect_vec();
        let num_free_vars = log2(items.len()) as usize;

        let mle_indices: Vec<MleIndex<F>> = ((0..num_free_vars).map(|_| MleIndex::Free)).collect();

        let current_mle =
            MultilinearExtension::new_from_evals(Evaluations::<F>::new(num_free_vars, items));
        Self {
            layer_id,
            mle: current_mle,
            mle_indices,
        }
    }

    /// Constructs a new `DenseMle` from a bookkeeping table represented by
    /// [`Vec<F>`] and [LayerId].
    ///
    /// # Example
    /// ```
    ///     use remainder::layer::LayerId;
    ///     use remainder_shared_types::Fr;
    ///     use remainder::mle::dense::DenseMle;
    ///
    ///     DenseMle::<Fr>::new_from_raw(vec![Fr::one()], LayerId::Input(0));
    /// ```
    pub fn new_from_raw(items: Vec<F>, layer_id: LayerId) -> Self {
        let num_free_vars = log2(items.len()) as usize;

        let mle_indices: Vec<MleIndex<F>> = ((0..num_free_vars).map(|_| MleIndex::Free)).collect();

        let current_mle =
            MultilinearExtension::new_from_evals(Evaluations::<F>::new(num_free_vars, items));

        Self {
            layer_id,
            mle: current_mle,
            mle_indices,
        }
    }

    /// Constructs a new [DenseMle] from a [MultilinearExtension], additionally
    /// being able to specify the prefix vars and layer ID.
    ///
    /// Optionally gives back an indexed [DenseMle].
    pub fn new_from_multilinear_extension(
        mle: MultilinearExtension<F>,
        layer_id: LayerId,
        prefix_vars: Option<Vec<bool>>,
        maybe_starting_var_index: Option<usize>,
    ) -> Self {
        let mle_indices: Vec<MleIndex<F>> = prefix_vars
            .unwrap_or_default()
            .into_iter()
            .map(|prefix_var| MleIndex::Fixed(prefix_var))
            .chain((0..mle.num_vars()).map(|_| MleIndex::Free))
            .collect();
        let mut ret = Self {
            layer_id,
            mle,
            mle_indices,
        };
        if let Some(starting_var_index) = maybe_starting_var_index {
            ret.index_mle_indices(starting_var_index);
        }
        ret
    }

    /// Merges the MLEs into a single MLE by simply concatenating them.
    pub fn combine_mles(mles: Vec<DenseMle<F>>) -> DenseMle<F> {
        let first_mle_num_vars = mles[0].num_free_vars();
        let all_same_num_vars = mles
            .iter()
            .all(|mle| mle.num_free_vars() == first_mle_num_vars);
        assert!(all_same_num_vars);
        let layer_id = mles[0].layer_id;
        let mle_flattened = mles.into_iter().flat_map(|mle| mle.into_iter());

        Self::new_from_iter(mle_flattened, layer_id)
    }

    /// Creates an expression from the current MLE.
    pub fn expression(self) -> Expression<F, ProverExpr> {
        Expression::<F, ProverExpr>::mle(self)
    }

    /// Returns the evaluation challenges for a fully-bound MLE.
    ///
    /// Note that this function panics if a particular challenge is neither
    /// fixed nor bound!
    pub fn get_bound_point(&self) -> Vec<F> {
        self.mle_indices()
            .iter()
            .map(|index| match index {
                MleIndex::Bound(chal, _) => *chal,
                MleIndex::Fixed(chal) => F::from(*chal as u64),
                _ => panic!("MLE index not bound"),
            })
            .collect()
    }
}

impl<F: Field> IntoIterator for DenseMle<F> {
    type Item = F;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        // TEMPORARY: get_evals_vector()
        self.mle.iter().collect::<Vec<F>>().into_iter()
    }
}

/// Creates a new [DenseMle] which is a "bound" version of the given
/// `previous_mle` to the (extension field) `challenge` at the formal
/// variable with the `index` label.
pub fn fix_variable_to_new_dense_mle<F: Field, E: ExtensionField<F>>(
    previous_mle: &DenseMle<F>,
    index: usize,
    challenge: E,
) -> DenseMle<E> {
    let new_multilinear_extension =
        fix_variable_at_index_to_bookkeeping_table_copy(&previous_mle.mle, index, challenge);

    // We want to keep the same variables, but with the `index`-th variable
    // fixed to the given challenge.
    let new_mle_vars = fix_variable_var_conversion(&previous_mle.mle_indices, index, challenge);

    DenseMle {
        layer_id: previous_mle.layer_id,
        mle: new_multilinear_extension,
        mle_indices: new_mle_vars,
    }
}

/// Given a slice of previous MLE variables [MleIndex<F>], creates a new list of
/// [MleIndex<E>] with the (free) one labeled `index` bound to `challenge`.
pub fn fix_variable_var_conversion<F: Field, E: ExtensionField<F>>(
    prev_mle_vars: &[MleIndex<F>],
    index: usize,
    challenge: E,
) -> Vec<MleIndex<E>> {
    // We want to keep the same variables, but with the `index`-th variable
    // fixed to the given challenge.
    prev_mle_vars
        .iter()
        .map(|var| match var {
            // Previously bound challenges stay the same, but are cast into `E`
            // elements.
            MleIndex::Bound(prev_bound_challenge_val, var_idx) => {
                MleIndex::Bound((*prev_bound_challenge_val).into(), *var_idx)
            }
            // Fixed 0/1 variable values stay the same.
            MleIndex::Fixed(fixed_val) => MleIndex::Fixed(*fixed_val),
            // Free variables stay the same unless they are getting bound
            // to the current challenge.
            MleIndex::Indexed(var_idx) => {
                if *var_idx == index {
                    return MleIndex::Bound(challenge, *var_idx);
                }
                MleIndex::Indexed(*var_idx)
            }
            MleIndex::Free => panic!("We should not be here prior to indexing"),
        })
        .collect()
}

/// Simple lift from [MleIndex<F>] to [MleIndex<E>].
impl<F: Field, E: ExtensionField<F>> LiftTo<MleIndex<E>> for MleIndex<F> {
    fn lift(&self) -> MleIndex<E> {
        match self {
            MleIndex::Fixed(val) => MleIndex::Fixed(*val),
            MleIndex::Free => MleIndex::Free,
            MleIndex::Indexed(var_idx) => MleIndex::Indexed(*var_idx),
            MleIndex::Bound(base_chal, var_idx) => MleIndex::Bound((*base_chal).into(), *var_idx),
        }
    }
}
