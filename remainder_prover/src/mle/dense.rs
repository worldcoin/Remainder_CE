#[cfg(test)]
mod tests;

use std::{
    fmt::Debug,
    iter::{Cloned, Map, Zip},
    marker::PhantomData,
};

use ark_std::log2;
use itertools::{repeat_n, Itertools};

use serde::{Deserialize, Serialize};

use super::{mle_enum::MleEnum, Mle, MleIndex};
use crate::{
    builders::layer_builder::batched::combine_mles,
    claims::{wlx_eval::ClaimMle, Claim},
    layer::LayerId,
    mle::evals::{Evaluations, MultilinearExtension},
};
use crate::{
    claims::{ClaimError, YieldClaim},
    expression::{generic_expr::Expression, prover_expr::ProverExpr},
    layer::{combine_mle_refs::combine_mle_refs, LayerError},
};
use remainder_shared_types::FieldExt;

/// An implementation of an [Mle] using a dense representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseMle<F> {
    /// The underlying data.
    pub mle: Vec<F>,

    /// Number of iterated variables.
    pub num_iterated_vars: usize,

    /// The ID of the layer this data belongs to.
    pub layer_id: LayerId,

    /// Any prefix bits that must be added to any [MleRef]s yielded by this MLE.
    pub prefix_bits: Option<Vec<MleIndex<F>>>,

    /// below are fields originally belonging to DenseMle

    /// A representation of the MLE on its current state.
    pub current_mle: MultilinearExtension<F>,
    /// The MleIndices `current_mle`.
    pub mle_indices: Vec<MleIndex<F>>,

    /// The original MLE (that does not get destructively modified
    /// when fixing a variable).
    // TODO(Makis): Need to find a way to skip only the `evals` field inside the
    // original MLE.
    // #[serde(skip)]
    // #[serde(default = "MultilinearExtension::new_zero")]
    pub original_mle: MultilinearExtension<F>,
    /// The original mle indices (not modified during fix var)
    pub original_mle_indices: Vec<MleIndex<F>>,
    /// A marker that keeps track of if this MleRef is indexed.
    pub indexed: bool,
}

impl<F: FieldExt> Mle<F> for DenseMle<F> {
    fn num_iterated_vars(&self) -> usize {
        self.num_iterated_vars
    }

    fn get_padded_evaluations(&self) -> Vec<F> {
        let size: usize = 1 << log2(self.mle.len());
        let padding = size - self.mle.len();

        self.mle
            .iter()
            .cloned()
            .chain(repeat_n(F::ZERO, padding))
            .collect()
    }

    fn set_prefix_bits(&mut self, new_bits: Option<Vec<MleIndex<F>>>) {
        self.prefix_bits = new_bits.clone();
        if let Some(mut all_bits) = new_bits {
            all_bits.extend(self.mle_indices.clone());
            self.mle_indices = all_bits.clone();
            self.original_mle_indices = all_bits;
        }
    }

    fn get_prefix_bits(&self) -> Option<Vec<MleIndex<F>>> {
        self.prefix_bits.clone()
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn bookkeeping_table(&self) -> &[F] {
        self.current_mle.get_evals_vector()
    }

    fn original_bookkeeping_table(&self) -> &[F] {
        self.original_mle.get_evals_vector()
    }

    fn mle_indices(&self) -> &[MleIndex<F>] {
        &self.mle_indices
    }

    fn original_mle_indices(&self) -> &Vec<MleIndex<F>> {
        &self.original_mle_indices
    }

    fn num_vars(&self) -> usize {
        self.current_mle.num_vars()
    }

    fn original_num_vars(&self) -> usize {
        self.original_mle.num_vars()
    }

    fn indexed(&self) -> bool {
        self.indexed
    }

    fn fix_variable_at_index(&mut self, indexed_bit_index: usize, point: F) -> Option<Claim<F>> {
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
                    } else if let MleIndex::IndexedBit(current_bit_index) = *mle_index {
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
        debug_assert!(1 <= bit_count && bit_count <= self.num_vars());

        self.current_mle.fix_variable_at_index(bit_count - 1, point);

        if self.num_vars() == 0 {
            let fixed_claim_return = Claim::new(
                self.mle_indices
                    .iter()
                    .map(|index| index.val().unwrap())
                    .collect_vec(),
                self.current_mle.value(),
            );
            Some(fixed_claim_return)
        } else {
            None
        }
    }

    /// Ryan's note -- I assume this function updates the bookkeeping tables as
    /// described by [Tha13].
    fn fix_variable(&mut self, round_index: usize, challenge: F) -> Option<Claim<F>> {
        // --- Bind the current indexed bit to the challenge value ---
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::IndexedBit(round_index) {
                mle_index.bind_index(challenge);
            }
        }

        self.current_mle.fix_variable(challenge);

        if self.num_vars() == 0 {
            let fixed_claim_return = Claim::new(
                self.mle_indices
                    .iter()
                    .map(|index| index.val().unwrap())
                    .collect_vec(),
                self.current_mle.value(),
            );
            Some(fixed_claim_return)
        } else {
            None
        }
    }

    fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        let mut new_indices = 0;
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::Iterated {
                *mle_index = MleIndex::IndexedBit(curr_index + new_indices);
                new_indices += 1;
            }
        }

        self.indexed = true;
        curr_index + new_indices
    }

    fn get_layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn push_mle_indices(&mut self, new_indices: &[MleIndex<F>]) {
        self.mle_indices.append(&mut new_indices.to_vec());
    }

    fn get_enum(self) -> MleEnum<F> {
        MleEnum::Dense(self)
    }
}

impl<F: FieldExt> YieldClaim<F, ClaimMle<F>> for DenseMle<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, crate::layer::LayerError> {
        if self.bookkeeping_table().len() != 1 {
            return Err(LayerError::ClaimError(ClaimError::MleRefMleError));
        }
        let mle_indices: Result<Vec<F>, _> = self
            .mle_indices
            .iter()
            .map(|index| {
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::MleRefMleError))
            })
            .collect();
        Ok(vec![ClaimMle::new(
            mle_indices?,
            self.bookkeeping_table()[0],
            None,
            Some(self.layer_id),
            Some(self.clone().get_enum()),
        )])
    }
}

impl<F: FieldExt> DenseMle<F> {
    /// Constructs a new `DenseMle` from an iterator over items of the [MleAble]
    /// type `T`.
    ///
    /// # Example
    /// ```
    ///     use remainder::layer::LayerId;
    ///     use remainder_shared_types::Fr;
    ///     use remainder::mle::dense::DenseMle;
    ///
    ///     DenseMle::<Fr>::new_from_iter(vec![Fr::one()].into_iter(), LayerId::Input(0), None);
    /// ```
    pub fn new_from_iter(
        iter: impl Iterator<Item = F>,
        layer_id: LayerId,
        prefix_bits: Option<Vec<MleIndex<F>>>,
    ) -> Self {
        let items = iter.collect_vec();
        let num_iterated_vars = log2(items.len()) as usize;

        let mle_indices: Vec<MleIndex<F>> = prefix_bits
            .clone()
            .into_iter()
            .flatten()
            .chain((0..num_iterated_vars).map(|_| MleIndex::Iterated))
            .collect();

        let current_mle =
            MultilinearExtension::new(Evaluations::<F>::new(num_iterated_vars, items.clone()));
        Self {
            mle: items,
            num_iterated_vars,
            layer_id,
            prefix_bits,
            current_mle: current_mle.clone(),
            mle_indices: mle_indices.clone(),
            original_mle: current_mle,
            original_mle_indices: mle_indices,
            indexed: false,
        }
    }

    /// Constructs a new `DenseMle` from any valid representation of the
    /// [MleAble] type `T`.
    ///
    /// # Example
    /// ```
    ///     use remainder::layer::LayerId;
    ///     use remainder_shared_types::Fr;
    ///     use remainder::mle::dense::DenseMle;
    ///
    ///     DenseMle::<Fr>::new_from_raw(vec![Fr::one()], LayerId::Input(0), None);
    /// ```
    pub fn new_from_raw(
        items: Vec<F>,
        layer_id: LayerId,
        prefix_bits: Option<Vec<MleIndex<F>>>,
    ) -> Self {
        let num_iterated_vars = log2(items.len()) as usize;

        let mle_indices: Vec<MleIndex<F>> = prefix_bits
            .clone()
            .into_iter()
            .flatten()
            .chain((0..num_iterated_vars).map(|_| MleIndex::Iterated))
            .collect();

        let current_mle =
            MultilinearExtension::new(Evaluations::<F>::new(num_iterated_vars, items.clone()));

        Self {
            mle: items,
            num_iterated_vars,
            layer_id,
            prefix_bits,
            current_mle: current_mle.clone(),
            mle_indices: mle_indices.clone(),
            original_mle: current_mle,
            original_mle_indices: mle_indices,
            indexed: false,
        }
    }

    pub fn expression(self) -> Expression<F, ProverExpr> {
        Expression::mle(self)
    }
}

impl<F: FieldExt> IntoIterator for DenseMle<F> {
    type Item = F;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.mle.into_iter()
    }
}

/// Takes the individual bookkeeping tables from the `MleRefs` within an MLE and
/// merges them with padding, using a little-endian representation merge
/// strategy.
///
/// # Requires / Panics
/// *All* MleRefs should be of the same size, otherwise panics.
pub fn get_padded_evaluations_for_list<F: FieldExt, const L: usize>(items: &[Vec<F>; L]) -> Vec<F> {
    // All the items within should be the same size.
    let max_size = items.iter().map(|mle_ref| mle_ref.len()).max().unwrap();
    assert!(items.iter().all(|mle_ref| mle_ref.len() == max_size));

    let part_size = 1 << log2(max_size);
    let part_count = 2_u32.pow(log2(L)) as usize;

    // Number of "part" slots which need to filled with padding.
    let padding_count = part_count - L;
    let total_size = part_size * part_count;
    let total_padding: usize = total_size - max_size * part_count;

    (0..max_size)
        .flat_map(|index| {
            items
                .iter()
                .map(move |item| *item.get(index).unwrap_or(&F::ZERO))
                .chain(repeat_n(F::ZERO, padding_count))
        })
        .chain(repeat_n(F::ZERO, total_padding))
        .collect()
}

impl<F: FieldExt> DenseMle<F> {
    /// Splits the MLE into a new MLE with a tuple of size 2 as its element.
    pub fn split(self) -> [DenseMle<F>; 2] {
        let first_iter = self.mle.clone().into_iter().step_by(2);
        let second_iter = self.mle.into_iter().skip(1).step_by(2);

        [
            DenseMle::new_from_iter(first_iter, self.layer_id, self.prefix_bits.clone()),
            DenseMle::new_from_iter(second_iter, self.layer_id, self.prefix_bits.clone()),
        ]
    }

    /// Constructs a `DenseMle` with `mle_len` evaluations, all equal to
    /// `F::ONE`.
    pub fn one(
        mle_len: usize,
        layer_id: LayerId,
        prefix_bits: Option<Vec<MleIndex<F>>>,
    ) -> DenseMle<F> {
        let ones_vec: Vec<F> = (0..mle_len).map(|_| F::ONE).collect();
        DenseMle::new_from_raw(ones_vec, layer_id, prefix_bits)
    }

    /// Combines a batch of `DenseMle<F,>`s into a single `DenseMle<F,>`
    /// appropriately, such that the bit ordering is
    /// `(batched_bits, mle_ref_bits, iterated_bits)`.
    ///
    /// TODO!(ende): refactor
    pub fn combine_mle_batch(mle_batch: Vec<DenseMle<F>>) -> DenseMle<F> {
        let batched_bits = log2(mle_batch.len());

        let mle_batch_ref_combined = mle_batch.into_iter().map(|x| x).collect_vec();

        let mle_batch_ref_combined_ref =
            combine_mles(mle_batch_ref_combined, batched_bits as usize);

        DenseMle::new_from_raw(
            mle_batch_ref_combined_ref
                .current_mle
                .get_evals_vector()
                .clone(),
            LayerId::Input(0),
            None,
        )
    }
}
