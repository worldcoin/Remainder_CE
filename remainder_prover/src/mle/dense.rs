use std::{
    fmt::Debug,
    iter::{Cloned, Map, Zip},
    marker::PhantomData,
};

use ark_std::log2;
// use derive_more::{From, Into};
use itertools::{repeat_n, Itertools};

use rayon::prelude::ParallelIterator;
use serde::{Deserialize, Serialize};

use super::{mle_enum::MleEnum, Mle, MleAble, MleIndex, MleRef};
use crate::{
    claims::{wlx_eval::ClaimMle, Claim},
    layer::{batched::combine_mles, LayerId},
    mle::evals::{Evaluations, MultilinearExtension},
};
use crate::{
    claims::{ClaimError, YieldClaim},
    expression::{generic_expr::Expression, prover_expr::ProverExpr},
    layer::{combine_mle_refs::combine_mle_refs, LayerError},
};
use remainder_shared_types::FieldExt;

#[derive(Clone, Debug, Serialize, Deserialize)]
///An [Mle] that is dense
pub struct DenseMle<F, T: Send + Sync + Clone + Debug + MleAble<F>> {
    ///The underlying data
    pub mle: T::Repr,
    num_iterated_vars: usize,
    ///The layer_id this data belongs to
    pub layer_id: LayerId,
    ///Any prefix bits that must be added to any MleRefs yielded by this Mle
    pub prefix_bits: Option<Vec<MleIndex<F>>>,
    ///marker
    pub _marker: PhantomData<F>,
}

impl<F: FieldExt, T> Mle<F> for DenseMle<F, T>
where
    // Self: IntoIterator<Item = T> + FromIterator<T>
    T: Send + Sync + Clone + Debug + MleAble<F>,
{
    fn num_iterated_vars(&self) -> usize {
        self.num_iterated_vars
    }
    fn get_padded_evaluations(&self) -> Vec<F> {
        T::get_padded_evaluations(&self.mle)
    }

    fn set_prefix_bits(&mut self, new_bits: Option<Vec<MleIndex<F>>>) {
        self.prefix_bits = new_bits;
    }

    fn get_prefix_bits(&self) -> Option<Vec<MleIndex<F>>> {
        self.prefix_bits.clone()
    }
}

impl<F: FieldExt, T: Send + Sync + Clone + Debug + MleAble<F>> DenseMle<F, T> {
    pub fn new_from_iter(
        iter: impl Iterator<Item = T>,
        layer_id: LayerId,
        prefix_bits: Option<Vec<MleIndex<F>>>,
    ) -> Self {
        let items = T::from_iter(iter);
        let num_vars = T::num_vars(&items);
        Self {
            mle: items,
            num_iterated_vars: num_vars,
            layer_id,
            prefix_bits,
            _marker: PhantomData,
        }
    }

    pub fn new_from_raw(
        items: T::Repr,
        layer_id: LayerId,
        prefix_bits: Option<Vec<MleIndex<F>>>,
    ) -> Self {
        let num_vars = T::num_vars(&items);
        Self {
            mle: items,
            num_iterated_vars: num_vars,
            layer_id,
            prefix_bits,
            _marker: PhantomData,
        }
    }
}

impl<'a, F: FieldExt, T: Send + Sync + Clone + Debug + MleAble<F>> IntoIterator
    for &'a DenseMle<F, T>
{
    type Item = T;

    type IntoIter = T::IntoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        T::to_iter(&self.mle)
    }
}

// impl<F: FieldExt> FromIterator<F> for DenseMle<F, F> {
//     fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
//         let evaluations = iter.into_iter().collect_vec();

//         let num_vars = log2(evaluations.len()) as usize;

//         Self {
//             mle: evaluations,
//             num_vars,
//             layer_id: None,
//             prefix_bits: None,
//             _marker: PhantomData,
//         }
//     }
// }

/// Takes the individual bookkeeping tables from the MleRefs within an MLE
/// and merges them with padding, using a little-endian representation
/// merge strategy. Assumes that ALL MleRefs are the same size.
pub fn get_padded_evaluations_for_list<F: FieldExt, const L: usize>(items: &[Vec<F>; L]) -> Vec<F> {
    // --- All the items within should be the same size ---
    let max_size = items.iter().map(|mle_ref| mle_ref.len()).max().unwrap();

    let part_size = 1 << log2(max_size);
    let part_count = 2_u32.pow(log2(L)) as usize;

    // --- Number of "part" slots which need to filled with padding ---
    let padding_count = part_count - L;
    let total_size = part_size * part_count;
    let total_padding: usize = total_size - max_size * part_count;

    // items.into_iter().cloned().map(|mut items| {
    //     let padding = part_size - items.len();
    //     items.extend(repeat_n(F::zero(), padding));
    //     items
    // }).flatten().chain(repeat_n(F::zero(), padding_count * part_size)).collect()

    (0..max_size)
        .flat_map(|index| {
            items
                .iter()
                .map(move |item| *item.get(index).unwrap_or(&F::zero()))
                .chain(repeat_n(F::zero(), padding_count))
        })
        .chain(repeat_n(F::zero(), total_padding))
        .collect()
}

impl<F: FieldExt> MleAble<F> for F {
    type Repr = Vec<F>;
    type IntoIter<'a> = Cloned<std::slice::Iter<'a, F>>;

    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F> {
        let size: usize = 1 << log2(items.len());
        let padding = size - items.len();

        items
            .iter()
            .cloned()
            .chain(repeat_n(F::zero(), padding))
            .collect()
    }

    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr {
        iter.into_iter().collect_vec()
    }

    fn to_iter(items: &Self::Repr) -> Self::IntoIter<'_> {
        items.iter().cloned()
    }

    fn num_vars(items: &Self::Repr) -> usize {
        log2(items.len()) as usize
    }
}

impl<F: FieldExt> DenseMle<F, F> {
    ///Creates a flat DenseMle from a Vec<F>
    // pub fn new(mle: Vec<F>) -> Self {
    //     let num_vars = log2(mle.len()) as usize;
    //     Self {
    //         mle,
    //         num_vars,
    //         layer_id: None,
    //         prefix_bits: None,
    //         _marker: PhantomData,
    //     }
    // }

    ///Creates a DenseMleRef from this DenseMle
    pub fn mle_ref(&self) -> DenseMleRef<F> {
        let mle_indices: Vec<MleIndex<F>> = self
            .prefix_bits
            .clone()
            .into_iter()
            .flatten()
            .chain((0..self.num_iterated_vars()).map(|_| MleIndex::Iterated))
            .collect();

        let mle: MultilinearExtension<F> = MultilinearExtension::new(Evaluations::<F>::new(
            self.num_iterated_vars,
            self.mle.clone(),
        ));

        DenseMleRef {
            current_mle: mle.clone(),
            original_mle: mle,
            mle_indices: mle_indices.clone(),
            original_mle_indices: mle_indices,
            layer_id: self.layer_id,
            indexed: false,
        }
    }

    ///Splits the mle into a new mle with a tuple of size 2 as it's element
    pub fn split(&self, padding: F) -> DenseMle<F, Tuple2<F>> {
        DenseMle::new_from_iter(
            self.mle
                .chunks(2)
                .map(|items| (items[0], items.get(1).cloned().unwrap_or(padding)).into()),
            self.layer_id,
            self.prefix_bits.clone(),
        )
    }

    ///Splits the mle into a new mle with a tuple of size 2 as it's element
    pub fn split_tree(&self, num_split: usize) -> DenseMle<F, TupleTree<F>> {
        let mut first_half = vec![];
        let mut second_half = vec![];
        self.mle
            .clone()
            .into_iter()
            .enumerate()
            .for_each(|(idx, elem)| {
                if (idx % (num_split * 2)) < (num_split) {
                    first_half.push(elem);
                } else {
                    second_half.push(elem);
                }
            });

        DenseMle::new_from_raw(
            [first_half, second_half],
            self.layer_id,
            self.prefix_bits.clone(),
        )
    }

    pub fn one(
        mle_len: usize,
        layer_id: LayerId,
        prefix_bits: Option<Vec<MleIndex<F>>>,
    ) -> DenseMle<F, F> {
        let mut one_vec = vec![];
        for _ in 0..mle_len {
            one_vec.push(F::one())
        }
        DenseMle::new_from_raw(one_vec, layer_id, prefix_bits)
    }

    /// To combine a batch of `DenseMle<F, F>` into a single `DenseMle<F, F>`
    /// appropriately, such that the bit ordering is (batched_bits, (mle_ref_bits), iterated_bits)
    ///
    /// TODO!(ende): refactor
    pub fn combine_mle_batch(mle_batch: Vec<DenseMle<F, F>>) -> DenseMle<F, F> {
        let batched_bits = log2(mle_batch.len());

        let mle_batch_ref_combined = mle_batch.into_iter().map(|x| x.mle_ref()).collect_vec();

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

#[derive(Debug, Clone)]
///Newtype around a tuple of field elements
pub struct Tuple2<F: FieldExt>(pub (F, F));

impl<F: FieldExt> MleAble<F> for Tuple2<F> {
    type Repr = [Vec<F>; 2];

    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F> {
        get_padded_evaluations_for_list(items)
    }

    type IntoIter<'a> = Map<Zip<std::slice::Iter<'a, F>, std::slice::Iter<'a, F>>, fn((&F, &F)) -> Self> where Self: 'a;

    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr {
        let iter = iter.into_iter();
        let (first, second): (Vec<F>, Vec<F>) = iter.map(|x| (x.0 .0, x.0 .1)).unzip();
        [first, second]
    }

    fn to_iter(items: &Self::Repr) -> Self::IntoIter<'_> {
        items[0]
            .iter()
            .zip(items[1].iter())
            .map(|(first, second)| Tuple2((*first, *second)))
    }

    fn num_vars(items: &Self::Repr) -> usize {
        log2(items[0].len() + items[1].len()) as usize
    }
}

impl<F: FieldExt> From<(F, F)> for Tuple2<F> {
    fn from(value: (F, F)) -> Self {
        Self(value)
    }
}

//TODO!(Fix this so that it clones less)
// impl<'a, F: FieldExt> IntoIterator for &'a DenseMle<F, Tuple2<F>> {
//     type Item = (F, F);

//     type IntoIter = Zip<Cloned<std::slice::Iter<'a, F>>, Cloned<std::slice::Iter<'a, F>>>;

//     fn into_iter(self) -> Self::IntoIter {
//         self.mle[0].iter().cloned().zip(self.mle[1].iter().cloned())
//     }
// }

// impl<F: FieldExt> FromIterator<Tuple2<F>> for DenseMle<F, Tuple2<F>> {
//     fn from_iter<T: IntoIterator<Item = Tuple2<F>>>(iter: T) -> Self {
//         let iter = iter.into_iter();
//         let (first, second): (Vec<F>, Vec<F>) = iter.map(|x| (x.0 .0, x.0 .1)).unzip();

//         let num_vars: usize = log2(first.len() + second.len()) as usize;

//         Self {
//             mle: [first, second],
//             num_vars,
//             layer_id: None,
//             prefix_bits: None,
//             _marker: PhantomData,
//         }
//     }
// }

impl<F: FieldExt> DenseMle<F, Tuple2<F>> {
    ///Gets an MleRef to the first element in the tuple
    pub fn first(&'_ self) -> DenseMleRef<F> {
        // --- Number of *remaining* iterated variables ---
        let new_num_iterated_vars = self.num_iterated_vars - 1;

        let mle_indices = self
            .prefix_bits
            .clone()
            .into_iter()
            .flatten()
            .chain(
                std::iter::once(MleIndex::Fixed(false))
                    .chain(repeat_n(MleIndex::Iterated, new_num_iterated_vars)),
            )
            .collect_vec();

        let mle =
            MultilinearExtension::new(Evaluations::new(new_num_iterated_vars, self.mle[0].clone()));

        DenseMleRef {
            current_mle: mle.clone(),
            original_mle: mle,
            mle_indices: mle_indices.clone(),
            original_mle_indices: mle_indices,
            layer_id: self.layer_id,
            indexed: false,
        }
    }

    ///Gets an MleRef to the second element in the tuple
    pub fn second(&'_ self) -> DenseMleRef<F> {
        let new_num_iterated_vars = self.num_iterated_vars - 1;
        let mle_indices = self
            .prefix_bits
            .clone()
            .into_iter()
            .flatten()
            .chain(
                std::iter::once(MleIndex::Fixed(true))
                    .chain(repeat_n(MleIndex::Iterated, new_num_iterated_vars)),
            )
            .collect_vec();

        let mle =
            MultilinearExtension::new(Evaluations::new(new_num_iterated_vars, self.mle[1].clone()));
        DenseMleRef {
            current_mle: mle.clone(),
            original_mle: mle,
            mle_indices: mle_indices.clone(),
            original_mle_indices: mle_indices,
            layer_id: self.layer_id,
            indexed: false,
        }
    }

    /// To combine a batch of `DenseMle<F, Tuple2<F>>` into a single `DenseMle<F, F>`
    /// appropriately, such that the bit ordering is (batched_bits, mle_ref_bits, iterated_bits)
    ///
    /// TODO!(ende): refactor
    pub fn combine_mle_batch(tuple2_mle_batch: Vec<DenseMle<F, Tuple2<F>>>) -> DenseMle<F, F> {
        let batched_bits = log2(tuple2_mle_batch.len());

        let tuple2_mle_batch_ref_combined = tuple2_mle_batch
            .into_iter()
            .map(|x| combine_mle_refs(vec![x.first(), x.second()]).mle_ref())
            .collect_vec();

        let tuple2_mle_batch_ref_combined_ref =
            combine_mles(tuple2_mle_batch_ref_combined, batched_bits as usize);

        DenseMle::new_from_raw(
            tuple2_mle_batch_ref_combined_ref
                .current_mle
                .get_evals_vector()
                .clone(),
            LayerId::Input(0),
            None,
        )
    }
}

#[derive(Debug, Clone)]
///Newtype around a tuple of field elements
pub struct TupleTree<F: FieldExt>(pub (F, F));

impl<F: FieldExt> MleAble<F> for TupleTree<F> {
    type Repr = [Vec<F>; 2];

    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F> {
        get_padded_evaluations_for_list(items)
    }

    type IntoIter<'a> = Map<Zip<std::slice::Iter<'a, F>, std::slice::Iter<'a, F>>, fn((&F, &F)) -> Self> where Self: 'a;

    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr {
        let iter = iter.into_iter();
        let (first, second): (Vec<F>, Vec<F>) = iter.map(|x| (x.0 .0, x.0 .1)).unzip();
        [first, second]
    }

    fn to_iter(items: &Self::Repr) -> Self::IntoIter<'_> {
        items[0]
            .iter()
            .zip(items[1].iter())
            .map(|(first, second)| TupleTree((*first, *second)))
    }

    fn num_vars(items: &Self::Repr) -> usize {
        log2(items[0].len() + items[1].len()) as usize
    }
}

impl<F: FieldExt> From<(F, F)> for TupleTree<F> {
    fn from(value: (F, F)) -> Self {
        Self(value)
    }
}

impl<F: FieldExt> DenseMle<F, TupleTree<F>> {
    ///Gets an MleRef to the first element in the tuple
    pub fn first(&'_ self, splitter: usize) -> DenseMleRef<F> {
        // --- Number of *remaining* iterated variables ---
        let new_num_iterated_vars = self.num_iterated_vars - 1;

        let mle_indices = self
            .prefix_bits
            .clone()
            .into_iter()
            .flatten()
            .chain(repeat_n(MleIndex::Iterated, splitter).chain(
                std::iter::once(MleIndex::Fixed(false)).chain(repeat_n(
                    MleIndex::Iterated,
                    new_num_iterated_vars - splitter,
                )),
            ))
            .collect_vec();

        let mle =
            MultilinearExtension::new(Evaluations::new(new_num_iterated_vars, self.mle[0].clone()));

        DenseMleRef {
            current_mle: mle.clone(),
            original_mle: mle,
            mle_indices: mle_indices.clone(),
            original_mle_indices: mle_indices,
            layer_id: self.layer_id,
            indexed: false,
        }
    }

    ///Gets an MleRef to the second element in the tuple
    pub fn second(&'_ self, splitter: usize) -> DenseMleRef<F> {
        let new_num_iterated_vars = self.num_iterated_vars - 1;
        let mle_indices = self
            .prefix_bits
            .clone()
            .into_iter()
            .flatten()
            .chain(repeat_n(MleIndex::Iterated, splitter).chain(
                std::iter::once(MleIndex::Fixed(true)).chain(repeat_n(
                    MleIndex::Iterated,
                    new_num_iterated_vars - splitter,
                )),
            ))
            .collect_vec();

        let mle =
            MultilinearExtension::new(Evaluations::new(new_num_iterated_vars, self.mle[1].clone()));

        DenseMleRef {
            current_mle: mle.clone(),
            original_mle: mle,
            mle_indices: mle_indices.clone(),
            original_mle_indices: mle_indices,
            layer_id: self.layer_id,
            indexed: false,
        }
    }
}

// --------------------------- MleRef stuff ---------------------------

/// An [MleRef] that is dense
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct DenseMleRef<F: FieldExt> {
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

    // /// Number of non-fixed variables within this MLE
    // /// (warning: this gets modified destructively DURING sumcheck)
    // pub num_vars: usize,
    // /// Number of non-fixed variables originally, doesn't get modifier
    // pub original_num_vars: usize,
    /// The layer this MleRef is a reference to.
    pub layer_id: LayerId,
    /// A marker that keeps track of if this MleRef is indexed.
    pub indexed: bool,
}

impl<F: FieldExt> DenseMleRef<F> {
    ///Convienence function for wrapping this in an Expression
    pub fn expression(self) -> Expression<F, ProverExpr> {
        Expression::mle(self)
    }

    pub fn num_vars(&self) -> usize {
        self.current_mle.num_vars()
    }

    pub fn original_num_vars(&self) -> usize {
        self.original_mle.num_vars()
    }
}

impl<F: FieldExt> MleRef for DenseMleRef<F> {
    type F = F;

    fn bookkeeping_table(&self) -> &[F] {
        self.current_mle.get_evals_vector()
    }

    fn original_bookkeeping_table(&self) -> &[Self::F] {
        self.original_mle.get_evals_vector()
    }

    fn mle_indices(&self) -> &[MleIndex<Self::F>] {
        &self.mle_indices
    }

    fn original_mle_indices(&self) -> &Vec<MleIndex<Self::F>> {
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

    fn fix_variable_at_index(
        &mut self,
        indexed_bit_index: usize,
        point: Self::F,
    ) -> Option<Claim<Self::F>> {
        // Bind the `MleIndex::IndexedBit(index)` to the challenge `point`.

        // First, find the bit corresponding to `index` and compute its absolute
        // index. For example, if `mle_indices` is equal to
        // `[MleIndex::Fixed(0), MleIndex::Bound(42, 0), MleIndex::IndexedBit(1), MleIndex::Bound(17, 2) MleIndex::IndexedBit(3))]`
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
    fn fix_variable(&mut self, round_index: usize, challenge: Self::F) -> Option<Claim<Self::F>> {
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

    fn push_mle_indices(&mut self, new_indices: &[MleIndex<Self::F>]) {
        self.mle_indices.append(&mut new_indices.to_vec());
    }

    fn get_enum(self) -> MleEnum<Self::F> {
        MleEnum::Dense(self)
    }
}

impl<F: FieldExt> YieldClaim<F, ClaimMle<F>> for DenseMleRef<F> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use remainder_shared_types::Fr;

    // ======== `fix_variable` tests ========

    #[test]
    ///test fixing variables in an mle with two variables
    fn fix_variable_twovars() {
        let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
        let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.fix_variable(1, Fr::from(1));

        let mle_vec_exp = vec![Fr::from(2), Fr::from(3)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);
        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
    }
    #[test]
    ///test fixing variables in an mle with three variables
    fn fix_variable_threevars() {
        let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.fix_variable(1, Fr::from(3));

        let mle_vec_exp = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);
        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
    }

    #[test]
    ///test nested fixing variables in an mle with three variables
    fn fix_variable_nested() {
        let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.fix_variable(1, Fr::from(3));
        mle_ref.fix_variable(2, Fr::from(2));

        let mle_vec_exp = vec![Fr::from(6), Fr::from(11)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);
        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
    }

    #[test]
    ///test nested fixing all the wayyyy
    fn fix_variable_full() {
        let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        let _ = mle_ref.index_mle_indices(0);
        mle_ref.fix_variable(0, Fr::from(3));
        mle_ref.fix_variable(1, Fr::from(2));
        mle_ref.fix_variable(2, Fr::from(4));

        let mle_vec_exp = vec![Fr::from(26)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);
        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
    }

    // ======== `fix_variable_at_index` tests ========

    #[test]
    ///test fixing variables in an mle with two variables
    fn smart_fix_variable_two_vars_forward() {
        let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.index_mle_indices(0);

        // Fix 1st variable to 1.
        mle_ref.fix_variable_at_index(0, Fr::from(1));

        let mle_vec_exp = vec![Fr::from(2), Fr::from(3)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 2nd variable to 1.
        mle_ref.fix_variable_at_index(1, Fr::from(1));

        let mle_vec_exp = vec![Fr::from(3)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
    }

    #[test]
    fn smart_fix_variable_two_vars_backwards() {
        let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.index_mle_indices(0);

        // Fix 2nd variable to 1.
        mle_ref.fix_variable_at_index(1, Fr::from(1));

        let mle_vec_exp = vec![Fr::from(1), Fr::from(3)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 1st variable to 1.
        mle_ref.fix_variable_at_index(0, Fr::from(1));

        let mle_vec_exp = vec![Fr::from(3)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
    }

    #[test]
    ///test fixing variables in an mle with three variables
    fn smart_fix_variable_three_vars_123() {
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.index_mle_indices(0);

        // Fix 1st variable to 3.
        mle_ref.fix_variable_at_index(0, Fr::from(3));

        let mle_vec_exp = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 2nd variable to 4.
        mle_ref.fix_variable_at_index(1, Fr::from(4));

        let mle_vec_exp = vec![Fr::from(6), Fr::from(13)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 3rd variable to 5.
        mle_ref.fix_variable_at_index(2, Fr::from(5));

        let mle_vec_exp = vec![Fr::from(41)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
    }

    #[test]
    ///test fixing variables in an mle with three variables
    fn smart_fix_variable_three_vars_132() {
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.index_mle_indices(0);

        // Fix 1st variable to 3.
        mle_ref.fix_variable_at_index(0, Fr::from(3));

        let mle_vec_exp = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 3rd variable to 5.
        mle_ref.fix_variable_at_index(2, Fr::from(5));

        let mle_vec_exp = vec![Fr::from(21), Fr::from(26)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 2nd variable to 4.
        mle_ref.fix_variable_at_index(1, Fr::from(4));

        let mle_vec_exp = vec![Fr::from(41)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
    }

    #[test]
    ///test fixing variables in an mle with three variables
    fn smart_fix_variable_three_vars_213() {
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.index_mle_indices(0);

        // Fix 2nd variable to 4.
        mle_ref.fix_variable_at_index(1, Fr::from(4));

        let mle_vec_exp = vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(7)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 1st variable to 3.
        mle_ref.fix_variable_at_index(0, Fr::from(3));

        let mle_vec_exp = vec![Fr::from(6), Fr::from(13)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 3rd variable to 5.
        mle_ref.fix_variable_at_index(2, Fr::from(5));

        let mle_vec_exp = vec![Fr::from(41)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
    }

    #[test]
    ///test fixing variables in an mle with three variables
    fn smart_fix_variable_three_vars_231() {
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.index_mle_indices(0);

        // Fix 2nd variable to 4.
        mle_ref.fix_variable_at_index(1, Fr::from(4));

        let mle_vec_exp = vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(7)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 3rd variable to 5.
        mle_ref.fix_variable_at_index(2, Fr::from(5));

        let mle_vec_exp = vec![Fr::from(20), Fr::from(27)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 1st variable to 3.
        mle_ref.fix_variable_at_index(0, Fr::from(3));

        let mle_vec_exp = vec![Fr::from(41)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
    }

    #[test]
    ///test fixing variables in an mle with three variables
    fn smart_fix_variable_three_vars_312() {
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.index_mle_indices(0);

        // Fix 3rd variable to 5.
        mle_ref.fix_variable_at_index(2, Fr::from(5));

        let mle_vec_exp = vec![Fr::from(0), Fr::from(7), Fr::from(5), Fr::from(12)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 1st variable to 3.
        mle_ref.fix_variable_at_index(0, Fr::from(3));

        let mle_vec_exp = vec![Fr::from(21), Fr::from(26)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 2nd variable to 4.
        mle_ref.fix_variable_at_index(1, Fr::from(4));

        let mle_vec_exp = vec![Fr::from(41)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
    }
    #[test]

    ///test fixing variables in an mle with three variables
    fn smart_fix_variable_three_vars_321() {
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.index_mle_indices(0);

        // Fix 3rd variable to 5.
        mle_ref.fix_variable_at_index(2, Fr::from(5));

        let mle_vec_exp = vec![Fr::from(0), Fr::from(7), Fr::from(5), Fr::from(12)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 2nd variable to 4.
        mle_ref.fix_variable_at_index(1, Fr::from(4));

        let mle_vec_exp = vec![Fr::from(20), Fr::from(27)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

        // Fix 1st variable to 3.
        mle_ref.fix_variable_at_index(0, Fr::from(3));

        let mle_vec_exp = vec![Fr::from(41)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
    }

    #[test]

    // ======== ========

    fn create_dense_mle_from_vec() {
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
            Fr::from(7),
        ];

        //DON'T do this normally, it clones the vec, if you have a flat MLE just use Mle::new
        let mle_iter =
            DenseMle::new_from_iter(mle_vec.clone().into_iter(), LayerId::Input(0), None);

        let mle_new: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);

        assert!(mle_iter.mle == mle_new.mle);
        assert!(
            mle_iter.num_iterated_vars() == 3 && mle_new.num_iterated_vars() == 3,
            "Num vars must be the log_2 of the length of the vector"
        );
    }

    #[test]
    fn create_dense_tuple_mle_from_vec() {
        let tuple_vec = vec![
            (Fr::from(0), Fr::from(1)),
            (Fr::from(2), Fr::from(3)),
            (Fr::from(4), Fr::from(5)),
            (Fr::from(6), Fr::from(7)),
        ];

        let mle = DenseMle::new_from_iter(
            tuple_vec.clone().into_iter().map(Tuple2::from),
            LayerId::Input(0),
            None,
        );

        let (first, second): (Vec<Fr>, Vec<_>) = tuple_vec.into_iter().unzip();

        assert!(mle.mle == [first, second]);
        assert!(mle.num_iterated_vars() == 3);
    }

    #[test]
    fn create_dense_mle_ref_from_flat_mle() {
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
            Fr::from(7),
        ];

        let mle: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec.clone(), LayerId::Input(0), None);

        let mle_ref: DenseMleRef<Fr> = mle.mle_ref();

        assert!(
            mle_ref.mle_indices == vec![MleIndex::Iterated, MleIndex::Iterated, MleIndex::Iterated]
        );
        assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_vec);
    }

    #[test]
    fn create_dense_mle_ref_from_tuple_mle() {
        let tuple_vec = vec![
            (Fr::from(0), Fr::from(1)),
            (Fr::from(2), Fr::from(3)),
            (Fr::from(4), Fr::from(5)),
            (Fr::from(6), Fr::from(7)),
        ];

        let mle = DenseMle::new_from_iter(
            tuple_vec.into_iter().map(Tuple2::from),
            LayerId::Input(0),
            None,
        );

        let first = mle.first();
        let second = mle.second();

        assert!(
            first.mle_indices
                == vec![
                    MleIndex::Fixed(false),
                    MleIndex::Iterated,
                    MleIndex::Iterated
                ]
        );
        assert!(
            second.mle_indices
                == vec![
                    MleIndex::Fixed(true),
                    MleIndex::Iterated,
                    MleIndex::Iterated
                ]
        );

        assert!(first.bookkeeping_table() == &[Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(6)]);
        assert!(
            second.bookkeeping_table() == &[Fr::from(1), Fr::from(3), Fr::from(5), Fr::from(7)]
        );
    }
}
